"""One window for the whole masking pass: pick a dataset, get points, check them, run SAM3.

Points live in their own file, data/points/<dataset>.json, separate from anything Gemini
wrote, and every click is written there straight away -- there is no save button and
nothing to lose by closing the tab.

Three ways to get a point onto an item, and they mix freely:

  automatically  scripts/auto_point.py answers every episode from the gripper and from
                 what left the table. Fast and free, and wrong often enough to be worth
                 looking at, which is what the rest of this window is for.
  by clicking    scrub to a frame where the item is visible, click it. The click also
                 fixes that camera's seed frame, so each camera can be answered at its
                 own instant -- the overview one while the item is still on the table,
                 the wrist one once it is in the gripper.
  by correcting  clicks accumulate and the toggle decides what they mean, so a mask that
                 caught half the item gets the rest clicked in, and one that spilled over
                 gets it carved back out.

Under each view is what SAM3 makes of that camera's points, which needs demo/server.py
running. Without it the previews stay empty and everything else still works.

Masking is the same tracking scripts/mask.py does: the seed frame is segmented from the
points, then the episode is tracked backwards to its start and forwards to its end,
because the model has no notion of time and a reversed clip is just another clip to it.

Anything slow goes to the queue at the bottom, which takes more work at any time, running
or not: mask a dataset, move on to the next one, put that one behind it. The window stays
usable throughout. Masking goes one dataset at a time, sharing the one GPU; generating
points goes four at a time beside it, since that is decoding video. Jobs can be cancelled
one by one, and one that fails leaves the rest of the queue standing.

Reviewing a dataset that is already masked (data/masked/<name>, e.g. pulled off the Hub with
scripts/fetch_pairs.py) is the same window: the masked clip of the open episode plays at the
top, one per camera, because a wrong mask is obvious in motion and invisible in a frame.
An episode that is wrong gets flagged (saved into the points file as "bad"), its points get
clicked right on the clear frames below, and "Re-mask flagged episodes" masks only those
again and splices them into the masked copy, leaving every good episode's pixels as they
were. Flags come off when the re-mask lands and the episode is marked `remasked`, so it
can be looked at once more.

Usage: python scripts/annotate.py [--dataset data/clear/<name>]
"""
import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path

import av
import cv2
import numpy as np
import gradio as gr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import auto_point  # noqa: E402
import mask as mask_script  # noqa: E402
from correct_point import sam_preview  # noqa: E402
from find_point_with_gemini import (  # noqa: E402
    as_points, camera_name, episode_offsets, load_episodes, mark, save_preview, seed_frame,
    video_keys,
)

CLEAR_DIR = REPO_ROOT / "data" / "clear"
MASKED_DIR = REPO_ROOT / "data" / "masked"
# Browser-playable cuts of the masked copy, one per episode per camera, made on demand and
# thrown away whenever the masked copy changes.
CLIPS_DIR = REPO_ROOT / "data" / "clips"


class Cancelled(Exception):
    """Raised inside a job once it has been asked to stop."""


class Job:
    """One dataset's worth of long work: its own bar, its own stop, its own outcome.

    A job knows how much work it is (`total`, in the unit the work really spends its time
    on -- episodes read for point generation, frames written for masking) and counts its
    way there, so the bar moves smoothly and the estimate extrapolates from work really
    done rather than from a guess made before it started.
    """

    def __init__(self, ident: int, kind: str, lane: str, name: str, key: str,
                 dataset: str, total: int, run):
        self.id, self.kind, self.lane = ident, kind, lane
        self.name, self.key, self.dataset = name, key, dataset
        self.total, self.done, self.run = total, 0, run
        self.state = "queued"    # queued | running | done | failed | stopped
        self.note = ""
        self.started = self.finished = None
        self.stopping = False

    def advance(self, units: int = 1) -> None:
        """Passed into the work as its `report`. Counting and cancelling are the same call
        on purpose: anything slow enough to need a bar is slow enough to need a way out of
        it, and this is the one place every such job already passes through."""
        if self.stopping:
            raise Cancelled
        self.done += units    # only ever the job's own worker thread

    @property
    def waiting(self) -> bool:
        return self.state == "queued"

    @property
    def active(self) -> bool:
        return self.state in ("queued", "running")

    @property
    def share(self) -> float:
        if self.state == "done":
            return 1.0
        return max(0.0, min(1.0, self.done / self.total)) if self.total else 0.0

    @property
    def left(self) -> float | None:
        """Seconds still to go, or None while there is nothing to extrapolate from."""
        if self.state != "running" or not self.done or not self.started:
            return None
        return (time.time() - self.started) / self.done * max(0, self.total - self.done)

    def words(self) -> str:
        if self.state == "queued":
            return f"queued -- {self.total:,} to do"
        if self.state == "running":
            left = self.left
            return (f"{self.done:,} of {self.total:,} ({self.share:.0%})"
                    + (f", {duration(left)} left" if left else ", measuring"))
        spent = duration((self.finished or time.time()) - self.started) if self.started else ""
        return self.note or (f"done in {spent}" if spent else self.state)


class Queue:
    """Every long job of the session, in the order they were asked for.

    Work can be added at any time, including while something is running: masking one
    dataset, moving on to the next and putting that one behind it is how this window is
    meant to be used, and nothing here ever refuses work because it is busy.

    Two rails, because the two kinds of work are held up by different things. Masking is
    the GPU, shared with the previews, so one at a time -- four streams at once only make
    all four slower. Generating points is decoding video, which is per core, so four at
    once. A mask job and a point job never wait for each other, which is what lets a
    dataset be prepared while another is being masked.

    Jobs of this session are kept after they finish so the window can show what happened,
    and the whole thing lives in memory: closing the process drops the queue with it.
    """

    LANES = {"mask": 1, "points": 4}

    def __init__(self):
        self.cond = threading.Condition()
        self.jobs = []
        self.next_id = 1
        self.staffed = set()

    # ------------------------------------------------------------------ putting work in
    def submit(self, kind: str, lane: str, entries: list) -> tuple:
        """entries: [{"name", "key", "dataset", "total", "run"}]. `key` is what a dataset
        cannot be queued twice under, so pressing "Mask every dataset" twice adds the ones
        that are new and leaves the rest where they are in the line.

        Returns (jobs added, names refused as already queued)."""
        added, dupes = [], []
        with self.cond:
            taken = {j.key for j in self.jobs if j.active}
            for entry in entries:
                if entry["key"] in taken:
                    dupes.append(entry["name"])
                    continue
                taken.add(entry["key"])
                job = Job(self.next_id, kind, lane, entry["name"], entry["key"],
                          entry["dataset"], entry["total"], entry["run"])
                self.next_id += 1
                self.jobs.append(job)
                added.append(job)
            self.cond.notify_all()
        self.staff(lane)
        return added, dupes

    def staff(self, lane: str) -> None:
        """Start that rail's workers, once per session. They are daemons and they idle on
        the condition when there is nothing to do, so an empty queue costs nothing."""
        with self.cond:
            if lane in self.staffed:
                return
            self.staffed.add(lane)
            workers = self.LANES[lane]
        for _ in range(workers):
            threading.Thread(target=self.work, args=(lane,), daemon=True).start()

    def work(self, lane: str) -> None:
        while True:
            with self.cond:
                job = None
                while job is None:
                    job = next((j for j in self.jobs if j.lane == lane and j.waiting), None)
                    if job is None:
                        self.cond.wait()
                job.state, job.started = "running", time.time()
            self.run_one(job)
            with self.cond:
                self.cond.notify_all()

    def run_one(self, job: Job) -> None:
        try:
            job.run(job.advance)
            job.state = "done"
            job.note = f"done in {duration(time.time() - job.started)}"
        except Cancelled:
            job.state, job.note = "stopped", "stopped, and left unfinished"
        except (Exception, SystemExit) as exc:
            # SystemExit and not just Exception: mask.py is a script as well as a library
            # and says no that way, and one dataset saying no must leave the queue standing.
            job.state = "failed"
            job.note = f"**failed** -- {type(exc).__name__}: {exc}"
        finally:
            job.finished = time.time()

    # ------------------------------------------------------------------ taking work out
    def cancel(self, ids) -> int:
        """A job still waiting is dropped and never starts; the one running is asked to
        stop, which it does at its next episode, leaving that dataset unfinished. Either
        way the rest of the queue carries on."""
        wanted = {int(i) for i in ids}
        with self.cond:
            hit = [j for j in self.jobs if j.id in wanted and j.active]
            for job in hit:
                if job.waiting:
                    job.state, job.note = "stopped", "dropped before it started"
                    job.finished = time.time()
                else:
                    job.stopping = True
        return len(hit)

    def drop_waiting(self) -> int:
        with self.cond:
            ids = [j.id for j in self.jobs if j.waiting]
        return self.cancel(ids)

    def stop_all(self) -> int:
        with self.cond:
            ids = [j.id for j in self.jobs if j.active]
        return self.cancel(ids)

    # ---------------------------------------------------------------------- looking at it
    def active(self) -> list:
        with self.cond:
            return [j for j in self.jobs if j.active]

    def touching(self, dataset: str) -> bool:
        """Is anything still going to write this dataset? The window waits for that before
        picking its points file back up off disk."""
        return any(j.dataset == dataset for j in self.active())

    def choices(self) -> list:
        """(label, id) of everything that can still be cancelled."""
        return [(f"{j.name} -- {j.state}", j.id) for j in self.active()]

    def view(self) -> tuple:
        """(html table, one line of words) for the window to show. Everything still going
        or waiting, and the last few that finished, because a failure has to be readable
        after the fact -- this is the only place a job's outcome is reported."""
        with self.cond:
            jobs = list(self.jobs)
        active = [j for j in jobs if j.active]
        recent = [j for j in jobs if not j.active][-4:]
        if not active and not recent:
            return "", ""
        rows = "".join(row(j) for j in active + recent)
        table = ('<table style="width:100%;border-collapse:collapse;font-size:0.9em">'
                 + rows + "</table>")
        running = [j for j in active if j.state == "running"]
        if not active:
            return table, "the queue is empty"
        done = sum(j.done for j in active)
        total = sum(j.total for j in active)
        lefts = [j.left for j in running if j.left]
        # The mask rail is a line, so its jobs' times add up; the point rail runs several
        # at once, and the longest of those is what the queue is really waiting for.
        ahead = sum(j.total - j.done for j in active if j.lane == "mask")
        rate = max(((j.done / (time.time() - j.started)) for j in running
                    if j.lane == "mask" and j.done), default=0)
        eta = max(lefts, default=0)
        if rate:
            eta = max(eta, ahead / rate)
        share = f"{done / total:.0%}" if total else "0%"
        return table, (f"**{len(running)} running, {len(active) - len(running)} queued** -- "
                       f"{done:,} of {total:,} ({share} of what is in the queue)"
                       + (f", about {duration(eta)} left" if eta else ", measuring"))


def duration(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


STATE_COLOUR = {"running": "#2a78d6", "queued": "#9a9790", "done": "#6f8f5f",
                "failed": "#b4443a", "stopped": "#b07a2a"}


def bar(share: float, colour: str) -> str:
    return (f'<div style="background:#e6e4df;border-radius:4px;height:12px;width:100%">'
            f'<div style="background:{colour};border-radius:4px;height:12px;'
            f'width:{max(0.0, min(1.0, share)) * 100:.2f}%"></div></div>')


def row(job: Job) -> str:
    colour = STATE_COLOUR.get(job.state, "#9a9790")
    cell = "padding:3px 8px 3px 0;vertical-align:middle;border:none"
    return (f'<tr><td style="{cell};width:1%;white-space:nowrap">'
            f'<b style="color:{colour}">{job.kind}</b> {job.name}</td>'
            f'<td style="{cell};width:40%">{bar(job.share, colour)}</td>'
            f'<td style="{cell};white-space:nowrap;opacity:0.8">{job.words()}</td></tr>')


QUEUE = Queue()


def datasets() -> list:
    return sorted(p.name for p in CLEAR_DIR.glob("*") if (p / "meta" / "info.json").exists())


def blank_file(dataset_dir: Path, keys: list, episodes) -> dict:
    return {"dataset": dataset_dir.name, "cameras": keys,
            "episodes": {str(int(i)): {"frame": 0, "frames": {}, "points": {}}
                         for i in sorted(episodes.index)}}


def seeded_file(dataset_dir: Path, keys: list, episodes) -> dict:
    """A new points file for `dataset_dir`: empty, unless data/points/<name>.map.json says
    which episodes of an earlier recording these are (a Hub copy with episodes dropped and
    renumbered), in which case the earlier recording's points are carried over under the
    new numbers, so a review starts from what masked the dataset rather than from nothing."""
    content = blank_file(dataset_dir, keys, episodes)
    map_path = auto_point.points_path(dataset_dir).with_suffix(".map.json")
    if not map_path.exists():
        return content
    index_map = json.loads(map_path.read_text())
    source_path = REPO_ROOT / "data" / "points" / f"{index_map['original']}.json"
    if not source_path.exists():
        return content
    source = json.loads(source_path.read_text())["episodes"]
    for new, old in index_map["hub_to_original"].items():
        if str(old) in source and new in content["episodes"]:
            content["episodes"][new] = json.loads(json.dumps(source[str(old)]))
    content["seeded_from"] = index_map["original"]
    return content


def load_state(name: str) -> dict:
    """Everything the window needs for one dataset, with its points file created if new."""
    dataset_dir = CLEAR_DIR / name
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    keys = video_keys(info)
    episodes = episode_offsets(load_episodes(dataset_dir), info).set_index("episode_index")
    path = auto_point.points_path(dataset_dir)
    if path.exists():
        points_file = json.loads(path.read_text())
    else:
        points_file = seeded_file(dataset_dir, keys, episodes)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(points_file, indent=2))
    return {"name": name, "dataset_dir": dataset_dir, "info": info, "keys": keys,
            "episodes": episodes, "path": path, "file": points_file,
            "fps": info["fps"],
            # Frames are decoded one at a time by seeking, so switching episodes costs
            # nothing; the last few decoded frames are kept for the scrub slider.
            "frame_cache": {}, "frame_lock": threading.Lock(),
            # Clips are cut by ffmpeg, one process per camera, so two at once is right.
            "clip_executor": ThreadPoolExecutor(max_workers=2), "clip_futures": {},
            "clip_lock": threading.Lock(), "precut": False,
            "job": {"running": False, "log": ""}}


def points_file_for(name: str) -> tuple:
    """(path, contents) of one dataset's points file, created empty if it has none yet.
    Reads from disk rather than from the open window, so a queued dataset needs no state."""
    dataset_dir = CLEAR_DIR / name
    path = auto_point.points_path(dataset_dir)
    if path.exists():
        return path, json.loads(path.read_text())
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    episodes = episode_offsets(load_episodes(dataset_dir), info).set_index("episode_index")
    content = seeded_file(dataset_dir, video_keys(info), episodes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2))
    return path, content


def episode_count(name: str) -> int:
    return len(load_episodes(CLEAR_DIR / name))


def generate_points(name: str, wrist: tuple, region: tuple, report) -> int:
    """Answer one whole dataset and merge it into its points file. Returns how many
    episodes were left alone because somebody had already clicked them."""
    found = auto_point.detect(CLEAR_DIR / name, None, wrist, region, report=report)
    path, content = points_file_for(name)
    kept = {e for e, ann in content["episodes"].items() if ann.get("by_hand")}
    content["cameras"] = found["cameras"]
    content["episodes"].update({e: a for e, a in found["episodes"].items() if e not in kept})
    path.write_text(json.dumps(content, indent=2))
    return len(kept & set(found["episodes"]))


def entry(state: dict, index: int) -> dict:
    return state["file"]["episodes"].setdefault(
        str(index), {"frame": 0, "frames": {}, "points": {}})


def write(state: dict) -> None:
    state["path"].write_text(json.dumps(state["file"], indent=2))


def is_complete(state: dict, index: int) -> bool:
    points = entry(state, index)["points"]
    return all(as_points(points.get(key)) for key in state["keys"])


def episode_length(state: dict, index: int) -> int:
    return int(state["episodes"].loc[index, "length"])


def progress(state: dict) -> str:
    indices = sorted(int(k) for k in state["file"]["episodes"])
    done = [i for i in indices if is_complete(state, i)]
    todo = [i for i in indices if i not in set(done)]
    line = f"**{len(done)}/{len(indices)} episodes answered for in every camera.**"
    if todo:
        line += f" Left: {', '.join(str(i) for i in todo[:12])}" + (" ..." if len(todo) > 12 else "")
    bad = flagged(state)
    if bad:
        line += (f"  \n**{len(bad)} flagged as bad:** {', '.join(str(i) for i in bad[:30])}"
                 + (" ..." if len(bad) > 30 else ""))
    redone = [i for i in indices if entry(state, i).get("remasked") and not entry(state, i).get("bad")]
    if redone:
        line += f"  \n{len(redone)} re-masked since the Hub copy, worth a second look: " + \
                ", ".join(str(i) for i in redone[:30]) + (" ..." if len(redone) > 30 else "")
    if not has_masked(state):
        line += "  \n*(no masked copy under data/masked yet, so no clips to play)*"
    return line


def flagged(state: dict) -> list:
    return sorted(int(k) for k, a in state["file"]["episodes"].items() if a.get("bad"))


def has_masked(state: dict) -> bool:
    return (MASKED_DIR / state["name"] / "meta" / "info.json").exists()


# ------------------------------------------------------------------ frames, one at a time
def video_file(state: dict, root: Path, index: int, key: str) -> tuple:
    """(path, start timestamp in seconds) of one episode's stretch of one camera's file."""
    ep = state["episodes"].loc[index]
    rel = state["info"]["video_path"].format(
        video_key=key, chunk_index=int(ep[f"videos/{key}/chunk_index"]),
        file_index=int(ep[f"videos/{key}/file_index"]))
    return root / rel, float(ep[f"videos/{key}/from_timestamp"])


def frame_at(state: dict, index: int, key: str, at: int) -> np.ndarray:
    """One clear frame of one camera, decoded by seeking to it.

    These recordings have a keyframe every second frame, so a seek lands within one frame
    of the target and the decode is a few milliseconds; the whole episode never has to be
    read. This is what makes switching episodes instant, where decoding every frame into
    memory first took the best part of ten seconds.
    """
    at = max(0, min(at, episode_length(state, index) - 1))
    cache_key = (index, key, at)
    with state["frame_lock"]:
        hit = state["frame_cache"].get(cache_key)
    if hit is not None:
        return hit
    path, start = video_file(state, CLEAR_DIR / state["name"], index, key)
    target = start + at / state["fps"]
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        container.seek(int(target * av.time_base), backward=True, any_frame=False)
        frame = None
        for candidate in container.decode(stream):
            frame = candidate
            if candidate.time is not None and candidate.time >= target - 0.5 / state["fps"]:
                break
        out = frame.to_ndarray(format="rgb24")
    finally:
        container.close()
    with state["frame_lock"]:
        cache = state["frame_cache"]
        if len(cache) > 64:
            cache.clear()
        cache[cache_key] = out
    return out


# ------------------------------------------------------------------------------ clips
def clip_path(state: dict, index: int, key: str) -> Path:
    return CLIPS_DIR / state["name"] / f"ep{index:04d}_{camera_name(key)}.mp4"


# One lock per clip, so the same clip is never cut twice at once. Two cuts of one clip is
# not a waste of a second, it is a broken file: they used to share one temporary name and
# the two ffmpegs wrote over each other, leaving an mp4 whose header says one thing and
# whose body is another -- valid enough to hand to the browser, empty enough to play as
# black. Reached whenever two windows are open on one dataset, or one is refreshed while
# the first cut of the recording is still going, because each window has a pool of its own.
CUT_LOCKS, CUT_LOCKS_GUARD = {}, threading.Lock()


def cut_lock(path: Path) -> threading.Lock:
    with CUT_LOCKS_GUARD:
        return CUT_LOCKS.setdefault(str(path), threading.Lock())


def sound_mp4(path: Path) -> bool:
    """Is this a whole mp4? Walks the boxes -- every one says its own length, so a file cut
    short, or written into by two processes at once, contradicts itself within a few reads.
    Pure arithmetic over a few dozen bytes, cheap enough to ask before every clip is served,
    which is the point: a clip that was already broken on disk has to be noticed and cut
    again, not handed to the browser because the file happens to exist."""
    try:
        size = path.stat().st_size
    except OSError:
        return False

    def walk(handle, start: int, end: int, depth: int) -> bool:
        seen, off = [], start
        while off < end:
            handle.seek(off)
            header = handle.read(8)
            if len(header) < 8:
                return False
            length = int.from_bytes(header[:4], "big")
            name = header[4:8]
            if length == 0:          # "to the end of the file", legal for the last box
                length = end - off
            if length < 8 or off + length > end:
                return False
            seen.append(name)
            if name in (b"moov", b"trak", b"mdia") and depth < 3:
                if not walk(handle, off + 8, off + length, depth + 1):
                    return False
            off += length
        if depth == 0:
            return b"ftyp" in seen and b"moov" in seen and b"mdat" in seen
        return True

    try:
        with path.open("rb") as handle:
            return walk(handle, 0, size, 0)
    except OSError:
        return False


def cut_clip(state: dict, index: int, key: str) -> Path | None:
    """The masked copy's frames of one episode, as an h264 clip a browser will play.

    Cut with ffmpeg from the masked file by the timestamps the episode parquet holds; the
    masked copy has a keyframe every second frame, so the seek is exact to within one frame
    and the cut takes about a second, not a decode of the whole file. Every frame is kept,
    at the recording's own rate, so a click on the clip names a frame exactly; slowing the
    playback is the player's business.
    """
    if not has_masked(state):
        return None
    out = clip_path(state, index, key)
    with cut_lock(out):
        if out.exists() and sound_mp4(out):
            return out
        src, start = video_file(state, MASKED_DIR / state["name"], index, key)
        out.parent.mkdir(parents=True, exist_ok=True)
        # A name of this cut's own, never the clip's: two cuts that do meet write to two
        # files and the rename picks a winner, rather than writing over one another.
        tmp = out.with_suffix(f".{threading.get_ident():x}.part.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.4f}",
                 "-i", str(src),
                 "-frames:v", str(episode_length(state, index)), "-an",
                 # even sides for yuv420p h264; the overview camera is 355 wide
                 "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(tmp)],
                check=True, capture_output=True)
            if not sound_mp4(tmp):
                raise RuntimeError(f"ffmpeg wrote an unreadable clip for episode {index}")
            tmp.replace(out)
        finally:
            tmp.unlink(missing_ok=True)
    return out


def clips(state: dict, index: int) -> list:
    """Every camera's clip of one episode; one already being cut is waited for, not cut twice."""
    # The episode on screen must not wait behind the background queue: a clip still queued
    # is pulled out of it and cut right here, one that is already being cut is waited for.
    pending, now = [], []
    for key in state["keys"]:
        with state["clip_lock"]:
            future = state["clip_futures"].pop((index, key), None)
        if future is None or future.cancel():
            now.append(key)
        else:
            pending.append(future)
    with ThreadPoolExecutor(max_workers=max(1, len(now))) as pool:
        inline = [pool.submit(cut_clip, state, index, key) for key in now]
    paths = []
    for future in inline + pending:
        try:
            paths.append(future.result())
        # CancelledError as well as Exception: it is not an Exception, and a pool shut down
        # under a superseded window raises it here. Either way one clip that will not cut
        # must not take the window down.
        except (Exception, CancelledError) as exc:
            print(f"clip failed: {type(exc).__name__}: {exc}")
            paths.append(None)
    # keep camera order: inline ones came first in `now`, then the pending ones
    order = now + [key for key in state["keys"] if key not in now]
    by_key = dict(zip(order, paths))
    return [by_key.get(key) for key in state["keys"]]


def schedule_clips(state: dict, index: int) -> None:
    if not has_masked(state):
        return
    with state["clip_lock"]:
        for key in state["keys"]:
            if (index, key) not in state["clip_futures"] and not sound_mp4(clip_path(state, index, key)):
                state["clip_futures"][(index, key)] = state["clip_executor"].submit(cut_clip, state, index, key)


def precut(state: dict, start: int) -> None:
    """Queue every episode's clips, nearest to `start` first, the first time a dataset is
    opened: about a second each, so a whole recording is ready within a minute or two and
    from then on every episode opens at once. The order matters more than the total --
    the annotator walks forward, so the episodes ahead go first."""
    if state["precut"] or not has_masked(state):
        return
    state["precut"] = True
    indices = sorted(int(k) for k in state["file"]["episodes"])
    ahead = [i for i in indices if i >= start] + [i for i in indices if i < start]
    for i in ahead:
        schedule_clips(state, i)


def forget_clips(name: str) -> None:
    """After the masked copy changed, every clip of it is stale."""
    shutil.rmtree(CLIPS_DIR / name, ignore_errors=True)


def clip_url(path: Path) -> str:
    # Gradio serves files under allowed_paths at this route; the mtime defeats the
    # browser's cache once a re-mask has rewritten the clip.
    return f"/gradio_api/file={path}?v={int(path.stat().st_mtime)}"


# JS for the clip players. The players are plain <video> elements with the browser's own
# controls (the seek bar, like any video site). A click on the picture is a click on that
# frame: a transparent layer over the picture -- and not over the control strip at the bottom,
# which stays the browser's -- reads where the player stands and where the click landed, and
# hands both to Python through a hidden textbox. Gradio's video component cannot say where
# in the picture a click was, which is why the players are hand-made.
HEAD = """
<script>
window.clipClick = function (ev, slot, fps) {
  const v = ev.currentTarget.previousElementSibling, r = v.getBoundingClientRect();
  const x = (ev.clientX - r.left) / r.width, y = (ev.clientY - r.top) / r.height;
  v.pause();
  const box = document.querySelector('#clip-click textarea');
  box.value = JSON.stringify({slot: slot, frame: Math.round(v.currentTime * fps), x: x, y: y, t: Date.now()});
  box.dispatchEvent(new Event('input', {bubbles: true}));
  box.dispatchEvent(new Event('change', {bubbles: true}));
};
</script>
<style>
.grid-row { display: flex; gap: 12px; align-items: stretch; height: min(42vh, 460px); }
.tile { position: relative; height: 100%; flex: 0 0 auto; }
.tile video, .tile img { height: 100%; width: auto; display: block; background: #000; border-radius: 4px; }
.tile .hit { position: absolute; left: 0; top: 0; right: 0; bottom: 40px; cursor: crosshair; }
.tile .tag { position: absolute; left: 6px; top: 6px; font: 600 12px sans-serif; color: #fff;
             background: rgba(0,0,0,.55); padding: 1px 6px; border-radius: 3px; pointer-events: none; }
.grid-note { color: #888; padding: 8px; }
</style>
"""


def aspect(state: dict, key: str) -> str:
    """CSS aspect ratio of one camera, from its declared shape, so a tile has its final width
    before the video's metadata arrives (and even if it never does)."""
    _, height, width = state["info"]["features"][key]["shape"]
    return f"aspect-ratio: {width} / {height}"


def clips_html(state: dict, paths: list) -> str:
    """Top row: the masked clip of every camera, playing, same height side by side."""
    if not any(paths):
        return ("<div class='grid-note'>no masked copy of this dataset under data/masked, "
                "so nothing to play</div>")
    fps = state["fps"]
    tiles = []
    for slot, (key, path) in enumerate(zip(state["keys"], paths)):
        if path is None:
            continue
        tiles.append(
            f'<div class="tile">'
            f'<video src="{clip_url(path)}" style="{aspect(state, key)}" controls autoplay loop muted playsinline></video>'
            f'<div class="hit" onclick="clipClick(event,{slot},{fps})"></div>'
            f'<span class="tag">masked now -- {camera_name(key)} -- click the item to correct</span>'
            f'</div>')
    return f'<div class="grid-row">{"".join(tiles)}</div>'


def as_data_url(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def previews_html(state: dict, index: int) -> str:
    """Bottom row: for every camera, its seed frame masked by SAM3 from the current points,
    with the points drawn on it -- what the re-mask will start from. Same heights as the
    clips above, so the two rows line up column by column."""
    ann = entry(state, index)
    tiles, notes = [], []
    for key in state["keys"]:
        seed = seed_frame(ann, key)
        frame = frame_at(state, index, key, seed)
        points = as_points(ann["points"].get(key))
        preview, note = sam_preview(frame, points)
        if note:
            notes.append(note)
        shown = mark(preview if preview is not None else frame, points)
        label = (f"after correction -- {camera_name(key)} -- frame {seed}, "
                 f"{len(points)} point{'s' if len(points) != 1 else ''}"
                 + ("" if preview is not None else " (no SAM3 preview)"))
        tiles.append(f'<div class="tile"><img src="{as_data_url(shown)}" style="{aspect(state, key)}">'
                     f'<span class="tag">{label}</span></div>')
    html = f'<div class="grid-row">{"".join(tiles)}</div>'
    if notes:
        html += f"<div class='grid-note'>{' '.join(sorted(set(notes)))}</div>"
    return html


def describe(state: dict, index: int) -> str:
    ann = entry(state, index)
    parts = []
    if ann.get("bad"):
        parts.append("**FLAGGED BAD**")
    elif ann.get("remasked"):
        parts.append(f"re-masked x{len(ann['remasked'])}")
    for key in state["keys"]:
        points = as_points(ann["points"].get(key))
        plus = sum(1 for p in points if p[2])
        shown = "none" if not points else f"{plus}+" + (f"/{len(points) - plus}-" if len(points) > plus else "")
        parts.append(f"{camera_name(key)}: {shown} @ frame {seed_frame(ann, key)}")
    return f"**episode {index}** -- " + ", ".join(parts)


def build_ui(initial: str | None) -> gr.Blocks:
    names = datasets()
    if not names:
        raise SystemExit(f"no datasets under {CLEAR_DIR}; fetch one with scripts/fetch_dataset.py")
    start = initial if initial in names else names[0]

    # gr.State cannot carry a lock or a thread pool, so the window's own state lives here.
    # "reload": datasets a queued job is rewriting, picked back up off disk once it lands.
    holder = {"state": load_state(start), "reload": set()}

    def adopt(name: str) -> dict:
        """Open a dataset, and let go of the one that was open. Letting go matters: each
        state carries a pool that is cutting clips, and a window refreshed in the middle of
        a recording's first cut would otherwise leave two pools cutting the same clips."""
        old = holder.get("state")
        holder["state"] = load_state(name)
        if old is not None:
            old["clip_executor"].shutdown(wait=False, cancel_futures=True)
        return holder["state"]
    max_cameras = 4

    with gr.Blocks(title="Masking points", head=HEAD) as demo:
        gr.Markdown("### Masking points\nTop: the masked clip of the episode, playing. "
                    "**Click the item on the clip** when the mask is wrong: that frame and spot "
                    "become the point, and the row below shows that frame masked from it. "
                    "Flag the episodes that are wrong, then re-mask only those.")
        with gr.Row():
            dataset = gr.Dropdown(names, value=start, label="Dataset", scale=2)
            points_path = gr.Markdown()
        counter = gr.Markdown()

        with gr.Accordion("Generate points automatically", open=False):
            gr.Markdown("From the gripper (wrist cameras) and from what left the table "
                        "(overview cameras). Overwrites the points of the episodes it covers.")
            with gr.Row():
                scope = gr.Radio(["this episode", "whole dataset", "all datasets"],
                                 value="this episode", label="Scope", scale=1)
                wrist_point = gr.Textbox(",".join(map(str, auto_point.WRIST_POINT)), scale=1,
                                         label="Wrist point x,y (where the held item sits)")
                side_roi = gr.Textbox(",".join(map(str, auto_point.SIDE_ROI)), scale=1,
                                      label="Overview region x0,y0,x1,y1 (the table)")
            auto_btn = gr.Button("Generate", variant="primary")
            auto_log = gr.Markdown()

        with gr.Row():
            prev_btn = gr.Button("< prev", scale=0)
            episode = gr.Dropdown([], label="Episode", scale=1)
            next_btn = gr.Button("next >", scale=0)
            todo_btn = gr.Button("next unanswered >>", scale=0)
            bad_btn = gr.Button("flag as BAD", variant="stop", scale=0)
            next_bad_btn = gr.Button("next flagged >>", variant="primary", scale=0)
            mode = gr.Radio(["+ add to mask", "- subtract from mask"], value="+ add to mask",
                            label="What a click does", scale=0)

        status = gr.Markdown()
        clip_row = gr.HTML()       # the masked clips, playing
        preview_row = gr.HTML()    # the same cameras, seed frame masked from the current points
        click_box = gr.Textbox(visible="hidden", elem_id="clip-click")  # "hidden": in the DOM, unlike False
        with gr.Row():
            undo_btns = [gr.Button(visible=False, size="sm") for _ in range(max_cameras)]
            clear_btns = [gr.Button(visible=False, size="sm") for _ in range(max_cameras)]

        with gr.Accordion("Run SAM3", open=False):
            gr.Markdown("Tracks each camera backwards from its seed frame to the start of "
                        "the episode and forwards to the end. Needs demo/server.py running.")
            with gr.Row():
                remask_btn = gr.Button("Re-mask the flagged episodes of this dataset",
                                       variant="primary")
                remask_all_btn = gr.Button("Re-mask the flagged episodes of every dataset",
                                           variant="primary")
                one_btn = gr.Button("Mask this episode into preview clips")
                all_btn = gr.Button("Mask this dataset")
                every_btn = gr.Button("Mask every dataset")
                refresh_btn = gr.Button("refresh status", scale=0)
            gr.Markdown("Re-masking takes only the episodes flagged as bad, masks them again "
                        "from the clear recording and splices them into `data/masked/<name>`; "
                        "the previous copy moves to `data/masked_prev/`. Good episodes keep "
                        "their pixels. Flags come off once it lands. \"Every dataset\" walks "
                        "all of `data/clear` in turn, taking the flags from each points file.")
            with gr.Row():
                # Tick the ones wanted and press the button: for when two machines split
                # the work, or one dataset is still being clicked while the rest can go.
                chosen = gr.CheckboxGroup(datasets(), label="or only these datasets", scale=3)
                chosen_btn = gr.Button("Mask the ticked datasets", scale=1)
            gr.Markdown("A whole dataset goes to `data/masked/<name>` and nowhere else -- "
                        "nothing is uploaded. These go to the queue below: one dataset is "
                        "masked at a time, because they share one GPU, and any number can "
                        "wait behind it. More can be queued while one is running, and the "
                        "window stays usable meanwhile -- so queue this one, carry on "
                        "clicking the next, and queue that one too.")
            mask_log = gr.Markdown()
            videos = [gr.Video(visible=False) for _ in range(max_cameras)]

        with gr.Accordion("Queue", open=True):
            gr.Markdown("Long work runs behind the window, which stays usable while it "
                        "does: queue a dataset, carry on clicking the next one, queue that "
                        "one too. Masking goes one at a time, sharing the one GPU; "
                        "generating points goes four at a time beside it.")
            queue_note = gr.Markdown()
            queue_bars = gr.HTML()
            with gr.Row():
                cancel_pick = gr.Dropdown([], multiselect=True, label="cancel these", scale=3)
                cancel_btn = gr.Button("cancel the ticked jobs", scale=0)
                drop_btn = gr.Button("drop everything not yet started", scale=0)
                stop_btn = gr.Button("stop everything", variant="stop", scale=0)
            # The ticker owns queue_note, so what the buttons have to say goes here instead,
            # where it stays put rather than being overwritten two seconds later.
            queue_msg = gr.Markdown()
        ticker = gr.Timer(2.0)

        # ---------------------------------------------------------------- handlers
        def on_dataset(name):
            st = adopt(name)
            indices = sorted(int(k) for k in st["file"]["episodes"])
            precut(st, indices[0])
            buttons = []
            for kind in ("undo last point", "clear points"):
                buttons += [gr.Button(f"{kind} -- {camera_name(st['keys'][i])}", visible=True)
                            if i < len(st["keys"]) else gr.Button(visible=False)
                            for i in range(max_cameras)]
            return (gr.Dropdown(choices=indices, value=indices[0]),
                    f"points file: `{st['path'].relative_to(REPO_ROOT)}`", progress(st), *buttons)

        def flag_label(ann):
            return gr.Button("unflag (it is fine)" if ann.get("bad") else "flag as BAD",
                             variant="secondary" if ann.get("bad") else "stop")

        def show_episode(index):
            st = holder["state"]
            index = int(index)
            paths = clips(st, index)
            indices = sorted(int(k) for k in st["file"]["episodes"])
            position = indices.index(index)
            if position + 1 < len(indices):
                schedule_clips(st, indices[position + 1])
            return [describe(st, index), progress(st), clips_html(st, paths),
                    previews_html(st, index), flag_label(entry(st, index))]

        def apply(slot, index, at, new_points, adopt_frame):
            """Every edit is the same: replace one camera's point list, maybe adopt the
            frame as that camera's seed, save, and redraw the bottom row from the points."""
            st = holder["state"]
            key = st["keys"][slot]
            ann = entry(st, int(index))
            ann["points"][key] = new_points
            ann["by_hand"] = True    # so a later "whole dataset" run leaves it alone
            if adopt_frame:
                ann.setdefault("frames", {})[key] = int(at)
            ann["frame"] = int(ann.get("frame", 0))
            write(st)
            preview_dir = st["path"].with_suffix("")
            preview_dir.mkdir(parents=True, exist_ok=True)
            save_preview({k: frame_at(st, int(index), k, seed_frame(ann, k)) for k in st["keys"]},
                         ann["points"], st["keys"], preview_dir / f"ep{int(index):04d}.jpg")
            return (describe(st, int(index)) + "  *(saved)*", progress(st),
                    previews_html(st, int(index)))

        def on_clip_click(payload, index, how):
            """A click on the playing clip: the frame it stands on becomes the seed of that
            camera, the spot its point -- appended if the click is on the camera's seed frame
            already, otherwise starting the camera over there, because two points from two
            instants would describe two different scenes."""
            st = holder["state"]
            skip = [gr.skip()] * 3
            if not payload:
                return skip
            try:
                click = json.loads(payload)
                slot, at = int(click["slot"]), int(click["frame"])
                x, y = float(click["x"]), float(click["y"])
            except (ValueError, KeyError, TypeError):
                return skip
            if slot >= len(st["keys"]):
                return skip
            key = st["keys"][slot]
            ann = entry(st, int(index))
            at = max(0, min(at, episode_length(st, int(index)) - 1))
            label = 0 if how.startswith("-") else 1
            existing = as_points(ann["points"].get(key)) if at == seed_frame(ann, key) else []
            return apply(slot, index, at, existing + [[round(x, 4), round(y, 4), label]], True)

        def on_undo(slot):
            def handler(index):
                st = holder["state"]
                key = st["keys"][slot]
                ann = entry(st, int(index))
                return apply(slot, index, seed_frame(ann, key),
                             as_points(ann["points"].get(key))[:-1], False)
            return handler

        def on_clear(slot):
            def handler(index):
                st = holder["state"]
                key = st["keys"][slot]
                ann = entry(st, int(index))
                return apply(slot, index, seed_frame(ann, key), [], False)
            return handler

        def on_flag(index):
            """Toggle the bad flag; saved at once like a click. A flagged episode with no
            point yet will be refused by the re-mask, which says so, rather than here."""
            st = holder["state"]
            ann = entry(st, int(index))
            ann["bad"] = not ann.get("bad", False)
            write(st)
            return describe(st, int(index)), progress(st), flag_label(ann)

        def next_bad(index):
            st = holder["state"]
            bad = flagged(st)
            return next((i for i in bad if i > int(index)), bad[0] if bad else int(index))

        def step(index, delta):
            st = holder["state"]
            indices = sorted(int(k) for k in st["file"]["episodes"])
            return indices[max(0, min(len(indices) - 1, indices.index(int(index)) + delta))]

        def next_todo(index):
            st = holder["state"]
            indices = sorted(int(k) for k in st["file"]["episodes"])
            todo = [i for i in indices if not is_complete(st, i)]
            return next((i for i in todo if i > int(index)), todo[0] if todo else int(index))

        def on_auto(index, which, wrist, roi):
            st = holder["state"]
            try:
                settings = (tuple(float(v) for v in wrist.split(",")),
                            tuple(float(v) for v in roi.split(",")))
            except ValueError:
                return ("**failed:** the wrist point and the region are comma separated "
                        "numbers", progress(st))
            if which != "this episode":
                return queue_points(st, which, *settings)
            try:
                found = auto_point.detect(st["dataset_dir"], [int(index)], *settings)
            except Exception as exc:
                return f"**failed:** {type(exc).__name__}: {exc}", progress(st)
            kept = [e for e, ann in st["file"]["episodes"].items()
                    if ann.get("by_hand") and e in found["episodes"]]
            fresh = {e: ann for e, ann in found["episodes"].items() if e not in kept}
            st["file"]["episodes"].update(fresh)
            write(st)
            answered = sum(1 for e in fresh.values()
                           if all(e["points"].get(k) for k in st["keys"]))
            note = f" {len(kept)} already clicked by hand were left as they are." if kept else ""
            return (f"{len(fresh)} episodes written, {answered} with every "
                    f"camera answered for.{note} Check them below.", progress(st))

        def queue_points(st, which, wrist, region):
            """A whole dataset takes minutes and four take the best part of a quarter of
            an hour, so both go to the queue and the window stays usable. Point jobs run
            four at a time and never wait behind masking: this work is decoding video,
            which is per core, and it is the other rail of the queue."""
            names = [st["name"]] if which == "whole dataset" else datasets()
            entries = []
            for name in names:
                def one(report, name=name):
                    try:
                        return generate_points(name, wrist, region, report)
                    finally:
                        holder["reload"].add(name)

                entries.append({"name": name, "key": f"points:{name}", "dataset": name,
                                "total": episode_count(name), "run": one})
            added, dupes = QUEUE.submit("points", "points", entries)
            note = (f"queued: {', '.join(j.name for j in added)}. Watch the queue at the "
                    "bottom; the window stays usable meanwhile." if added
                    else "nothing added")
            if dupes:
                note += f"  \nalready queued, left where they are: {', '.join(dupes)}"
            return note, progress(st)

        def queue_masking(names):
            """Straight onto the mask rail, whatever else is going: one dataset is masked
            at a time because they share one SAM3 server on one GPU, but any number can
            wait behind it, and they can be added while one is running. Which is the
            point -- press this, carry on with the next dataset, press it again."""
            ready = [n for n in names if not (MASKED_DIR / n).exists()]
            skipped = [f"{n} (already under data/masked)" for n in names if n not in ready]
            entries = []
            for name in ready:
                path, _ = points_file_for(name)
                def one(report, path=path, name=name):
                    if (MASKED_DIR / name).exists():
                        raise RuntimeError(f"{name} was masked by somebody else in the "
                                           f"meantime; remove data/masked/{name} to redo it")
                    try:
                        mask_script.main(str(path), report)
                    finally:
                        forget_clips(name)
                        holder["reload"].add(name)

                entries.append({"name": name, "key": f"mask:{name}", "dataset": name,
                                "total": mask_script.planned_frames(str(path)), "run": one})
            added, dupes = QUEUE.submit("mask", "mask", entries)
            skipped += [f"{n} (already in the queue)" for n in dupes]
            if not added:
                return ("nothing added -- " + "; ".join(skipped) if skipped
                        else "nothing to do")
            fresh = {j.id for j in added}
            waiting = len([j for j in QUEUE.active()
                           if j.lane == "mask" and j.id not in fresh])
            note = (f"queued: {', '.join(j.name for j in added)} -> `data/masked/`, nothing "
                    "is uploaded."
                    + (f" {waiting} mask job(s) ahead in the line." if waiting > 0 else "")
                    + " Keep clicking meanwhile; more can be queued at any time.")
            if skipped:
                note += f"  \nskipped: {'; '.join(skipped)}"
            return note

        def queue_remask(names):
            """Every dataset in `names` that has flagged episodes and a masked copy joins
            the mask rail, behind whatever is already on it: they share one SAM3 server on
            one GPU. Flags are read from the points files on disk, so a dataset never
            opened in this window counts too, and which episodes are flagged is read when
            the job is queued -- flagging more afterwards means queueing it again once
            this one has landed."""
            entries, skipped = [], []
            for name in names:
                path, content = points_file_for(name)
                bad = mask_script.flagged_episodes(content)
                if not bad:
                    skipped.append(f"{name} (nothing flagged)")
                    continue
                if not (MASKED_DIR / name / "meta" / "info.json").exists():
                    skipped.append(f"{name} (no masked copy under data/masked)")
                    continue

                def one(report, path=path, name=name):
                    try:
                        mask_script.remask_flagged(str(path), report)
                    finally:
                        forget_clips(name)
                        holder["reload"].add(name)

                entries.append({"name": f"{name}: episodes {', '.join(map(str, bad))}",
                                "key": f"mask:{name}", "dataset": name,
                                "total": mask_script.planned_remask_frames(str(path)),
                                "run": one})
            added, dupes = QUEUE.submit("re-mask", "mask", entries)
            skipped += [f"{n.split(':')[0]} (already in the queue)" for n in dupes]
            if not added:
                return "nothing to re-mask -- " + "; ".join(skipped or ["no datasets"])
            note = ("queued:  \n" + "  \n".join(j.name for j in added)
                    + f"  \n{sum(j.total for j in added):,} frames to write in all. Watch the "
                    "queue at the bottom; open an episode again once its dataset lands to see "
                    "the new clip.")
            if skipped:
                note += "  \nskipped: " + "; ".join(skipped)
            return note

        def tick(picked):
            """Every couple of seconds while the window is open: cheap, it only reads
            counters. It also picks the open dataset back up off disk once everything that
            was going to rewrite it has finished, so the view is never of a stale file --
            and only then, because a reload in the middle of a run would show a half
            written file and throw away what is on screen."""
            html, note = QUEUE.view()
            name = holder["state"]["name"]
            if name in holder["reload"] and not QUEUE.touching(name):
                holder["reload"].discard(name)
                adopt(name)
            choices = QUEUE.choices()
            live = {ident for _, ident in choices}
            return (html, note,
                    gr.Dropdown(choices=choices, value=[i for i in (picked or []) if i in live]),
                    progress(holder["state"]))

        def on_cancel(picked):
            """Ticked jobs only. One still waiting never starts; the one running stops at
            its next episode and leaves that dataset unfinished, and either way everything
            else in the queue carries on."""
            if not picked:
                return "tick the jobs to cancel first"
            return (f"cancelling {QUEUE.cancel(picked)} job(s); one already running stops "
                    "at its next episode and leaves that dataset unfinished")

        def on_drop():
            return (f"dropped {QUEUE.drop_waiting()} job(s) that had not started; "
                    "what is running carries on")

        def on_stop():
            """Stops between episodes, not between datasets: whatever is being masked now
            is left half written and has to be removed from data/masked before it can be
            run again. Everything still queued behind it is dropped too."""
            QUEUE.stop_all()
            return ("stopping between episodes; the dataset in progress will be incomplete "
                    "and the rest of the queue is dropped")

        def start_job(index, one):
            st = holder["state"]
            """SAM3 is slow enough that the window must not wait on it: the run goes to a
            thread and its outcome is collected by the status button."""
            if st["job"]["running"]:
                return "a masking run is already going", *[gr.Video(visible=False)] * max_cameras
            st["job"] = {"running": True, "log": "started ..."}

            def run():
                try:
                    if one:
                        paths = mask_script.preview_episode(str(st["path"]), int(index))
                        st["job"] = {"running": False, "log": "done", "paths": paths}
                    else:
                        mask_script.main(str(st["path"]))
                        st["job"] = {"running": False,
                                     "log": f"done -> data/masked/{st['name']}"}
                except Exception as exc:
                    st["job"] = {"running": False,
                                 "log": f"**failed:** {type(exc).__name__}: {exc}\n\n```\n"
                                        + traceback.format_exc()[-1500:] + "\n```"}
            threading.Thread(target=run, daemon=True).start()
            note = ("Masking one episode, a minute or two per camera."
                    if one else "Masking the whole dataset. This takes hours; the window can "
                                "be closed, the run carries on.")
            return note, *[gr.Video(visible=False)] * max_cameras

        def job_status():
            st = holder["state"]
            job = st["job"]
            paths = job.get("paths") or []
            return (("running ..." if job["running"] else job["log"]),
                    *[gr.Video(value=str(paths[i]), visible=True,
                               label=camera_name(st["keys"][i]))
                      if i < len(paths) else gr.Video(visible=False)
                      for i in range(max_cameras)])

        # ------------------------------------------------------------------- wiring
        dataset_outs = [episode, points_path, counter, *undo_btns, *clear_btns]
        dataset.change(on_dataset, dataset, dataset_outs)
        demo.load(on_dataset, dataset, dataset_outs)

        episode_outs = [status, counter, clip_row, preview_row, bad_btn]
        episode.change(show_episode, episode, episode_outs)
        prev_btn.click(lambda i: step(i, -1), episode, episode)
        next_btn.click(lambda i: step(i, +1), episode, episode)
        todo_btn.click(next_todo, episode, episode)
        bad_btn.click(on_flag, episode, [status, counter, bad_btn])
        next_bad_btn.click(next_bad, episode, episode)
        remask_btn.click(lambda: queue_remask([holder["state"]["name"]]), None, mask_log)
        remask_all_btn.click(lambda: queue_remask(datasets()), None, mask_log)
        edit_outs = [status, counter, preview_row]
        # .change, not .input: Gradio fires .input only for keystrokes it saw itself, while a
        # value set from JS is picked up as a change.
        click_box.change(on_clip_click, [click_box, episode, mode], edit_outs)
        for slot in range(max_cameras):
            undo_btns[slot].click(on_undo(slot), episode, edit_outs)
            clear_btns[slot].click(on_clear(slot), episode, edit_outs)

        auto_btn.click(on_auto, [episode, scope, wrist_point, side_roi],
                       [auto_log, counter]).then(show_episode, episode, episode_outs)
        one_btn.click(lambda i: start_job(i, True), episode, [mask_log, *videos])
        all_btn.click(lambda: queue_masking([holder["state"]["name"]]), None, mask_log)
        every_btn.click(lambda: queue_masking(datasets()), None, mask_log)
        chosen_btn.click(lambda names: queue_masking(list(names)) if names
                         else "tick at least one dataset", chosen, mask_log)
        refresh_btn.click(job_status, None, [mask_log, *videos])
        cancel_btn.click(on_cancel, cancel_pick, queue_msg)
        drop_btn.click(on_drop, None, queue_msg)
        stop_btn.click(on_stop, None, queue_msg)
        ticker.tick(tick, cancel_pick, [queue_bars, queue_note, cancel_pick, counter])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", help="directory name under data/clear to open first")
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    name = Path(args.dataset).name if args.dataset else None
    # allowed_paths: the clip players load their mp4s straight from data/clips.
    build_ui(name).launch(server_name="0.0.0.0", server_port=args.port,
                          allowed_paths=[str(CLIPS_DIR)])
