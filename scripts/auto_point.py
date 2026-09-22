"""Points without a human: one per camera per episode, from the gripper and from what
left the table.

A pick-and-place episode hands us two facts for free.

The gripper closes on the item and opens again to release it, and that is recorded in
the dataset. Read from the end: the last time the gripper opens is the release, so the
stretch of frames just before it is the robot carrying the item. Anywhere inside that
stretch the item sits between the fingers, which in a wrist camera is a fixed place in
the frame -- the same pixel in every episode, because the camera is bolted to the hand.

The overview camera gets no such gift, so it is answered by what changed. Before the
grasp the item lies on the table; after the release it is in the bin. So a pixel of the
item looks different in every frame before the grasp from how it looks in every frame
after the release. The arm also differs between frames, but the arm moves: pick a dozen
frames from before the grasp and a handful from after the release, and for every pixel
take the SMALLEST difference over all the pairs. The arm, present in some frames and not
in others, scores its smallest pair at nearly zero. The item scores high in every pair.
The largest blob of what survives is where the item lay. The earlier version compared
frame 0 with the release frame alone, and failed on every episode that began with the
arm already over the table -- most of amanuel-2 -- because it had no way to tell the arm
from the item. What still fools it is the arm standing over the item's spot for the
whole approach: then the item is never seen on the table and the surviving blob is the
gripper. That blob gives itself away -- tall, and faint, because a hovering gripper
jitters -- and such an episode is answered at the LIFT frame instead, a moment after the
grasp, where the item hangs under the fingers in plain view: it is the lowest part of
what the arm brought into the picture.

That leaves each camera seeded at its own instant, which is what `frames` in the json is
for. Everything the detector is not sure about is left as a null point for a human to
answer in scripts/annotate.py.

Every video file is decoded exactly once, whatever number of episodes it holds, and the
files go to a pool of processes, so a whole recording is answered in a minute or two.

Usage:
    python scripts/auto_point.py data/clear/jr-pnp-tomek-1 [--episode N] [--workers N]
Output: data/points/<name>.json
"""
import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from find_point_with_gemini import (  # noqa: E402
    camera_name, episode_offsets, file_columns, load_episodes, video_keys, video_rel_path,
)

# Where the item sits in a wrist view once it is in the gripper. One number for the whole
# dataset, because the camera does not move relative to the fingers. Override per rig.
# Below and right of the frame's centre: the camera looks past the fingers rather than
# straight down them, so a centred point lands on the left edge of anything wide. Pushed
# further than this it starts to miss narrow things -- a syringe held upright is about
# a tenth of the frame across, and 0.58 already grazes its edge.
WRIST_POINT = (0.55, 0.62)
# The part of an overview frame the item can be in, as x0,y0,x1,y1 in 0..1: the table in
# front of the bin, and nothing else. The top edge is set by the bin, not by the arm: it
# has to stay below the bin's contents, because an item ARRIVING in the bin changes
# between before and after exactly as one leaving the table does, and the arm no longer
# matters since the comparison takes the minimum over many frames. At 0.55 it clears the
# bin's front wall in all five jr-pnp recordings and still reaches the far edge of the
# table, where amanuel-2 keeps a third of its items; the old 0.66 cut those off. The
# right edge stops the region before the table ends: past it is a strip of shadow that
# changes with the arm and holds no item, and items are only ever in front of the bin,
# never beside it. Look at this once for any new camera placement; the GUI draws it over
# a frame, so a bad one shows before a whole dataset runs on it.
SIDE_ROI = (0.0, 0.55, 0.75, 1.0)
# A difference blob smaller than this share of the whole frame is noise, not an item. Of
# the frame and not of the region, so that narrowing the region tightens what is searched
# without also lowering the bar for what counts as an item: measured against the region,
# every shrink quietly admits more noise, and a wrong point is worse than none -- the GUI
# offers up an unanswered episode to be clicked, and a confident wrong one it does not.
MIN_BLOB = 0.00028
# When the largest blob is taller than this share of the frame, or its difference is weaker
# than this (0..255), it is more likely the gripper than an item: an item is compact and
# differs strongly from the bare table, while a gripper that hung over the item for the
# whole approach is tall, and jitters enough that its smallest difference is faint. For
# such an episode the item is never visible on the table, so the answer moves to the LIFT
# frame -- a moment after the grasp, when the item hangs under the fingers in plain view.
# Measured on 745 verified episodes: right answers had a median strength of 90 and a
# height at the 90th percentile of 44 px of 300; gripper answers a median strength of 34.
SUSPECT_HEIGHT = 0.15
SUSPECT_STRENGTH = 30
# How many frames after the gripper closes to look for the item hanging in it, and what a
# pixel has to differ by (0..255) from the emptied scene to count as arm-or-item there.
LIFT_FRAMES = 30
SILHOUETTE = 40
# How many frames to compare: spread over everything before the grasp, and over
# everything after the release. A dozen before is enough for the arm to have moved off
# any one spot at least once; more only costs decoding.
N_PRE, N_POST = 12, 6
# The first post-release frame is taken this many frames after the gripper opens, so the
# item has actually dropped out of view rather than still hanging from the fingers.
POST_GAP = 5
# How far into the carrying stretch to seed the wrist camera. The middle is furthest from
# both the grasp and the release, so the item is least likely to be half in the fingers.
HOLD_FRACTION = 0.5
# A carry has to end somewhere near the end: the robot picks, crosses to the bin and lets
# go, so the release cannot fall in the opening moments. A stretch that does is the gripper
# sitting closed and empty before the episode starts in earnest, and seeding the wrist
# camera inside it puts the point on bare table. Across the four jr-pnp recordings every
# real carry released after 58% of the episode and every false one before 19%, so the line
# sits between them with room on both sides.
MIN_RELEASE = 0.35


def tool_column(data: pd.DataFrame) -> str:
    for name in data.columns:
        if "tool" in name and "timestamp" not in name and name.startswith("observation"):
            return name
    raise SystemExit(f"no gripper column among {list(data.columns)}; pass --tool-column")


def scalar(values) -> np.ndarray:
    return np.array([float(np.asarray(v).ravel()[0]) for v in values])


def carry_window(tool: np.ndarray, min_frames: int = 25) -> tuple[int, int] | None:
    """(first, last) frame of the last stretch with the gripper closed, read from the end.

    Reading backwards matters. A closed gripper is not the same as a holding one -- the
    recordings start with long closed-but-empty stretches, and the longest closed stretch
    in an episode is sometimes one of those, so the longest is the wrong one to take.

    The last one is usually the carry, since it is the one the release ends, but not
    always: in tomek-1 episode 8 the gripper closes again on the way home, after the box
    is already in the bin, and that late empty squeeze is what this returns. Nothing in
    the gripper signal separates it from a real carry, so an episode whose chosen stretch
    is not also its longest is worth a look; there were five such episodes in the four
    jr-pnp recordings.
    """
    if tool.size < min_frames:
        return None
    closed = tool > (tool.min() + tool.max()) / 2
    floor = MIN_RELEASE * tool.size
    end = None
    for i in range(len(closed) - 1, 0, -1):
        if end is None and closed[i - 1] and not closed[i]:
            if i - 1 < floor:
                return None                   # too early to be a carry: closed and empty
            end = i - 1                       # the release: closed here, open next
        elif end is not None and not closed[i]:
            start = i + 1
            return (start, end) if end - start >= min_frames else None
    if end is not None and end >= min_frames:
        return 0, end                         # closed from the very first frame
    return None


def table_difference(before: np.ndarray, after: np.ndarray, roi: tuple) -> dict | None:
    """Where the biggest thing left the table: {"point": [x, y] in 0..1 of the whole frame,
    "height": blob height as a share of the frame, "strength": mean difference in the
    blob, "bbox": blob box in pixels}, or None when nothing worth the name changed.

    `before` holds frames from before the grasp and `after` frames from after the
    release, each (n, height, width, 3). A pixel's score is the smallest difference
    over every before/after pair, so only what differs in ALL of them -- the item --
    survives; see the module docstring. Frames are blurred first so sensor noise and a
    pixel of camera shake do not, and the threshold is Otsu's rather than a constant
    because the two rooms are lit differently. The point is the centre of the blob's
    strongest half, weighted by score: a shadow the item casts differs faintly and an
    item strongly, so the plain centroid of the two together drifted onto bare table.
    """
    height, width = before.shape[1:3]
    x0, y0, x1, y1 = (int(roi[0] * width), int(roi[1] * height),
                      int(roi[2] * width), int(roi[3] * height))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    def prepared(frames):
        return np.stack([cv2.GaussianBlur(f[y0:y1, x0:x1], (5, 5), 0) for f in frames]) \
                 .astype(np.int16)

    pre, post = prepared(before), prepared(after)
    score = None
    for frame in post:
        pair = np.abs(pre - frame[None]).max(-1).min(0)      # min over pre frames
        score = pair if score is None else np.minimum(score, pair)
    score = score.astype(np.uint8)
    _, binary = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < MIN_BLOB * width * height:
        return None
    inside = labels == biggest
    ys, xs = np.where(inside)
    weight = score[inside].astype(float)
    strong = weight >= 0.5 * weight.max()
    cx = np.average(xs[strong], weights=weight[strong])
    cy = np.average(ys[strong], weights=weight[strong])
    if not inside[int(round(cy)), int(round(cx))]:
        nearest = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
        cx, cy = xs[nearest], ys[nearest]
    return {"point": [round((x0 + cx) / width, 4), round((y0 + cy) / height, 4)],
            "height": stats[biggest, cv2.CC_STAT_HEIGHT] / height,
            "strength": float(weight.mean()),
            "bbox": [int(x0 + stats[biggest, cv2.CC_STAT_LEFT]),
                     int(y0 + stats[biggest, cv2.CC_STAT_TOP]),
                     int(stats[biggest, cv2.CC_STAT_WIDTH]),
                     int(stats[biggest, cv2.CC_STAT_HEIGHT])]}


def lifted_item(frame: np.ndarray, after: np.ndarray, bbox: list) -> list | None:
    """Where the item hangs in the gripper at the lift frame, as [x, y] in 0..1.

    Against the emptied scene, everything the arm brought into the frame stands out as
    one silhouette, and the item is its lowest part: the gripper points down and the
    item hangs below the fingers. Only the silhouette near where the gripper stood at
    frame 0 (`bbox`, the suspect blob) is considered, since the arm's shadow and other
    moved things stand out too. The point sits a few pixels above the lowest row so it
    is on the item rather than on its edge.
    """
    height, width = frame.shape[:2]
    blurred = cv2.GaussianBlur(frame, (5, 5), 0).astype(np.int16)
    silhouette = np.min([np.abs(blurred - cv2.GaussianBlur(a, (5, 5), 0).astype(np.int16))
                         .max(-1) for a in after], axis=0)
    binary = (silhouette > SILHOUETTE).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    bx, by, bw, bh = bbox
    x0, x1 = max(0, bx - 15), min(width, bx + bw + 15)
    y0, y1 = max(0, by - 60), min(height, by + bh + 25)   # the gripper rose; the item hangs a little lower than the fingers did
    count, labels, _, _ = cv2.connectedComponentsWithStats(binary, 8)
    best = None
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        near = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
        if near.sum() >= 30 and (best is None or near.sum() > best[0]):
            best = (near.sum(), xs[near], ys[near])
    if best is None:
        return None
    _, xs, ys = best
    bottom = ys.max()
    x = float(np.median(xs[ys >= bottom - 5]))
    y = float(max(bottom - 6, ys.min()))
    return [round(x / width, 4), round(y / height, 4)]


def sample_frames(start: int, release: int, length: int) -> tuple[list, list]:
    """Which frames of an episode to compare: before the grasp, after the release."""
    pre = np.linspace(0, max(start - 1, 0), N_PRE).round().astype(int)
    post = np.linspace(min(release + POST_GAP, length - 1), length - 1, N_POST) \
             .round().astype(int)
    return sorted(set(pre.tolist())), sorted(set(post.tolist()))


def read_frames(dataset_dir: Path, info: dict, episode, key: str, wanted: list) -> dict:
    """{frame index within the episode: rgb frame} for the few frames asked for.

    Decoded from the start of the file rather than sought to: seeking these clips is not
    reliable. For one episode; `detect` does not use this, it decodes a file at a time.
    """
    chunk_col, file_col = file_columns(key)
    path = dataset_dir / video_rel_path(info, key, episode[chunk_col], episode[file_col])
    offset, length = int(episode[f"offset:{key}"]), int(episode["length"])
    wanted = sorted({min(max(0, w), length - 1) for w in wanted})
    out, container = {}, av.open(str(path))
    try:
        decoder = container.decode(video=0)
        for _ in range(offset):
            next(decoder)
        position = 0
        for want in wanted:
            while position < want:
                next(decoder)
                position += 1
            out[want] = next(decoder).to_ndarray(format="rgb24")
            position += 1
    finally:
        container.close()
    return out


def is_wrist(key: str) -> bool:
    """Whether this camera rides on the hand, and so is answered by the gripper rather
    than by what left the table."""
    name = camera_name(key)
    return "wrist" in name or "hand" in name


def answer_file(job: dict) -> dict:
    """{episode: (point or None, seed frame)} for every episode in one overview-camera
    file, decoding it once.

    Runs in a worker process. `job` names the file and, per episode, its frame offset in
    the file and which frames to compare; the frames wanted are kept as they stream past
    and everything else is dropped, so memory is a few frames per episode. The lift frame
    is decoded along with the rest so that a suspect answer can be replaced without a
    second pass over the file.
    """
    wanted = {}
    for ep in job["episodes"]:
        for frame in ep["pre"] + ep["post"] + [ep["lift"]]:
            wanted.setdefault(ep["offset"] + frame, None)
    last = max(wanted)
    with av.open(job["path"]) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                wanted[i] = frame.to_ndarray(format="rgb24")
            if i >= last:
                break
    out = {}
    for ep in job["episodes"]:
        before = np.stack([wanted[ep["offset"] + f] for f in ep["pre"]])
        after = np.stack([wanted[ep["offset"] + f] for f in ep["post"]])
        found = table_difference(before, after, tuple(job["roi"]))
        if found is None:
            out[ep["episode"]] = (None, 0)
        elif found["height"] > SUSPECT_HEIGHT or found["strength"] < SUSPECT_STRENGTH:
            lifted = lifted_item(wanted[ep["offset"] + ep["lift"]], after, found["bbox"])
            out[ep["episode"]] = (lifted, ep["lift"] if lifted else 0)
        else:
            out[ep["episode"]] = (found["point"], 0)
    return out


def detect(dataset_dir: Path, episodes: list | None = None, wrist_point=WRIST_POINT,
           side_roi=SIDE_ROI, hold_fraction=HOLD_FRACTION, tool_col=None, report=None,
           workers: int | None = None) -> dict:
    """The annotation file this dataset would get with nobody clicking anything.

    `report`, if given, is called with a number of episodes each time some are done.
    """
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    keys = video_keys(info)
    meta = episode_offsets(load_episodes(dataset_dir), info).set_index("episode_index")
    data = pd.concat([pd.read_parquet(f)
                      for f in sorted((dataset_dir / "data").rglob("*.parquet"))])
    column = tool_col or tool_column(data)

    wanted = episodes if episodes is not None else sorted(int(i) for i in meta.index)
    out, skipped, jobs = {}, {"no carry": 0, "no difference": 0, "at the lift": 0}, {}
    for index in wanted:
        episode = meta.loc[index]
        rows = data[data.episode_index == index].sort_values("frame_index")
        window = carry_window(scalar(rows[column].to_numpy()))
        length = int(episode["length"])
        entry = {"frame": 0, "frames": {}, "points": {}}
        if window is None:
            # No carry to read: still worth looking at the table, since an item that
            # was there at the start and gone at the end is an item all the same. The
            # wrist camera stays unanswered, as there is no frame to trust for it.
            skipped["no carry"] += 1
            start, release = int(0.4 * length), int(0.9 * length)
        else:
            start, release = window
            hold = int(start + hold_fraction * (release - start))
            entry["frame"] = hold
            for key in keys:
                if is_wrist(key):
                    entry["frames"][key] = hold
                    entry["points"][key] = [[wrist_point[0], wrist_point[1], 1]]
        pre, post = sample_frames(start, release, length)
        lift = min(start + LIFT_FRAMES, release)
        for key in keys:
            if is_wrist(key):
                continue
            entry["frames"][key] = 0
            chunk_col, file_col = file_columns(key)
            path = dataset_dir / video_rel_path(info, key, episode[chunk_col],
                                                episode[file_col])
            job = jobs.setdefault((key, str(path)), {"path": str(path), "key": key,
                                                     "roi": list(side_roi), "episodes": []})
            job["episodes"].append({"episode": index, "offset": int(episode[f"offset:{key}"]),
                                    "pre": pre, "post": post, "lift": lift})
        out[str(index)] = entry

    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 4) // 2))
    # spawn, not fork: this also runs inside the GUI's server threads, and a forked copy
    # of a threaded process is not a safe place to decode video from.
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for job, answers in zip(jobs.values(), pool.map(answer_file, jobs.values())):
            for index, (point, seed) in answers.items():
                if point is None:
                    skipped["no difference"] += 1
                else:
                    out[str(index)]["points"][job["key"]] = [[point[0], point[1], 1]]
                    out[str(index)]["frames"][job["key"]] = seed
                    skipped["at the lift"] += bool(seed)
            if report:
                report(len(answers))

    print(f"{len(wanted)} episodes | no carry stretch found: {skipped['no carry']} | "
          f"overview camera undecided: {skipped['no difference']} | "
          f"answered at the lift frame: {skipped['at the lift']}")
    return {"dataset": dataset_dir.name, "cameras": keys, "episodes": out}


def points_path(dataset_dir: Path) -> Path:
    return REPO_ROOT / "data" / "points" / f"{dataset_dir.name}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="e.g. data/clear/jr-pnp-tomek-1")
    parser.add_argument("--episode", type=int, action="append",
                        help="only this episode; repeatable")
    parser.add_argument("--wrist-point", default=",".join(map(str, WRIST_POINT)),
                        help="x,y in 0..1 where the held item sits in a wrist view")
    parser.add_argument("--side-roi", default=",".join(map(str, SIDE_ROI)),
                        help="x0,y0,x1,y1 in 0..1: the part of an overview frame to compare")
    parser.add_argument("--hold-fraction", type=float, default=HOLD_FRACTION)
    parser.add_argument("--tool-column", default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="processes decoding video files; default half the cores")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    found = detect(dataset_dir, args.episode,
                   tuple(float(v) for v in args.wrist_point.split(",")),
                   tuple(float(v) for v in args.side_roi.split(",")),
                   args.hold_fraction, args.tool_column, workers=args.workers)
    out = args.out or points_path(dataset_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Merge rather than replace. --episode asks about a few episodes, not for a file
    # holding only those: overwriting would throw away every other episode's answer,
    # including ones a person clicked, and the loss is silent.
    if out.exists():
        existing = json.loads(out.read_text())
        kept = {e for e, ann in existing["episodes"].items() if ann.get("by_hand")}
        existing["cameras"] = found["cameras"]
        existing["episodes"].update({e: a for e, a in found["episodes"].items()
                                     if e not in kept})
        found = existing
        if kept & set(found["episodes"]):
            print(f"kept {len(kept & set(found['episodes']))} episodes answered by hand")
    out.write_text(json.dumps(found, indent=2))
    print(f"Wrote {out} ({len(found['episodes'])} episodes)")


if __name__ == "__main__":
    main()
