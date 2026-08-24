"""
Place and fix the points that scripts/mask.py masks with.

Pick an episode and you get every camera's view of the same frame. Scrub to a frame
where the item is clearly visible, then click on it in each view; the click saves that
camera's point and the frame you are looking at back into the json. Every camera needs
at least one point, so scrub until the item is visible in all of them.

Clicks accumulate, and the toggle above the views decides what a click means: "add"
grows the mask, "subtract" carves out of it. That is how a mask that only caught part
of the item -- SAM3 reading a texture edge as the object's edge -- gets fixed: click the
part it missed, watch the preview grow, carve back anything it over-reached.

Under each view sits what SAM3 actually segmented from that click, so a point that the
model reads differently than the annotator meant shows up immediately instead of six
hours into masking. That part needs demo/server.py running; without it the previews
just stay empty and everything else still works.

Works on a json from scripts/find_point_with_gemini.py or an empty one from
scripts/init_annotations.py.

Usage: python scripts/correct_point.py data/annotations/<name>.json
"""
import argparse
import concurrent.futures
import json
import sys
import threading
from pathlib import Path

import av
import cv2
import gradio as gr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from demo.app import close_session, new_session, predict  # noqa: E402
from find_point_with_gemini import (  # noqa: E402
    as_points, camera_name, episode_offsets, file_columns, load_episodes, mark,
    save_preview, video_rel_path,
)


def load_state(path: Path) -> dict:
    ann_file = json.loads(path.read_text())
    dataset_dir = REPO_ROOT / "data" / "clear" / ann_file["dataset"]
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    episodes = episode_offsets(load_episodes(dataset_dir), info).set_index("episode_index")
    return {"path": path, "file": ann_file, "keys": ann_file["cameras"],
            "dataset_dir": dataset_dir, "info": info, "episodes": episodes,
            "prefetch": None, "prefetch_lock": threading.Lock(),
            "prefetch_executor": concurrent.futures.ThreadPoolExecutor(max_workers=1)}


def _decode_camera(state: dict, ep, key: str) -> tuple:
    """One camera's jpeg frames for episode `ep` -- the unit of work read_episode
    fans out across threads, one per camera, since each reads its own file."""
    chunk_col, file_col = file_columns(key)
    src = state["dataset_dir"] / video_rel_path(state["info"], key, ep[chunk_col], ep[file_col])
    container = av.open(str(src))
    try:
        frames = container.decode(video=0)
        for _ in range(int(ep[f"offset:{key}"])):
            next(frames)
        return key, [encode(next(frames).to_ndarray(format="rgb24"))
                     for _ in range(int(ep["length"]))]
    finally:
        container.close()


def read_episode(state: dict, episode_index: int) -> dict:
    """{camera key: [jpeg bytes per frame]} for one episode.

    Decoded from the start of each file: seeking on these clips is not reliable, and
    the files are cut at 200 MB so the run-up is a few thousand frames, a handful of
    seconds. Kept as jpeg rather than arrays -- 15x less RAM, and decoding the one
    frame on screen is free. Cameras are separate files, so they decode in parallel
    threads instead of one after another.
    """
    ep = state["episodes"].loc[episode_index]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(state["keys"])) as pool:
        return dict(pool.map(lambda key: _decode_camera(state, ep, key), state["keys"]))


def take_prefetched(state: dict, episode_index: int) -> dict | None:
    """The pending or finished result of a schedule_prefetch call for this episode, if
    there is one -- waiting for it here is never worse than decoding from scratch."""
    with state["prefetch_lock"]:
        pending = state["prefetch"]
        if not pending or pending["index"] != episode_index:
            return None
        state["prefetch"] = None
    return pending["future"].result()


def schedule_prefetch(state: dict, episode_index: int) -> None:
    """Kick off decoding `episode_index` in the background, so it is ready by the time
    the annotator clicks their way to it. A prefetch the annotator never lands on (they
    went somewhere else instead) just finishes unused -- wasted CPU, bounded to one
    episode, never a correctness issue, since decoding does not touch the points."""
    with state["prefetch_lock"]:
        pending = state["prefetch"]
        if pending and pending["index"] == episode_index:
            return
        future = state["prefetch_executor"].submit(read_episode, state, episode_index)
        state["prefetch"] = {"index": episode_index, "future": future}


def encode(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def decode(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sam_preview(frame: np.ndarray, points):
    """What SAM3 makes of these points together, as the server's red overlay.

    A throwaway session per call: one frame in, one frame out, nothing to track. The
    server may not be running -- that is not worth interrupting the annotation for, so
    the failure comes back as a message instead of an exception.
    """
    if not points:
        return None, ""
    try:
        session_id = new_session()
    except Exception as e:
        return None, f"  *(no SAM preview: {type(e).__name__} -- is demo/server.py running?)*"
    try:
        return predict(session_id, frame, points=points), ""
    except Exception as e:
        return None, f"  *(SAM preview failed: {type(e).__name__})*"
    finally:
        close_session(session_id)


def is_complete(state: dict, episode_index: int) -> bool:
    """Every camera has at least one point."""
    points = state["file"]["episodes"][str(episode_index)]["points"]
    return all(as_points(points.get(key)) for key in state["keys"])


def progress(state: dict) -> str:
    indices = sorted(int(k) for k in state["file"]["episodes"])
    done = [i for i in indices if is_complete(state, i)]
    todo = [i for i in indices if i not in set(done)]
    line = f"**{len(done)}/{len(indices)} episodes done.**"
    if todo:
        line += f" Left to do: {', '.join(str(i) for i in todo[:12])}"
        line += " ..." if len(todo) > 12 else ""
    return line


def save(state: dict, episode_index: int, frame_idx: int, points: dict, frames: dict) -> None:
    """Write the annotation back to the json and redraw that episode's preview jpeg."""
    state["file"]["episodes"][str(episode_index)] = {"frame": frame_idx, "points": points}
    state["path"].write_text(json.dumps(state["file"], indent=2))

    preview_dir = state["path"].with_suffix("")
    preview_dir.mkdir(parents=True, exist_ok=True)
    save_preview({k: decode(frames[k][frame_idx]) for k in state["keys"]}, points,
                 state["keys"], preview_dir / f"ep{episode_index:04d}.jpg")


def describe(state: dict, episode_index: int, frame_idx: int, points: dict) -> str:
    def one(key):
        pts = as_points(points.get(key))
        if not pts:
            return "**?**"
        plus = sum(1 for p in pts if p[2])
        return f"{plus}+" + (f"/{len(pts) - plus}-" if len(pts) > plus else "")

    shown = ", ".join(f"{camera_name(k)}: {one(k)}" for k in state["keys"])
    return f"**episode {episode_index}, frame {frame_idx}** -- {shown}"


def build_ui(state: dict) -> gr.Blocks:
    keys = state["keys"]
    indices = sorted(int(k) for k in state["file"]["episodes"])

    def show_episode(episode_index):
        """Load an episode and jump to the frame its annotation names."""
        episode_index = int(episode_index)
        ann = state["file"]["episodes"][str(episode_index)]
        frames = take_prefetched(state, episode_index) or read_episode(state, episode_index)
        length = len(frames[keys[0]])
        frame_idx = min(ann["frame"], length - 1)
        points = {k: ann["points"][k] for k in keys if k in ann["points"]}
        previews, notes = zip(*[sam_preview(decode(frames[k][frame_idx]), points.get(k)) for k in keys])
        next_index = step(episode_index, +1)
        if next_index != episode_index:
            schedule_prefetch(state, next_index)
        return [frames, points, gr.Slider(maximum=length - 1, value=frame_idx),
                describe(state, episode_index, frame_idx, points) + "".join(set(notes)),
                progress(state),
                *[mark(decode(frames[k][frame_idx]), points.get(k)) for k in keys],
                *previews]

    def show_frame(frames, frame_idx, points, episode_index):
        """Scrubbing moves the views; the SAM previews below belong to the last click,
        so they are cleared rather than left showing a mask for another frame."""
        if not frames:
            return [None] * (2 * len(keys) + 1)
        frame_idx = int(frame_idx)
        return [describe(state, int(episode_index), frame_idx, points),
                *[mark(decode(frames[k][frame_idx]), points.get(k)) for k in keys],
                *[None] * len(keys)]

    def apply(key, frames, frame_idx, episode_index, points, new_list):
        """Every edit is the same: replace one camera's point list, adopt the frame, save,
        and show what SAM3 segments from all of them together."""
        if not frames:
            raise gr.Error("Pick an episode first.")
        frame_idx, episode_index = int(frame_idx), int(episode_index)
        points = {**points, key: new_list}
        save(state, episode_index, frame_idx, points, frames)
        frame = decode(frames[key][frame_idx])
        preview, note = sam_preview(frame, new_list)
        return (points, describe(state, episode_index, frame_idx, points) + "  *(saved)*" + note,
                progress(state), mark(frame, new_list), preview)

    def on_click(key):
        def handler(frames, frame_idx, episode_index, points, mode, evt: gr.SelectData):
            """A click appends to this camera's list -- clicks accumulate, the toggle
            decides whether this one grows the mask or carves out of it."""
            frame = decode(frames[key][int(frame_idx)])
            h, w = frame.shape[:2]
            label = 0 if mode.startswith("-") else 1
            point = [round(evt.index[0] / w, 4), round(evt.index[1] / h, 4), label]
            return apply(key, frames, frame_idx, episode_index, points,
                         as_points(points.get(key)) + [point])
        return handler

    def on_clear(key):
        def handler(frames, frame_idx, episode_index, points):
            return apply(key, frames, frame_idx, episode_index, points, [])
        return handler

    def on_undo(key):
        def handler(frames, frame_idx, episode_index, points):
            return apply(key, frames, frame_idx, episode_index, points,
                         as_points(points.get(key))[:-1])
        return handler

    def step(episode_index, delta):
        pos = indices.index(int(episode_index)) + delta
        return indices[max(0, min(len(indices) - 1, pos))]

    def next_todo(episode_index):
        """First episode after this one that still has a camera unanswered for, else
        the first such episode anywhere, else stay put."""
        todo = [i for i in indices if not is_complete(state, i)]
        return next((i for i in todo if i > int(episode_index)), todo[0] if todo else int(episode_index))

    with gr.Blocks(title=f"Points -- {state['path'].name}") as demo:
        frames_state, points_state = gr.State({}), gr.State({})
        gr.Markdown(f"**{state['path']}** -- {len(indices)} episodes, {len(keys)} cameras. "
                    "Scrub to a frame where the item is visible in all views, then click it in "
                    "each one. The row below each view is what SAM3 segments from those clicks -- "
                    "keep clicking the parts it misses until the mask covers the whole item.")
        counter = gr.Markdown()
        mode = gr.Radio(["+ add to mask", "- subtract from mask"], value="+ add to mask",
                        label="What a click does", scale=0)
        with gr.Row():
            prev_btn = gr.Button("< prev", scale=0)
            episode = gr.Dropdown(indices, value=indices[0], label="Episode", scale=1)
            next_btn = gr.Button("next >", scale=0)
            todo_btn = gr.Button("next unannotated >>", variant="primary", scale=0)

        images, previews, undo_btns, clear_btns = [], [], [], []
        with gr.Row():
            for key in keys:
                with gr.Column():
                    images.append(gr.Image(label=camera_name(key), type="numpy", interactive=False))
                    with gr.Row():
                        undo_btns.append(gr.Button("undo last point", size="sm"))
                        clear_btns.append(gr.Button("clear points", size="sm"))
                    previews.append(gr.Image(label=f"{camera_name(key)} -- SAM3 mask",
                                             type="numpy", interactive=False))
        frame_slider = gr.Slider(0, 1, step=1, value=0, label="Frame")
        status = gr.Markdown()

        load_outs = [frames_state, points_state, frame_slider, status, counter, *images, *previews]
        episode.change(show_episode, episode, load_outs)
        demo.load(show_episode, episode, load_outs)
        prev_btn.click(lambda i: step(i, -1), episode, episode)
        next_btn.click(lambda i: step(i, +1), episode, episode)
        todo_btn.click(next_todo, episode, episode)
        frame_slider.change(show_frame, [frames_state, frame_slider, points_state, episode],
                            [status, *images, *previews])
        for i, key in enumerate(keys):
            edit_ins = [frames_state, frame_slider, episode, points_state]
            edit_outs = [points_state, status, counter, images[i], previews[i]]
            images[i].select(on_click(key), edit_ins + [mode], edit_outs)
            undo_btns[i].click(on_undo(key), edit_ins, edit_outs)
            clear_btns[i].click(on_clear(key), edit_ins, edit_outs)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotations", help="e.g. data/annotations/ind-iso-4.json")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    build_ui(load_state(Path(args.annotations).resolve())).launch(
        server_name="0.0.0.0", server_port=args.port)
