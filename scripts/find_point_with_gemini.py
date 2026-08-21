"""
Phase 1 of masking a lerobot dataset: ask Gemini for one point per camera per
episode and write them to a json, so they can be eyeballed before any GPU time
is spent. Phase 2 is scripts/mask.py; scripts/correct_point.py fixes bad points.

Two phases per episode. First the overview ("side") camera is asked to find the
item, exactly as before. Then each remaining camera gets its own call: the
overview frame with the answer marked on it, plus its own frame, and the question
"where is that same item here" -- because the cameras are in different places, so
the same item sits somewhere else in every view. A camera where the item is not
visible is left with a null point and stays unmasked.

Two ways to pick the frame Gemini looks at:
  default        frame 0 -- the item sticking out of the crate.
  --middle-frame the middle of the episode -- the item the robot is holding.
                 A much easier question, but the point lands mid-grasp.

Needs demo/server.py running (for its /gemini endpoint).

Usage: python scripts/find_point_with_gemini.py data/clear/<name> [--middle-frame]
Output: data/annotations/<name>.json + one preview jpeg per episode next to it
"""
import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from demo.app import SERVER, rgb_to_b64  # noqa: E402

# Gemini's spatial convention is [y, x] normalized to 0..1000.
# Frame 0: the item is still in its crate, so most of the prompt is spent ruling
# out the other crates and items in the shot.
FIRST_FRAME_PROMPT = (
    "Your primary goal is to identify the single, main crate located in "
    "the foreground of the image, directly underneath the robot's tool. "
    "Within *only* that specific crate, find the one item that sticks out / "
    "protrudes beyond the top edge of that crate. "
    "Strictly follow these rules: "
    "1. Focus *only* on the items within or on the edge of the one main foreground crate. "
    "2. Completely ignore all background containers, other crates, and background items. "
    "3. Ignore the robot arm, its tools, and items on the distant floor. Point at the part of the item that is not overlapped by robot arm or gripper or any tool."
    "Point at the part of the item in the primary crate that sticks out."
)
# Mid-episode: the item is in the gripper, which makes it the only thing it can be.
MIDDLE_FRAME_PROMPT = (
    "The robot's vaccum gripper is holding exactly one item. "
    "Point at that item -- not at the gripper, the arm, or anything in the crates. "
    "If the item is partly hidden by the gripper, point at a visible part of it."
    "Point at the part of the item that is not overlapped by robot arm or gripper or any tool."
)
# Phase two: a follow-up turn in the same conversation, one per remaining camera.
# The overview frame and Gemini's own answer are still in the context, so the item
# can be referred to instead of described again. Asking for every camera at once in
# a single turn does not work -- the answers come back as the same coordinates for
# all of them, which cannot be right for cameras in different places.
MATCH_PROMPT = (
    "Here is the same instant from a camera mounted on the robot's gripper. "
    "Point at the same physical item you just pointed at in the previous image. "
    "This camera is somewhere else entirely, so the item sits in a different place and at a "
    "different size here -- work out where it is, do not reuse your previous coordinates. "
    "Point at a part of the item that is not covered by the robot arm, the gripper or a tool. "
    "Set visible=false if that item is out of frame or fully hidden in this image."
)
POINT_SCHEMA = {
    "type": "object",
    "properties": {
        "point": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
    },
    "required": ["point"],
}
MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "visible": {"type": "boolean"},
        "point": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
    },
    "required": ["visible"],
}


def video_keys(info: dict) -> list:
    """The dataset's camera keys, e.g. ["observation.images.side", ...]."""
    return [k for k, v in info["features"].items() if v["dtype"] == "video"]


def camera_name(key: str) -> str:
    """observation.images.wrist_left -> wrist_left (what Gemini is asked about)."""
    return key.rsplit(".", 1)[-1]


def reference_key(keys: list) -> str:
    """The camera that gets asked first, and whose answer the others are matched to:
    the overview one, since the whole scene is in it."""
    for key in keys:
        if camera_name(key) == "side":
            return key
    return keys[0]


def load_episodes(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def video_rel_path(info: dict, video_key: str, chunk_index: int, file_index: int) -> str:
    return info["video_path"].format(video_key=video_key, chunk_index=chunk_index, file_index=file_index)


def episode_offsets(episodes: pd.DataFrame, info: dict) -> pd.DataFrame:
    """Add `offset:<camera key>`: the frame an episode starts at inside that camera's file.

    Several episodes share one physical file per camera, and the parquet already
    records where each one starts as a timestamp, so no need to add up lengths.
    """
    out = episodes.copy()
    for key in video_keys(info):
        out[f"offset:{key}"] = (out[f"videos/{key}/from_timestamp"] * info["fps"]).round().astype(int)
    return out


def file_columns(video_key: str) -> tuple:
    return f"videos/{video_key}/chunk_index", f"videos/{video_key}/file_index"


def ask_gemini(prompt: str, images: list, schema: dict, history: list = None) -> dict:
    """`history` is earlier turns as [{"role", "prompt", "images"}], with rgb arrays."""
    payload = {"prompt": prompt, "images": [rgb_to_b64(i) for i in images], "schema": schema}
    if history:
        payload["history"] = [{"role": t["role"], "prompt": t.get("prompt", ""),
                               "images": [rgb_to_b64(i) for i in t.get("images", [])]}
                              for t in history]
    r = requests.post(f"{SERVER}/gemini", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["result"]


def to_xy(point) -> list:
    """Gemini answers [y, x]; the json keeps [x, y] in 0..1.

    The scale is not stable: the first call answers in 0..1000 as documented, the
    matching call has come back in 0..1, so pick the scale from the values. A real
    0..1000 answer on an edge (a 0 or a 1) would be read as 0..1 here, which is one
    corner pixel off -- cheaper than trusting either convention.
    """
    y, x = point
    scale = 1000 if max(abs(y), abs(x)) > 1 else 1
    return [round(x / scale, 4), round(y / scale, 4)]


def detect_points(frames: dict, keys: list, prompt: str) -> dict:
    """{camera key: [x, y] or None} for one episode, as a short conversation: ask the
    overview camera, then ask each remaining camera to find that same item, with the
    first question and answer still in the context.

    Each follow-up replays only the first exchange, so the cameras cannot copy each
    other's coordinates.
    """
    ref = reference_key(keys)
    answer = ask_gemini(prompt, [frames[ref]], POINT_SCHEMA)
    points = {ref: to_xy(answer["point"])}
    history = [{"role": "user", "prompt": prompt, "images": [frames[ref]]},
               {"role": "model", "prompt": json.dumps(answer)}]

    for key in keys:
        if key == ref:
            continue
        reply = ask_gemini(MATCH_PROMPT, [frames[key]], MATCH_SCHEMA, history)
        points[key] = to_xy(reply["point"]) if reply.get("visible") and reply.get("point") else None
    return {key: points[key] for key in keys}  # keep the dataset's camera order


def grab_frames(src_path: Path, wanted: list) -> dict:
    """{frame position inside the file -> rgb frame}, in one forward pass.

    Decoding the file once beats seeking per episode: the episodes in a file cover
    it end to end anyway, and av's seek on these clips is not reliable.
    """
    out = {}
    todo = sorted(wanted)
    container = av.open(str(src_path))
    try:
        for i, frame in enumerate(container.decode(video=0)):
            if not todo:
                break
            if i == todo[0]:
                out[i] = frame.to_ndarray(format="rgb24")
                todo.pop(0)
    finally:
        container.close()
    return out


def collect_frames(dataset_dir: Path, info: dict, episodes: pd.DataFrame, middle_frame: bool,
                   bar=None) -> tuple:
    """{episode index: {camera key: frame}}, {episode index: frame index within the episode}."""
    frames, seeds = {}, {}
    for _, ep in episodes.iterrows():
        seeds[int(ep["episode_index"])] = int(ep["length"]) // 2 if middle_frame else 0

    for key in video_keys(info):
        chunk_col, file_col = file_columns(key)
        for (chunk_idx, file_idx), group in episodes.groupby([chunk_col, file_col]):
            targets = {int(ep[f"offset:{key}"]) + seeds[int(ep["episode_index"])]: int(ep["episode_index"])
                       for _, ep in group.iterrows()}
            src = dataset_dir / video_rel_path(info, key, chunk_idx, file_idx)
            for pos, frame in grab_frames(src, list(targets)).items():
                frames.setdefault(targets[pos], {})[key] = frame
            if bar:
                bar.update(len(group))
    return frames, seeds


def mark(frame: np.ndarray, point) -> np.ndarray:
    """A copy of the frame with a red dot on the point (nothing drawn if point is None)."""
    out = frame.copy()
    if point is None:
        return out
    h, w = out.shape[:2]
    r = max(4, w // 60)
    centre = (round(point[0] * w), round(point[1] * h))
    cv2.circle(out, centre, r, (255, 0, 0), -1)
    cv2.circle(out, centre, r, (255, 255, 255), 1)
    return out


def save_preview(frames: dict, points: dict, keys: list, path: Path) -> None:
    """All cameras side by side, each with its point drawn, as one jpeg. Browse the
    folder as thumbnails and a bad point is obvious without opening anything."""
    height = max(frames[k].shape[0] for k in keys)
    tiles = []
    for key in keys:
        tile = mark(frames[key], points.get(key))
        if tile.shape[0] != height:  # match heights so they can sit in one row
            scale = height / tile.shape[0]
            tile = cv2.resize(tile, (round(tile.shape[1] * scale), height))
        # a camera can be missing from `points` entirely: not answered for yet
        label = camera_name(key) + ("" if points.get(key) else
                                    " (not visible)" if key in points else " (?)")
        cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(tile)
    cv2.imwrite(str(path), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def main(dataset_dir: str, middle_frame: bool, episode: int | None) -> Path:
    dataset_dir = Path(dataset_dir).resolve()
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    keys = video_keys(info)
    prompt = MIDDLE_FRAME_PROMPT if middle_frame else FIRST_FRAME_PROMPT

    episodes = episode_offsets(load_episodes(dataset_dir), info)
    name = dataset_dir.name
    if episode is not None:
        # a one-episode json of its own, so a test run cannot clobber a full one
        episodes = episodes[episodes["episode_index"] == episode]
        if episodes.empty:
            raise SystemExit(f"no episode {episode} in {dataset_dir}")
        name = f"_test_{name}_episode_{episode}"
    out_path = REPO_ROOT / "data" / "annotations" / f"{name}.json"
    preview_dir = out_path.with_suffix("")
    preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(episodes)} episodes x {len(keys)} cameras: {camera_name(reference_key(keys))} "
          f"is asked first, the rest are matched to it")
    with tqdm(total=len(episodes) * len(keys), desc="reading frames", unit="frame") as bar:
        frames, seeds = collect_frames(dataset_dir, info, episodes, middle_frame, bar)

    annotations = {}
    for episode_index in tqdm(sorted(frames), desc="asking gemini", unit="ep"):
        points = detect_points(frames[episode_index], keys, prompt)
        annotations[episode_index] = {"frame": seeds[episode_index], "points": points}
        save_preview(frames[episode_index], points, keys, preview_dir / f"ep{episode_index:04d}.jpg")

    out_path.write_text(json.dumps({
        "dataset": dataset_dir.name,
        "cameras": keys,
        "prompt": prompt,
        "match_prompt": MATCH_PROMPT,
        "episodes": {str(k): annotations[k] for k in sorted(annotations)},
    }, indent=2))
    print(f"Wrote {out_path} ({len(annotations)} episodes)")
    print(f"Previews in {preview_dir}. Fix bad points with: "
          f"python scripts/correct_point.py {out_path.relative_to(REPO_ROOT)}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", nargs="?", help="e.g. data/clear/ind-iso-1")
    parser.add_argument("--middle-frame", action="store_true",
                        help="point at the item held by the robot in the middle frame, "
                             "instead of the item sticking out of the crate in frame 0")
    parser.add_argument("--episode", type=int, metavar="N",
                        help="annotate only episode N, into its own json (for a test run)")
    args = parser.parse_args()

    if args.dataset_dir:
        main(args.dataset_dir, args.middle_frame, args.episode)
    else:
        parser.error("dataset_dir is required")
