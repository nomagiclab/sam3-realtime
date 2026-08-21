"""
Phase 2 of masking a lerobot dataset: take the points from
scripts/find_point_with_gemini.py (fixed up in scripts/correct_point.py) and let
SAM3 mask every camera with them. Everything else (parquet, meta) is copied
unchanged.

Each camera is masked from its own point, since it sees the item from its own
angle. A camera with a null point (the item is not visible there) is copied
through unmasked.

An episode is seeded at whatever frame the json names. If that is frame 0 the
episode is just tracked forward. Otherwise it is tracked twice from the seed --
backwards over the frames before it, forwards over the frames after it -- because
the model has no notion of time, so a reversed clip is just another clip to it.

Needs demo/server.py running (SAM3).

Usage: python scripts/mask.py data/annotations/<name>.json [--episode N]
Output: data/masked/<name>   (or one preview mp4 per camera with --episode)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import av
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from demo.app import H264Writer, close_session, new_session, predict  # noqa: E402
from find_point_with_gemini import (  # noqa: E402
    as_points, camera_name, episode_offsets, file_columns, load_episodes, video_rel_path,
)


def track(frames: list, points: list) -> list:
    """Mask a run of frames in one session, prompting with `points` on the first one.
    Returns the server's overlays, in the order the frames were given."""
    session_id = new_session()
    try:
        return [predict(session_id, frame, points=points if i == 0 else None)
                for i, frame in enumerate(frames)]
    finally:
        close_session(session_id)


def mask_episode(frames: list, seed: int, prompt, bar=None) -> list:
    """Mask one episode's frames, seeded at index `seed`. Returns one frame out per
    frame in -- the frames untouched if there is no prompt (item not visible here).

    `prompt` is whatever the json holds for this camera: one point, or a list of points
    with labels that the annotator refined the mask with.

    The backward half is fed to the model in reverse and flipped back afterwards;
    the seed frame goes through both sessions and the forward copy is the one kept.
    """
    points = as_points(prompt)
    if not points:
        out = frames
    elif seed == 0:
        out = track(frames, points)
    else:
        backward = track(frames[seed::-1], points)        # seed .. 0
        forward = track(frames[seed:], points)            # seed .. end
        out = backward[:0:-1] + forward                   # drop backward's seed copy
    if bar:
        bar.update(len(out))  # after both halves: the seed frame is masked twice, kept once
    return out


def episode_frames(frames_iter, length: int) -> list:
    """Pull one episode's frames off a container's decoder."""
    # ponytail: whole episode in RAM (~360 MB worst case here); stream it in halves
    # if a dataset ever shows up with much longer episodes.
    return [next(frames_iter).to_ndarray(format="rgb24") for _ in range(length)]


def mask_video_file(src_path: Path, dst_path: Path, episodes: pd.DataFrame, video_key: str,
                    annotations: dict, fps: float) -> None:
    """episodes: the rows (one per episode) packed into this one physical video file.

    One decoder for the whole file: the episodes in it are back to back, so each
    one's frames start exactly where the previous one stopped.
    """
    container = av.open(str(src_path))
    frames = container.decode(video=0)
    writer = H264Writer(str(dst_path), fps)
    try:
        with tqdm(total=int(episodes["length"].sum()), desc=dst_path.name, unit="frame") as bar:
            for _, ep in episodes.sort_values("episode_index").iterrows():
                ann = annotations[int(ep["episode_index"])]
                for frame in mask_episode(episode_frames(frames, int(ep["length"])),
                                          ann["frame"], ann["points"].get(video_key), bar):
                    writer.write(frame)
    finally:
        writer.close()
        container.close()


def report_points(annotations: dict, keys: list) -> None:
    """Say out loud what will not be masked, before spending hours on what will.

    A camera nobody answered for and one deliberately marked "not visible" both end up
    copied through untouched, and they are indistinguishable in the output, so the only
    place to catch a half-finished annotation is here.
    """
    for key in keys:
        never = sorted(i for i, a in annotations.items() if key not in a["points"])
        if never:
            print(f"WARNING {camera_name(key)}: {len(never)} episodes were never answered for "
                  f"(e.g. {never[:5]}) -- they will be copied through unmasked")
    unmasked = {camera_name(key): sum(1 for a in annotations.values() if not a["points"].get(key))
                for key in keys}
    print(f"Episodes left unmasked per camera: {unmasked}")


def main(annotations_path: str) -> None:
    ann_file = json.loads(Path(annotations_path).read_text())
    annotations = {int(k): v for k, v in ann_file["episodes"].items()}
    keys = ann_file["cameras"]
    dataset_dir = REPO_ROOT / "data" / "clear" / ann_file["dataset"]
    out_dir = REPO_ROOT / "data" / "masked" / ann_file["dataset"]
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists, remove it first")

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    episodes = episode_offsets(load_episodes(dataset_dir), info)
    missing = set(episodes["episode_index"].astype(int)) - set(annotations)
    if missing:
        raise SystemExit(f"{len(missing)} episodes have no point in the json, "
                         f"e.g. {sorted(missing)[:5]} -- annotate them first")

    report_points(annotations, keys)

    print(f"Copying {dataset_dir} -> {out_dir}")
    shutil.copytree(dataset_dir, out_dir)

    info_path = out_dir / "meta" / "info.json"
    out_info = json.loads(info_path.read_text())
    for key in keys:
        out_info["features"][key]["info"]["video.codec"] = "h264"  # re-encoded below, was av1
    info_path.write_text(json.dumps(out_info, indent=4))

    for key in keys:
        chunk_col, file_col = file_columns(key)
        for (chunk_idx, file_idx), group in episodes.groupby([chunk_col, file_col]):
            rel = video_rel_path(info, key, chunk_idx, file_idx)
            fps = info["features"][key]["info"]["video.fps"]
            unmasked = sum(1 for _, ep in group.iterrows()
                           if not annotations[int(ep["episode_index"])]["points"].get(key))
            print(f"Masking {rel} ({len(group)} episodes, {int(group['length'].sum())} frames"
                  + (f", {unmasked} without a point" if unmasked else "") + ")")
            mask_video_file(dataset_dir / rel, out_dir / rel, group, key, annotations, fps)

    print(f"Done -> {out_dir}")


def preview_episode(annotations_path: str, episode_index: int) -> list:
    """Mask one episode into a standalone mp4 per camera, without touching the dataset."""
    ann_file = json.loads(Path(annotations_path).read_text())
    ann = ann_file["episodes"][str(episode_index)]
    dataset_dir = REPO_ROOT / "data" / "clear" / ann_file["dataset"]

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    episodes = episode_offsets(load_episodes(dataset_dir), info)
    ep = episodes[episodes["episode_index"] == episode_index].iloc[0]

    out_paths = []
    for key in ann_file["cameras"]:
        chunk_col, file_col = file_columns(key)
        src = dataset_dir / video_rel_path(info, key, ep[chunk_col], ep[file_col])
        out_path = (REPO_ROOT / "data" / "masked" /
                    f"_test_{ann_file['dataset']}_episode_{episode_index}_{camera_name(key)}.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        container = av.open(str(src))
        writer = H264Writer(str(out_path), info["features"][key]["info"]["video.fps"])
        try:
            frames = container.decode(video=0)
            for _ in range(int(ep[f"offset:{key}"])):
                next(frames)
            with tqdm(total=int(ep["length"]), desc=out_path.name, unit="frame") as bar:
                for frame in mask_episode(episode_frames(frames, int(ep["length"])),
                                          ann["frame"], ann["points"].get(key), bar):
                    writer.write(frame)
        finally:
            writer.close()
            container.close()
        out_paths.append(out_path)

    print("Wrote " + ", ".join(str(p) for p in out_paths))
    return out_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotations", nargs="?", help="e.g. data/annotations/ind-iso-1.json")
    parser.add_argument("--episode", type=int, metavar="N",
                        help="mask just episode N into a preview mp4 per camera under "
                             "data/masked/, instead of the whole dataset")
    args = parser.parse_args()

    if not args.annotations:
        parser.error("annotations json is required")
    elif args.episode is not None:
        preview_episode(args.annotations, args.episode)
    else:
        main(args.annotations)
