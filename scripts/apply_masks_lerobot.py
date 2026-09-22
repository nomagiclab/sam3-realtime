import argparse
import json
import shutil
import sys
from pathlib import Path

import av
import pandas as pd
import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from demo.app import SERVER, VideoWriter, new_session, close_session, predict, rgb_to_b64  # noqa: E402

VIDEO_KEY = "observation.images.side"

# Hardcoded task: point at the item sticking out of its box. Gemini's spatial
# convention is [y, x] normalized to 0..1000.
DETECT_PROMPT = (
    "Your primary goal is to identify the single, main crate located in "
    "the foreground of the image, directly underneath the robot's tool. "
    "Within *only* that specific crate, find the one item that sticks out / "
    "protrudes beyond the top edge of that crate. "
    "Strictly follow these rules: "
    "1. Focus *only* on the items within or on the edge of the one main foreground crate. "
    "2. Completely ignore all background containers, other crates, and background items. "
    "3. Ignore the robot arm, its tools, and items on the distant floor. "
    "Point at the part of the item in the primary crate that sticks out."
)
DETECT_SCHEMA = {
    "type": "object",
    "properties": {
        "point": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
    },
    "required": ["point"],
}


def load_episodes(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def video_rel_path(info: dict, chunk_index: int, file_index: int) -> str:
    return info["video_path"].format(video_key=VIDEO_KEY, chunk_index=chunk_index, file_index=file_index)


def detect_first_point(frame) -> list:
    r = requests.post(f"{SERVER}/gemini", json={
        "images": [rgb_to_b64(frame)],
        "prompt": DETECT_PROMPT,
        "schema": DETECT_SCHEMA,
    }, timeout=60)
    r.raise_for_status()
    y1000, x1000 = r.json()["result"]["point"]
    return [x1000 / 1000, y1000 / 1000]


def mask_video_file(src_path: Path, dst_path: Path, episodes: pd.DataFrame, fps: float) -> None:
    """episodes: the rows (one per episode) packed into this one physical video file."""
    container = av.open(str(src_path))
    frames = container.decode(video=0)
    writer = VideoWriter(str(dst_path), fps)
    total = int(episodes["length"].sum())
    try:
        with tqdm(total=total, desc=dst_path.name, unit="frame") as bar:
            for _, ep in episodes.sort_values("episode_index").iterrows():
                session_id = new_session()
                try:
                    for i in range(int(ep["length"])):
                        frame = next(frames).to_ndarray(format="rgb24")
                        point = detect_first_point(frame) if i == 0 else None
                        writer.write(predict(session_id, frame, point=point))
                        bar.update(1)
                finally:
                    close_session(session_id)
    finally:
        writer.close()
        container.close()


def main(dataset_dir: str) -> None:
    dataset_dir = Path(dataset_dir).resolve()
    out_dir = REPO_ROOT / "data" / "masked" / dataset_dir.name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists, remove it first")

    print(f"Copying {dataset_dir} -> {out_dir}")
    shutil.copytree(dataset_dir, out_dir)

    info_path = out_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    fps = info["features"][VIDEO_KEY]["info"]["video.fps"]
    info["features"][VIDEO_KEY]["info"]["video.codec"] = "h264"  # re-encoded below, was av1
    info_path.write_text(json.dumps(info, indent=4))

    episodes = load_episodes(dataset_dir)
    chunk_col, file_col = f"videos/{VIDEO_KEY}/chunk_index", f"videos/{VIDEO_KEY}/file_index"
    for (chunk_idx, file_idx), group in episodes.groupby([chunk_col, file_col]):
        rel = video_rel_path(info, chunk_idx, file_idx)
        print(f"Masking {rel} ({len(group)} episodes, {int(group['length'].sum())} frames)")
        mask_video_file(dataset_dir / rel, out_dir / rel, group, fps)

    print(f"Done -> {out_dir}")


def test_episode(dataset_dir: str, episode_index: int) -> Path:
    """Quick check: mask just one episode and write it as a standalone preview mp4,
    without touching the dataset. Use this before running main() on a whole dataset."""
    dataset_dir = Path(dataset_dir).resolve()
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    fps = info["features"][VIDEO_KEY]["info"]["video.fps"]

    episodes = load_episodes(dataset_dir)
    chunk_col, file_col = f"videos/{VIDEO_KEY}/chunk_index", f"videos/{VIDEO_KEY}/file_index"
    ep = episodes[episodes["episode_index"] == episode_index].iloc[0]
    same_file = episodes[(episodes[chunk_col] == ep[chunk_col]) & (episodes[file_col] == ep[file_col])]
    skip = int(same_file.loc[same_file["episode_index"] < episode_index, "length"].sum())

    src = dataset_dir / video_rel_path(info, ep[chunk_col], ep[file_col])
    out_path = REPO_ROOT / "data" / "masked" / f"_test_{dataset_dir.name}_episode_{episode_index}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    container = av.open(str(src))
    frames = container.decode(video=0)
    for _ in range(skip):
        next(frames)

    writer = VideoWriter(str(out_path), fps)
    session_id = new_session()
    try:
        for i in tqdm(range(int(ep["length"])), desc=out_path.name, unit="frame"):
            frame = next(frames).to_ndarray(format="rgb24")
            point = detect_first_point(frame) if i == 0 else None
            writer.write(predict(session_id, frame, point=point))
    finally:
        close_session(session_id)
        writer.close()
        container.close()

    print(f"Wrote {out_path}")
    return out_path


def _selftest() -> None:
    info = {"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    assert video_rel_path(info, 0, 0) == "videos/observation.images.side/chunk-000/file-000.mp4"
    assert video_rel_path(info, 2, 7) == "videos/observation.images.side/chunk-002/file-007.mp4"
    print("ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", nargs="?", help="e.g. data/clear/iso_alfred_day1")
    parser.add_argument("--test-episode", type=int, metavar="N",
                         help="mask just episode N into a standalone preview mp4 under data/masked/, "
                              "instead of processing the whole dataset")
    parser.add_argument("--selftest", action="store_true", help="run the offline self-check and exit")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
    elif args.test_episode is not None:
        if not args.dataset_dir:
            parser.error("dataset_dir is required")
        test_episode(args.dataset_dir, args.test_episode)
    elif args.dataset_dir:
        main(args.dataset_dir)
    else:
        parser.error("dataset_dir is required")
