
import argparse
import json
import shutil
import sys
from pathlib import Path

import av
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from find_point_with_gemini import load_episodes, video_keys, video_rel_path  # noqa: E402



def check(dataset_dir: Path) -> list:
    """Everything that would make this dataset a lie. Empty list means it is fine."""
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    episodes = load_episodes(dataset_dir)
    problems = []
    for key in video_keys(info):
        chunk_col, file_col = f"videos/{key}/chunk_index", f"videos/{key}/file_index"
        for (chunk_idx, file_idx), group in episodes.groupby([chunk_col, file_col]):
            rel = video_rel_path(info, key, chunk_idx, file_idx)
            want = int(group["length"].sum())
            try:
                container = av.open(str(dataset_dir / rel))
                stream = container.streams.video[0]
                got, codec = stream.frames, stream.codec_context.name
                container.close()
            except Exception as e:
                problems.append(f"{rel}: unreadable ({type(e).__name__})")
                continue
            if codec != "h264":
                problems.append(f"{rel}: codec {codec}, so mask.py never got to it")
            if got != want:
                problems.append(f"{rel}: {got} frames, expected {want}")
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push one masked dataset to the Hub.")
    parser.add_argument("name", help="e.g. ind-pnp-0, under data/masked/")
    parser.add_argument("--repo-id", help="default: nomagic/<name>-masked")
    args = parser.parse_args()

    DATASET_DIR = REPO_ROOT / "data" / "masked" / args.name
    ANNOTATIONS = REPO_ROOT / "data" / "annotations" / f"{args.name}.json"
    REPO_ID = args.repo_id or f"nomagic/{args.name}-masked"

    print(f"Checking {DATASET_DIR}")
    problems = check(DATASET_DIR)
    if problems:
        raise SystemExit("not uploading, this dataset is incomplete:\n  " + "\n  ".join(problems))
    print("  every camera is h264 with the expected frame count")

    shutil.copy(ANNOTATIONS, DATASET_DIR / "meta" / "masking_points.json")

    # `root` pointing at an existing dataset makes this a local load, no Hub round trip,
    # so the repo does not have to exist yet. push_to_hub then creates it, uploads,
    # writes the dataset card and -- the part that matters -- tags the commit with the
    # codebase version, which is the only revision LeRobotDataset ever downloads.
    LeRobotDataset(REPO_ID, root=DATASET_DIR).push_to_hub(private=True)
    print(f"Done -> https://huggingface.co/datasets/{REPO_ID}")
