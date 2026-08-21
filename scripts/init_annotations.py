"""
Start an annotation json for a dataset without asking Gemini anything: every episode
gets its middle frame and no points at all, ready to be filled in by hand in
scripts/correct_point.py.

The middle frame is only where the annotator lands first -- clicking in the GUI saves
whatever frame is on screen, so it moves with the first click.

Usage: python scripts/init_annotations.py data/clear/ind-iso-4
Output: data/annotations/<name>.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from find_point_with_gemini import camera_name, load_episodes, video_keys  # noqa: E402


def main(dataset_dir: str) -> Path:
    dataset_dir = Path(dataset_dir).resolve()
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    keys = video_keys(info)
    episodes = load_episodes(dataset_dir)

    out_path = REPO_ROOT / "data" / "annotations" / f"{dataset_dir.name}.json"
    if out_path.exists():
        # overwriting would silently throw away however many hours of clicking
        raise SystemExit(f"{out_path} already exists -- delete it first if you really mean to start over")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "dataset": dataset_dir.name,
        "cameras": keys,
        "episodes": {str(int(ep["episode_index"])): {"frame": int(ep["length"]) // 2, "points": {}}
                     for _, ep in episodes.sort_values("episode_index").iterrows()},
    }, indent=2))

    print(f"Wrote {out_path}: {len(episodes)} episodes x {len(keys)} cameras "
          f"({', '.join(camera_name(k) for k in keys)}), no points yet")
    print(f"Annotate them with: python scripts/correct_point.py {out_path.relative_to(REPO_ROOT)}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", help="e.g. data/clear/ind-iso-4")
    main(parser.parse_args().dataset_dir)
