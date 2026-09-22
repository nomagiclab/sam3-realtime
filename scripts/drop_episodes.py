"""Drop the episodes whose item is never visible from the side camera, and upload the
rest as <name>-target-visible-from-side.

Usage (from the LeTS venv, which has lerobot's dataset tools):
    python scripts/drop_episodes.py <source dir> <new repo id> <ep> <ep> ... [--push]

lerobot re-encodes only the video files that mix kept and dropped episodes, with the
encoder settings read from the source's info.json (libsvtav1, g=2, crf 30, preset 12):
the clean and the masked copy go through exactly the same step, so they still differ
only by the mask.
"""
import sys
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes

src, repo_id, *rest = sys.argv[1:]
push = "--push" in rest
drop = sorted(int(e) for e in rest if e != "--push")
out = Path.home() / "repos/sam3-realtime/data/visible" / repo_id.split("/")[-1]
ds = LeRobotDataset("nomagic/" + Path(src).name.replace("-masked", ""), root=src)
print(f"{Path(src).name}: {ds.meta.total_episodes} episodes, dropping {len(drop)}: {drop}", flush=True)
new = delete_episodes(ds, drop, output_dir=out, repo_id=repo_id)
print(f"wrote {out}: {new.meta.total_episodes} episodes, {new.meta.total_frames} frames", flush=True)
if push:
    new.push_to_hub(private=True, push_videos=True)
    print(f"pushed {repo_id}", flush=True)
