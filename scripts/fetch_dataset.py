"""Pull a lerobot dataset off the Hub into data/clear/<name>, where the rest of the
pipeline looks for it.

Everything downstream (auto_point.py, annotate.py, mask.py) reads a plain directory, so
this is the one step that talks to the Hub. Re-running it is cheap: the files already
there are not fetched again.

Usage: python scripts/fetch_dataset.py nomagic/jr-pnp-tomek-1 [--name other-name]
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]


def fetch(repo_id: str, name: str | None = None) -> Path:
    target = REPO_ROOT / "data" / "clear" / (name or repo_id.split("/")[-1])
    if target.exists():
        print(f"{target} already exists, leaving it alone")
        return target
    print(f"Downloading {repo_id} ...")
    # Into the Hub cache first, then copied: the cache stores files as symlinks into a
    # blob store, and av / the masking pass want a directory it can read and copy freely.
    local = snapshot_download(repo_id, repo_type="dataset")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local, target, symlinks=False)
    print(f"Done -> {target}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="e.g. nomagic/jr-pnp-tomek-1")
    parser.add_argument("--name", help="directory name under data/clear (default: the repo name)")
    fetch(**vars(parser.parse_args()))
