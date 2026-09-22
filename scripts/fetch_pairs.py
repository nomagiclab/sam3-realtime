"""Pull the five jerryrig recordings and their masked twins off the Hub, where the review
window (scripts/annotate.py) expects them:

    mim-chess-vlas/jr-pnp-N         -> data/clear/jr-pnp-N
    mim-chess-vlas/jr-pnp-N-masked  -> data/masked/jr-pnp-N

Re-running is cheap: a directory already there is left alone, and the Hub cache keeps
what was fetched. Usage: python scripts/fetch_pairs.py [N ...]
"""
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
ORG = "mim-chess-vlas"


def fetch(repo_id: str, target: Path) -> None:
    if (target / "meta" / "info.json").exists():
        print(f"{target} already there", flush=True)
        return
    local = snapshot_download(repo_id, repo_type="dataset", max_workers=8)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(local, target, symlinks=False, ignore=shutil.ignore_patterns(".cache"))
    print(f"{repo_id} -> {target}", flush=True)


if __name__ == "__main__":
    numbers = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5]
    jobs = []
    for n in numbers:
        jobs.append((f"{ORG}/jr-pnp-{n}", REPO_ROOT / "data" / "clear" / f"jr-pnp-{n}"))
        jobs.append((f"{ORG}/jr-pnp-{n}-masked", REPO_ROOT / "data" / "masked" / f"jr-pnp-{n}"))
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(lambda j: fetch(*j), jobs))
    print("ALL FETCHED", flush=True)
