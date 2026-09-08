"""Prefix every task of a masked dataset with "red masked".

mask.py starts by copying the clear dataset, so it brings the original prompts along and
this has to run again after every re-mask.

The prompt lives in two places: the index of meta/tasks.parquet (what training reads,
looked up positionally by task_index) and the `tasks` column of meta/episodes/*.parquet
(what the train/eval split groups by). Both move together.

    uv run scripts/retask_masked.py ind-pnp-0 ind-pnp-1     # dry run
    uv run scripts/retask_masked.py ind-pnp-0 --apply
"""
import argparse
import glob
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD, NEW = "Pick up the ", "Pick up the red masked "


def retask(dataset_dir: Path, dry_run: bool) -> None:
    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    tasks = pd.read_parquet(tasks_path)

    rename = {}
    for name in tasks.index:
        if name.startswith(NEW):
            continue                     # already done: running this twice changes nothing
        assert name.startswith(OLD), f"{dataset_dir}: unexpected task {name!r}"
        rename[name] = NEW + name[len(OLD):]
    if not rename:
        print(f"{dataset_dir.name}: nothing to do")
        return

    # rename in place: task_index is resolved with .iloc, so the row order must not move
    new_index = pd.Index([rename.get(n, n) for n in tasks.index], name=tasks.index.name)
    assert new_index.is_unique, f"{dataset_dir}: renaming collided two tasks"
    assert list(tasks["task_index"]) == list(range(len(tasks))), \
        f"{dataset_dir}: task_index is not 0..n-1, .iloc lookups would be wrong"

    episode_files = glob.glob(str(dataset_dir / "meta" / "episodes" / "**" / "*.parquet"),
                              recursive=True)
    episodes = [(f, pd.read_parquet(f)) for f in episode_files]
    for f, ep in episodes:
        unknown = {t for row in ep["tasks"] for t in row} - set(rename) - set(new_index)
        assert not unknown, f"{f}: tasks missing from tasks.parquet: {unknown}"

    print(f"{dataset_dir.name}: {len(rename)}/{len(tasks)} tasks, "
          f"{sum(len(e) for _, e in episodes)} episodes")
    print(f"   e.g. {next(iter(rename.values()))!r}")
    if dry_run:
        return

    tasks.index = new_index
    tasks.to_parquet(tasks_path)
    for f, ep in episodes:
        ep["tasks"] = [[rename.get(t, t) for t in row] for row in ep["tasks"]]
        ep.to_parquet(f, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", help="e.g. ind-pnp-0, under data/masked/")
    parser.add_argument("--apply", action="store_true", help="write, instead of a dry run")
    args = parser.parse_args()

    for name in args.names:
        retask(REPO_ROOT / "data" / "masked" / name, not args.apply)
    print("DRY RUN, nothing written -- pass --apply" if not args.apply else "written")
