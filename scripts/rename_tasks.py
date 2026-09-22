"""Rewrite the task strings of a local lerobot v3 dataset in place.

"Pick the Black square box ... and place it into the box"
 -> "Pick up the black square box ... and place it in the box"            (clean)
 -> "Pick up the red masked black square box ... and place it in the box" (--masked)

Only the first letter of the item description is lowercased; brand names inside it
(Dynamixel, Skittles, FILA) keep their case. Touches meta/tasks.parquet and the `tasks`
column of meta/episodes/**; data/ carries only task_index and needs nothing.
"""
import glob, re, sys
from pathlib import Path
import numpy as np, pandas as pd

root = Path(sys.argv[1]); masked = "--masked" in sys.argv
PAT = re.compile(r"^Pick(?: up)? the (?:red masked )?(.*?) and place it (?:into|in) the box$")

def fix(task: str) -> str:
    m = PAT.match(task)
    if not m:
        raise SystemExit(f"unexpected task string: {task!r}")
    item = m.group(1); item = item[0].lower() + item[1:]
    return f"Pick up the {'red masked ' if masked else ''}{item} and place it in the box"

p = root / "meta/tasks.parquet"; t = pd.read_parquet(p)
mapping = {old: fix(old) for old in t.index}
t.index = pd.Index([mapping[o] for o in t.index], name=t.index.name); t.to_parquet(p)
n = 0
for f in glob.glob(str(root / "meta/episodes/**/*.parquet"), recursive=True):
    e = pd.read_parquet(f)
    e["tasks"] = e["tasks"].apply(lambda arr: np.array([mapping.get(s, fix(s)) for s in arr], dtype=object)); e.to_parquet(f); n += len(e)
print(f"{root.name}: {len(mapping)} tasks, {n} episodes"); [print("  ", v) for v in list(mapping.values())[:3]]
