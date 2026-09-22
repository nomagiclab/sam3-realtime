"""Cut again the preview clips that are broken, without touching the window or what it runs.

The clips under data/clips are only a browser-playable copy of what is already in
data/masked, so a broken one costs a few seconds to make again and nothing else: the
masked recording it was cut from is not involved, and neither is SAM3. A broken clip is
one two ffmpegs wrote at once -- see `sound_mp4` in scripts/annotate.py for what that
leaves behind and how it is recognised.

Made to be run beside a window that is already open, and beside a masking run that must
not be interrupted:

  * it only ever touches clips that are already broken, and the open window will not cut
    those itself -- it sees a file of that name and leaves it alone, which is why they
    stayed broken in the first place;
  * each clip is written under a name of its own and moved into place in one step, so a
    player reading that file gets either the old bytes or the new ones;
  * two workers by default, to leave the machine to the masking.

Refresh the browser tab when it is done -- the clip's address carries its timestamp, so a
tab that is left open keeps showing what it already fetched.

Usage: python scripts/recut_clips.py [--dataset jr-pnp-5] [--workers 2] [--list]
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import annotate  # noqa: E402
from find_point_with_gemini import camera_name  # noqa: E402


def broken(name: str) -> list:
    """(episode, camera key) of every clip of this dataset that will not play, in order."""
    state = annotate.load_state(name)
    state["clip_executor"].shutdown(wait=False, cancel_futures=True)
    out = []
    for index in sorted(int(k) for k in state["file"]["episodes"]):
        for key in state["keys"]:
            path = annotate.clip_path(state, index, key)
            if path.exists() and not annotate.sound_mp4(path):
                out.append((index, key))
    return state, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", help="one dataset; the default is every dataset with clips")
    parser.add_argument("--workers", type=int, default=2,
                        help="clips cut at once; the default leaves the machine to the masking")
    parser.add_argument("--list", action="store_true", help="say what is broken and stop")
    args = parser.parse_args()

    names = ([args.dataset] if args.dataset
             else sorted(p.name for p in annotate.CLIPS_DIR.glob("*") if p.is_dir()))
    for name in names:
        if not (annotate.CLEAR_DIR / name / "meta" / "info.json").exists():
            print(f"{name}: no such dataset under data/clear, skipped")
            continue
        state, todo = broken(name)
        if not todo:
            print(f"{name}: every clip plays, nothing to do")
            continue
        print(f"{name}: {len(todo)} broken clip(s), first at episode {todo[0][0]}")
        if args.list:
            for index, key in todo:
                print(f"    episode {index:4d}  {camera_name(key)}")
            continue
        done = failed = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            results = pool.map(lambda job: cut(state, *job), todo)
            for index, key, error in results:
                if error:
                    failed += 1
                    print(f"    episode {index:4d}  {camera_name(key)}  FAILED -- {error}")
                else:
                    done += 1
                    if done % 25 == 0 or done + failed == len(todo):
                        print(f"    {done + failed} of {len(todo)} ...", flush=True)
        print(f"{name}: {done} cut again, {failed} still broken. Refresh the browser tab.")


def cut(state: dict, index: int, key: str) -> tuple:
    path = annotate.clip_path(state, index, key)
    path.unlink(missing_ok=True)    # so the cut cannot mistake the broken one for a clip
    try:
        annotate.cut_clip(state, index, key)
        return index, key, None
    except Exception as exc:
        return index, key, f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":
    main()
