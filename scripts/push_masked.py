"""Push the re-masked jr-pnp recordings back to the Hub, over the copies they came from.

Only the video files of the episodes that were re-masked differ, but a video file holds many
episodes, so whole files are uploaded; the Hub keeps the previous commit either way. The v3.0
tag is moved to the new commit by push_to_hub, which is the revision LeRobotDataset reads.

Usage: python scripts/push_masked.py [N ...]
"""
import logging
import sys
from pathlib import Path

from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("push")

REPO_ROOT = Path(__file__).resolve().parents[1]
ORG = "mim-chess-vlas"
api = HfApi()

numbers = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5]
# Smallest first, so a problem with the upload path shows up on the cheapest dataset.
for n in sorted(numbers, key=lambda i: (REPO_ROOT / "data/masked" / f"jr-pnp-{i}").stat().st_size):
    root = REPO_ROOT / "data/masked" / f"jr-pnp-{n}"
    repo_id = f"{ORG}/jr-pnp-{n}-masked"
    ds = LeRobotDataset(repo_id, root=root)   # local load: the folder is the dataset
    log.info("pushing %s from %s (%d episodes, %d frames)",
             repo_id, root, ds.meta.total_episodes, ds.meta.total_frames)
    ds.push_to_hub(private=True)
    info = api.dataset_info(repo_id)
    tags = {t.name: t.target_commit for t in api.list_repo_refs(repo_id, repo_type="dataset").tags}
    head = {b.name: b.target_commit for b in api.list_repo_refs(repo_id, repo_type="dataset").branches}["main"]
    assert tags.get("v3.0") == head, f"{repo_id}: v3.0 tag {tags.get('v3.0')} is not main {head}"
    local = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    remote = {s.rfilename for s in info.siblings}
    missing = local - remote - {".gitattributes"}
    assert not missing, f"{repo_id}: missing after upload: {sorted(missing)}"
    log.info("pushed %s (private=%s, %d files, tag v3.0 on head)", repo_id, info.private, len(remote))
log.info("all pushed")
