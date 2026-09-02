"""Self-check for the tool masks and the mask AND NOT tool arithmetic.

    uv run scripts/test_tool_mask.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MASK_DIR = REPO_ROOT / "demo" / "tool_masks"


def load(camera):
    """demo/server.tool_mask without importing the server (it builds SAM3 on import)."""
    mask = cv2.imread(str(MASK_DIR / f"{camera}.png"), cv2.IMREAD_GRAYSCALE)
    assert mask is not None, f"missing {camera}.png -- run scripts/find_tool_mask.py"
    return mask > 127


def main():
    left, right = load("wrist_left"), load("wrist_right")

    for name, mask in (("wrist_left", left), ("wrist_right", right)):
        # a tool covering nothing, or half the frame, means the threshold has drifted
        assert 0.01 < mask.mean() < 0.15, f"{name} covers {mask.mean():.1%} of the frame"
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        assert count == 2, f"{name} is {count - 1} blobs, the tool should be one"
        # the tool reaches in from the frame edge, it is not floating in the middle
        assert mask[:, 0].any() or mask[:, -1].any() or mask[0].any() or mask[-1].any()

    # the two wrists see the tool from opposite sides, so the masks must not coincide
    assert (left & right).mean() < 0.2 * min(left.mean(), right.mean())

    # mask AND NOT tool, the way predict() applies it: (N, H, W) against (H, W)
    masks = np.ones((2,) + left.shape, bool)
    out = masks & ~left
    assert not out[:, left].any(), "tool pixels survived the subtraction"
    assert out[:, ~left].all(), "pixels outside the tool were dropped"

    # an empty stack broadcasts too (no objects found on this frame)
    assert (np.zeros((0,) + left.shape, bool) & ~left).shape == (0,) + left.shape

    print("ok")


if __name__ == "__main__":
    main()
