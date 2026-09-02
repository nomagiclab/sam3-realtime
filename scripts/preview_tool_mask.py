"""Draw a tool mask over real frames, so you can see what it covers.

Reads the mask straight from demo/tool_masks/, so after hand-editing the png just run
this again to see the edit.

    uv run scripts/preview_tool_mask.py                        # wrist_right, 12 frames
    uv run scripts/preview_tool_mask.py --camera wrist_left --count 20
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from find_tool_mask import OUT_DIR, sample_frames  # noqa: E402


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The frame with the mask tinted and outlined, so both the covered pixels and the
    exact border are visible. Magenta, not the server's red -- a red-checkered tablecloth
    turns up in these scenes and the two are hard to tell apart."""
    out = frame.copy()
    out[mask] = (0.45 * out[mask] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 1)
    return out


def contact_sheet(tiles: list, columns: int) -> np.ndarray:
    """Tiles of one size laid out in a grid, 4 px of white between them."""
    h, w = tiles[0].shape[:2]
    rows = -(-len(tiles) // columns)
    sheet = np.full((rows * (h + 4) + 4, columns * (w + 4) + 4, 3), 255, np.uint8)
    for i, tile in enumerate(tiles):
        y, x = divmod(i, columns)
        sheet[4 + y * (h + 4):4 + y * (h + 4) + h, 4 + x * (w + 4):4 + x * (w + 4) + w] = tile
    return sheet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="wrist_right")
    p.add_argument("--count", type=int, default=12, help="how many frames to show")
    p.add_argument("--columns", type=int, default=4)
    p.add_argument("--out", type=Path, help="default: demo/tool_masks/<camera>_preview.png")
    args = p.parse_args()

    mask_path = OUT_DIR / f"{args.camera}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"no {mask_path} -- run scripts/find_tool_mask.py first")

    # 2 frames per video, far apart, then thin down to `count` spread over everything
    frames = sample_frames(args.camera, per_video=2, stride=997).astype(np.uint8)
    picked = frames[np.linspace(0, len(frames) - 1, args.count, dtype=int)]

    if mask.shape != picked.shape[1:3]:
        raise SystemExit(f"{mask_path} is {mask.shape}, the frames are {picked.shape[1:3]}")

    out = args.out or OUT_DIR / f"{args.camera}_preview.png"
    cv2.imwrite(str(out), contact_sheet([overlay(f, mask > 127) for f in picked], args.columns))
    print(f"{args.camera}: {len(picked)} frames, tool covers {(mask > 127).mean():.1%} -> {out}")


if __name__ == "__main__":
    main()
