"""Find the tool mask for a wrist camera.

The wrist camera is bolted to the arm, so the tool sits in the same pixels in every
frame while everything else moves: sample many frames and the tool is the place where
the pixels barely change. The score is std/mean rather than plain std, because the dark
vignette in the frame corners is low-variance too, but only for want of light -- divide
it out and the tool is the one region left.

    uv run scripts/find_tool_mask.py                 # both wrists -> demo/tool_masks/
    uv run scripts/find_tool_mask.py --debug         # also write the variance heatmaps
"""
import argparse
import glob
from pathlib import Path

import av
import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "demo" / "tool_masks"
# Only these two datasets: the `cmp_sophie` rigs mount the camera differently (landscape
# frames, no static tool at all) and the rest are not trusted to hold the same tool.
DATASET_GLOB = "data/clear/ind-iso-[36]"


def sample_frames(camera: str, per_video: int, stride: int) -> np.ndarray:
    """(N, H, W, 3) float32 frames, spread over every episode file of one camera."""
    frames = []
    for path in sorted(glob.glob(f"{DATASET_GLOB}/videos/observation.images.{camera}/chunk-000/*.mp4")):
        taken = 0
        with av.open(path) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i % stride:
                    continue
                frames.append(frame.to_ndarray(format="bgr24"))
                taken += 1
                if taken >= per_video:
                    break
    if not frames:
        raise SystemExit(f"no {camera} videos under {DATASET_GLOB}")
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        raise SystemExit(f"{camera}: mixed frame sizes {shapes}, masks would not line up")
    return np.stack(frames).astype(np.float32)


def tool_mask(frames: np.ndarray, threshold: float, margin: int):
    """((H, W) bool tool mask grown by `margin` px, (H, W) float score) for a frame stack."""
    # +8 keeps the near-black pixels from blowing the ratio up
    score = frames.std(axis=0).mean(axis=2) / (frames.mean(axis=0).mean(axis=2) + 8)
    still = (score < threshold).astype(np.uint8)

    # close pinholes (a specular highlight on the tool flickers), then drop the speckle
    # that survives elsewhere in the frame
    still = cv2.morphologyEx(still, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    still = cv2.morphologyEx(still, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # the tool is one piece reaching in from a frame edge: keep the biggest blob only
    count, labels, stats, _ = cv2.connectedComponentsWithStats(still, connectivity=8)
    if count < 2:
        raise SystemExit(f"no still region found (threshold {threshold} too low?)")
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == biggest).astype(np.uint8)

    # fill the tool's own holes: a gap the background shows through is still the tool,
    # and we never want a mask there either
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 1, cv2.FILLED)
    mask = mask.astype(bool)

    if margin:
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((margin * 2 + 1,) * 2, np.uint8)).astype(bool)
    return mask, score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cameras", nargs="+", default=["wrist_left", "wrist_right"])
    p.add_argument("--threshold", type=float, default=0.6, help="max per-pixel std/mean to count as tool")
    p.add_argument("--margin", type=int, default=6, help="px to grow the mask by")
    p.add_argument("--per-video", type=int, default=40)
    p.add_argument("--stride", type=int, default=97, help="keep 1 frame in this many")
    p.add_argument("--debug", action="store_true", help="also write variance + overlay previews")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for camera in args.cameras:
        frames = sample_frames(camera, args.per_video, args.stride)
        mask, score = tool_mask(frames, args.threshold, args.margin)
        out = OUT_DIR / f"{camera}.png"
        cv2.imwrite(str(out), mask.astype(np.uint8) * 255)
        print(f"{camera}: {len(frames)} frames, tool covers {mask.mean():.1%} of the frame -> {out}")

        if args.debug:
            heatmap = 255 * score / np.percentile(score, 99)
            cv2.imwrite(str(OUT_DIR / f"{camera}_score.png"), heatmap.clip(0, 255).astype(np.uint8))
            preview = np.median(frames, axis=0).astype(np.uint8)
            preview[mask] = (0.3 * preview[mask] + 0.7 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.imwrite(str(OUT_DIR / f"{camera}_overlay.png"), preview)


if __name__ == "__main__":
    main()
