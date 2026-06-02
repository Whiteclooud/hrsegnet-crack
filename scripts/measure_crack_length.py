#!/usr/bin/env python3
"""Measure crack centerline length from HrSegNet masks or probability maps."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List, Tuple


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTS
    )


def zhang_suen_thinning(mask: "np.ndarray") -> "np.ndarray":
    """Return a 1-pixel skeleton using Zhang-Suen thinning."""
    import numpy as np

    image = (mask > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(image, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )

            if step == 0:
                condition = (
                    (image == 1)
                    & (neighbors >= 2)
                    & (neighbors <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p6) == 0)
                    & ((p4 * p6 * p8) == 0)
                )
            else:
                condition = (
                    (image == 1)
                    & (neighbors >= 2)
                    & (neighbors <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p8) == 0)
                    & ((p2 * p6 * p8) == 0)
                )

            if condition.any():
                image[condition] = 0
                changed = True
    return image.astype(bool)


def filter_small_components(mask: "np.ndarray", min_area: int) -> Tuple["np.ndarray", int]:
    import cv2
    import numpy as np

    if min_area <= 1:
        return mask.astype(bool), 0

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    filtered = np.zeros(mask.shape, dtype=bool)
    removed = 0
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered[labels == label] = True
        else:
            removed += 1
    return filtered, removed


def skeleton_length_px(skeleton: "np.ndarray") -> float:
    import numpy as np

    skel = skeleton.astype(bool)
    if not skel.any():
        return 0.0

    length = 0.0
    # Count every 8-neighbor edge once.
    offsets = [
        (0, 1, 1.0),
        (1, 0, 1.0),
        (1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
    ]
    height, width = skel.shape
    for dy, dx, weight in offsets:
        y0 = max(0, -dy)
        y1 = min(height, height - dy)
        x0 = max(0, -dx)
        x1 = min(width, width - dx)
        current = skel[y0:y1, x0:x1]
        neighbor = skel[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        length += float(np.count_nonzero(current & neighbor)) * weight
    return length


def skeleton_component_lengths(skeleton: "np.ndarray") -> Tuple[int, List[float]]:
    import cv2
    import numpy as np

    labels_count, labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    lengths: List[float] = []
    for label in range(1, labels_count):
        component = labels == label
        lengths.append(skeleton_length_px(component))
    return labels_count - 1, lengths


def load_mask(path: Path, threshold: float) -> "np.ndarray":
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image >= int(round(threshold * 255))


def write_skeleton(path: Path, skeleton: "np.ndarray") -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), skeleton.astype(np.uint8) * 255)


def measure_one(
    path: Path,
    threshold: float,
    gsd_mm_per_px: float,
    min_area_px: int,
    skeleton_dir: Path | None,
) -> dict:
    mask = load_mask(path, threshold)
    filtered, removed_components = filter_small_components(mask, min_area_px)
    skeleton = zhang_suen_thinning(filtered)
    total_length_px = skeleton_length_px(skeleton)
    component_count, component_lengths = skeleton_component_lengths(skeleton)

    if skeleton_dir is not None:
        write_skeleton(skeleton_dir / f"{path.stem}_skeleton.png", skeleton)

    height, width = mask.shape
    mask_area_px = int(filtered.sum())
    length_mm = total_length_px * gsd_mm_per_px
    return {
        "image": path.name,
        "width": width,
        "height": height,
        "threshold": threshold,
        "gsd_mm_per_px": gsd_mm_per_px,
        "min_area_px": min_area_px,
        "removed_components": removed_components,
        "mask_area_px": mask_area_px,
        "mask_ratio": mask_area_px / float(width * height),
        "skeleton_components": component_count,
        "skeleton_length_px": total_length_px,
        "length_mm": length_mm,
        "length_cm": length_mm / 10.0,
        "length_m": length_mm / 1000.0,
        "max_component_length_px": max(component_lengths) if component_lengths else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Mask/prob image or folder.")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--threshold",
        default=0.4,
        type=float,
        help="Threshold for grayscale probability maps, in [0, 1].",
    )
    parser.add_argument(
        "--gsd-mm-per-px",
        default=1.075,
        type=float,
        help="Fixed ground sampling distance used for physical length conversion.",
    )
    parser.add_argument(
        "--min-area-px",
        default=0,
        type=int,
        help="Remove connected mask components smaller than this area before skeletonizing.",
    )
    parser.add_argument(
        "--skeleton-dir",
        type=Path,
        default=None,
        help="Optional folder for skeleton preview images.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    images = find_images(args.input)
    if not images:
        raise SystemExit(f"no mask/prob images found under: {args.input}")

    rows = [
        measure_one(
            path=path,
            threshold=args.threshold,
            gsd_mm_per_px=args.gsd_mm_per_px,
            min_area_px=args.min_area_px,
            skeleton_dir=args.skeleton_dir,
        )
        for path in images
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
