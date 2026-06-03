#!/usr/bin/env python3
"""Render threshold sweep views from existing probability maps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
STEM_SUFFIXES = ("_prob", "_mask", "_overlay", "_preview")


def parse_thresholds(value: str) -> List[float]:
    thresholds = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    for threshold in thresholds:
        if threshold < 0.0 or threshold > 1.0:
            raise argparse.ArgumentTypeError("thresholds must be in [0, 1]")
    return thresholds


def threshold_tag(threshold: float) -> str:
    return f"thr{int(round(threshold * 100)):02d}"


def find_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTS
    )


def base_stem(path: Path) -> str:
    stem = path.stem
    for suffix in STEM_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def index_images(image_dir: Path) -> dict[str, Path]:
    return {base_stem(path): path for path in find_images(image_dir)}


def make_overlay(image_bgr: "np.ndarray", mask: "np.ndarray", alpha: float) -> "np.ndarray":
    import cv2
    import numpy as np

    overlay = image_bgr.copy()
    red = np.zeros_like(image_bgr)
    red[:, :, 2] = 255
    overlay[mask] = cv2.addWeighted(image_bgr, 1.0 - alpha, red, alpha, 0)[mask]
    return overlay


def make_diff_overlay(
    image_bgr: "np.ndarray",
    added: "np.ndarray",
    removed: "np.ndarray",
    alpha: float,
) -> "np.ndarray":
    import cv2
    import numpy as np

    overlay = image_bgr.copy()
    added_color = np.zeros_like(image_bgr)
    removed_color = np.zeros_like(image_bgr)
    added_color[:, :, 1] = 220
    added_color[:, :, 2] = 255
    removed_color[:, :, 0] = 255

    if added.any():
        overlay[added] = cv2.addWeighted(
            image_bgr, 1.0 - alpha, added_color, alpha, 0
        )[added]
    if removed.any():
        overlay[removed] = cv2.addWeighted(
            overlay, 1.0 - alpha, removed_color, alpha, 0
        )[removed]
    return overlay


def make_preview(
    image_bgr: "np.ndarray",
    mask_u8: "np.ndarray",
    overlay: "np.ndarray",
    diff_overlay: "np.ndarray",
) -> "np.ndarray":
    import cv2
    import numpy as np

    mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    preview = np.concatenate([image_bgr, mask_bgr, overlay, diff_overlay], axis=1)
    max_width = 3200
    if preview.shape[1] > max_width:
        scale = max_width / preview.shape[1]
        preview = cv2.resize(
            preview,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    return preview


def write_threshold_outputs(
    prob_path: Path,
    image_path: Path,
    output_root: Path,
    threshold: float,
    baseline_threshold: float,
    alpha: float,
) -> dict[str, object]:
    import cv2
    import numpy as np

    stem = base_stem(prob_path)
    tag = threshold_tag(threshold)
    threshold_dir = output_root / tag
    masks_dir = threshold_dir / "masks"
    overlays_dir = threshold_dir / "overlays"
    previews_dir = threshold_dir / "previews"
    diffs_dir = threshold_dir / "diffs_from_baseline"
    for directory in (masks_dir, overlays_dir, previews_dir, diffs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    prob_u8 = cv2.imread(str(prob_path), cv2.IMREAD_GRAYSCALE)
    if image_bgr is None:
        raise ValueError(f"failed to read image: {image_path}")
    if prob_u8 is None:
        raise ValueError(f"failed to read probability map: {prob_path}")
    if image_bgr.shape[:2] != prob_u8.shape[:2]:
        image_bgr = cv2.resize(
            image_bgr,
            (prob_u8.shape[1], prob_u8.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    threshold_value = int(round(threshold * 255))
    baseline_value = int(round(baseline_threshold * 255))
    mask = prob_u8 >= threshold_value
    baseline_mask = prob_u8 >= baseline_value
    added = mask & ~baseline_mask
    removed = baseline_mask & ~mask

    mask_u8 = mask.astype(np.uint8) * 255
    added_removed_u8 = np.zeros((*mask.shape, 3), dtype=np.uint8)
    added_removed_u8[added] = (0, 220, 255)
    added_removed_u8[removed] = (255, 0, 0)

    overlay = make_overlay(image_bgr, mask, alpha)
    diff_overlay = make_diff_overlay(image_bgr, added, removed, alpha)
    preview = make_preview(image_bgr, mask_u8, overlay, diff_overlay)

    cv2.imwrite(str(masks_dir / f"{stem}_mask.png"), mask_u8)
    cv2.imwrite(str(overlays_dir / f"{stem}_overlay.jpg"), overlay)
    cv2.imwrite(str(previews_dir / f"{stem}_preview.jpg"), preview)
    cv2.imwrite(str(diffs_dir / f"{stem}_diff.png"), added_removed_u8)
    cv2.imwrite(str(diffs_dir / f"{stem}_diff_overlay.jpg"), diff_overlay)

    total_px = int(mask.size)
    return {
        "image": stem,
        "threshold": threshold,
        "mask_area_px": int(mask.sum()),
        "mask_ratio": float(mask.sum()) / total_px,
        "added_vs_baseline_px": int(added.sum()),
        "added_vs_baseline_ratio": float(added.sum()) / total_px,
        "removed_vs_baseline_px": int(removed.sum()),
        "removed_vs_baseline_ratio": float(removed.sum()) / total_px,
        "preview": str(previews_dir / f"{stem}_preview.jpg"),
        "diff_overlay": str(diffs_dir / f"{stem}_diff_overlay.jpg"),
    }


def iter_threshold_outputs(
    prob_paths: Iterable[Path],
    image_index: dict[str, Path],
    output_root: Path,
    thresholds: List[float],
    baseline_threshold: float,
    alpha: float,
) -> Iterable[dict[str, object]]:
    for prob_path in prob_paths:
        stem = base_stem(prob_path)
        image_path = image_index.get(stem)
        if image_path is None:
            raise ValueError(f"no matching base image for probability map: {prob_path}")
        for threshold in thresholds:
            yield write_threshold_outputs(
                prob_path=prob_path,
                image_path=image_path,
                output_root=output_root,
                threshold=threshold,
                baseline_threshold=baseline_threshold,
                alpha=alpha,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prob-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--thresholds",
        default="0.35,0.38,0.40,0.42,0.45",
        type=parse_thresholds,
        help="Comma-separated thresholds to render.",
    )
    parser.add_argument(
        "--baseline-threshold",
        default=0.40,
        type=float,
        help="Baseline threshold used for added/removed diff overlays.",
    )
    parser.add_argument("--alpha", default=0.55, type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prob_paths = find_images(args.prob_dir)
    if not prob_paths:
        raise SystemExit(f"no probability maps found under: {args.prob_dir}")

    image_index = index_images(args.image_dir)
    rows = list(
        iter_threshold_outputs(
            prob_paths=prob_paths,
            image_index=image_index,
            output_root=args.output,
            thresholds=args.thresholds,
            baseline_threshold=args.baseline_threshold,
            alpha=args.alpha,
        )
    )

    summary_path = args.output / "threshold_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
