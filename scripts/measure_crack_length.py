#!/usr/bin/env python3
"""Measure crack centerline length from HrSegNet masks or probability maps."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List, Tuple


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
STEM_SUFFIXES = ("_prob", "_mask", "_overlay", "_preview", "_skeleton")


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


def crack_width_metrics(
    mask: "np.ndarray",
    skeleton: "np.ndarray",
    skeleton_length: float,
    gsd_mm_per_px: float,
) -> dict:
    import cv2
    import numpy as np

    mask_area_px = int(mask.sum())
    mean_width_px_area = mask_area_px / skeleton_length if skeleton_length > 0 else 0.0

    if skeleton.any():
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        local_widths_px = distance[skeleton] * 2.0
        mean_width_px_dt = float(local_widths_px.mean())
        median_width_px_dt = float(np.median(local_widths_px))
        p95_width_px_dt = float(np.percentile(local_widths_px, 95))
        max_width_px_dt = float(local_widths_px.max())
    else:
        mean_width_px_dt = 0.0
        median_width_px_dt = 0.0
        p95_width_px_dt = 0.0
        max_width_px_dt = 0.0

    return {
        "mean_width_px_area": mean_width_px_area,
        "mean_width_mm_area": mean_width_px_area * gsd_mm_per_px,
        "mean_width_px_dt": mean_width_px_dt,
        "mean_width_mm_dt": mean_width_px_dt * gsd_mm_per_px,
        "median_width_px_dt": median_width_px_dt,
        "median_width_mm_dt": median_width_px_dt * gsd_mm_per_px,
        "p95_width_px_dt": p95_width_px_dt,
        "p95_width_mm_dt": p95_width_px_dt * gsd_mm_per_px,
        "max_width_px_dt": max_width_px_dt,
        "max_width_mm_dt": max_width_px_dt * gsd_mm_per_px,
    }


def skeleton_component_details(skeleton: "np.ndarray", mask: "np.ndarray") -> List[dict]:
    import cv2
    import numpy as np

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        skeleton.astype(np.uint8), connectivity=8
    )
    components: List[dict] = []
    for label in range(1, labels_count):
        component = labels == label
        local_widths_px = distance[component] * 2.0
        if local_widths_px.size:
            mean_width_px = float(local_widths_px.mean())
            median_width_px = float(np.median(local_widths_px))
            max_width_px = float(local_widths_px.max())
        else:
            mean_width_px = 0.0
            median_width_px = 0.0
            max_width_px = 0.0
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centroids[label]
        components.append(
            {
                "label": label,
                "length_px": skeleton_length_px(component),
                "skeleton_area_px": area,
                "mean_width_px": mean_width_px,
                "median_width_px": median_width_px,
                "max_width_px": max_width_px,
                "bbox": (x, y, w, h),
                "centroid": (float(cx), float(cy)),
            }
        )
    return components


def find_base_image(base_image_dir: Path | None, source_path: Path) -> Path | None:
    if base_image_dir is None:
        return None

    stem = base_stem(source_path)
    candidate_names = []
    for suffix in ("_overlay", "", "_preview", "_mask", "_prob"):
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            candidate_names.append(f"{stem}{suffix}{ext}")

    for name in candidate_names:
        candidate = base_image_dir / name
        if candidate.exists():
            return candidate

    matches = sorted(
        file
        for file in base_image_dir.glob(f"{stem}*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTS
    )
    return matches[0] if matches else None


def format_length(length_mm: float) -> str:
    if length_mm >= 1000.0:
        return f"{length_mm / 1000.0:.2f} m"
    if length_mm >= 10.0:
        return f"{length_mm / 10.0:.1f} cm"
    return f"{length_mm:.1f} mm"


def draw_text(
    image: "np.ndarray",
    text: str,
    x: int,
    y: int,
    scale: float = 0.9,
    color: Tuple[int, int, int] = (0, 255, 255),
) -> None:
    import cv2

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def write_annotated_overlay(
    source_path: Path,
    skeleton: "np.ndarray",
    components: List[dict],
    total_length_mm: float,
    mean_width_mm: float,
    gsd_mm_per_px: float,
    base_image_dir: Path | None,
    annotated_dir: Path,
    max_labels: int,
    min_label_length_mm: float,
) -> Path:
    import cv2
    import numpy as np

    height, width = skeleton.shape
    base_path = find_base_image(base_image_dir, source_path)
    image = cv2.imread(str(base_path), cv2.IMREAD_COLOR) if base_path is not None else None

    if image is None:
        image = np.zeros((height, width, 3), dtype=np.uint8)
    elif image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    annotated = image.copy()
    cyan = np.array([255, 255, 0], dtype=np.uint8)
    annotated[skeleton] = (
        0.35 * annotated[skeleton].astype(np.float32) + 0.65 * cyan.astype(np.float32)
    ).astype(np.uint8)

    draw_text(
        annotated,
        (
            f"total: {format_length(total_length_mm)} | "
            f"mean width: {mean_width_mm:.1f} mm | "
            f"GSD: {gsd_mm_per_px:.3f} mm/px"
        ),
        28,
        48,
        scale=1.1,
    )

    ranked = sorted(components, key=lambda item: item["length_px"], reverse=True)
    shown = 0
    for idx, component in enumerate(ranked, start=1):
        length_mm = component["length_px"] * gsd_mm_per_px
        if length_mm < min_label_length_mm:
            continue
        x, y, w, h = component["bbox"]
        cx, cy = component["centroid"]
        label_x = max(8, min(width - 280, int(cx) + 8))
        label_y = max(78, min(height - 12, int(cy) - 8))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 255), 2)
        cv2.circle(annotated, (int(cx), int(cy)), 6, (0, 255, 255), -1)
        draw_text(
            annotated,
            (
                f"#{idx} L={format_length(length_mm)} "
                f"W~{component['median_width_px'] * gsd_mm_per_px:.1f}mm"
            ),
            label_x,
            label_y,
            scale=0.8,
        )
        shown += 1
        if shown >= max_labels:
            break

    annotated_dir.mkdir(parents=True, exist_ok=True)
    out_path = annotated_dir / f"{base_stem(source_path)}_length_overlay.jpg"
    cv2.imwrite(str(out_path), annotated)
    return out_path


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
    base_image_dir: Path | None,
    annotated_dir: Path | None,
    max_labels: int,
    min_label_length_mm: float,
) -> dict:
    mask = load_mask(path, threshold)
    filtered, removed_components = filter_small_components(mask, min_area_px)
    skeleton = zhang_suen_thinning(filtered)
    total_length_px = skeleton_length_px(skeleton)
    components = skeleton_component_details(skeleton, filtered)
    width_metrics = crack_width_metrics(
        mask=filtered,
        skeleton=skeleton,
        skeleton_length=total_length_px,
        gsd_mm_per_px=gsd_mm_per_px,
    )

    if skeleton_dir is not None:
        write_skeleton(skeleton_dir / f"{path.stem}_skeleton.png", skeleton)

    annotated_path = ""
    if annotated_dir is not None:
        annotated_path = str(
            write_annotated_overlay(
                source_path=path,
                skeleton=skeleton,
                components=components,
                total_length_mm=total_length_px * gsd_mm_per_px,
                mean_width_mm=width_metrics["mean_width_mm_area"],
                gsd_mm_per_px=gsd_mm_per_px,
                base_image_dir=base_image_dir,
                annotated_dir=annotated_dir,
                max_labels=max_labels,
                min_label_length_mm=min_label_length_mm,
            )
        )

    height, width = mask.shape
    mask_area_px = int(filtered.sum())
    length_mm = total_length_px * gsd_mm_per_px
    component_lengths = [component["length_px"] for component in components]
    image_width_m = width * gsd_mm_per_px / 1000.0
    image_height_m = height * gsd_mm_per_px / 1000.0
    return {
        "image": path.name,
        "width": width,
        "height": height,
        "image_width_m": image_width_m,
        "image_height_m": image_height_m,
        "image_area_m2": image_width_m * image_height_m,
        "threshold": threshold,
        "gsd_mm_per_px": gsd_mm_per_px,
        "min_area_px": min_area_px,
        "removed_components": removed_components,
        "mask_area_px": mask_area_px,
        "mask_ratio": mask_area_px / float(width * height),
        "skeleton_components": len(components),
        "skeleton_length_px": total_length_px,
        "length_mm": length_mm,
        "length_cm": length_mm / 10.0,
        "length_m": length_mm / 1000.0,
        "max_component_length_px": max(component_lengths) if component_lengths else 0.0,
        **width_metrics,
        "annotated_overlay": annotated_path,
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
    parser.add_argument(
        "--base-image-dir",
        type=Path,
        default=None,
        help="Optional original/overlay image folder used as annotation background.",
    )
    parser.add_argument(
        "--annotated-dir",
        type=Path,
        default=None,
        help="Optional folder for length-annotated overlay images.",
    )
    parser.add_argument(
        "--max-labels",
        default=25,
        type=int,
        help="Maximum number of component length labels per annotated image.",
    )
    parser.add_argument(
        "--min-label-length-mm",
        default=0.0,
        type=float,
        help="Only annotate components with at least this physical length.",
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
            base_image_dir=args.base_image_dir,
            annotated_dir=args.annotated_dir,
            max_labels=args.max_labels,
            min_label_length_mm=args.min_label_length_mm,
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
