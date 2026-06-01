#!/usr/bin/env python3
"""Run tiled HrSegNet inference on a folder of images.

The script keeps project-specific inference outside the upstream HrSegNet
repository. It expects the official implementation to be available under
`third_party/HrSegNet4CrackSegmentation` unless `--official-repo` is supplied.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_triplet(value: str) -> List[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return [float(part) for part in parts]


def parse_pair(value: str) -> List[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated values")
    return [int(part) for part in parts]


def find_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    images = [
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(images)


def import_python_files(directory: Path) -> None:
    if not directory.exists():
        return
    for file in sorted(directory.glob("*.py")):
        if file.name.startswith("_"):
            continue
        module_name = f"_hrsegnet_{file.stem}"
        if module_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(module_name, file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


def read_yaml(path: Path) -> object:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def iter_nodes(node: object) -> Iterable[object]:
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from iter_nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_nodes(value)


def norm_from_config(config_path: Path) -> Optional[Tuple[List[float], List[float]]]:
    try:
        config = read_yaml(config_path)
    except Exception:
        return None
    for node in iter_nodes(config):
        if not isinstance(node, dict):
            continue
        if node.get("type") != "Normalize":
            continue
        mean = node.get("mean")
        std = node.get("std")
        if mean is not None and std is not None:
            return [float(v) for v in mean], [float(v) for v in std]
    return None


def load_model(
    official_repo: Path,
    config_path: Path,
    weights_path: Path,
    device: str,
):
    sys.path.insert(0, str(official_repo))
    import_python_files(official_repo / "models")

    import paddle

    try:
        from paddleseg.cvlibs import Config
    except ImportError:
        from paddleseg.cvlibs.config import Config

    paddle.set_device(device)
    cfg = Config(str(config_path))
    try:
        from paddleseg.cvlibs import SegBuilder

        model = SegBuilder(cfg).model
    except ImportError:
        # PaddleSeg 2.7 builds configured models directly through Config.model.
        model = cfg.model

    try:
        from paddleseg.utils import utils

        utils.load_entire_model(model, str(weights_path))
    except Exception:
        state = paddle.load(str(weights_path))
        if isinstance(state, dict):
            for key in ("model", "state_dict", "params"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        model.set_state_dict(state)

    model.eval()
    return model


def positions(size: int, crop: int, stride: int) -> List[int]:
    if size <= crop:
        return [0]
    values = list(range(0, size - crop + 1, stride))
    last = size - crop
    if values[-1] != last:
        values.append(last)
    return values


def pad_tile(tile: "np.ndarray", crop_h: int, crop_w: int) -> "np.ndarray":
    import cv2

    h, w = tile.shape[:2]
    pad_h = crop_h - h
    pad_w = crop_w - w
    if pad_h <= 0 and pad_w <= 0:
        return tile
    return cv2.copyMakeBorder(
        tile,
        0,
        max(0, pad_h),
        0,
        max(0, pad_w),
        borderType=cv2.BORDER_REPLICATE,
    )


def preprocess(tile_bgr: "np.ndarray", mean: List[float], std: List[float]) -> "np.ndarray":
    import cv2
    import numpy as np

    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    rgb = (rgb - np.array(mean, dtype="float32")) / np.array(std, dtype="float32")
    return rgb.transpose(2, 0, 1)[None, :, :, :]


def predict_tile(
    model,
    tile_bgr: "np.ndarray",
    mean: List[float],
    std: List[float],
    crack_class: int,
) -> "np.ndarray":
    import paddle
    import paddle.nn.functional as F
    import numpy as np

    tensor = paddle.to_tensor(preprocess(tile_bgr, mean, std))
    with paddle.no_grad():
        output = model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        if tuple(output.shape[-2:]) != tile_bgr.shape[:2]:
            output = F.interpolate(
                output,
                size=tile_bgr.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
        prob = F.softmax(output, axis=1)[:, crack_class, :, :]
    return prob.numpy()[0]


def infer_image(
    model,
    image_bgr: "np.ndarray",
    crop_size: Tuple[int, int],
    overlap: int,
    mean: List[float],
    std: List[float],
    crack_class: int,
) -> "np.ndarray":
    import numpy as np

    height, width = image_bgr.shape[:2]
    crop_h, crop_w = crop_size
    stride_h = max(1, crop_h - overlap)
    stride_w = max(1, crop_w - overlap)

    prob_sum = np.zeros((height, width), dtype="float32")
    count = np.zeros((height, width), dtype="float32")

    for y in positions(height, crop_h, stride_h):
        for x in positions(width, crop_w, stride_w):
            tile = image_bgr[y : min(y + crop_h, height), x : min(x + crop_w, width)]
            real_h, real_w = tile.shape[:2]
            padded = pad_tile(tile, crop_h, crop_w)
            prob = predict_tile(model, padded, mean, std, crack_class)[:real_h, :real_w]
            prob_sum[y : y + real_h, x : x + real_w] += prob
            count[y : y + real_h, x : x + real_w] += 1.0

    return prob_sum / np.maximum(count, 1.0)


def make_overlay(image_bgr: "np.ndarray", mask: "np.ndarray", alpha: float) -> "np.ndarray":
    import cv2
    import numpy as np

    overlay = image_bgr.copy()
    red = np.zeros_like(image_bgr)
    red[:, :, 2] = 255
    overlay[mask] = cv2.addWeighted(image_bgr, 1.0 - alpha, red, alpha, 0)[mask]
    return overlay


def make_preview(
    image_bgr: "np.ndarray", mask_u8: "np.ndarray", overlay: "np.ndarray"
) -> "np.ndarray":
    import cv2
    import numpy as np

    mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    preview = np.concatenate([image_bgr, mask_bgr, overlay], axis=1)
    max_width = 2400
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


def save_outputs(
    image_path: Path,
    image_bgr: "np.ndarray",
    prob: "np.ndarray",
    output_dir: Path,
    threshold: float,
    alpha: float,
) -> None:
    import cv2
    import numpy as np

    stem = image_path.stem
    masks_dir = output_dir / "masks"
    probs_dir = output_dir / "probs"
    overlays_dir = output_dir / "overlays"
    previews_dir = output_dir / "previews"
    for directory in (masks_dir, probs_dir, overlays_dir, previews_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mask = prob >= threshold
    mask_u8 = (mask.astype("uint8") * 255)
    prob_u8 = np.clip(prob * 255.0, 0, 255).astype("uint8")
    overlay = make_overlay(image_bgr, mask, alpha)
    preview = make_preview(image_bgr, mask_u8, overlay)

    cv2.imwrite(str(masks_dir / f"{stem}_mask.png"), mask_u8)
    cv2.imwrite(str(probs_dir / f"{stem}_prob.png"), prob_u8)
    cv2.imwrite(str(overlays_dir / f"{stem}_overlay.jpg"), overlay)
    cv2.imwrite(str(previews_dir / f"{stem}_preview.jpg"), preview)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Image file or folder.")
    parser.add_argument("--output", required=True, type=Path, help="Output folder.")
    parser.add_argument(
        "--official-repo",
        type=Path,
        default=Path("third_party/HrSegNet4CrackSegmentation"),
        help="Path to the official HrSegNet repository.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("third_party/HrSegNet4CrackSegmentation/configs/hrsegnetb48.yml"),
        help="PaddleSeg config file.",
    )
    parser.add_argument("--weights", required=True, type=Path, help="Model .pdparams file.")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"], help="Paddle device.")
    parser.add_argument("--crop-size", default="400,400", type=parse_pair)
    parser.add_argument("--overlap", default=96, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--crack-class", default=1, type=int)
    parser.add_argument("--alpha", default=0.55, type=float)
    parser.add_argument("--mean", type=parse_triplet, default=None)
    parser.add_argument("--std", type=parse_triplet, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    crop_size = (int(args.crop_size[0]), int(args.crop_size[1]))

    import cv2
    from tqdm import tqdm

    if not args.official_repo.exists():
        raise SystemExit(
            f"official repo not found: {args.official_repo}\n"
            "Run: bash scripts/bootstrap_third_party.sh"
        )
    if not args.config.exists():
        raise SystemExit(f"config not found: {args.config}")
    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")

    norm = norm_from_config(args.config)
    mean = args.mean or (norm[0] if norm else [0.5, 0.5, 0.5])
    std = args.std or (norm[1] if norm else [0.5, 0.5, 0.5])

    images = find_images(args.input)
    if not images:
        raise SystemExit(f"no images found under: {args.input}")

    model = load_model(args.official_repo, args.config, args.weights, args.device)

    args.output.mkdir(parents=True, exist_ok=True)
    for image_path in tqdm(images, desc="infer"):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"skip unreadable image: {image_path}", file=sys.stderr)
            continue
        prob = infer_image(
            model=model,
            image_bgr=image_bgr,
            crop_size=crop_size,
            overlap=args.overlap,
            mean=mean,
            std=std,
            crack_class=args.crack_class,
        )
        save_outputs(image_path, image_bgr, prob, args.output, args.threshold, args.alpha)


if __name__ == "__main__":
    main()
