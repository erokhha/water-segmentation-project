"""Independent full-TIFF water segmentation with 512 or 768 RGB windows.

The model architecture is imported unchanged from model.py. No historical
inference, training or audit scripts are imported.

Examples (run from repository root):
    python src/infer.py --image /path/to/rgb.tif \
      --checkpoint outputs/models/BRANCH/epoch_009.pth \
      --output outputs/checks/infer_B09_768 --window-size 768 \
      --stride 576 --threshold 0.55 --preflight

    python src/infer.py --image /path/to/rgb.tif \
      --checkpoint outputs/models/BRANCH/epoch_009.pth \
      --output outputs/checks/infer_B09_768 --window-size 768 \
      --stride 576 --threshold 0.55 --review-csv docs/stage6-review-windows.csv

Outputs: water_probability.tif (float32), water_mask.tif (uint8 0/1),
manifest.json and optional preview images. Existing output directories are
never overwritten. The entire image rectangle is processed; no boundary or
black-background footprint is inferred automatically.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.windows import Window
from PIL import Image
import torch

from dataset import MEAN, STD

ROOT = Path(__file__).resolve().parents[1]
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
EPOCH_RE = re.compile(r"epoch_(\d+)\.pth\Z")


@dataclass(frozen=True)
class InputInfo:
    image_path: Path
    checkpoint_path: Path
    image_sha256: str
    checkpoint_sha256: str
    width: int
    height: int
    crs: str | None
    transform: rasterio.Affine
    checkpoint_epoch: int | None
    checkpoint_run: str | None


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_expected_sha(actual: str, expected: str | None, label: str) -> None:
    if expected is not None and (
        SHA_RE.fullmatch(expected) is None or actual != expected
    ):
        raise ValueError(f"{label} SHA-256 differs or is malformed")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            name = "mps"
        elif torch.cuda.is_available():
            name = "cuda"
        else:
            name = "cpu"
    if name == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS is unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if name not in ("cpu", "mps", "cuda"):
        raise ValueError(f"Unsupported device: {name}")
    return torch.device(name)


def plan_window_origins(length: int, tile_size: int, stride: int) -> list[int]:
    """Same covering-window origin rule as the historical 512 inference."""
    if any(type(v) is not int for v in (length, tile_size, stride)) or \
            length <= 0 or tile_size <= 0 or not 0 < stride <= tile_size:
        raise ValueError("Invalid window geometry")
    if length <= tile_size:
        return [0]
    return sorted(set([*range(0, length - tile_size + 1, stride),
                       length - tile_size]))


def prepare_rgb_window(rgb: np.ndarray, tile_size: int) -> np.ndarray:
    """Normalize RGB exactly as dataset.py; zero-pad in normalized space."""
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or \
            rgb.shape[0] != 3 or rgb.dtype != np.uint8 or \
            type(tile_size) is not int or tile_size <= 0:
        raise ValueError("Expected uint8 CHW RGB and positive window size")
    _, height, width = rgb.shape
    if not 1 <= height <= tile_size or not 1 <= width <= tile_size:
        raise ValueError("RGB window dimensions differ")
    result = np.zeros((3, tile_size, tile_size), dtype=np.float32)
    result[:, :height, :width] = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return result


def make_blend_weights(height: int, width: int, left: int, right: int,
                       top: int, bottom: int) -> np.ndarray:
    """Original half-pixel linear feathering across overlapping windows."""
    if not all(type(v) is int for v in
               (height, width, left, right, top, bottom)) or \
            height <= 0 or width <= 0 or not 0 <= left <= width or \
            not 0 <= right <= width or not 0 <= top <= height or \
            not 0 <= bottom <= height:
        raise ValueError("Invalid blend window geometry")
    horizontal = np.ones(width, dtype=np.float32)
    vertical = np.ones(height, dtype=np.float32)
    if left:
        horizontal[:left] *= (np.arange(left, dtype=np.float32) + .5) / left
    if right:
        horizontal[-right:] *= (np.arange(right, 0, -1, dtype=np.float32) - .5) / right
    if top:
        vertical[:top] *= (np.arange(top, dtype=np.float32) + .5) / top
    if bottom:
        vertical[-bottom:] *= (np.arange(bottom, 0, -1, dtype=np.float32) - .5) / bottom
    return vertical[:, None] * horizontal[None, :]


def add_weighted_probability_window(
    probability: np.ndarray,
    weights: np.ndarray,
    x0: int,
    y0: int,
    weighted_sum_map: np.ndarray,
    weight_sum_map: np.ndarray,
    count_map: np.ndarray,
) -> None:
    if not isinstance(probability, np.ndarray) or not isinstance(weights, np.ndarray) or \
            probability.ndim != 2 or probability.dtype != np.float32 or \
            weights.shape != probability.shape or weights.dtype != np.float32:
        raise ValueError("Expected matching float32 probability and weight windows")
    if not all(isinstance(array, np.ndarray) and array.ndim == 2 for array in
               (weighted_sum_map, weight_sum_map, count_map)) or \
            weighted_sum_map.dtype != np.float32 or weight_sum_map.dtype != np.float32 or \
            not np.issubdtype(count_map.dtype, np.integer) or \
            weighted_sum_map.shape != weight_sum_map.shape or weighted_sum_map.shape != count_map.shape:
        raise ValueError("Invalid accumulator maps")
    if type(x0) is not int or type(y0) is not int or x0 < 0 or y0 < 0 or \
            probability.shape[0] == 0 or probability.shape[1] == 0 or \
            y0 + probability.shape[0] > weighted_sum_map.shape[0] or \
            x0 + probability.shape[1] > weighted_sum_map.shape[1]:
        raise ValueError("Blend window exceeds image")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)) or \
            not np.isfinite(weights).all() or np.any((weights <= 0) | (weights > 1)):
        raise ValueError("Probability or blend weights are invalid")
    region = np.s_[y0:y0 + probability.shape[0], x0:x0 + probability.shape[1]]
    if np.any(count_map[region] == np.iinfo(count_map.dtype).max):
        raise ValueError("Coverage counter would overflow")
    weighted_sum_map[region] += probability * weights
    weight_sum_map[region] += weights
    count_map[region] += 1


def model_state_hash(model: torch.nn.Module) -> str:
    """Detect changes to weights and BatchNorm buffers during inference."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu_tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu_tensor.dtype).encode("ascii"))
        digest.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
        digest.update(cpu_tensor.numpy().tobytes())
    return digest.hexdigest()


def load_model_and_inputs(
    image_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    expected_image_sha: str | None = None,
    expected_checkpoint_sha: str | None = None,
) -> tuple[torch.nn.Module, InputInfo]:
    """Strict checkpoint weights load; optimizer not required for prediction."""
    image_path = Path(image_path).expanduser().resolve(strict=True)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    if not image_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Image/checkpoint must be regular files")
    image_sha = file_hash(image_path)
    checkpoint_sha = file_hash(checkpoint_path)
    check_expected_sha(image_sha, expected_image_sha, "Image")
    check_expected_sha(checkpoint_sha, expected_checkpoint_sha, "Checkpoint")

    with rasterio.open(image_path) as image:
        if image.width <= 0 or image.height <= 0 or image.count != 3 or \
                image.dtypes != ("uint8",) * 3 or \
                image.colorinterp != (ColorInterp.red, ColorInterp.green, ColorInterp.blue):
            raise ValueError("TIFF must be three uint8 RGB bands in R, G, B order")
        width, height = image.width, image.height
        crs = str(image.crs) if image.crs is not None else None
        transform = image.transform

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not isinstance(saved.get("model_state_dict"), dict):
        raise ValueError("Checkpoint must contain model_state_dict")
    filename_match = EPOCH_RE.fullmatch(checkpoint_path.name)
    filename_epoch = int(filename_match.group(1)) if filename_match else None
    metadata_epoch = saved.get("continuation_epoch", saved.get("epoch"))
    if metadata_epoch is not None and (
        type(metadata_epoch) is not int or metadata_epoch < 1
    ):
        raise ValueError("Invalid checkpoint epoch metadata")
    if filename_epoch is not None and metadata_epoch is not None and \
            filename_epoch != metadata_epoch:
        raise ValueError("Checkpoint filename and saved epoch differ")
    config = saved.get("run_metadata")
    if config is not None and (
        not isinstance(config, dict) or
        ("run" in config and config["run"] != checkpoint_path.parent.name)
    ):
        raise ValueError("Checkpoint run_metadata differs from containing run")
    if config is not None and "train_manifest_sha256" in config and \
            saved.get("train_manifest_sha256") != config["train_manifest_sha256"]:
        raise ValueError("Checkpoint Train manifest identity differs")
    if config is not None and "source_checkpoint_sha256" in config and \
            saved.get("source_parent_sha256") != config["source_checkpoint_sha256"]:
        raise ValueError("Checkpoint source lineage differs")
    if isinstance(saved.get("history"), list) and metadata_epoch is not None and \
            len(saved["history"]) != metadata_epoch:
        raise ValueError("Checkpoint history length differs")

    # model.py is intentionally unchanged. Import it only when needed.
    from model import create_model
    model = create_model()
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.to(device).requires_grad_(False).eval()
    if any(module.training for module in model.modules()) or \
            any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("Loaded model is nonfinite or not in eval mode")
    info = InputInfo(
        image_path=image_path, checkpoint_path=checkpoint_path,
        image_sha256=image_sha, checkpoint_sha256=checkpoint_sha,
        width=width, height=height, crs=crs, transform=transform,
        checkpoint_epoch=metadata_epoch if metadata_epoch is not None else filename_epoch,
        checkpoint_run=config.get("run") if config is not None else checkpoint_path.parent.name,
    )
    return model, info



def check_first_window(model: torch.nn.Module, info: InputInfo,
                       device: torch.device, tile_size: int) -> None:
    """Preflight one real RGB window without changing weights or BN buffers."""
    before = model_state_hash(model)
    with rasterio.open(info.image_path) as image, torch.inference_mode():
        rgb = image.read((1, 2, 3), window=Window(
            0, 0, min(tile_size, info.width), min(tile_size, info.height)))
        sample = torch.from_numpy(prepare_rgb_window(rgb, tile_size)[None]).to(device)
        logits = model(sample)
        if tuple(logits.shape) != (1, 1, tile_size, tile_size) or \
                not bool(torch.isfinite(logits).all()):
            raise ValueError("Model forward for selected window size is invalid")
    if model_state_hash(model) != before:
        raise RuntimeError("Preflight forward changed model or BatchNorm state")


def read_review_sectors(path: Path, width: int, height: int) -> list[tuple]:
    """Validate optional sector CSV before creating inference output files."""
    path = Path(path).expanduser().resolve(strict=True)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Review CSV has no sectors")
    seen: set[str] = set()
    checked = []
    for row in rows:
        cell = row.get("cell", "")
        if re.fullmatch(r"[A-Za-z0-9_-]+", cell) is None or cell in seen:
            raise ValueError("Review sector name is invalid or repeated")
        seen.add(cell)
        try:
            x0, y0, w, h = (int(row[k]) for k in ("x0", "y0", "width", "height"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Review sector geometry is invalid: {cell}") from exc
        if x0 < 0 or y0 < 0 or w <= 0 or h <= 0 or x0 + w > width or y0 + h > height:
            raise ValueError(f"Review sector exceeds image: {cell}")
        role = row.get("role", "diagnostic")
        if role not in ("diagnostic", "holdout", ""):
            raise ValueError(f"Unknown review sector role: {cell}")
        checked.append((cell, role or "diagnostic", x0, y0, w, h))
    for i, a in enumerate(checked):
        for b in checked[i + 1:]:
            if a[1] != b[1] and \
                    max(a[2], b[2]) < min(a[2] + a[4], b[2] + b[4]) and \
                    max(a[3], b[3]) < min(a[3] + a[5], b[3] + b[5]):
                raise ValueError("Diagnostic and holdout sector regions overlap")
    return checked


def render_views(info: InputInfo, output: Path, review_sectors: list[tuple],
                 overview_width: int) -> dict:
    """Render optional overview and diagnostic sectors from saved rasters."""
    views = output / "previews"
    views.mkdir(exist_ok=False)
    mask_path = output / "water_mask.tif"
    probability_path = output / "water_probability.tif"
    overlay_color = np.array([0, 190, 255], dtype=np.float32)
    rendered = []
    with rasterio.open(info.image_path) as image, \
            rasterio.open(mask_path) as mask, \
            rasterio.open(probability_path) as probability:
        if (image.width, image.height) != (mask.width, mask.height) or \
                (mask.width, mask.height) != (probability.width, probability.height):
            raise RuntimeError("Preview raster dimensions differ")
        if overview_width > 0:
            scale = min(1.0, overview_width / info.width)
            size = (max(1, round(info.width * scale)),
                    max(1, round(info.height * scale)))
            rgb = np.moveaxis(image.read(
                (1, 2, 3), out_shape=(3, size[1], size[0]),
                resampling=Resampling.average), 0, -1)
            coverage = mask.read(
                1, out_shape=(size[1], size[0]),
                resampling=Resampling.average, out_dtype="float32")
            if not np.isfinite(coverage).all() or not np.all((coverage >= 0) & (coverage <= 1)):
                raise RuntimeError("Invalid overview coverage")
            colored = np.rint(rgb.astype(np.float32) *
                              (1 - .45 * coverage[..., None]) +
                              overlay_color * (.45 * coverage[..., None]))
            Image.fromarray(rgb).save(views / "full_rgb.jpg", quality=92)
            Image.fromarray(np.clip(colored, 0, 255).astype(np.uint8)).save(
                views / "full_overlay.jpg", quality=92)
        for cell, role, x0, y0, w, h in review_sectors:
            if role == "holdout":
                continue
            window = Window(x0, y0, w, h)
            rgb = np.moveaxis(image.read((1, 2, 3), window=window), 0, -1)
            binary = mask.read(1, window=window)
            if rgb.shape != (h, w, 3) or binary.shape != (h, w) or \
                    not np.isin(binary, (0, 1)).all():
                raise RuntimeError(f"Invalid diagnostic sector {cell}")
            sector_dir = views / "sectors" / cell
            sector_dir.mkdir(parents=True, exist_ok=False)
            overlay = rgb.astype(np.float32)
            overlay[binary == 1] = .55 * overlay[binary == 1] + .45 * overlay_color
            Image.fromarray(rgb).save(sector_dir / "rgb.png")
            Image.fromarray(binary * np.uint8(255)).save(sector_dir / "mask.png")
            Image.fromarray(overlay.astype(np.uint8)).save(sector_dir / "overlay.png")
            (sector_dir / "coordinates.json").write_text(
                json.dumps({"cell": cell, "x0": x0, "y0": y0,
                            "width": w, "height": h}) + "\n", encoding="utf-8")
            rendered.append(cell)
    summary = {
        "overview": overview_width > 0,
        "review_sector_count": len(rendered),
        "review_sectors": rendered,
    }
    (views / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def run_full_inference(
    model: torch.nn.Module,
    info: InputInfo,
    output: Path,
    device: torch.device,
    tile_size: int,
    stride: int,
    threshold: float,
    *,
    review_sectors: list[tuple] | None = None,
    overview_width: int = 0,
    review_csv: Path | None = None,
) -> dict:
    """Merge every overlapping prediction with verified positive feather weights."""
    if tile_size not in (512, 768) or not 0 < stride <= tile_size or \
            not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Unsupported inference window, stride, or threshold")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    before_state = model_state_hash(model)
    if any(module.training for module in model.modules()):
        raise ValueError("Model must be in eval mode")
    if file_hash(info.image_path) != info.image_sha256 or \
            file_hash(info.checkpoint_path) != info.checkpoint_sha256:
        raise ValueError("Input bytes changed after validation")

    x_origins = plan_window_origins(info.width, tile_size, stride)
    y_origins = plan_window_origins(info.height, tile_size, stride)
    count = len(x_origins) * len(y_origins)
    output.mkdir(parents=True, exist_ok=False)
    probability_path = output / "water_probability.tif"
    mask_path = output / "water_mask.tif"
    minimum_count = np.iinfo(np.uint16).max
    maximum_count = 0
    water_pixels = 0
    uncovered = 0
    with tempfile.TemporaryDirectory(prefix=".infer_accumulators_", dir=output.parent) as workspace:
        workspace = Path(workspace)
        weighted = np.memmap(workspace / "sum.bin", dtype="float32", mode="w+",
                             shape=(info.height, info.width))
        weights = np.memmap(workspace / "weights.bin", dtype="float32", mode="w+",
                            shape=(info.height, info.width))
        counts = np.memmap(workspace / "count.bin", dtype="uint16", mode="w+",
                           shape=(info.height, info.width))
        weighted[:] = 0
        weights[:] = 0
        counts[:] = 0
        with rasterio.open(info.image_path) as image, torch.inference_mode():
            if (image.width, image.height) != (info.width, info.height):
                raise RuntimeError("Input TIFF dimensions changed")
            for index, (yi, xi) in enumerate(
                ((i, j) for i in range(len(y_origins)) for j in range(len(x_origins))), 1
            ):
                y0, x0 = y_origins[yi], x_origins[xi]
                w = min(tile_size, info.width - x0)
                h = min(tile_size, info.height - y0)
                rgb = image.read((1, 2, 3), window=Window(x0, y0, w, h))
                tensor = torch.from_numpy(prepare_rgb_window(rgb, tile_size)[None]).to(device)
                logits = model(tensor)
                if tuple(logits.shape) != (1, 1, tile_size, tile_size) or \
                        not bool(torch.isfinite(logits).all()):
                    raise RuntimeError(f"Invalid model logits at window {index}")
                probability = logits.sigmoid()[0, 0, :h, :w].cpu().numpy().astype(np.float32)
                left = x_origins[xi - 1] + tile_size - x0 if xi else 0
                right = x0 + w - x_origins[xi + 1] if xi + 1 < len(x_origins) else 0
                top = y_origins[yi - 1] + tile_size - y0 if yi else 0
                bottom = y0 + h - y_origins[yi + 1] if yi + 1 < len(y_origins) else 0
                feather = make_blend_weights(h, w, left, right, top, bottom)
                add_weighted_probability_window(probability, feather, x0, y0,
                                                weighted, weights, counts)
                if index % 50 == 0 or index == count:
                    print(f"Windows: {index}/{count}", flush=True)
        weighted.flush()
        weights.flush()
        counts.flush()

        # Same GeoTIFF types, compression, and 512-row verification as legacy.
        # Preserve source georeferencing when present; legacy source had none.
        profile = dict(driver="GTiff", width=info.width, height=info.height,
                       count=1, tiled=True, blockxsize=512, blockysize=512,
                       compress="deflate", transform=info.transform)
        if info.crs is not None:
            profile["crs"] = info.crs
        with rasterio.open(probability_path, "w", dtype="float32", **profile) as dst_prob, \
                rasterio.open(mask_path, "w", dtype="uint8", **profile) as dst_mask:
            for y in range(0, info.height, 512):
                h = min(512, info.height - y)
                window = Window(0, y, info.width, h)
                coverage = np.asarray(counts[y:y + h, :])
                minimum_count = min(minimum_count, int(coverage.min()))
                maximum_count = max(maximum_count, int(coverage.max()))
                uncovered += int(np.count_nonzero(coverage == 0))
                if np.any(coverage == 0):
                    raise RuntimeError("TIFF has uncovered pixels")
                sum_of_weights = np.asarray(weights[y:y + h, :])
                if not np.isfinite(sum_of_weights).all() or np.any(sum_of_weights <= 0):
                    raise RuntimeError("TIFF has pixels without positive blend weights")
                merged = np.asarray(weighted[y:y + h, :]) / sum_of_weights
                if not np.isfinite(merged).all() or np.any((merged < -1e-6) | (merged > 1 + 1e-6)):
                    raise RuntimeError("Merged probabilities are invalid")
                merged = np.clip(merged, 0, 1).astype(np.float32)
                binary = (merged >= threshold).astype(np.uint8)
                water_pixels += int(binary.sum())
                dst_prob.write(merged, 1, window=window)
                dst_mask.write(binary, 1, window=window)
        # Reopen the files and verify every stripe, not just the in-memory masks.
        with rasterio.open(probability_path) as prob, rasterio.open(mask_path) as mask:
            if (prob.width, prob.height, prob.count, prob.dtypes) != \
                    (info.width, info.height, 1, ("float32",)) or \
                    (mask.width, mask.height, mask.count, mask.dtypes) != \
                    (info.width, info.height, 1, ("uint8",)):
                raise RuntimeError("Saved rasters have wrong geometry or types")
            for y in range(0, info.height, 512):
                window = Window(0, y, info.width, min(512, info.height - y))
                p, m = prob.read(1, window=window), mask.read(1, window=window)
                if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or \
                        not np.array_equal(m, (p >= threshold).astype(np.uint8)):
                    raise RuntimeError("Saved probability and binary mask disagree")
        del weighted, weights, counts

    after_state = model_state_hash(model)
    if before_state != after_state or \
            file_hash(info.image_path) != info.image_sha256 or \
            file_hash(info.checkpoint_path) != info.checkpoint_sha256:
        raise RuntimeError("Image, checkpoint or model state changed during inference")
    preview = None
    if overview_width > 0 or review_sectors:
        preview = render_views(info, output, review_sectors or [], overview_width)
    manifest = {
        "status": "complete",
        "purpose": "unified_water_tiff_inference",
        "merge_rule": "linear_half_pixel_feather",
        "image": str(info.image_path),
        "image_sha256": info.image_sha256,
        "checkpoint": str(info.checkpoint_path),
        "checkpoint_sha256": info.checkpoint_sha256,
        "checkpoint_epoch": info.checkpoint_epoch,
        "checkpoint_run": info.checkpoint_run,
        "width": info.width,
        "height": info.height,
        "crs": info.crs,
        "device": str(device),
        "tile_size": tile_size,
        "stride": stride,
        "overlap": tile_size - stride,
        "threshold": threshold,
        "window_count": count,
        "count_min": int(minimum_count),
        "count_max": int(maximum_count),
        "uncovered_pixels": uncovered,
        "predicted_water_pixels": water_pixels,
        "state_sha256_before": before_state,
        "state_sha256_after": after_state,
        "probability": str(probability_path),
        "mask": str(mask_path),
        "preview": preview,
        "review_csv": str(review_csv.resolve()) if review_csv is not None else None,
        "black_exterior_policy": "No automatic footprint mask; full rectangular TIFF is segmented",
        "not_final_vector_layer": True,
        "torch": str(torch.__version__),
        "rasterio": str(rasterio.__version__),
        "inference_code_sha256": file_hash(Path(__file__)),
        "model_code_sha256": file_hash(ROOT / "src/model.py"),
        "dataset_code_sha256": file_hash(ROOT / "src/dataset.py"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Three-band RGB uint8 TIFF")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Full epoch checkpoint with model_state_dict")
    parser.add_argument("--output", type=Path, required=True, help="New, non-existing result directory")
    parser.add_argument("--window-size", "--tile-size", dest="tile_size", type=int,
                        choices=(512, 768), required=True)
    parser.add_argument("--stride", type=int, default=None,
                        help="Default: 384 for 512, 576 for 768")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Default: 0.55; pass 0.65 for old 512 diagnostic")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--checkpoint-sha256", default=None)
    parser.add_argument("--image-sha256", default=None)
    parser.add_argument("--preflight", action="store_true", help="Validate without writing outputs")
    parser.add_argument("--review-csv", type=Path, default=None,
                        help="Optional cell/x0/y0/width/height CSV for sector previews")
    parser.add_argument("--overview-width", type=int, default=0,
                        help="If positive, render full-image RGB/overlay JPEG previews")
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold) or not 0 < args.threshold < 1:
        parser.error("threshold must be finite and between 0 and 1")
    if args.overview_width < 0:
        parser.error("overview-width must be nonnegative")
    stride = args.stride if args.stride is not None else (384 if args.tile_size == 512 else 576)
    if not 0 < stride <= args.tile_size:
        parser.error("stride must be between 1 and window-size")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    device = resolve_device(args.device)
    model, info = load_model_and_inputs(
        args.image, args.checkpoint, device,
        args.image_sha256, args.checkpoint_sha256)
    if output == info.image_path or output == info.checkpoint_path:
        raise ValueError("Cannot overwrite input files")
    sectors = read_review_sectors(args.review_csv, info.width, info.height) \
        if args.review_csv is not None else []
    check_first_window(model, info, device, args.tile_size)
    x = plan_window_origins(info.width, args.tile_size, stride)
    y = plan_window_origins(info.height, args.tile_size, stride)
    summary = {
        "status": "preflight_ok",
        "image_sha256": info.image_sha256,
        "checkpoint_sha256": info.checkpoint_sha256,
        "checkpoint_run": info.checkpoint_run,
        "checkpoint_epoch": info.checkpoint_epoch,
        "image_size": [info.width, info.height],
        "tile_size": args.tile_size,
        "stride": stride,
        "threshold": args.threshold,
        "window_count": len(x) * len(y),
        "device": str(device),
        "review_sectors": len(sectors),
        "output_absent": True,
        "sample_forward_ok": True,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.preflight:
        return
    result = run_full_inference(
        model, info, output, device,
        args.tile_size, stride, args.threshold,
        review_sectors=sectors, overview_width=args.overview_width,
        review_csv=args.review_csv)
    print(json.dumps({
        "status": result["status"],
        "output": str(output),
        "window_count": result["window_count"],
        "water_pixels": result["predicted_water_pixels"],
        "probability": result["probability"],
        "mask": result["mask"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
