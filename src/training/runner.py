from __future__ import annotations

import hashlib
import math
import os
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.losses import weighted_bce
from training.history import (
    THRESHOLDS,
    counts_for_thresholds,
    metric_rows,
    write_csv,
)


def state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        cpu = tensor.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str((tuple(cpu.shape), cpu.dtype)).encode("utf-8"))
        h.update(cpu.numpy())
    return h.hexdigest()


def configure_training_mode(model: torch.nn.Module, freeze_bn: bool) -> None:
    model.train()
    if freeze_bn:
        layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        if not layers:
            raise ValueError("No BatchNorm2d layers to freeze")
        for layer in layers:
            layer.eval()


def train_epoch(
    model,
    loader,
    optimizer,
    loss_function,
    device,
    freeze_bn: bool,
    max_batches: int = 0,
) -> dict:
    if max_batches < 0:
        raise ValueError("max_batches must be nonnegative")
    configure_training_mode(model, freeze_bn)
    weighted_loss = 0.0
    valid_pixels = 0
    water_pixels = 0
    batches = 0
    for batch_number, (images, masks, valid) in enumerate(loader, start=1):
        if max_batches and batch_number > max_batches:
            break
        known = int(valid.sum().item())
        if known == 0:
            continue
        water_count = int(((masks > 0.5) & (valid > 0.5)).sum().item())
        images, masks, valid = images.to(device), masks.to(device), valid.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if logits.shape != masks.shape:
            raise RuntimeError("Model output and mask shapes differ")
        pixel_loss = loss_function(logits, masks)
        loss = (pixel_loss * valid).sum() / valid.sum()
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        weighted_loss += loss_value * known
        valid_pixels += known
        water_pixels += water_count
        batches += 1
        if batch_number % 100 == 0:
            print(f"  Train: {batch_number}/{len(loader)} batches", flush=True)
    if valid_pixels == 0:
        raise RuntimeError("No valid pixels in training epoch")
    return {
        "loss": weighted_loss / valid_pixels,
        "batches": batches,
        "water_fraction": water_pixels / valid_pixels,
    }


def epoch_order(count: int, *, seed: int, epoch: int) -> list[int]:
    if (
        type(count) is not int
        or count < 1
        or type(seed) is not int
        or seed < 0
        or type(epoch) is not int
        or epoch < 1
    ):
        raise ValueError("Invalid epoch order arguments")
    return torch.randperm(
        count, generator=torch.Generator().manual_seed(seed + epoch * 1000003)
    ).tolist()


def first_step_report(source_step: int):
    sent = False

    def after_step(optimizer, args, kwargs):
        nonlocal sent
        if sent:
            return
        steps = {int(state["step"].item()) for state in optimizer.state.values()}
        if steps != {source_step + 1}:
            raise RuntimeError("First optimizer step differs from source + 1")
        print(
            f"FIRST_OPTIMIZER_STEP completed: AdamW step={source_step + 1}", flush=True
        )
        sent = True

    return after_step


@torch.inference_mode()
def evaluate_mode(model, dataset, device, mode):
    if mode not in ("running", "current_batch"):
        raise ValueError("mode must be running or current_batch")
    model.eval()
    bn_layers = [
        layer for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm2d)
    ]
    if not bn_layers:
        raise ValueError("No BatchNorm2d layers in model")
    if any(not layer.track_running_stats for layer in bn_layers):
        raise ValueError("BatchNorm2d without running statistics")

    saved = [
        (
            layer,
            {
                name: buffer.detach().clone()
                for name, buffer in layer.named_buffers(recurse=False)
            },
        )
        for layer in bn_layers
    ]
    counts = np.zeros((len(THRESHOLDS), 3), dtype=np.int64)
    region_counts = {}
    region_valid = {}
    total_valid = 0
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    try:
        if mode == "current_batch":
            for layer in bn_layers:
                layer.train()
        for i, (image, mask, valid) in enumerate(loader):
            logits = model(image.to(device))
            if logits.shape != mask.shape or not torch.isfinite(logits).all().item():
                raise RuntimeError("Invalid model output or shape")
            probability = logits.sigmoid()[0, 0].cpu().numpy()
            truth = mask[0, 0].numpy() > 0.5
            known = valid[0, 0].numpy() > 0.5
            values, n_valid = counts_for_thresholds(probability, truth, known)
            counts += values
            total_valid += n_valid
            sample_index, _, _ = dataset.windows[i]
            region = dataset.samples[sample_index]["region"]
            region_counts.setdefault(region, np.zeros_like(counts))
            region_valid[region] = region_valid.get(region, 0) + n_valid
            region_counts[region] += values
            if (i + 1) % 250 == 0:
                print(f"  {mode}: {i + 1}/{len(dataset)} windows", flush=True)
    finally:
        model.eval()
        for layer, buffers in saved:
            for name, buffer in layer.named_buffers(recurse=False):
                buffer.copy_(buffers[name])

    rows = metric_rows(counts, total_valid)
    regional = []
    for region in sorted(region_counts):
        if region_valid[region] > 0:
            for row in metric_rows(region_counts[region], region_valid[region]):
                regional.append({"region": region, **row})
    return rows, regional, len(loader)


def run_training_epochs(
    *,
    model,
    optimizer,
    dataset,
    validation,
    device,
    output: Path,
    config: dict,
    history: list[dict],
    start_epoch: int,
    water_weight: float,
    land_weight: float,
    step_before: int,
    epoch_for_local: Callable[[int], int],
    save_complete_epoch: Callable,
) -> None:
    steps = config["train_unique_pairs"]
    epochs = config["epochs"]
    threshold = config["threshold"]
    validation_windows = config["validation_windows"]

    loss = partial(weighted_bce, water_weight=water_weight, land_weight=land_weight)
    hook = optimizer.register_step_post_hook(first_step_report(step_before))
    try:
        for epoch in range(start_epoch, epochs + 1):
            scheduled = epoch_for_local(epoch)
            np.random.seed(44 + scheduled * 1000003 + 17)
            torch.manual_seed(44 + scheduled * 1000003 + 29)
            loader = DataLoader(
                dataset,
                batch_size=1,
                sampler=epoch_order(steps, seed=44, epoch=scheduled),
                num_workers=0,
                generator=torch.Generator().manual_seed(44 + scheduled * 1000003 + 41),
            )
            trained = train_epoch(model, loader, optimizer, loss, device, True)
            if trained["batches"] != steps or not math.isfinite(float(trained["loss"])):
                raise RuntimeError("Incomplete or nonfinite Train epoch")
            metrics, regions, windows = evaluate_mode(
                model, validation, device, "running"
            )
            selected = [row for row in metrics if row["threshold"] == threshold]
            selected_regions = [row for row in regions if row["threshold"] == threshold]
            if (
                windows != validation_windows
                or len(selected) != 1
                or len(selected_regions) != 11
                or any(
                    not math.isfinite(float(selected[0][key]))
                    for key in ("iou", "precision", "recall")
                )
                or any(
                    sum(int(r[key]) for r in selected_regions) != int(selected[0][key])
                    for key in ("tp", "fp", "fn")
                )
            ):
                raise RuntimeError("Incomplete Validation")
            threshold_tmp = output / f".epoch_{epoch:03d}_thresholds.{os.getpid()}.tmp"
            region_tmp = output / f".epoch_{epoch:03d}_regions.{os.getpid()}.tmp"
            write_csv(threshold_tmp, metrics)
            write_csv(region_tmp, regions)
            os.replace(threshold_tmp, output / f"epoch_{epoch:03d}_thresholds.csv")
            os.replace(region_tmp, output / f"epoch_{epoch:03d}_regions.csv")
            score = selected[0]
            history.append(
                {
                    "epoch": epoch,
                    "logical_epoch": scheduled,
                    "updates": steps,
                    "train_loss": trained["loss"],
                    "train_water_fraction": trained["water_fraction"],
                    "val_iou_055": score["iou"],
                    "val_precision_055": score["precision"],
                    "val_recall_055": score["recall"],
                    "val_tp_055": score["tp"],
                    "val_fp_055": score["fp"],
                    "val_fn_055": score["fn"],
                }
            )
            save_complete_epoch(output, epoch, model, optimizer, history, config)
            print(
                f"Epoch {epoch}/{epochs}: steps={steps}, "
                f"Val IoU={score['iou']:.6f}, "
                f"Precision={score['precision']:.6f}, "
                f"Recall={score['recall']:.6f}, "
                f"TP={score['tp']}, FP={score['fp']}, FN={score['fn']}",
                flush=True,
            )
    finally:
        hook.remove()
