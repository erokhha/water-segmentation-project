from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable

import torch

from training.history import write_csv

from training.history import read_history, check_epoch_csv
from training.state import full_state_matches


def _check_resume_files(
    output: Path,
    paths: list[Path],
    saved: dict,
    rows: list[dict],
    config: dict,
    max_epochs: int,
) -> tuple[int, list[dict]]:
    epoch = saved.get("continuation_epoch")
    history = saved.get("history")
    if saved.get("run_metadata") != config:
        raise ValueError("Resume run metadata differs")
    if saved.get("source_parent_sha256") != config["source_checkpoint_sha256"]:
        raise ValueError("Resume source checkpoint differs")
    if saved.get("train_manifest_sha256") != config["train_manifest_sha256"]:
        raise ValueError("Resume Train manifest differs")
    if type(epoch) is not int or not 1 <= epoch <= max_epochs:
        raise ValueError("Resume epoch is outside configured range")
    if [path.name for path in paths] != [
        f"epoch_{n:03d}.pth" for n in range(1, epoch + 1)
    ]:
        raise ValueError("Resume checkpoint sequence is incomplete")
    if not isinstance(history, list) or len(history) != epoch or len(rows) != epoch:
        raise ValueError("Resume history length differs")
    for kind in ("thresholds", "regions"):
        actual = [path.name for path in sorted(output.glob(f"epoch_*_{kind}.csv"))]
        expected = [f"epoch_{n:03d}_{kind}.csv" for n in range(1, epoch + 1)]
        if actual != expected:
            raise ValueError(f"Resume {kind} CSV sequence is incomplete")
    return epoch, history


def _check_resume_record(
    record: dict, row: dict, epoch: int, expected_logical_epoch: int, updates: int
) -> None:
    if (
        record.get("epoch") != epoch
        or record.get("logical_epoch") != expected_logical_epoch
        or record.get("updates") != updates
    ):
        raise ValueError(f"Resume history schedule differs at epoch {epoch}")
    if set(record) != set(row) or any(
        str(value) != row[key] for key, value in record.items()
    ):
        raise ValueError(f"Resume history CSV differs at epoch {epoch}")
    if any(
        not math.isfinite(float(record[key]))
        for key in ("val_iou_055", "val_precision_055", "val_recall_055")
    ):
        raise ValueError(f"Resume metrics are nonfinite at epoch {epoch}")


def restore_epoch(
    output: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    *,
    epoch_for_local: Callable[[int], int],
    max_epochs: int,
    sync_best: Callable,
) -> tuple[int, list[dict]]:
    """Restore only own contiguous full epochs with matching CSV and AdamW."""
    output = Path(output)
    paths = sorted(output.glob("epoch_*.pth"))
    if not paths:
        raise ValueError("No complete epoch checkpoint")
    try:
        saved = torch.load(paths[-1], map_location="cpu", weights_only=True)
        rows = read_history(output / "history.csv")
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        raise ValueError("Resume checkpoint/history unreadable") from exc
    epoch, history = _check_resume_files(output, paths, saved, rows, config, max_epochs)
    for n, (record, row) in enumerate(zip(history, rows, strict=True), 1):
        _check_resume_record(
            record, row, n, epoch_for_local(n), config["train_unique_pairs"]
        )
        try:
            check_epoch_csv(
                output,
                n,
                record,
                config,
                threshold=config["threshold"],
                validation_windows=config["validation_windows"],
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Resume Validation CSV differs") from exc
    try:
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
    except (RuntimeError, ValueError, TypeError, KeyError) as exc:
        raise ValueError("Resume strict U-Net/AdamW load failed") from exc
    expected_step = (
        config["source_optimizer_step"] + epoch * config["train_unique_pairs"]
    )
    if not full_state_matches(model, optimizer, expected_step, config["learning_rate"]):
        raise ValueError("Resume U-Net/AdamW state nonfinite or step differs")
    sync_best(output, history)
    return epoch + 1, history


def _sync_best_full(output: Path, history: list[dict]) -> None:
    """Point the best alias at the earliest maximum IoU at threshold 0.55."""
    best_epoch = min(
        range(1, len(history) + 1),
        key=lambda e: (-float(history[e - 1]["val_iou_055"]), e),
    )
    source = output / f"epoch_{best_epoch:03d}.pth"
    if not source.is_file():
        raise RuntimeError("Best complete checkpoint missing")
    best = output / f"best_full_epoch_{best_epoch:03d}.pth"
    if best.exists() and not best.samefile(source):
        raise ValueError("Best complete checkpoint alias differs")
    if not best.exists():
        temporary = output / f".best_full_epoch_{best_epoch:03d}.{os.getpid()}.tmp"
        try:
            os.link(source, temporary)
            os.replace(temporary, best)
        finally:
            temporary.unlink(missing_ok=True)
    for old in output.glob("best_full_epoch_*.pth"):
        if old != best:
            old.unlink()


def save_complete_epoch(
    output: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: list[dict],
    config: dict,
) -> Path:
    """Atomically save a full model+AdamW checkpoint and threshold-0.55 history."""
    output = Path(output)
    metric_names = ("val_iou_055", "val_precision_055", "val_recall_055")
    if (
        type(epoch) is not int
        or not 1 <= epoch <= config["epochs"]
        or len(history) != epoch
        or history[-1].get("epoch") != epoch
        or history[-1].get("updates") != config["train_unique_pairs"]
        or any(
            not math.isfinite(float(r[key])) for r in history for key in metric_names
        )
        or config.get("keep_all_full") is not True
    ):
        raise ValueError("Incomplete epoch inputs")
    path = output / f"epoch_{epoch:03d}.pth"
    if path.exists():
        raise FileExistsError(path)
    expected_step = (
        config["source_optimizer_step"] + epoch * config["train_unique_pairs"]
    )
    if (
        not optimizer.state
        or len(optimizer.param_groups) != 1
        or {int(s["step"].item()) for s in optimizer.state.values()} != {expected_step}
    ):
        raise ValueError("Incomplete AdamW epoch step")
    temporary = output / f".epoch_{epoch:03d}.{os.getpid()}.tmp"
    payload = {
        "source_parent_sha256": config["source_checkpoint_sha256"],
        "train_manifest_sha256": config["train_manifest_sha256"],
        "continuation_epoch": epoch,
        "run_metadata": config,
        "history": history,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    try:
        torch.save(payload, temporary)
        check = torch.load(temporary, map_location="cpu", weights_only=True)
        if (
            check.get("run_metadata") != config
            or check.get("continuation_epoch") != epoch
            or check.get("history") != history
            or not check.get("optimizer_state_dict", {}).get("state")
            or set(check.get("model_state_dict", {})) != set(model.state_dict())
        ):
            raise RuntimeError("Saved complete checkpoint verification failed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    history_tmp = output / f".history.{os.getpid()}.csv"
    write_csv(history_tmp, history)
    os.replace(history_tmp, output / "history.csv")
    _sync_best_full(output, history)
    return path
