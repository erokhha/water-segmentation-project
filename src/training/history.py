from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows without imposing a history-specific schema."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_history(path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError("Run history is absent or empty")
    return rows


def check_epoch_csv(
    output: Path,
    epoch: int,
    record: dict,
    config: dict,
    *,
    threshold: float,
    validation_windows: int,
) -> None:
    """Match saved validation files to the embedded full-epoch history."""
    thresholds = read_history(output / f"epoch_{epoch:03d}_thresholds.csv")
    regions = read_history(output / f"epoch_{epoch:03d}_regions.csv")
    score = [row for row in thresholds if float(row["threshold"]) == threshold]
    selected_regions = [row for row in regions if float(row["threshold"]) == threshold]
    if (
        len(score) != 1
        or not selected_regions
        or (
            config.get("validation_windows") == validation_windows
            and len(selected_regions) != 11
        )
        or any(
            score[0][key] != str(record[history_key])
            for key, history_key in (
                ("iou", "val_iou_055"),
                ("precision", "val_precision_055"),
                ("recall", "val_recall_055"),
                ("tp", "val_tp_055"),
                ("fp", "val_fp_055"),
                ("fn", "val_fn_055"),
            )
        )
        or any(
            sum(int(r[key]) for r in selected_regions) != int(score[0][key])
            for key in ("tp", "fp", "fn")
        )
    ):
        raise ValueError("Resume threshold/region CSV differs from history")


THRESHOLDS = np.array([i / 20 for i in range(1, 20)], dtype=np.float32)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Нет строк для сохранения")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def counts_for_thresholds(probability, truth, known):
    """Счётчики по всем порогам без массива размером 19 x H x W."""
    values = probability[known]
    targets = truth[known]
    # В bin j попадают оценки, прошедшие j порогов; равенство учитываем как >=.
    bins = np.searchsorted(THRESHOLDS, values, side="right")
    positive_hist = np.bincount(bins[targets], minlength=len(THRESHOLDS) + 1)
    negative_hist = np.bincount(bins[~targets], minlength=len(THRESHOLDS) + 1)
    tp = np.cumsum(positive_hist[::-1], dtype=np.int64)[::-1][1:]
    fp = np.cumsum(negative_hist[::-1], dtype=np.int64)[::-1][1:]
    fn = int(targets.sum()) - tp
    return np.stack((tp, fp, fn), axis=1), int(known.sum())


def metric_rows(counts, valid_count: int) -> list[dict]:
    rows = []
    if valid_count <= 0:
        raise ValueError("Нет размеченных пикселей")
    for i, threshold in enumerate(THRESHOLDS):
        tp, fp, fn = map(int, counts[i])
        union = tp + fp + fn
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "iou": tp / union if union else None,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "predicted_water_fraction": (tp + fp) / valid_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "valid_pixels": valid_count,
            }
        )
    return rows
