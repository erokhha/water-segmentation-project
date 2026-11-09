from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from training.history import read_history
from training.state import full_state_matches


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_history(path: Path) -> list[dict[str, str]]:
    rows = read_history(path)
    if not {"epoch", "updates"}.issubset(rows[0]):
        raise ValueError("Source history schema/rows differ")
    return rows


def load_full_source(
    source: Path,
    expected_sha: str,
    source_epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_config: dict,
    expected_step: int,
    expected_config_sha: str | None = None,
    expected_history_sha: str | None = None,
) -> dict:
    source = Path(source)
    if (
        type(source_epoch) is not int
        or source_epoch < 1
        or type(expected_step) is not int
        or expected_step < 1
        or not source.is_file()
        or checksum(source) != expected_sha
        or source.name != f"epoch_{source_epoch:03d}.pth"
    ):
        raise ValueError("Source checkpoint path/epoch/SHA differs")
    config_path, history_path = (
        source.parent / "config.json",
        source.parent / "history.csv",
    )
    if (
        expected_config_sha is not None and checksum(config_path) != expected_config_sha
    ) or (
        expected_history_sha is not None
        and checksum(history_path) != expected_history_sha
    ):
        raise ValueError("Source config/history SHA differs")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows = _read_history(history_path)
        saved = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise ValueError("Source config/history/checkpoint unreadable") from exc
    if (
        not isinstance(config, dict)
        or any(config.get(key) != value for key, value in expected_config.items())
        or len(rows) < source_epoch
        or len(rows) > config.get("epochs", -1)
        or any(
            int(row["epoch"]) != n
            or int(row["updates"]) != config["train_unique_pairs"]
            for n, row in enumerate(rows, 1)
        )
    ):
        raise ValueError("Source config/history identity differs")
    history = saved.get("history")
    if (
        saved.get("continuation_epoch") != source_epoch
        or saved.get("run_metadata") != config
        or (
            "train_manifest_sha256" in config
            and saved.get("train_manifest_sha256") != config["train_manifest_sha256"]
        )
        or (
            "source_checkpoint_sha256" in config
            and saved.get("source_parent_sha256") != config["source_checkpoint_sha256"]
        )
        or not isinstance(history, list)
        or len(history) != source_epoch
        or any(
            row.get("epoch") != n
            or row.get("updates") != config["train_unique_pairs"]
            or any(rows[n - 1].get(key) != str(value) for key, value in row.items())
            for n, row in enumerate(history, 1)
        )
        or "model_state_dict" not in saved
        or "optimizer_state_dict" not in saved
    ):
        raise ValueError("Source full checkpoint metadata/history differs")
    try:
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
    except (RuntimeError, ValueError, TypeError, KeyError) as exc:
        raise ValueError("Source model/AdamW strict load failed") from exc
    if not full_state_matches(model, optimizer, expected_step, config["learning_rate"]):
        raise ValueError("Source model/AdamW nonfinite or step differs")
    return {
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": expected_sha,
        "source_config_sha256": checksum(config_path),
        "source_history_sha256": checksum(history_path),
        "source_optimizer_step": expected_step,
        "source_epoch": source_epoch,
        "source_run": source.parent.name,
    }
