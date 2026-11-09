from __future__ import annotations

from pathlib import Path


from training.sources import checksum

ROOT = Path(__file__).resolve().parents[2]


FIRST_RUN = "glh_coast5406_from09_mixed768_40_20261102_001"


SECOND_RUN = "glh_coast5406_from09_mixed768_best40_20261102_001"


SOURCE_SHA = "f2eb899e7cd718f884f0add35f4948ae5f27b63bcdaf657f50342b949a246b7b"


TRAIN_SHA = "460833ec51f30d706c991b9b330f569f32d89f1a9266d42197da976e953c6e9f"


VAL_SHA = "5c285e82048e9733ee8e8428ad743a850beba7ac6ce6e9611b539018db56ce3c"


CONTEXT_INDEX = ROOT / "data/processed/glh_coast_context768_20261102_001/index.csv"


EPOCHS = 40


STEPS = 5406


VAL_SCENES = 371


VAL_WINDOWS = 1337


THRESHOLD = 0.55


MIXED_POLICY = "native_768_context_center_512_supervision_else_512"


CONTEXT_INDEX_SHA = "dfb8bf32411f6fca8856d66e83aa198efcf74b7d55ef86e85b963e19d5812899"


def logical_epoch(stage: int, local_epoch: int) -> int:
    """Use 290..329, then 330..369 without repeating the first-stage order."""
    if (
        type(stage) is not int
        or stage not in (1, 2)
        or type(local_epoch) is not int
        or (not 1 <= local_epoch <= EPOCHS)
    ):
        raise ValueError("stage must be 1/2 and local_epoch must be 1..40")
    return 289 + (stage - 1) * EPOCHS + local_epoch


def build_run_config(stage: int, source: dict, data_config: dict, counts: dict) -> dict:
    """Pin source, unchanged factors and exact mixed-context membership."""
    logical_epoch(stage, 1)
    if (
        data_config.get("train_manifest_sha256") != TRAIN_SHA
        or data_config.get("oem_val_sha256") != VAL_SHA
        or data_config.get("train_unique_pairs") != STEPS
        or (not isinstance(counts, dict))
        or (sum(counts.get("native768_counts", {}).values()) != 2844)
        or (sum(counts.get("fallback512_counts", {}).values()) != 2562)
        or (source.get("source_checkpoint_sha256") is None)
        or (source.get("source_optimizer_step") is None)
    ):
        raise ValueError("Run source/data/context identity differs")
    run = FIRST_RUN if stage == 1 else SECOND_RUN
    source_path = Path(source["source_checkpoint"])
    if not source_path.is_relative_to(ROOT):
        raise ValueError("Source checkpoint is outside project")
    return {
        "run": run,
        "purpose": "epoch09_mixed_768_context_40_stage_" + str(stage),
        "epochs": EPOCHS,
        "start_epoch": 1,
        "source_run": source["source_run"],
        "source_epoch": source["source_epoch"],
        "source_logical_epoch": (
            289 if stage == 1 else logical_epoch(1, source["source_epoch"])
        ),
        "source_checkpoint": str(source_path.relative_to(ROOT)),
        "source_checkpoint_sha256": source["source_checkpoint_sha256"],
        "source_config_sha256": source["source_config_sha256"],
        "source_history_sha256": source["source_history_sha256"],
        "source_optimizer_step": source["source_optimizer_step"],
        "optimizer_restored": True,
        "optimizer_restarted": False,
        "learning_rate": 5e-06,
        "batch_size": 1,
        "seed": 44,
        "freeze_batchnorm": True,
        "pos_weight": 1,
        "negative_weight": 1.2,
        "threshold": THRESHOLD,
        "train_unique_pairs": STEPS,
        "validation_scenes": VAL_SCENES,
        "validation_windows": VAL_WINDOWS,
        "train_manifest_sha256": TRAIN_SHA,
        "external_file_index_sha256": data_config["external_file_index_sha256"],
        "validation_csv_sha256": VAL_SHA,
        "context_index": str(CONTEXT_INDEX.relative_to(ROOT)),
        "context_index_sha256": CONTEXT_INDEX_SHA,
        "mixed_train_policy": MIXED_POLICY,
        "native768_counts": counts["native768_counts"],
        "fallback512_counts": counts["fallback512_counts"],
        "sampling_schedule_base_epoch": 289 if stage == 1 else 329,
        "run_code_sha256": checksum(Path(__file__)),
        "base_context_code_sha256": checksum(ROOT / "src/training/mixed_data.py"),
        "external_context_code_sha256": checksum(ROOT / "src/training/mixed_data.py"),
        "model_code_sha256": checksum(ROOT / "src/model.py"),
        "dataset_code_sha256": checksum(ROOT / "src/dataset.py"),
        "augmentation_code_sha256": checksum(ROOT / "src/geometry_augmentation.py"),
        "validation_code_sha256": checksum(ROOT / "src/training/runner.py"),
        "loss_code_sha256": checksum(ROOT / "src/training/losses.py"),
        "train_epoch_code_sha256": checksum(ROOT / "src/training/runner.py"),
        "keep_all_full": True,
        "new_source_semantic_review_complete": False,
        "glh_explicit_file_rights_confirmed": False,
    }
