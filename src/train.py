from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import torch

from dataset import WaterDataset
from model import create_model
from training import checkpoints, mixed_data, preflight, runner, sources
from training.losses import weighted_bce

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "outputs" / "models"
RUN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,110}\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
EPOCH_RE = re.compile(r"epoch_([0-9]+)\.pth\Z")

SEED = 44
LR = 5e-6
THRESHOLD = 0.55
BATCH_SIZE = 1
VAL_SCENES = 371
VAL_WINDOWS = 1337
TRAIN_PAIRS = 5406


def resolve_device(name: str) -> torch.device:
    if name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS device is unavailable")
    elif name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is unavailable")
    elif name != "cpu":
        raise ValueError(f"Unsupported device: {name}")
    return torch.device(name)


def source_paths(arg: str, expected_sha: str | None) -> tuple[Path, int, str]:
    path = Path(arg).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(MODELS.resolve()):
        raise ValueError("Source must be a file under outputs/models")
    match = EPOCH_RE.fullmatch(path.name)
    if not match or int(match.group(1)) < 1:
        raise ValueError("Source must be a full epoch_XXX.pth checkpoint")
    actual_sha = sources.checksum(path)
    if expected_sha is not None and (
        not SHA_RE.fullmatch(expected_sha) or actual_sha != expected_sha
    ):
        raise ValueError("Source checkpoint SHA-256 differs")
    return path, int(match.group(1)), actual_sha


def inspect_source(path: Path, epoch: int, sha: str) -> tuple[dict, str, str, int]:
    """Read only enough to validate lineage before calling strict source loader."""
    config_path = path.parent / "config.json"
    history_path = path.parent / "history.csv"
    config_sha = sources.checksum(config_path)
    history_sha = sources.checksum(history_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("run") != path.parent.name:
        raise ValueError("Parent source config/run differs")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("continuation_epoch") != epoch:
        raise ValueError("Source is not a complete local-epoch checkpoint")
    if saved.get("run_metadata") != config:
        raise ValueError("Source run_metadata differs from config.json")
    state = saved.get("optimizer_state_dict")
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("state"), dict)
        or not state["state"]
    ):
        raise ValueError("Source has no AdamW state")
    try:
        steps = {int(item["step"].item()) for item in state["state"].values()}
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Malformed source AdamW steps") from exc
    if len(steps) != 1 or next(iter(steps)) < 1:
        raise ValueError("Source AdamW steps differ")
    del saved
    return config, config_sha, history_sha, steps.pop()


def prepare_data(mode: str):
    base_rows, external_rows, data_config = mixed_data.load_screened_train_rows(ROOT)
    if (
        (len(base_rows), len(external_rows)) != (4023, 1383)
        or data_config.get("train_unique_pairs") != TRAIN_PAIRS
        or data_config.get("train_manifest_sha256") != mixed_data.TRAIN_SHA
        or data_config.get("oem_val_sha256") != mixed_data.VAL_SHA
    ):
        raise ValueError("Frozen screened Train/Validation differs")

    validation_csv = (ROOT / data_config["oem_val"]).resolve(strict=True)
    if (
        not validation_csv.is_relative_to(ROOT)
        or sources.checksum(validation_csv) != mixed_data.VAL_SHA
    ):
        raise ValueError("Frozen Validation CSV SHA differs")

    if mode == "standard":
        dataset = mixed_data.build_standard_training_dataset(
            ROOT, base_rows, external_rows, data_config
        )
        counts = None
    elif mode == "mixed":
        dataset, counts = mixed_data.build_mixed_training_dataset(
            ROOT, base_rows, external_rows, data_config
        )
    else:
        raise ValueError("Unsupported window mode")

    try:
        validation = WaterDataset(
            validation_csv, ROOT / "data/raw/openearthmap", mode="val"
        )
        if (
            len(dataset),
            len(dataset.rows),
            len(validation.samples),
            len(validation),
        ) != (TRAIN_PAIRS, TRAIN_PAIRS, VAL_SCENES, VAL_WINDOWS):
            raise ValueError("Unexpected Train/Validation coverage")
        if len({(r["source"], r["pair_id"]) for r in dataset.rows}) != TRAIN_PAIRS:
            raise ValueError("Duplicated training pairs")
        if set(Counter(r["source"] for r in dataset.rows)) != {
            "OEM",
            "CAID",
            "LandCover",
            "xBD",
            "FLAIR",
            "GLH-Water",
            "CoastTrain_NAIP11",
        }:
            raise ValueError("Missing training source")
        if len({row["region"] for row in validation.samples}) != 11:
            raise ValueError("Unexpected Validation region count")
        return dataset, validation, data_config, counts
    except BaseException:
        dataset.close()
        raise


def preflight_tensors(
    dataset,
    mode: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    water_weight: float,
    land_weight: float,
) -> list[str]:
    """Check every source, plus the selected weighted loss without a step."""
    if mode == "mixed":
        seen = mixed_data._check_training_tensors(dataset, model, device)
        preflight.check_arm_loss(
            dataset, model, optimizer, device, water_weight, land_weight
        )
        return seen

    original_hash = runner.state_hash(model)
    step_before = {int(v["step"].item()) for v in optimizer.state.values()}
    seen: set[str] = set()
    model.eval()
    try:
        with torch.inference_mode():
            for idx, row in enumerate(dataset.rows):
                if row["source"] in seen:
                    continue
                image, water, valid = dataset[idx]
                if (
                    image.shape != (3, 512, 512)
                    or water.shape != (1, 512, 512)
                    or valid.shape != (1, 512, 512)
                    or not int(valid.sum())
                ):
                    raise ValueError("Invalid standard 512 training tensor")
                logits = model(image.unsqueeze(0).to(device))
                if logits.shape != (1, 1, 512, 512) or not torch.isfinite(logits).all():
                    raise ValueError("Invalid standard model output")
                seen.add(row["source"])
                if len(seen) == 7:
                    break
        if len(seen) != 7:
            raise ValueError("Missing source in standard preflight")
        image, water, valid = dataset[0]
        runner.configure_training_mode(model, True)
        logits = model(image.unsqueeze(0).to(device))
        pixels = weighted_bce(
            logits, water.unsqueeze(0).to(device), water_weight, land_weight
        )
        v = valid.unsqueeze(0).to(device)
        loss = (pixels * v).sum() / v.sum()
        if not torch.isfinite(loss):
            raise ValueError("Standard preflight loss is nonfinite")
        loss.backward()
        if not any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise ValueError("Invalid standard preflight gradients")
    finally:
        model.zero_grad(set_to_none=True)
        model.eval()
    if (
        runner.state_hash(model) != original_hash
        or {int(v["step"].item()) for v in optimizer.state.values()} != step_before
    ):
        raise ValueError("Standard preflight changed model/AdamW state")
    return sorted(seen)


def run_config(
    args: argparse.Namespace, source: dict, data_config: dict, counts: dict | None
) -> dict:
    """New run metadata; no historical SHA of obsolete source code is required."""
    data = {
        "run": args.run,
        "purpose": "unified_5406_continuation",
        "epochs": args.epochs,
        "start_epoch": 1,
        "source_run": source["source_run"],
        "source_epoch": source["source_epoch"],
        "source_checkpoint": str(Path(source["source_checkpoint"]).relative_to(ROOT)),
        "source_checkpoint_sha256": source["source_checkpoint_sha256"],
        "source_config_sha256": source["source_config_sha256"],
        "source_history_sha256": source["source_history_sha256"],
        "source_optimizer_step": source["source_optimizer_step"],
        "optimizer_restored": True,
        "optimizer_restarted": False,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "freeze_batchnorm": True,
        "pos_weight": args.water_weight,
        "negative_weight": args.land_weight,
        "threshold": THRESHOLD,
        "train_unique_pairs": TRAIN_PAIRS,
        "train_manifest_sha256": data_config["train_manifest_sha256"],
        "external_file_index_sha256": data_config["external_file_index_sha256"],
        "validation_csv_sha256": data_config["oem_val_sha256"],
        "validation_scenes": VAL_SCENES,
        "validation_windows": VAL_WINDOWS,
        "oem_val": data_config["oem_val"],
        "window_mode": args.mode,
        "mixed_train_policy": (
            mixed_data.MIXED_POLICY if args.mode == "mixed" else "selected_512_only"
        ),
        "native768_counts": counts["native768_counts"] if counts is not None else {},
        "fallback512_counts": (
            counts["fallback512_counts"] if counts is not None else {}
        ),
        "context_index_sha256": (
            mixed_data.CONTEXT_INDEX_SHA if args.mode == "mixed" else None
        ),
        "sampling_schedule_base_epoch": args.schedule_base_epoch,
        "device": str(args.device),
        "keep_all_full": True,
        "run_code_sha256": sources.checksum(Path(__file__)),
        "model_code_sha256": sources.checksum(ROOT / "src/model.py"),
        "dataset_code_sha256": sources.checksum(ROOT / "src/dataset.py"),
        "augmentation_code_sha256": sources.checksum(
            ROOT / "src/geometry_augmentation.py"
        ),
        "training_code_sha256": {
            path.relative_to(ROOT / "src/training").as_posix(): sources.checksum(path)
            for path in sorted((ROOT / "src/training").rglob("*.py"))
        },
    }
    if args.mode == "mixed":
        data["context_index"] = str(mixed_data.CONTEXT_INDEX.relative_to(ROOT))
    return data


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--source",
        required=True,
        help="Full checkpoint outputs/models/NAME/epoch_NNN.pth",
    )
    cli.add_argument(
        "--source-sha256", default=None, help="Optional expected SHA-256 of source .pth"
    )
    cli.add_argument(
        "--run", required=True, help="New run directory name under outputs/models"
    )
    cli.add_argument(
        "--mode",
        choices=("standard", "mixed"),
        required=True,
        help="512 only, or 768/512 with 512 supervised pixels",
    )
    cli.add_argument(
        "--epochs", type=int, required=True, help="Number of NEW local epochs"
    )
    cli.add_argument(
        "--schedule-base-epoch",
        type=int,
        required=True,
        help="Logical epoch preceding the new run's local epoch 1",
    )
    cli.add_argument("--water-weight", type=float, required=True)
    cli.add_argument("--land-weight", type=float, required=True)
    cli.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    action = cli.add_mutually_exclusive_group()
    action.add_argument(
        "--preflight",
        action="store_true",
        help="Validate source, datasets and forward/backward without training",
    )
    action.add_argument(
        "--resume",
        action="store_true",
        help="Resume ONLY a run previously started by this entrypoint",
    )
    return cli


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if not RUN_RE.fullmatch(args.run) or args.run in (".", ".."):
        raise ValueError("Invalid new run name")
    if args.epochs < 1 or args.schedule_base_epoch < 0:
        raise ValueError("Epochs and schedule must be positive/nonnegative")
    if any(
        not math.isfinite(w) or w <= 0 for w in (args.water_weight, args.land_weight)
    ):
        raise ValueError("Class weights must be positive finite values")
    if args.schedule_base_epoch + args.epochs > 1_000_000:
        raise ValueError("Logical schedule out of range")

    output = MODELS / args.run
    if args.resume:
        if not output.is_dir():
            raise ValueError("No existing run to resume")
    elif output.exists():
        raise FileExistsError(
            f"Run directory already exists: {output}. Choose a new --run."
        )

    source_path, epoch, sha = source_paths(args.source, args.source_sha256)
    if output.resolve() == source_path.parent:
        raise ValueError("Cannot train into source checkpoint directory")
    source_config, source_cfg_sha, source_hist_sha, source_step = inspect_source(
        source_path, epoch, sha
    )
    device = resolve_device(args.device)
    if source_config.get("learning_rate") != LR:
        raise ValueError(
            "Source AdamW learning rate differs; optimizer reset is not allowed"
        )
    if source_config.get("train_unique_pairs") != TRAIN_PAIRS:
        raise ValueError(
            "Expected a full source checkpoint for the screened 5406 pairs"
        )

    model = create_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    source = sources.load_full_source(
        source_path,
        sha,
        epoch,
        model,
        optimizer,
        source_config,
        source_step,
        source_cfg_sha,
        source_hist_sha,
    )
    initial_hash = runner.state_hash(model)

    dataset, validation, data_config, counts = prepare_data(args.mode)
    try:
        config = run_config(args, source, data_config, counts)
        if args.resume:
            saved_config = json.loads(
                (output / "config.json").read_text(encoding="utf-8")
            )
            if saved_config != config:
                raise ValueError("Resume configuration or code/data SHA differs")
        seen = preflight_tensors(
            dataset,
            args.mode,
            model,
            optimizer,
            device,
            args.water_weight,
            args.land_weight,
        )
        if runner.state_hash(model) != initial_hash:
            raise RuntimeError("Preflight changed model weights or BatchNorm buffers")
        print(
            json.dumps(
                {
                    "preflight": "OK",
                    "mode": args.mode,
                    "run": args.run,
                    "source": config["source_checkpoint"],
                    "source_sha256": sha,
                    "source_optimizer_step": source_step,
                    "train_pairs": len(dataset),
                    "source_counts": dict(Counter(r["source"] for r in dataset.rows)),
                    "native768_counts": config["native768_counts"],
                    "fallback512_counts": config["fallback512_counts"],
                    "forward_sources": seen,
                    "validation_scenes": len(validation.samples),
                    "validation_windows": len(validation),
                    "water_weight": args.water_weight,
                    "land_weight": args.land_weight,
                    "schedule_base_epoch": args.schedule_base_epoch,
                    "epochs": args.epochs,
                    "device": str(device),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        if args.preflight:
            return

        if args.resume:
            start_epoch, history = checkpoints.restore_epoch(
                output,
                model,
                optimizer,
                config,
                epoch_for_local=lambda e: args.schedule_base_epoch + e,
                max_epochs=args.epochs,
                sync_best=checkpoints._sync_best_full,
            )
            if start_epoch > args.epochs:
                print("All requested local epochs already completed.", flush=True)
                return
        else:
            output.mkdir(parents=True, exist_ok=False)
            (output / "config.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            start_epoch, history = 1, []

        step_before = source_step + (start_epoch - 1) * TRAIN_PAIRS
        runner.run_training_epochs(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            validation=validation,
            device=device,
            output=output,
            config=config,
            history=history,
            start_epoch=start_epoch,
            water_weight=args.water_weight,
            land_weight=args.land_weight,
            step_before=step_before,
            epoch_for_local=lambda e: args.schedule_base_epoch + e,
            save_complete_epoch=checkpoints.save_complete_epoch,
        )
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
