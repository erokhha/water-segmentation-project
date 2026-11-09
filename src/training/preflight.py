
from __future__ import annotations


import torch

from training.losses import negative_weighted_bce_12, weighted_bce
from training.runner import configure_training_mode, state_hash

EXPECTED_SOURCES = {
    "OEM",
    "CAID",
    "LandCover",
    "xBD",
    "FLAIR",
    "GLH-Water",
    "CoastTrain_NAIP11",
}


def _valid_triplet(image, water, valid) -> bool:
    size = image.shape[-1]
    if size not in (512, 768):
        return False
    if (tuple(image.shape), tuple(water.shape), tuple(valid.shape)) != (
        (3, size, size),
        (1, size, size),
        (1, size, size),
    ):
        return False
    if any(
        t.dtype != torch.float32
        or not t.is_contiguous()
        or not bool(torch.isfinite(t).all())
        for t in (image, water, valid)
    ):
        return False
    if not bool(torch.all((water == 0) | (water == 1))):
        return False
    if not bool(torch.all((valid == 0) | (valid == 1))):
        return False
    return 0 < int(valid.sum()) <= (512 * 512 if size == 768 else size * size)


def _first_native_index(dataset) -> int:
    return next(
        index for index, size in enumerate(dataset.base.context_sizes) if size == 768
    )


def _check_training_tensors(
    dataset, model: torch.nn.Module, device: torch.device
) -> list[str]:
    """Forward one row per source and backward once at 768 without stepping."""
    original = state_hash(model)
    seen = set()
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(dataset.rows):
            source = row["source"]
            if source in seen:
                continue
            image, water, valid = dataset[index]
            if not _valid_triplet(image, water, valid):
                raise ValueError(f"Invalid mixed source tensor: {source}")
            size = image.shape[-1]
            logits = model(image[None].to(device))
            if tuple(logits.shape) != (1, 1, size, size) or not bool(
                torch.isfinite(logits).all()
            ):
                raise ValueError(f"Invalid mixed source forward: {source}")
            seen.add(source)
            if len(seen) == 7:
                break
    if seen != EXPECTED_SOURCES or state_hash(model) != original:
        raise ValueError("Mixed source coverage/model state differs")
    first_native = _first_native_index(dataset)
    image, water, valid = dataset[first_native]
    configure_training_mode(model, True)
    logits = model(image[None].to(device))
    loss = negative_weighted_bce_12(logits, water[None].to(device))
    value = (loss * valid[None].to(device)).sum() / valid.sum().to(device)
    if not bool(torch.isfinite(value)):
        raise ValueError("Mixed 768 backward loss is nonfinite")
    value.backward()
    if not any(
        (
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        )
    ):
        raise ValueError("Mixed 768 backward gradients are nonfinite")
    model.zero_grad(set_to_none=True)
    model.eval()
    if state_hash(model) != original:
        raise ValueError("Mixed preflight changed model state")
    return sorted(seen)


def check_arm_loss(
    dataset,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    water_weight: float,
    land_weight: float,
) -> None:
    """Exercise the selected loss on a real 768 row without optimizer.step."""
    before = state_hash(model)
    steps = {int(state["step"].item()) for state in optimizer.state.values()}
    first_native = _first_native_index(dataset)
    image, water, valid = dataset[first_native]
    configure_training_mode(model, True)
    logits = model(image[None].to(device))
    pixel_loss = weighted_bce(logits, water[None].to(device), water_weight, land_weight)
    value = (pixel_loss * valid[None].to(device)).sum() / valid.sum().to(device)
    if not bool(torch.isfinite(value)):
        raise ValueError("Selected arm loss is nonfinite")
    value.backward()
    if not any(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
    ):
        raise ValueError("Selected arm gradients are nonfinite")
    model.zero_grad(set_to_none=True)
    model.eval()
    if (
        state_hash(model) != before
        or {int(state["step"].item()) for state in optimizer.state.values()} != steps
    ):
        raise ValueError("Arm preflight changed model or AdamW state")
