
import math

import torch
import torch.nn.functional as F


def weighted_bce(logits: torch.Tensor, target: torch.Tensor,
                 water_weight: float, land_weight: float) -> torch.Tensor:
    """Unreduced BCE; existing train_epoch applies and normalizes valid_mask."""
    if logits.shape != target.shape or not logits.is_floating_point() or \
            not target.is_floating_point() or \
            not all(isinstance(w, (float, int)) and math.isfinite(float(w))
                    and float(w) > 0 for w in (water_weight, land_weight)) or \
            not bool(torch.all((target == 0) | (target == 1))):
        raise ValueError("Invalid binary targets, shapes or class weights")
    base = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return base * (water_weight * target + land_weight * (1 - target))


NEGATIVE_WEIGHT = 1.2


def negative_weighted_bce_12(logits: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
    """Per-pixel BCE: land=1.2, water=1; train_epoch applies valid_mask."""
    if logits.shape != target.shape or not logits.is_floating_point() or \
            not target.is_floating_point():
        raise ValueError("Logits/target shape or dtype differs")
    if not bool(torch.all((target == 0) | (target == 1)).item()):
        raise ValueError("Target must be binary before valid masking")
    pixels = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none")
    return pixels * (1.0 + (1.0 - target) * (NEGATIVE_WEIGHT - 1.0))
