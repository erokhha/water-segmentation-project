from __future__ import annotations

import torch


def full_state_matches(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_step: int,
    learning_rate: float,
) -> bool:
    if not optimizer.state or len(optimizer.param_groups) != 1:
        return False
    if optimizer.param_groups[0]["lr"] != learning_rate:
        return False
    if {int(state["step"].item()) for state in optimizer.state.values()} != {
        expected_step
    }:
        return False
    if any(
        not bool(torch.isfinite(value).all())
        for value in model.state_dict().values()
        if value.is_floating_point()
    ):
        return False
    return not any(
        not bool(torch.isfinite(value).all())
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
