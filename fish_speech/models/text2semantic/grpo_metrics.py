from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from fish_speech.models.text2semantic.grpo_checkpoint import scalar
from fish_speech.models.text2semantic.grpo_rollout import GRPORolloutRecord


class GRPOMetricsError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class _Logger(Protocol):
    last_step_metrics: dict[str, float]

    def log(self, name: str, value: Tensor, **kwargs: bool) -> None: ...


@dataclass(frozen=True, slots=True)
class GRPOStepStats:
    rewards: Tensor
    slow_active: Tensor
    fast_active: Tensor
    slow_ratio: Tensor
    fast_ratio: Tensor
    slow_kl: Tensor
    fast_kl: Tensor


def reject_empty_actions(record: GRPORolloutRecord) -> None:
    slow_complete = record.slow_mask.any(dim=2)
    fast_complete = record.fast_mask.flatten(start_dim=2).any(dim=2)
    if not slow_complete.all().item() or not fast_complete.all().item():
        raise GRPOMetricsError("rollout produced no active Slow/Fast actions")


def log_grpo_step(
    module: _Logger, loss: Tensor, record: GRPORolloutRecord, stats: GRPOStepStats
) -> None:
    values = {
        "train/loss": loss,
        "train/reward_mean": stats.rewards.mean(),
        "train/reward_variance": stats.rewards.var(unbiased=False),
        "train/slow_active": stats.slow_active.float(),
        "train/fast_active": stats.fast_active.float(),
        "train/slow_ratio": stats.slow_ratio,
        "train/fast_ratio": stats.fast_ratio,
        "train/slow_kl_delta": stats.slow_kl,
        "train/fast_kl_delta": stats.fast_kl,
        "train/eos_rate": record.eos.float().mean(),
        "train/truncation_rate": record.truncated.float().mean(),
    }
    module.last_step_metrics = {
        f"last_{name.removeprefix('train/')}": scalar(value)
        for name, value in values.items()
    }
    for name, value in values.items():
        module.log(
            name, value, on_step=True, logger=True, prog_bar=name == "train/loss"
        )


def centered_no_std_rewards(group_size: int, device: torch.device) -> Tensor:
    return torch.arange(group_size, device=device, dtype=torch.float32).unsqueeze(0)
