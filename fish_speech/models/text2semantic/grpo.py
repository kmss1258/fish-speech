"""Fish-local GRPO loss baseline, not the exact Fish S2 objective."""

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

CLIP_EPSILON: Final[float] = 0.2


class GRPOLossInputError(ValueError):
    field: str
    reason: str

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field} {reason}")


@dataclass(frozen=True, slots=True)
class GRPOLossInputs:
    rewards: Tensor
    current_slow_logprobs: Tensor
    behavior_slow_logprobs: Tensor
    reference_slow_logprobs: Tensor
    slow_mask: Tensor
    current_fast_logprobs: Tensor
    behavior_fast_logprobs: Tensor
    reference_fast_logprobs: Tensor
    fast_mask: Tensor


def grpo_loss(inputs: GRPOLossInputs, *, kl_weight: float) -> Tensor:
    if inputs.rewards.ndim != 2:
        raise GRPOLossInputError("rewards", "must have rank 2 with shape [B, G]")
    if inputs.current_slow_logprobs.ndim != 3:
        raise GRPOLossInputError(
            "current_slow_logprobs", "must have rank 3 with shape [B, G, T]"
        )
    if inputs.current_fast_logprobs.ndim != 4:
        raise GRPOLossInputError(
            "current_fast_logprobs", "must have rank 4 with shape [B, G, T, K]"
        )
    if inputs.behavior_slow_logprobs.shape != inputs.current_slow_logprobs.shape:
        raise GRPOLossInputError(
            "behavior_slow_logprobs",
            "shape must exactly match current_slow_logprobs",
        )
    if inputs.reference_slow_logprobs.shape != inputs.current_slow_logprobs.shape:
        raise GRPOLossInputError(
            "reference_slow_logprobs",
            "shape must exactly match current_slow_logprobs",
        )
    if inputs.slow_mask.shape != inputs.current_slow_logprobs.shape:
        raise GRPOLossInputError(
            "slow_mask", "shape must exactly match current_slow_logprobs"
        )
    if inputs.behavior_fast_logprobs.shape != inputs.current_fast_logprobs.shape:
        raise GRPOLossInputError(
            "behavior_fast_logprobs",
            "shape must exactly match current_fast_logprobs",
        )
    if inputs.reference_fast_logprobs.shape != inputs.current_fast_logprobs.shape:
        raise GRPOLossInputError(
            "reference_fast_logprobs",
            "shape must exactly match current_fast_logprobs",
        )
    if inputs.fast_mask.shape != inputs.current_fast_logprobs.shape:
        raise GRPOLossInputError(
            "fast_mask", "shape must exactly match current_fast_logprobs"
        )
    if inputs.current_slow_logprobs.shape[:2] != inputs.rewards.shape:
        raise GRPOLossInputError(
            "rewards",
            "shape [B, G] must match current_slow_logprobs first two dimensions",
        )
    if inputs.current_fast_logprobs.shape[:2] != inputs.rewards.shape:
        raise GRPOLossInputError(
            "rewards",
            "shape [B, G] must match current_fast_logprobs first two dimensions",
        )
    if inputs.slow_mask.dtype != torch.bool:
        raise GRPOLossInputError("slow_mask", "must have dtype torch.bool")
    if inputs.fast_mask.dtype != torch.bool:
        raise GRPOLossInputError("fast_mask", "must have dtype torch.bool")
    if not inputs.slow_mask.any().item():
        raise GRPOLossInputError("slow_mask", "must contain at least one active action")
    if not inputs.fast_mask.any().item():
        raise GRPOLossInputError("fast_mask", "must contain at least one active action")

    with torch.autocast(
        device_type=inputs.current_slow_logprobs.device.type, enabled=False
    ):
        rewards = inputs.rewards.detach().float()
        advantages = rewards - rewards.mean(dim=1, keepdim=True)

        slow_current = inputs.current_slow_logprobs.float()[inputs.slow_mask]
        slow_behavior = inputs.behavior_slow_logprobs.detach().float()[inputs.slow_mask]
        slow_reference = inputs.reference_slow_logprobs.detach().float()[
            inputs.slow_mask
        ]
        slow_advantages = advantages.unsqueeze(-1).expand_as(
            inputs.current_slow_logprobs
        )[inputs.slow_mask]
        slow_ratio = torch.exp(slow_current - slow_behavior)
        slow_clipped_ratio = slow_ratio.clamp(
            min=1.0 - CLIP_EPSILON, max=1.0 + CLIP_EPSILON
        )
        slow_policy = -torch.minimum(
            slow_ratio * slow_advantages,
            slow_clipped_ratio * slow_advantages,
        )
        slow_delta = slow_reference - slow_current
        slow_k3 = torch.expm1(slow_delta) - slow_delta
        slow_tokens = slow_policy + kl_weight * slow_k3
        slow_loss = slow_tokens.mean(dtype=torch.float32)

        fast_current = inputs.current_fast_logprobs.float()[inputs.fast_mask]
        fast_behavior = inputs.behavior_fast_logprobs.detach().float()[inputs.fast_mask]
        fast_reference = inputs.reference_fast_logprobs.detach().float()[
            inputs.fast_mask
        ]
        fast_advantages = (
            advantages.unsqueeze(-1)
            .unsqueeze(-1)
            .expand_as(inputs.current_fast_logprobs)[inputs.fast_mask]
        )
        fast_ratio = torch.exp(fast_current - fast_behavior)
        fast_clipped_ratio = fast_ratio.clamp(
            min=1.0 - CLIP_EPSILON, max=1.0 + CLIP_EPSILON
        )
        fast_policy = -torch.minimum(
            fast_ratio * fast_advantages,
            fast_clipped_ratio * fast_advantages,
        )
        fast_delta = fast_reference - fast_current
        fast_k3 = torch.expm1(fast_delta) - fast_delta
        fast_tokens = fast_policy + kl_weight * fast_k3
        fast_loss = fast_tokens.mean(dtype=torch.float32)

        return (slow_loss + fast_loss) / 2
