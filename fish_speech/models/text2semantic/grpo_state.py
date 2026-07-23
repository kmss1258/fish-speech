from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn


class LoraAdapterStateError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class LoraAdapterSnapshot:
    tensors: Mapping[str, torch.Tensor]

    def restore(self, model: nn.Module) -> None:
        targets = _validated_lora_tensors(model, self.tensors)
        for name, tensor in self.tensors.items():
            target = targets[name]
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))


@dataclass(frozen=True, slots=True)
class _LoraMergeState:
    module: nn.Module
    training: bool
    merge_weights: bool
    merged: bool


def snapshot_lora_adapter(
    model: nn.Module,
    *,
    state_dict: Mapping[str, torch.Tensor] | None = None,
    prefix: str = "",
) -> LoraAdapterSnapshot:
    source = model.state_dict() if state_dict is None else state_dict
    source_tensors = _strip_lora_prefix(source, prefix)
    _validated_lora_tensors(model, source_tensors)
    return LoraAdapterSnapshot(
        MappingProxyType(
            {
                name: tensor.detach().cpu().clone()
                for name, tensor in source_tensors.items()
            }
        )
    )


@contextmanager
def frozen_lora_reference(
    model: nn.Module,
    reference: LoraAdapterSnapshot,
) -> Iterator[None]:
    policy = snapshot_lora_adapter(model)
    merge_states = _lora_merge_states(model)
    if any(state.merged for state in merge_states):
        raise LoraAdapterStateError("LoRA reference context requires unmerged weights")
    training_states = {name: module.training for name, module in model.named_modules()}
    requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    try:
        reference.restore(model)
        for state in merge_states:
            setattr(state.module, "merge_weights", False)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        yield
    finally:
        _unmerge_lora_modules(merge_states)
        for state in merge_states:
            setattr(state.module, "merge_weights", False)
        policy.restore(model)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(requires_grad[name])
        _restore_training_states(model, training_states)
        for state in merge_states:
            setattr(state.module, "merge_weights", state.merge_weights)
            setattr(state.module, "merged", state.merged)


def _strip_lora_prefix(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    stripped: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if "lora_" not in name:
            continue
        if prefix and name.startswith(prefix):
            stripped[name.removeprefix(prefix)] = tensor
        else:
            stripped[name] = tensor
    return stripped


def _validated_lora_tensors(
    model: nn.Module,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    expected = _model_lora_tensors(model)
    if not expected:
        raise LoraAdapterStateError("model has no LoRA adapter tensors")
    if set(tensors) != set(expected):
        missing = sorted(set(expected) - set(tensors))
        extra = sorted(set(tensors) - set(expected))
        raise LoraAdapterStateError(
            f"LoRA key topology mismatch; missing={missing}; extra={extra}"
        )
    for name, tensor in tensors.items():
        if tensor.shape != expected[name].shape:
            raise LoraAdapterStateError(
                f"LoRA tensor shape mismatch for {name}: "
                f"expected {tuple(expected[name].shape)}, got {tuple(tensor.shape)}"
            )
    return expected


def _model_lora_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor for name, tensor in model.state_dict().items() if "lora_" in name
    }


def _lora_merge_states(model: nn.Module) -> list[_LoraMergeState]:
    return [
        _LoraMergeState(
            module=module,
            training=module.training,
            merge_weights=bool(getattr(module, "merge_weights")),
            merged=bool(getattr(module, "merged")),
        )
        for module in model.modules()
        if hasattr(module, "merge_weights") and hasattr(module, "merged")
    ]


def _unmerge_lora_modules(states: list[_LoraMergeState]) -> None:
    for state in states:
        if bool(getattr(state.module, "merged")):
            setattr(state.module, "merge_weights", True)
            state.module.train(True)


def _restore_training_states(
    model: nn.Module,
    training_states: Mapping[str, bool],
) -> None:
    for name, module in model.named_modules():
        module.train(training_states[name])
