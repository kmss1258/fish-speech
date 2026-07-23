import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.modules.module import _IncompatibleKeys

from fish_speech.models.text2semantic.grpo_state import (
    LoraAdapterSnapshot,
    snapshot_lora_adapter,
)

Scalar = str | int | float | bool
MetadataValue = Scalar | list[int] | list[str] | dict[str, Scalar | list[str]]
Metadata = dict[str, MetadataValue]
RESUME_METADATA_KEYS = (
    "base_revision",
    "reference_adapter_sha256",
    "sft_adapter_ckpt_sha256",
    "provenance_manifest_sha256",
    "adapter_config",
    "reward_mode",
    "group_size",
    "max_new_tokens",
    "eos_token_id",
    "rollout_seed",
    "kl_weight",
)


def is_adapter_only_state(state_dict: Mapping[str, Tensor]) -> bool:
    return bool(state_dict) and all(
        name.startswith("model.") and "lora_" in name for name in state_dict
    )


def restore_adapter_only_state(
    model: nn.Module, state_dict: Mapping[str, Tensor]
) -> _IncompatibleKeys:
    snapshot = snapshot_lora_adapter(model, state_dict=state_dict, prefix="model.")
    snapshot.restore(model)
    return _IncompatibleKeys([], [])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sft_adapter(
    model: nn.Module,
    path: Path,
    expected_sha256: str,
    error: Callable[[str], Exception],
) -> LoraAdapterSnapshot:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise error("SFT adapter checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    if not isinstance(state, dict):
        raise error(
            "SFT adapter checkpoint must be a state_dict or Lightning checkpoint"
        )
    return snapshot_lora_adapter(model, state_dict=state, prefix="model.")


def validate_resume_metadata(
    metadata: Mapping[str, MetadataValue],
    expected: Mapping[str, MetadataValue],
    error: Callable[[str], Exception],
) -> None:
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise error(f"GRPO {name} mismatch")


def resume_metadata(metadata: Mapping[str, MetadataValue]) -> Metadata:
    return {name: metadata[name] for name in RESUME_METADATA_KEYS}


def adapter_sha256(snapshot: LoraAdapterSnapshot) -> str:
    digest = hashlib.sha256()
    for name in sorted(snapshot.tensors):
        tensor = snapshot.tensors[name].contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def scalar(value: Tensor) -> float:
    return float(value.detach().cpu().item())
