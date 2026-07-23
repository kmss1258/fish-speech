from collections.abc import Callable, Iterable, Mapping, MutableMapping
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from fish_speech.models.text2semantic.grpo import GRPOLossInputs, grpo_loss
from fish_speech.models.text2semantic.grpo_checkpoint import (
    adapter_sha256,
    is_adapter_only_state,
    load_sft_adapter,
    resume_metadata,
    restore_adapter_only_state,
    validate_resume_metadata,
)
from fish_speech.models.text2semantic.grpo_metrics import (
    GRPOMetricsError,
    GRPOStepStats,
    centered_no_std_rewards,
    log_grpo_step,
    reject_empty_actions,
)
from fish_speech.models.text2semantic.grpo_rollout import (
    GRPORolloutConfig,
    GRPORolloutRecord,
    rollout_dual_ar,
    score_dual_ar_rollout,
)
from fish_speech.models.text2semantic.grpo_state import (
    frozen_lora_reference,
    snapshot_lora_adapter,
)
from fish_speech.models.text2semantic.llama import DualARTransformer

Scalar = str | int | float | bool
MetadataValue = Scalar | list[int] | list[str] | dict[str, Scalar | list[str]]
CheckpointValue = Tensor | dict[str, Tensor] | dict[str, MetadataValue]


class GRPOModuleInputError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class RewardGateError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("no validated Korean GRPO reward scorer is configured")


class TextToSemanticGRPO(L.LightningModule):
    def __init__(
        self,
        model: DualARTransformer,
        optimizer: Callable[[Iterable[nn.Parameter]], Optimizer],
        lr_scheduler: Callable[[Optimizer], LRScheduler],
        sft_adapter_ckpt_path: str,
        sft_adapter_ckpt_sha256: str,
        provenance_manifest_sha256: str,
        base_revision: str,
        group_size: int,
        max_new_tokens: int,
        eos_token_id: int,
        rollout_seed: int,
        kl_weight: float,
        mechanics_smoke: bool = False,
        allow_unvalidated_reward: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0,
        lora_targets: list[str] | None = None,
    ) -> None:
        super().__init__()
        if group_size < 2:
            raise GRPOModuleInputError("group_size must be at least 2")
        if max_new_tokens < 1:
            raise GRPOModuleInputError("max_new_tokens must be at least 1")
        self.model = model
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler
        self.sft_adapter_ckpt_sha256 = sft_adapter_ckpt_sha256
        self.provenance_manifest_sha256 = provenance_manifest_sha256
        self.base_revision = base_revision
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.rollout_seed = rollout_seed
        self.kl_weight = kl_weight
        self.mechanics_smoke = mechanics_smoke
        self.allow_unvalidated_reward = allow_unvalidated_reward
        self.adapter_config: dict[str, Scalar | list[str]] = {
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": lora_targets
            or ["attention", "mlp", "embeddings", "output"],
        }
        loaded = load_sft_adapter(
            self.model,
            Path(sft_adapter_ckpt_path),
            sft_adapter_ckpt_sha256,
            GRPOModuleInputError,
        )
        loaded.restore(self.model)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_("lora_" in name)
        self.reference_adapter = snapshot_lora_adapter(self.model)
        self.reference_adapter_sha256 = adapter_sha256(self.reference_adapter)
        self.last_eos: list[int] = []
        self.last_truncated: list[int] = []
        self.last_lengths: list[int] = []
        self.last_step_metrics: dict[str, float] = {}

    def build_rollout_prompt(self, batch: Mapping[str, Tensor]) -> Tensor:
        inputs = batch["inputs"]
        labels = batch["labels"]
        if inputs.shape[0] != 1:
            raise GRPOModuleInputError("GRPO rollout requires batch size 1")
        semantic = (labels[:, 0] >= self.model.config.semantic_begin_id) & (
            labels[:, 0] <= self.model.config.semantic_end_id
        )
        positions = semantic[0].nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            raise GRPOModuleInputError("batch has no semantic target in labels[:, 0]")
        first_index = int(positions[0].item())
        return inputs[:, :, : first_index + 1]

    def training_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Tensor:
        self._check_reward_gate()
        self.model.train()
        behavior = snapshot_lora_adapter(self.model)
        prompt = self.build_rollout_prompt(batch)
        record = rollout_dual_ar(self.model, prompt, self._rollout_config(batch_idx))
        self.last_eos = [int(value) for value in record.eos.flatten().tolist()]
        self.last_truncated = [
            int(value) for value in record.truncated.flatten().tolist()
        ]
        self.last_lengths = [int(value) for value in record.lengths.flatten().tolist()]
        behavior.restore(self.model)
        _reject_empty_actions(record)
        with frozen_lora_reference(self.model, self.reference_adapter), torch.no_grad():
            reference = score_dual_ar_rollout(self.model, record)
        behavior.restore(self.model)
        self.model.train()
        current = score_dual_ar_rollout(self.model, record)
        rewards = self._rewards(record)
        inputs = GRPOLossInputs(
            rewards=rewards,
            current_slow_logprobs=current.slow_logprobs,
            behavior_slow_logprobs=record.behavior_slow_logprobs,
            reference_slow_logprobs=reference.slow_logprobs,
            slow_mask=record.slow_mask,
            current_fast_logprobs=current.fast_logprobs,
            behavior_fast_logprobs=record.behavior_fast_logprobs,
            reference_fast_logprobs=reference.fast_logprobs,
            fast_mask=record.fast_mask,
        )
        loss = grpo_loss(inputs, kl_weight=self.kl_weight)
        stats = GRPOStepStats(
            rewards=rewards,
            slow_active=record.slow_mask.sum(),
            fast_active=record.fast_mask.sum(),
            slow_ratio=torch.exp(
                current.slow_logprobs.detach() - record.behavior_slow_logprobs
            )
            .masked_select(record.slow_mask)
            .mean(),
            fast_ratio=torch.exp(
                current.fast_logprobs.detach() - record.behavior_fast_logprobs
            )
            .masked_select(record.fast_mask)
            .mean(),
            slow_kl=(reference.slow_logprobs - current.slow_logprobs.detach())
            .masked_select(record.slow_mask)
            .mean(),
            fast_kl=(reference.fast_logprobs - current.fast_logprobs.detach())
            .masked_select(record.fast_mask)
            .mean(),
        )
        log_grpo_step(self, loss, record, stats)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        parameters = [
            parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        optimizer = self.optimizer_builder(parameters)
        scheduler = self.lr_scheduler_builder(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        if is_adapter_only_state(state_dict):
            return restore_adapter_only_state(self.model, state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def on_save_checkpoint(
        self, checkpoint: MutableMapping[str, CheckpointValue]
    ) -> None:
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            return
        for name in list(state_dict):
            if "lora_" not in name:
                state_dict.pop(name)
        checkpoint["grpo_metadata"] = self._metadata()

    def on_load_checkpoint(self, checkpoint: Mapping[str, CheckpointValue]) -> None:
        metadata = checkpoint.get("grpo_metadata")
        if not isinstance(metadata, dict):
            raise GRPOModuleInputError("GRPO checkpoint is missing grpo_metadata")
        validate_resume_metadata(
            metadata, self._resume_metadata(), GRPOModuleInputError
        )

    def _rollout_config(self, batch_idx: int) -> GRPORolloutConfig:
        return GRPORolloutConfig(
            self.group_size,
            self.max_new_tokens,
            self.eos_token_id,
            self.rollout_seed + batch_idx * self.group_size,
        )

    def _rewards(self, record: GRPORolloutRecord) -> Tensor:
        return centered_no_std_rewards(self.group_size, record.full_sequence.device)

    def _check_reward_gate(self) -> None:
        if not (self.mechanics_smoke and self.allow_unvalidated_reward):
            raise RewardGateError()

    def _metadata(self) -> dict[str, MetadataValue]:
        return {
            "base_revision": self.base_revision,
            "reference_adapter_sha256": self.reference_adapter_sha256,
            "sft_adapter_ckpt_sha256": self.sft_adapter_ckpt_sha256,
            "provenance_manifest_sha256": self.provenance_manifest_sha256,
            "adapter_config": self.adapter_config,
            "reward_mode": "mechanics_smoke" if self.mechanics_smoke else "fail_closed",
            "group_size": self.group_size,
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.eos_token_id,
            "rollout_seed": self.rollout_seed,
            "kl_weight": self.kl_weight,
            "last_eos": self.last_eos,
            "last_truncated": self.last_truncated,
            "last_lengths": self.last_lengths,
            **self.last_step_metrics,
        }

    def _resume_metadata(self) -> dict[str, MetadataValue]:
        return resume_metadata(self._metadata())


def _reject_empty_actions(record: GRPORolloutRecord) -> None:
    try:
        reject_empty_actions(record)
    except GRPOMetricsError as exc:
        raise GRPOModuleInputError(str(exc)) from exc
