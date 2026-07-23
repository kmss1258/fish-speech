import hashlib
import tempfile
import unittest
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from fish_speech.models.text2semantic.llama import DualARModelArgs, DualARTransformer
from fish_speech.models.text2semantic.lora import LoraConfig, setup_lora


def _tiny_model() -> DualARTransformer:
    torch.manual_seed(31)
    model = DualARTransformer(
        DualARModelArgs(
            vocab_size=16,
            n_layer=1,
            n_head=2,
            dim=16,
            intermediate_size=32,
            n_local_heads=2,
            head_dim=8,
            max_seq_len=8,
            codebook_size=8,
            num_codebooks=3,
            semantic_begin_id=8,
            semantic_end_id=15,
            use_gradient_checkpointing=False,
            n_fast_layer=1,
            fast_dim=16,
            fast_n_head=2,
            fast_n_local_heads=2,
            fast_head_dim=8,
            fast_intermediate_size=32,
        )
    )
    setup_lora(
        model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules=["mlp"])
    )
    return model


def _adapter_checkpoint(model: DualARTransformer, path: Path) -> None:
    torch.save(
        {"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, path
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module(path: Path):
    from fish_speech.models.text2semantic.grpo_module import TextToSemanticGRPO

    return TextToSemanticGRPO(
        model=_tiny_model(),
        optimizer=torch.optim.AdamW,
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda _: 1.0
        ),
        sft_adapter_ckpt_path=str(path),
        sft_adapter_ckpt_sha256=_file_sha256(path),
        provenance_manifest_sha256="a" * 64,
        base_revision="tiny-base",
        group_size=2,
        max_new_tokens=1,
        eos_token_id=-1,
        rollout_seed=41,
        kl_weight=0.1,
        mechanics_smoke=True,
        allow_unvalidated_reward=True,
    )


def _batch() -> dict[str, torch.Tensor]:
    inputs = torch.tensor([[[1, 2, 10], [0, 0, 2], [0, 0, 1], [0, 0, 4]]])
    labels = torch.tensor([[[-100, 10, 13], [-100, 2, 5], [-100, 1, 3], [-100, 4, 6]]])
    return {
        "inputs": inputs,
        "attention_masks": torch.zeros((1, 3), dtype=torch.bool),
        "labels": labels,
    }


class _OneBatch(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return _batch()


class GRPOModuleTest(unittest.TestCase):
    def test_one_lightning_step_updates_lora_only_and_checkpoints_metadata(
        self,
    ) -> None:
        # Given
        from fish_speech.models.text2semantic.grpo_module import TextToSemanticGRPO

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.ckpt"
            seeded = _tiny_model()
            _adapter_checkpoint(seeded, path)
            model = _tiny_model()
            module = TextToSemanticGRPO(
                model=model,
                optimizer=torch.optim.AdamW,
                lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda _: 1.0
                ),
                sft_adapter_ckpt_path=str(path),
                sft_adapter_ckpt_sha256=_file_sha256(path),
                provenance_manifest_sha256="a" * 64,
                base_revision="tiny-base",
                group_size=2,
                max_new_tokens=1,
                eos_token_id=-1,
                rollout_seed=41,
                kl_weight=0.1,
                mechanics_smoke=True,
                allow_unvalidated_reward=True,
            )
            before = {
                name: parameter.detach().clone()
                for name, parameter in module.named_parameters()
            }
            loader = DataLoader(_OneBatch(), batch_size=None)

            # When
            trainer = L.Trainer(
                accelerator="cpu",
                devices=1,
                max_steps=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            trainer.fit(module, train_dataloaders=loader)
            checkpoint = {"state_dict": module.state_dict()}
            module.on_save_checkpoint(checkpoint)

            # Then
            changed = {
                name
                for name, parameter in module.named_parameters()
                if not torch.equal(before[name], parameter)
            }
            self.assertTrue(
                any(
                    name.startswith("model.layers.") and ".feed_forward." in name
                    for name in changed
                )
            )
            self.assertTrue(
                any(
                    name.startswith("model.fast_layers.") and ".feed_forward." in name
                    for name in changed
                )
            )
            self.assertTrue(all("lora_" in name for name in changed))
            self.assertTrue(all("lora_" in name for name in checkpoint["state_dict"]))
            metadata = checkpoint["grpo_metadata"]
            self.assertEqual(metadata["base_revision"], "tiny-base")
            self.assertEqual(metadata["provenance_manifest_sha256"], "a" * 64)
            self.assertEqual(metadata["reward_mode"], "mechanics_smoke")
            self.assertEqual(metadata["group_size"], 2)
            self.assertIn("reference_adapter_sha256", metadata)
            self.assertEqual(metadata["last_reward_mean"], 0.5)
            self.assertEqual(metadata["last_reward_variance"], 0.25)
            for name in [
                "last_loss",
                "last_slow_active",
                "last_fast_active",
                "last_slow_ratio",
                "last_fast_ratio",
                "last_slow_kl_delta",
                "last_fast_kl_delta",
                "last_eos_rate",
                "last_truncation_rate",
            ]:
                self.assertTrue(torch.isfinite(torch.tensor(metadata[name])).item())

    def test_rejects_any_completion_with_empty_slow_or_fast_actions(self) -> None:
        # Given
        from fish_speech.models.text2semantic.grpo_module import (
            GRPOModuleInputError,
            _reject_empty_actions,
        )
        from fish_speech.models.text2semantic.grpo_rollout import GRPORolloutRecord

        slow_mask = torch.tensor([[[True, False], [False, False]]])
        fast_mask = slow_mask.unsqueeze(-1).expand(1, 2, 2, 2)
        record = GRPORolloutRecord(
            full_sequence=torch.zeros((1, 2, 4, 3), dtype=torch.long),
            prompt_length=1,
            eos_token_id=-1,
            generated_frames=torch.zeros((1, 2, 2, 3), dtype=torch.long),
            slow_tokens=torch.zeros((1, 2, 2), dtype=torch.long),
            behavior_slow_logprobs=torch.zeros((1, 2, 2)),
            behavior_fast_logprobs=torch.zeros((1, 2, 2, 2)),
            slow_mask=slow_mask,
            fast_mask=fast_mask,
            eos=torch.zeros((1, 2), dtype=torch.bool),
            truncated=torch.ones((1, 2), dtype=torch.bool),
            lengths=torch.tensor([[1, 0]]),
            seeds=(1, 2),
        )

        # When / Then
        with self.assertRaises(GRPOModuleInputError):
            _reject_empty_actions(record)

    def test_reward_gate_and_batch_contract_fail_closed(self) -> None:
        # Given
        from fish_speech.models.text2semantic.grpo_module import (
            GRPOModuleInputError,
            RewardGateError,
            TextToSemanticGRPO,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.ckpt"
            _adapter_checkpoint(_tiny_model(), path)
            module = TextToSemanticGRPO(
                model=_tiny_model(),
                optimizer=torch.optim.AdamW,
                lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda _: 1.0
                ),
                sft_adapter_ckpt_path=str(path),
                sft_adapter_ckpt_sha256=_file_sha256(path),
                provenance_manifest_sha256="a" * 64,
                base_revision="tiny-base",
                group_size=2,
                max_new_tokens=1,
                eos_token_id=-1,
                rollout_seed=41,
                kl_weight=0.1,
            )

            # When / Then
            with self.assertRaises(RewardGateError):
                module.training_step(_batch(), 0)
            bad = _batch()
            bad["labels"] = bad["labels"].masked_fill(bad["labels"] >= 8, -100)
            with self.assertRaises(GRPOModuleInputError):
                module.build_rollout_prompt(bad)


if __name__ == "__main__":
    unittest.main()
