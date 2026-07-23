import hashlib
import tempfile
import unittest
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
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


def _adapter_checkpoint(path: Path) -> None:
    model = _tiny_model()
    torch.save(
        {"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, path
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module(path: Path, **overrides):
    from fish_speech.models.text2semantic.grpo_module import TextToSemanticGRPO

    args = {
        "model": _tiny_model(),
        "optimizer": torch.optim.AdamW,
        "lr_scheduler": lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda _: 1.0
        ),
        "sft_adapter_ckpt_path": str(path),
        "sft_adapter_ckpt_sha256": _file_sha256(path),
        "provenance_manifest_sha256": "a" * 64,
        "base_revision": "tiny-base",
        "group_size": 2,
        "max_new_tokens": 1,
        "eos_token_id": -1,
        "rollout_seed": 41,
        "kl_weight": 0.1,
        "mechanics_smoke": True,
        "allow_unvalidated_reward": True,
    }
    args.update(overrides)
    return TextToSemanticGRPO(**args)


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


class GRPOCheckpointTest(unittest.TestCase):
    def test_rejects_sft_checkpoint_hash_mismatch_before_load(self) -> None:
        # Given
        from fish_speech.models.text2semantic.grpo_module import GRPOModuleInputError

        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "adapter.ckpt"
            adapter.write_bytes(b"not a trusted checkpoint")

            # When / Then
            with self.assertRaises(GRPOModuleInputError):
                _module(adapter, sft_adapter_ckpt_sha256="bad")

    def test_lightning_resume_loads_adapter_only_state_and_optimizer(self) -> None:
        # Given
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter.ckpt"
            _adapter_checkpoint(adapter)
            loader = DataLoader(_OneBatch(), batch_size=None)
            checkpoint = ModelCheckpoint(
                dirpath=root / "ckpts", every_n_train_steps=1, save_last=True
            )
            first = _module(adapter)
            first_trainer = L.Trainer(
                accelerator="cpu",
                devices=1,
                max_steps=1,
                logger=False,
                callbacks=[checkpoint],
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            first_trainer.fit(first, train_dataloaders=loader)
            fresh = _module(adapter)
            before = {
                name: parameter.detach().clone()
                for name, parameter in fresh.named_parameters()
                if "lora_" in name
            }

            # When
            second_trainer = L.Trainer(
                accelerator="cpu",
                devices=1,
                max_steps=2,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            second_trainer.fit(
                fresh, train_dataloaders=loader, ckpt_path=checkpoint.last_model_path
            )

            # Then
            saved = torch.load(
                checkpoint.last_model_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(second_trainer.global_step, 2)
            self.assertTrue(all("lora_" in name for name in saved["state_dict"]))
            self.assertEqual(
                saved["grpo_metadata"]["sft_adapter_ckpt_sha256"], _file_sha256(adapter)
            )
            self.assertTrue(
                any(
                    not torch.equal(before[name], parameter)
                    for name, parameter in fresh.named_parameters()
                    if "lora_" in name
                )
            )
            steps = [
                state["step"].item()
                for state in second_trainer.optimizers[0].state.values()
                if "step" in state
            ]
            self.assertGreaterEqual(max(steps), 2)

    def test_incomplete_full_state_dict_still_fails_strict_load(self) -> None:
        # Given
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "adapter.ckpt"
            _adapter_checkpoint(adapter)
            module = _module(adapter)
            state = module.state_dict()
            state.pop(next(name for name in state if "lora_" not in name))

            # When / Then
            with self.assertRaises(RuntimeError):
                module.load_state_dict(state)

    def test_resume_rejects_metadata_mismatches(self) -> None:
        # Given
        from fish_speech.models.text2semantic.grpo_module import GRPOModuleInputError

        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "adapter.ckpt"
            _adapter_checkpoint(adapter)
            module = _module(adapter)
            checkpoint = {"grpo_metadata": module._metadata()}
            mismatches = {
                "base_revision": "other-base",
                "reference_adapter_sha256": "bad",
                "sft_adapter_ckpt_sha256": "bad",
                "provenance_manifest_sha256": "bad",
                "adapter_config": {
                    "r": 2,
                    "lora_alpha": 99,
                    "lora_dropout": 0,
                    "target_modules": ["mlp"],
                },
                "reward_mode": "fail_closed",
                "group_size": 3,
                "max_new_tokens": 2,
                "eos_token_id": 0,
                "rollout_seed": 42,
                "kl_weight": 0.2,
            }

            # When / Then
            for key, value in mismatches.items():
                bad = {
                    "grpo_metadata": dict(checkpoint["grpo_metadata"], **{key: value})
                }
                with self.subTest(key=key):
                    with self.assertRaises(GRPOModuleInputError):
                        module.on_load_checkpoint(bad)


if __name__ == "__main__":
    unittest.main()
