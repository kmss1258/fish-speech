import unittest

import loralib as lora
import torch

from fish_speech.models.text2semantic.grpo_state import (
    LoraAdapterStateError,
    frozen_lora_reference,
    snapshot_lora_adapter,
)
from fish_speech.models.text2semantic.llama import DualARModelArgs, DualARTransformer
from fish_speech.models.text2semantic.lora import LoraConfig, setup_lora


class ReferenceContextTestError(RuntimeError):
    pass


def _lora_parameters(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
        if "lora_" in name
    }


def _lora_modules(module: torch.nn.Module) -> list[torch.nn.Module]:
    return [child for child in module.modules() if hasattr(child, "merge_weights")]


def _base_tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
        if "lora_" not in name
    }


class GRPOStateTest(unittest.TestCase):
    def test_snapshot_clones_cpu_tensors_when_optimizer_mutates_policy(self) -> None:
        # Given
        torch.manual_seed(1)
        layer = lora.Linear(3, 2, r=2, lora_alpha=4, bias=False)
        lora.mark_only_lora_as_trainable(layer, bias="none")
        snapshot = snapshot_lora_adapter(layer)
        saved = {name: tensor.clone() for name, tensor in snapshot.tensors.items()}
        optimizer = torch.optim.AdamW(
            [parameter for parameter in layer.parameters() if parameter.requires_grad],
            lr=0.5,
            weight_decay=0.0,
        )

        # When
        optimizer.zero_grad(set_to_none=True)
        layer(torch.tensor([[1.0, -2.0, 0.5]])).sum().backward()
        optimizer.step()

        # Then
        self.assertTrue(
            any(
                not torch.equal(saved[name], parameter.detach())
                for name, parameter in layer.named_parameters()
                if "lora_" in name
            )
        )
        for name, tensor in snapshot.tensors.items():
            self.assertEqual(tensor.device.type, "cpu")
            torch.testing.assert_close(tensor, saved[name])

    def test_reference_context_restores_policy_after_success_and_failure(self) -> None:
        # Given
        torch.manual_seed(2)
        layer = lora.Linear(3, 2, r=2, lora_alpha=4, bias=False)
        lora.mark_only_lora_as_trainable(layer, bias="none")
        layer.train()
        policy_snapshot = snapshot_lora_adapter(layer)
        with torch.no_grad():
            for name, parameter in layer.named_parameters():
                if "lora_" in name:
                    parameter.add_(0.25)
        reference_snapshot = snapshot_lora_adapter(layer)
        snapshot_lora_adapter(layer, state_dict=policy_snapshot.tensors).restore(layer)
        base_before = _base_tensors(layer)

        # When
        with frozen_lora_reference(layer, reference_snapshot):
            self.assertFalse(layer.training)
            self.assertTrue(
                all(not parameter.requires_grad for parameter in layer.parameters())
            )
            self.assertTrue(
                all(not child.merge_weights for child in _lora_modules(layer))
            )
            self.assertTrue(all(not child.merged for child in _lora_modules(layer)))
            with torch.no_grad():
                for name, parameter in layer.named_parameters():
                    if "lora_" in name:
                        torch.testing.assert_close(
                            parameter, reference_snapshot.tensors[name]
                        )
                        parameter.add_(1.0)

        # Then
        self.assertTrue(layer.training)
        self.assertTrue(all(child.merge_weights for child in _lora_modules(layer)))
        self.assertTrue(all(not child.merged for child in _lora_modules(layer)))
        for name, parameter in layer.named_parameters():
            if "lora_" in name:
                torch.testing.assert_close(parameter, policy_snapshot.tensors[name])
                self.assertTrue(parameter.requires_grad)
        for name, tensor in _base_tensors(layer).items():
            self.assertTrue(torch.equal(tensor, base_before[name]))

        # When / Then
        with self.assertRaisesRegex(ReferenceContextTestError, "boom"):
            with frozen_lora_reference(layer, reference_snapshot):
                raise ReferenceContextTestError("boom")
        for name, parameter in layer.named_parameters():
            if "lora_" in name:
                torch.testing.assert_close(parameter, policy_snapshot.tensors[name])
        for name, tensor in _base_tensors(layer).items():
            self.assertTrue(torch.equal(tensor, base_before[name]))

    def test_snapshot_rejects_missing_extra_and_wrong_shape_lora_keys(self) -> None:
        # Given
        layer = lora.Linear(3, 2, r=2, lora_alpha=4, bias=False)
        snapshot = snapshot_lora_adapter(layer)
        prefixed = {
            f"model.{name}": tensor for name, tensor in snapshot.tensors.items()
        }

        # When
        lightning_snapshot = snapshot_lora_adapter(
            layer,
            state_dict=prefixed,
            prefix="model.",
        )

        # Then
        self.assertEqual(set(lightning_snapshot.tensors), set(snapshot.tensors))
        missing = dict(snapshot.tensors)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(LoraAdapterStateError, "LoRA key topology"):
            snapshot_lora_adapter(layer, state_dict=missing)
        extra = dict(snapshot.tensors)
        extra["extra.lora_A"] = torch.zeros(1)
        with self.assertRaisesRegex(LoraAdapterStateError, "LoRA key topology"):
            snapshot_lora_adapter(layer, state_dict=extra)
        wrong_shape = dict(snapshot.tensors)
        first_key = next(iter(wrong_shape))
        wrong_shape[first_key] = torch.zeros(1)
        with self.assertRaisesRegex(LoraAdapterStateError, first_key):
            snapshot_lora_adapter(layer, state_dict=wrong_shape)

    def test_broad_r8_alpha16_fish_lora_topology_snapshots_with_prefix(self) -> None:
        # Given
        torch.manual_seed(3)
        model = DualARTransformer(
            DualARModelArgs(
                vocab_size=16,
                n_layer=1,
                n_head=2,
                dim=16,
                intermediate_size=32,
                n_local_heads=2,
                head_dim=8,
                max_seq_len=4,
                codebook_size=8,
                num_codebooks=3,
                semantic_begin_id=8,
                semantic_end_id=15,
                use_gradient_checkpointing=False,
                n_fast_layer=1,
            )
        )
        setup_lora(model, LoraConfig(r=8, lora_alpha=16))
        prefixed_state = {
            f"model.{name}": tensor for name, tensor in model.state_dict().items()
        }

        # When
        snapshot = snapshot_lora_adapter(
            model, state_dict=prefixed_state, prefix="model."
        )

        # Then
        self.assertEqual(
            set(snapshot.tensors), set(snapshot_lora_adapter(model).tensors)
        )
        self.assertTrue(
            any(name.startswith("embeddings.") for name in snapshot.tensors)
        )
        self.assertTrue(
            any(name.startswith("fast_layers.") for name in snapshot.tensors)
        )
        self.assertTrue(
            all(tensor.device.type == "cpu" for tensor in snapshot.tensors.values())
        )


if __name__ == "__main__":
    unittest.main()
