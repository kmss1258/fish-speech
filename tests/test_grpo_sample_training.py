import unittest

import torch

from fish_speech.models.text2semantic.grpo import GRPOLossInputs, grpo_loss
from fish_speech.models.text2semantic.llama import (
    DualARModelArgs,
    DualARTransformer,
)
from fish_speech.models.text2semantic.lora import LoraConfig, setup_lora


class GRPOSampleTrainingTest(unittest.TestCase):
    def test_shifted_sample_updates_only_slow_and_fast_mlp_lora(self) -> None:
        # Given
        torch.manual_seed(7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = DualARModelArgs(
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
            fast_dim=16,
            fast_n_head=2,
            fast_n_local_heads=2,
            fast_head_dim=8,
            fast_intermediate_size=32,
        )
        model = DualARTransformer(config)
        setup_lora(
            model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0,
                target_modules=["mlp"],
            ),
        )
        model.to(device)
        model.train()
        inputs = torch.tensor(
            [
                [[1, 2, 10, 13], [0, 0, 2, 5], [0, 0, 1, 3], [0, 0, 4, 6]],
                [[1, 2, 9, 14], [0, 0, 1, 6], [0, 0, 2, 4], [0, 0, 3, 7]],
            ],
            device=device,
        )
        labels = torch.tensor(
            [
                [
                    [-100, 10, 13, 3],
                    [-100, 2, 5, -100],
                    [-100, 1, 3, -100],
                    [-100, 4, 6, -100],
                ],
                [
                    [-100, 9, 14, 3],
                    [-100, 1, 6, -100],
                    [-100, 2, 4, -100],
                    [-100, 3, 7, -100],
                ],
            ],
            device=device,
        )
        key_padding_mask = torch.zeros((2, 4), dtype=torch.bool, device=device)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.1,
            weight_decay=0.0,
        )
        parameter_snapshots = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        # When
        outputs = model(inputs, labels=labels, key_padding_mask=key_padding_mask)
        token_targets = labels[:, 0]
        semantic_mask = (token_targets >= 8) & (token_targets <= 15)
        safe_token_targets = token_targets.masked_fill(~semantic_mask, 0)
        current_slow = (
            outputs.token_logits.log_softmax(dim=-1)
            .gather(-1, safe_token_targets.unsqueeze(-1))
            .squeeze(-1)
            .reshape(1, 2, 4)
        )
        slow_mask = semantic_mask.reshape(1, 2, 4)

        fast_targets = labels[:, 1:].permute(0, 2, 1)[semantic_mask]
        selected_fast = (
            outputs.codebook_logits.log_softmax(dim=-1)
            .gather(-1, fast_targets.unsqueeze(-1))
            .squeeze(-1)
        )
        flat_fast_mask = semantic_mask.unsqueeze(-1).expand(-1, -1, 2)
        current_fast = (
            torch.zeros((2, 4, 2), device=device)
            .masked_scatter(flat_fast_mask, selected_fast[:, 1:])
            .reshape(1, 2, 4, 2)
        )
        fast_mask = flat_fast_mask.reshape(1, 2, 4, 2)
        optimizer.zero_grad(set_to_none=True)
        loss = grpo_loss(
            GRPOLossInputs(
                rewards=torch.tensor([[0.0, 2.0]], device=device),
                current_slow_logprobs=current_slow,
                behavior_slow_logprobs=current_slow.detach().clone(),
                reference_slow_logprobs=current_slow.detach().clone(),
                slow_mask=slow_mask,
                current_fast_logprobs=current_fast,
                behavior_fast_logprobs=current_fast.detach().clone(),
                reference_fast_logprobs=current_fast.detach().clone(),
                fast_mask=fast_mask,
            ),
            kl_weight=0.1,
        )
        loss.backward()
        optimizer.step()

        # Then
        self.assertEqual(semantic_mask.sum().item(), 4)
        self.assertEqual(tuple(outputs.token_logits.shape), (2, 4, 16))
        self.assertEqual(tuple(outputs.codebook_logits.shape), (4, 3, 8))
        self.assertEqual(tuple(current_slow.shape), (1, 2, 4))
        self.assertEqual(tuple(current_fast.shape), (1, 2, 4, 2))
        torch.testing.assert_close(fast_targets[:, 0], token_targets[semantic_mask] - 8)
        torch.testing.assert_close(
            current_fast[fast_mask], selected_fast[:, 1:].reshape(-1)
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss).item())

        changed = {
            name
            for name, parameter in model.named_parameters()
            if not torch.equal(parameter_snapshots[name], parameter.detach())
        }
        self.assertTrue(
            any(
                name.startswith("layers.") and ".feed_forward." in name
                for name in changed
            )
        )
        self.assertTrue(
            any(
                name.startswith("fast_layers.") and ".feed_forward." in name
                for name in changed
            )
        )
        self.assertTrue(all("lora_" in name for name in changed))
        for name, parameter in model.named_parameters():
            if "lora_" not in name:
                self.assertTrue(torch.equal(parameter_snapshots[name], parameter))
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
