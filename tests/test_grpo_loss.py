import unittest
from dataclasses import replace

import loralib as lora
import torch

from fish_speech.models.text2semantic.grpo import (
    GRPOLossInputError,
    GRPOLossInputs,
    grpo_loss,
)


def _valid_inputs(
    rewards: torch.Tensor, slow: torch.Tensor, fast: torch.Tensor
) -> GRPOLossInputs:
    return GRPOLossInputs(
        rewards=rewards,
        current_slow_logprobs=slow,
        behavior_slow_logprobs=slow.detach().clone(),
        reference_slow_logprobs=slow.detach().clone(),
        slow_mask=torch.ones_like(slow, dtype=torch.bool),
        current_fast_logprobs=fast,
        behavior_fast_logprobs=fast.detach().clone(),
        reference_fast_logprobs=fast.detach().clone(),
        fast_mask=torch.ones_like(fast, dtype=torch.bool),
    )


class GRPOLossTest(unittest.TestCase):
    def test_centers_rewards_and_applies_masks_clipping_and_equal_branch_weight(
        self,
    ) -> None:
        # Given
        slow_ratio = torch.tensor([[[2.0, 0.5], [2.0, 0.5]]], dtype=torch.float32)
        slow_behavior = torch.full_like(slow_ratio, -2.0)
        slow_current = slow_behavior + torch.log(slow_ratio)
        fast_ratio = torch.tensor(
            [[[[0.5, 1.0], [1.0, 1.0]], [[0.5, 2.0], [1.0, 2.0]]]],
            dtype=torch.float32,
        )
        fast_behavior = torch.full_like(fast_ratio, -2.0)
        fast_current = fast_behavior + torch.log(fast_ratio)
        inputs = replace(
            _valid_inputs(torch.tensor([[1.0, 3.0]]), slow_current, fast_current),
            behavior_slow_logprobs=slow_behavior,
            reference_slow_logprobs=torch.full_like(slow_current, -2.3),
            slow_mask=torch.tensor([[[True, False], [True, True]]]),
            behavior_fast_logprobs=fast_behavior,
            reference_fast_logprobs=torch.full_like(fast_current, -1.6),
            fast_mask=torch.tensor(
                [[[[True, True], [False, False]], [[True, False], [True, True]]]]
            ),
        )

        # When
        loss = grpo_loss(inputs, kl_weight=0.0)

        # Then
        self.assertAlmostEqual(loss.item(), -0.04, places=6)

    def test_sampled_k3_is_computed_in_fp32(self) -> None:
        # Given
        slow_current = torch.tensor(
            [[[-1.75], [-2.5]]], dtype=torch.float16, requires_grad=True
        )
        slow_reference = torch.tensor(
            [[[-1.25], [-3.0]]], dtype=torch.float16, requires_grad=True
        )
        fast_current = torch.tensor(
            [[[[-2.0, -1.5]], [[-2.5, -1.0]]]],
            dtype=torch.float16,
            requires_grad=True,
        )
        fast_reference = torch.tensor(
            [[[[-1.75, -1.75]], [[-1.5, -2.0]]]],
            dtype=torch.float16,
            requires_grad=True,
        )
        inputs = replace(
            _valid_inputs(
                torch.tensor([[2.0, 2.0]], dtype=torch.float16),
                slow_current,
                fast_current,
            ),
            behavior_slow_logprobs=torch.zeros_like(slow_current),
            reference_slow_logprobs=slow_reference,
            behavior_fast_logprobs=torch.zeros_like(fast_current),
            reference_fast_logprobs=fast_reference,
        )
        slow_delta = slow_reference.detach().float() - slow_current.detach().float()
        fast_delta = fast_reference.detach().float() - fast_current.detach().float()
        expected = 0.35 * (
            (torch.expm1(slow_delta) - slow_delta).mean()
            + (torch.expm1(fast_delta) - fast_delta).mean()
        )

        # When
        loss = grpo_loss(inputs, kl_weight=0.7)

        # Then
        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        slow_gradient = slow_current.grad
        fast_gradient = fast_current.grad
        assert slow_gradient is not None and fast_gradient is not None
        self.assertGreater(torch.count_nonzero(slow_gradient).item(), 0)
        self.assertGreater(torch.count_nonzero(fast_gradient).item(), 0)
        self.assertIsNone(slow_reference.grad)
        self.assertIsNone(fast_reference.grad)

    def test_equal_reward_group_has_zero_policy_gradient(self) -> None:
        # Given
        slow_current = torch.tensor(
            [[[-1.2], [-0.6]], [[-0.9], [-1.3]]], requires_grad=True
        )
        fast_current = torch.tensor(
            [[[[-1.1]], [[-0.8]]], [[[-0.7]], [[-1.4]]]], requires_grad=True
        )
        inputs = replace(
            _valid_inputs(
                torch.tensor([[1.0, 1.0], [5.0, 5.0]]),
                slow_current,
                fast_current,
            ),
            behavior_slow_logprobs=torch.zeros_like(slow_current),
            reference_slow_logprobs=slow_current.detach() + 0.5,
            behavior_fast_logprobs=torch.zeros_like(fast_current),
            reference_fast_logprobs=fast_current.detach() - 0.5,
        )

        # When
        loss = grpo_loss(inputs, kl_weight=0.0)
        loss.backward()

        # Then
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(loss.item(), 0.0)
        slow_gradient = slow_current.grad
        fast_gradient = fast_current.grad
        assert slow_gradient is not None
        assert fast_gradient is not None
        torch.testing.assert_close(slow_gradient, torch.zeros_like(slow_current))
        torch.testing.assert_close(fast_gradient, torch.zeros_like(fast_current))

    def test_adamw_step_updates_lora_tensors_only(self) -> None:
        # Given
        torch.manual_seed(0)
        layer = lora.Linear(2, 3, r=2, lora_alpha=4, bias=False)
        lora.mark_only_lora_as_trainable(layer, bias="none")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in layer.parameters() if parameter.requires_grad],
            lr=0.1,
            weight_decay=0.0,
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in layer.named_parameters()
        }
        logprobs = torch.log_softmax(
            layer(torch.tensor([[[1.0, -1.0], [-0.5, 2.0]]])), dim=-1
        )
        current_slow = logprobs[..., :1]
        current_fast = logprobs[..., 1:].unsqueeze(-2)
        inputs = _valid_inputs(torch.tensor([[0.0, 2.0]]), current_slow, current_fast)
        frozen_values = (
            inputs.behavior_slow_logprobs,
            inputs.behavior_fast_logprobs,
            inputs.reference_slow_logprobs,
            inputs.reference_fast_logprobs,
        )
        snapshots = tuple(value.clone() for value in frozen_values)

        # When
        optimizer.zero_grad()
        loss = grpo_loss(inputs, kl_weight=0.1)
        loss.backward()
        optimizer.step()

        # Then
        lora_changed = any(
            not torch.equal(before[name], parameter)
            for name, parameter in layer.named_parameters()
            if "lora_" in name
        )
        base_unchanged = all(
            torch.equal(before[name], parameter)
            for name, parameter in layer.named_parameters()
            if "lora_" not in name
        )
        self.assertTrue(lora_changed)
        self.assertTrue(base_unchanged)
        for value, snapshot in zip(frozen_values, snapshots, strict=True):
            torch.testing.assert_close(value, snapshot)
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad)

    def test_rejects_empty_enabled_action_masks(self) -> None:
        # Given
        slow = torch.zeros((1, 2, 1))
        fast = torch.zeros((1, 2, 1, 1))
        inputs = _valid_inputs(torch.tensor([[0.0, 1.0]]), slow, fast)
        slow_empty = replace(inputs, slow_mask=torch.zeros_like(slow, dtype=torch.bool))
        fast_empty = replace(inputs, fast_mask=torch.zeros_like(fast, dtype=torch.bool))

        # When / Then
        with self.assertRaisesRegex(ValueError, r"slow_mask.*at least one"):
            grpo_loss(slow_empty, kl_weight=0.0)
        with self.assertRaisesRegex(ValueError, r"fast_mask.*at least one"):
            grpo_loss(fast_empty, kl_weight=0.0)

    def test_masked_nonfinite_logprobs_do_not_affect_loss_or_gradients(
        self,
    ) -> None:
        # Given
        slow_mask = torch.tensor([[[True, False, False], [True, True, False]]])
        fast_mask = torch.tensor(
            [[[[True, False], [True, False]], [[True, True], [False, True]]]]
        )
        nonfinite = torch.tensor([torch.nan, torch.inf, -torch.inf])
        slow_current = torch.full(slow_mask.shape, -1.0)
        slow_behavior = slow_current.clone()
        slow_reference = slow_current.clone()
        fast_current = torch.full(fast_mask.shape, -1.0)
        fast_behavior = fast_current.clone()
        fast_reference = fast_current.clone()
        slow_current[~slow_mask] = nonfinite
        slow_behavior[~slow_mask] = nonfinite.roll(1)
        slow_reference[~slow_mask] = nonfinite.roll(2)
        fast_current[~fast_mask] = nonfinite
        fast_behavior[~fast_mask] = nonfinite.roll(1)
        fast_reference[~fast_mask] = nonfinite.roll(2)
        slow_current.requires_grad_()
        fast_current.requires_grad_()
        inputs = replace(
            _valid_inputs(torch.tensor([[0.0, 2.0]]), slow_current, fast_current),
            behavior_slow_logprobs=slow_behavior,
            reference_slow_logprobs=slow_reference,
            slow_mask=slow_mask,
            behavior_fast_logprobs=fast_behavior,
            reference_fast_logprobs=fast_reference,
            fast_mask=fast_mask,
        )

        # When
        loss = grpo_loss(inputs, kl_weight=0.3)
        loss.backward()

        # Then
        slow_gradient = slow_current.grad
        fast_gradient = fast_current.grad
        assert slow_gradient is not None
        assert fast_gradient is not None
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(slow_gradient).all().item())
        self.assertTrue(torch.isfinite(fast_gradient).all().item())
        self.assertEqual(
            torch.count_nonzero(slow_gradient.masked_select(~slow_mask)).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(fast_gradient.masked_select(~fast_mask)).item(), 0
        )

    def test_rejects_broadcastable_mask_and_reward_group_shape_mismatches(
        self,
    ) -> None:
        # Given
        inputs = _valid_inputs(
            torch.tensor([[0.0, 2.0]]),
            torch.full((1, 2, 3), -1.0),
            torch.full((1, 2, 2, 2), -1.0),
        )
        broadcastable_slow_mask = replace(
            inputs,
            slow_mask=torch.ones((1, 2, 1), dtype=torch.bool),
        )
        mismatched_rewards = replace(inputs, rewards=torch.zeros((1, 3)))

        # When / Then
        with self.assertRaisesRegex(
            GRPOLossInputError,
            r"slow_mask.*shape.*current_slow_logprobs",
        ):
            grpo_loss(broadcastable_slow_mask, kl_weight=0.0)
        with self.assertRaisesRegex(
            GRPOLossInputError,
            r"rewards.*shape.*\[B, G\]",
        ):
            grpo_loss(mismatched_rewards, kl_weight=0.0)


if __name__ == "__main__":
    unittest.main()
