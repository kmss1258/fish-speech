import unittest

import torch

from fish_speech.models.text2semantic.grpo_rollout import (
    GRPORolloutConfig,
    rollout_dual_ar,
    score_dual_ar_rollout,
)
from fish_speech.models.text2semantic.llama import DualARModelArgs, DualARTransformer


def _model() -> DualARTransformer:
    torch.manual_seed(5)
    return DualARTransformer(
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


class GRPORolloutTest(unittest.TestCase):
    def test_rollout_records_grouped_shapes_and_masks(self) -> None:
        # Given
        model = _model()
        prompt = torch.tensor(
            [[[1, 2], [0, 0], [0, 0], [0, 0]], [[1, 4], [0, 0], [0, 0], [0, 0]]]
        )

        # When
        record = rollout_dual_ar(
            model,
            prompt,
            GRPORolloutConfig(group_size=2, max_new_tokens=3, eos_token_id=3, seed=11),
        )

        # Then
        self.assertEqual(tuple(record.generated_frames.shape), (2, 2, 3, 3))
        self.assertEqual(tuple(record.behavior_slow_logprobs.shape), (2, 2, 3))
        self.assertEqual(tuple(record.behavior_fast_logprobs.shape), (2, 2, 3, 2))
        self.assertEqual(record.slow_mask.dtype, torch.bool)
        self.assertEqual(record.fast_mask.dtype, torch.bool)
        torch.testing.assert_close(
            record.generated_frames[..., 0][record.slow_mask] + 8,
            record.slow_tokens[record.slow_mask],
        )
        self.assertFalse(
            record.fast_mask[..., 0].logical_xor(record.fast_mask[..., 1]).any()
        )
        self.assertEqual(record.seeds, (11, 12, 13, 14))

    def test_score_matches_behavior_under_unchanged_policy(self) -> None:
        # Given
        model = _model()
        prompt = torch.tensor([[[1, 2], [0, 0], [0, 0], [0, 0]]])
        record = rollout_dual_ar(
            model,
            prompt,
            GRPORolloutConfig(group_size=2, max_new_tokens=3, eos_token_id=3, seed=17),
        )

        # When
        scores = score_dual_ar_rollout(model, record)

        # Then
        self.assertEqual(tuple(scores.slow_logprobs.shape), (1, 2, 3))
        self.assertEqual(tuple(scores.fast_logprobs.shape), (1, 2, 3, 2))
        torch.testing.assert_close(scores.slow_logprobs, record.behavior_slow_logprobs)
        torch.testing.assert_close(scores.fast_logprobs, record.behavior_fast_logprobs)

    def test_eos_truncation_and_empty_completion_masks(self) -> None:
        # Given
        model = _model()
        prompt = torch.tensor([[[1], [0], [0], [0]]])

        # When
        truncated = rollout_dual_ar(
            model,
            prompt,
            GRPORolloutConfig(group_size=1, max_new_tokens=1, eos_token_id=3, seed=23),
        )
        empty_eos_id = int(truncated.slow_tokens[0, 0, 0].item())
        empty = rollout_dual_ar(
            model,
            prompt,
            GRPORolloutConfig(
                group_size=1, max_new_tokens=1, eos_token_id=empty_eos_id, seed=23
            ),
        )

        # Then
        self.assertTrue(truncated.truncated[0, 0].item())
        self.assertEqual(truncated.lengths[0, 0].item(), 1)
        self.assertTrue(empty.eos[0, 0].item())
        self.assertEqual(empty.lengths[0, 0].item(), 0)
        self.assertFalse(empty.slow_mask.any().item())
        self.assertFalse(empty.fast_mask.any().item())


if __name__ == "__main__":
    unittest.main()
