import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


class GRPOConfigTest(unittest.TestCase):
    def test_grpo_config_composes_train_entrypoint_and_grpo_lora(self) -> None:
        # Given
        config_dir = Path("fish_speech/configs").resolve()

        # When
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            cfg = compose(
                config_name="text2semantic_grpo",
                overrides=[
                    "trainer.accelerator=cpu",
                    "trainer.devices=1",
                    "trainer.strategy=auto",
                    "trainer.precision=32-true",
                    "data.num_workers=0",
                    "sft_adapter_ckpt_sha256=431bfb6c7953833702d718c7bc99403cf2bcbff9fc5b1d5a6ef53f48c46a4035",
                    "provenance_manifest_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                ],
            )
            lora_cfg = compose(config_name="lora/r_8_alpha_16_grpo")

        # Then
        self.assertEqual(
            cfg.model._target_,
            "fish_speech.models.text2semantic.grpo_module.TextToSemanticGRPO",
        )
        self.assertEqual(cfg.data.batch_size, 1)
        self.assertEqual(cfg.model.group_size, 2)
        self.assertEqual(cfg.trainer.log_every_n_steps, 1)
        self.assertEqual(cfg.tokenizer.model_path, "checkpoints/s2-pro")
        self.assertEqual(cfg.model.eos_token_id, 151645)
        self.assertEqual(
            cfg.model.sft_adapter_ckpt_sha256,
            "431bfb6c7953833702d718c7bc99403cf2bcbff9fc5b1d5a6ef53f48c46a4035",
        )
        self.assertEqual(cfg.model.provenance_manifest_sha256, "a" * 64)
        self.assertFalse(cfg.model.allow_unvalidated_reward)
        self.assertEqual(lora_cfg.lora.lora_dropout, 0)
        self.assertEqual(lora_cfg.lora.r, 8)
        self.assertEqual(lora_cfg.lora.lora_alpha, 16)
        self.assertEqual(
            instantiate(lora_cfg.lora).target_modules,
            ["attention", "mlp", "embeddings", "output"],
        )


if __name__ == "__main__":
    unittest.main()
