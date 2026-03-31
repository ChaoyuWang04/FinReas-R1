import unittest

from src.training.sft_trainer import SFTConfig, build_sft_command


class SFTTrainerTests(unittest.TestCase):
    def test_build_sft_command_includes_expected_openrlhf_flags(self) -> None:
        config = SFTConfig(
            save_path="experiments/sft",
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            dataset="data/processed/train_with_sys.jsonl",
            train_batch_size=128,
            micro_train_batch_size=1,
            max_epochs=1,
            max_len=12288,
            learning_rate=5e-6,
            zero_stage=3,
            deepspeed_include="localhost:0,1,2,3",
            input_key="context_messages",
            output_key="winner",
            apply_chat_template=True,
            flash_attn=True,
            gradient_checkpointing=True,
            packing_samples=True,
            adam_offload=True,
            bf16=True,
            save_steps=-1,
            logging_steps=1,
            eval_steps=-1,
        )

        command = build_sft_command(config)

        self.assertEqual(command[:4], ["deepspeed", "--include", "localhost:0,1,2,3", "--module"])
        self.assertIn("openrlhf.cli.train_sft", command)
        self.assertIn("--dataset", command)
        self.assertIn("data/processed/train_with_sys.jsonl", command)
        self.assertIn("--apply_chat_template", command)
        self.assertIn("--flash_attn", command)


if __name__ == "__main__":
    unittest.main()
