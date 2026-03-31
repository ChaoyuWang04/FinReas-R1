import argparse
import unittest

from src.eval.benchmark import build_eval_commands


class EvalBenchmarkTests(unittest.TestCase):
    def test_build_eval_commands_uses_explicit_roots(self) -> None:
        args = argparse.Namespace(
            model="foo/bar",
            model_save_name="demo-model",
            device="0,1",
            vllm_gpu_util=0.8,
            num_gpus=2,
            max_tokens=2048,
            result_dir="experiments/eval",
            reward_bench_root="/tmp/reward-bench",
            rm_bench_root="/tmp/rm-bench",
            rmb_root="/tmp/rmb",
            trust_remote_code=True,
            dry_run=True,
        )

        commands = build_eval_commands(args)

        self.assertEqual(commands[0].cwd.as_posix(), "/tmp/reward-bench")
        self.assertEqual(commands[1].cwd.as_posix(), "/tmp/rm-bench")
        self.assertEqual(commands[5].cwd.as_posix(), "/tmp/rmb")
        self.assertIn("--model", commands[0].argv)
        self.assertIn("demo-model", commands[-1].argv)


if __name__ == "__main__":
    unittest.main()
