"""Evaluation entrypoint for external reward-model benchmarks.

This module only owns the orchestration layer. Benchmark code and datasets stay
outside the repository and are referenced via configurable checkout paths.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_RESULTS_DIR = Path("experiments/eval")
DEFAULT_BENCHMARK_ROOT = Path("dependencies/benchmarks")


@dataclass(frozen=True)
class EvalCommand:
    name: str
    cwd: Path
    argv: tuple[str, ...]


def _non_empty_args(items: Iterable[str | None]) -> list[str]:
    """Drop unset optional CLI fragments before building subprocess commands."""
    return [item for item in items if item]


def build_eval_commands(args: argparse.Namespace) -> list[EvalCommand]:
    """Translate high-level evaluation arguments into concrete benchmark commands.

    The repository owns this orchestration layer only. Each returned command
    points at an external benchmark checkout plus the result directory inside
    ``experiments/eval``.
    """
    result_dir = Path(args.result_dir)
    reward_bench_root = _resolve_root(
        explicit=args.reward_bench_root,
        env_var="FINRAG_REWARD_BENCH_ROOT",
        fallback=DEFAULT_BENCHMARK_ROOT / "reward-bench",
    )
    rm_bench_root = _resolve_root(
        explicit=args.rm_bench_root,
        env_var="FINRAG_RM_BENCH_ROOT",
        fallback=DEFAULT_BENCHMARK_ROOT / "RM-Bench",
    )
    rmb_root = _resolve_root(
        explicit=args.rmb_root,
        env_var="FINRAG_RMB_ROOT",
        fallback=DEFAULT_BENCHMARK_ROOT / "RMB-Reward-Model-Benchmark",
    )

    model_args = _non_empty_args(
        [
            "--model",
            args.model,
            "--model_save_name",
            args.model_save_name,
            "--vllm_gpu_util",
            str(args.vllm_gpu_util),
            "--num_gpus",
            str(args.num_gpus),
            "--max_tokens",
            str(args.max_tokens),
        ]
    )
    if args.trust_remote_code:
        model_args.append("--trust_remote_code")

    commands = [
        EvalCommand(
            name="reward-bench",
            cwd=reward_bench_root,
            argv=tuple(
                [
                    "python",
                    "scripts/run_generative.py",
                    *model_args,
                    "--meta_result_save_dir",
                    str(result_dir),
                ]
            ),
        ),
        EvalCommand(
            name="rm-bench-total-dataset-1",
            cwd=rm_bench_root,
            argv=tuple(
                [
                    "python",
                    "scripts/run_generative.py",
                    *model_args,
                    "--datapath",
                    "data/total_dataset_1.json",
                    "--META_RESULT_SAVE_DIR",
                    str(result_dir),
                ]
            ),
        ),
        EvalCommand(
            name="rm-bench-total-dataset-2",
            cwd=rm_bench_root,
            argv=tuple(
                [
                    "python",
                    "scripts/run_generative.py",
                    *model_args,
                    "--datapath",
                    "data/total_dataset_2.json",
                    "--META_RESULT_SAVE_DIR",
                    str(result_dir),
                ]
            ),
        ),
        EvalCommand(
            name="rm-bench-total-dataset-3",
            cwd=rm_bench_root,
            argv=tuple(
                [
                    "python",
                    "scripts/run_generative.py",
                    *model_args,
                    "--datapath",
                    "data/total_dataset_3.json",
                    "--META_RESULT_SAVE_DIR",
                    str(result_dir),
                ]
            ),
        ),
        EvalCommand(
            name="rm-bench-process-results",
            cwd=rm_bench_root,
            argv=tuple(
                [
                    "python",
                    "scripts/process_final_result.py",
                    "--model_save_name",
                    args.model_save_name,
                    "--model",
                    args.model,
                    "--meta_result_save_dir",
                    str(result_dir),
                ]
            ),
        ),
        EvalCommand(
            name="rmb-pairwise-harmlessness",
            cwd=rmb_root,
            argv=tuple(
                [
                    "python",
                    "eval/scripts/run_generative.py",
                    *model_args,
                    "--meta_result_save_dir",
                    str(result_dir),
                    "--dataset_dir",
                    "RMB_dataset/Pairwise_set/Harmlessness",
                ]
            ),
        ),
        EvalCommand(
            name="rmb-pairwise-helpfulness",
            cwd=rmb_root,
            argv=tuple(
                [
                    "python",
                    "eval/scripts/run_generative.py",
                    *model_args,
                    "--meta_result_save_dir",
                    str(result_dir),
                ]
            ),
        ),
        EvalCommand(
            name="rmb-bon-helpfulness",
            cwd=rmb_root,
            argv=tuple(
                [
                    "python",
                    "eval/scripts/run_generative_bestofn.py",
                    *model_args,
                    "--meta_result_save_dir",
                    str(result_dir),
                    "--dataset",
                    "RMB_dataset/BoN_set/Helpfulness",
                ]
            ),
        ),
        EvalCommand(
            name="rmb-bon-harmlessness",
            cwd=rmb_root,
            argv=tuple(
                [
                    "python",
                    "eval/scripts/run_generative_bestofn.py",
                    *model_args,
                    "--meta_result_save_dir",
                    str(result_dir),
                    "--dataset",
                    "RMB_dataset/BoN_set/Harmlessness",
                ]
            ),
        ),
        EvalCommand(
            name="rmb-process-results",
            cwd=rmb_root,
            argv=tuple(
                [
                    "python",
                    "eval/scripts/process_final_result.py",
                    "--model_save_name",
                    args.model_save_name,
                    "--meta_result_save_dir",
                    str(result_dir),
                ]
            ),
        ),
    ]
    return commands


def build_parser() -> argparse.ArgumentParser:
    """Define the user-facing CLI for the evaluation orchestration layer."""
    parser = argparse.ArgumentParser(
        description="Run the project-owned evaluation orchestration against external benchmark checkouts.",
    )
    parser.add_argument("--model", required=True, help="Model path or Hugging Face identifier.")
    parser.add_argument("--model-save-name", required=True, help="Logical model name used in result folders.")
    parser.add_argument("--device", default="0,1,2,3,4,5,6,7", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.90)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--reward-bench-root")
    parser.add_argument("--rm-bench-root")
    parser.add_argument("--rmb-root")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run or print the external benchmark commands for one model checkpoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    commands = build_eval_commands(args)
    if args.dry_run:
        for command in commands:
            print(f"[{command.name}] (cwd={command.cwd}) {' '.join(command.argv)}")
        return 0

    missing = [command.cwd for command in commands if not command.cwd.exists()]
    if missing:
        parser.error(
            "Benchmark checkout(s) not found. Configure the benchmark roots or download them as described in docs/evaluation.md: "
            + ", ".join(str(path) for path in sorted(set(missing)))
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.device
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    for command in commands:
        print(f"[run] {command.name}: {' '.join(command.argv)}")
        subprocess.run(command.argv, cwd=command.cwd, env=env, check=True)
    return 0


def _resolve_root(explicit: str | None, env_var: str, fallback: Path) -> Path:
    """Resolve a benchmark checkout path from CLI override, env var, or default."""
    if explicit:
        return Path(explicit)
    env_value = os.getenv(env_var)
    if env_value:
        return Path(env_value)
    return fallback


if __name__ == "__main__":
    raise SystemExit(main())
