"""OpenRLHF SFT launcher for reasoning distillation.

Unlike ``src.training.rl_trainer``, this file does not implement training
itself. Its job is to translate repository-friendly config fields into the
exact ``deepspeed --module openrlhf.cli.train_sft`` command needed for the
distillation stage.

Input:
- ``src/training/config/sft_1.7b.yaml`` plus optional CLI overrides

Output:
- either a printed shell command in ``--dry-run`` mode
- or an executed OpenRLHF SFT job that writes checkpoints to ``output_dir``
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SFTConfig:
    """Minimal typed view of the SFT hyperparameters we expose locally."""
    save_path: str
    model_name_or_path: str
    dataset: str
    train_batch_size: int
    micro_train_batch_size: int
    max_epochs: int
    max_len: int
    learning_rate: float
    zero_stage: int
    deepspeed_include: str | None
    input_key: str
    output_key: str
    apply_chat_template: bool
    flash_attn: bool
    gradient_checkpointing: bool
    packing_samples: bool
    adam_offload: bool
    bf16: bool
    save_steps: int
    logging_steps: int
    eval_steps: int


DEFAULT_SFT_CONFIG = SFTConfig(
    # save_path: where distilled checkpoints and trainer state are written.
    save_path="experiments/round2/sft_qwen2_5_7b",
    # model_name_or_path: base instruct model used to start distillation.
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    # dataset: JSONL prepared by prompt injection, aligned with project schema.
    dataset="data/processed/train_with_sys.jsonl",
    # train_batch_size: global batch size across all devices.
    train_batch_size=128,
    # micro_train_batch_size: per-step micro batch, lowered first when OOM occurs.
    micro_train_batch_size=1,
    max_epochs=1,
    # max_len: training sequence length ceiling; raise only if hardware allows it.
    max_len=12288,
    # learning_rate: common SFT default for full-model fine-tuning is around 1e-6~1e-5.
    learning_rate=5e-6,
    # zero_stage: OpenRLHF / DeepSpeed memory sharding level, 3 is the usual large-model default.
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


def load_config(config_path: str) -> SFTConfig:
    """Load YAML config and fall back to built-in defaults if OmegaConf is absent."""
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        return DEFAULT_SFT_CONFIG

    raw = OmegaConf.load(config_path)
    trainer = raw.get("trainer", {})
    data = raw.get("data", {})
    return SFTConfig(
        save_path=trainer.get("output_dir", DEFAULT_SFT_CONFIG.save_path),
        model_name_or_path=trainer.get("model_name_or_path", DEFAULT_SFT_CONFIG.model_name_or_path),
        dataset=data.get("train_file", DEFAULT_SFT_CONFIG.dataset),
        train_batch_size=int(trainer.get("train_batch_size", DEFAULT_SFT_CONFIG.train_batch_size)),
        micro_train_batch_size=int(trainer.get("micro_train_batch_size", DEFAULT_SFT_CONFIG.micro_train_batch_size)),
        max_epochs=int(trainer.get("max_epochs", DEFAULT_SFT_CONFIG.max_epochs)),
        max_len=int(trainer.get("max_len", DEFAULT_SFT_CONFIG.max_len)),
        learning_rate=float(trainer.get("learning_rate", DEFAULT_SFT_CONFIG.learning_rate)),
        zero_stage=int(trainer.get("zero_stage", DEFAULT_SFT_CONFIG.zero_stage)),
        deepspeed_include=trainer.get("deepspeed_include", DEFAULT_SFT_CONFIG.deepspeed_include),
        input_key=data.get("input_key", DEFAULT_SFT_CONFIG.input_key),
        output_key=data.get("output_key", DEFAULT_SFT_CONFIG.output_key),
        apply_chat_template=bool(trainer.get("apply_chat_template", DEFAULT_SFT_CONFIG.apply_chat_template)),
        flash_attn=bool(trainer.get("flash_attn", DEFAULT_SFT_CONFIG.flash_attn)),
        gradient_checkpointing=bool(trainer.get("gradient_checkpointing", DEFAULT_SFT_CONFIG.gradient_checkpointing)),
        packing_samples=bool(trainer.get("packing_samples", DEFAULT_SFT_CONFIG.packing_samples)),
        adam_offload=bool(trainer.get("adam_offload", DEFAULT_SFT_CONFIG.adam_offload)),
        bf16=bool(trainer.get("bf16", DEFAULT_SFT_CONFIG.bf16)),
        save_steps=int(trainer.get("save_steps", DEFAULT_SFT_CONFIG.save_steps)),
        logging_steps=int(trainer.get("logging_steps", DEFAULT_SFT_CONFIG.logging_steps)),
        eval_steps=int(trainer.get("eval_steps", DEFAULT_SFT_CONFIG.eval_steps)),
    )


def build_sft_command(config: SFTConfig) -> list[str]:
    """Convert local config fields into an OpenRLHF-compatible deepspeed command."""
    command = ["deepspeed"]
    if config.deepspeed_include:
        command.extend(["--include", config.deepspeed_include])
    command.extend(
        [
            "--module",
            "openrlhf.cli.train_sft",
            "--save_path",
            config.save_path,
            "--save_steps",
            str(config.save_steps),
            "--logging_steps",
            str(config.logging_steps),
            "--eval_steps",
            str(config.eval_steps),
            "--train_batch_size",
            str(config.train_batch_size),
            "--micro_train_batch_size",
            str(config.micro_train_batch_size),
            "--pretrain",
            config.model_name_or_path,
            "--max_epochs",
            str(config.max_epochs),
            "--max_len",
            str(config.max_len),
            "--zero_stage",
            str(config.zero_stage),
            "--learning_rate",
            str(config.learning_rate),
            "--dataset",
            config.dataset,
            "--input_key",
            config.input_key,
            "--output_key",
            config.output_key,
        ]
    )
    if config.bf16:
        command.append("--bf16")
    if config.apply_chat_template:
        command.append("--apply_chat_template")
    if config.flash_attn:
        command.append("--flash_attn")
    if config.gradient_checkpointing:
        command.append("--gradient_checkpointing")
    if config.packing_samples:
        command.append("--packing_samples")
    if config.adam_offload:
        command.append("--adam_offload")
    return command


def build_parser() -> argparse.ArgumentParser:
    """Expose a small CLI for local overrides and dry-run inspection."""
    parser = argparse.ArgumentParser(description="Launch OpenRLHF SFT training.")
    parser.add_argument(
        "--config",
        default="src/training/config/sft_1.7b.yaml",
        help="Path to the YAML config used to build the OpenRLHF command.",
    )
    parser.add_argument("--save-path")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--dataset")
    parser.add_argument("--deepspeed-include")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by scripts under ``scripts/distill``."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.save_path:
        config = dataclass_replace(config, save_path=args.save_path)
    if args.model_name_or_path:
        config = dataclass_replace(config, model_name_or_path=args.model_name_or_path)
    if args.dataset:
        config = dataclass_replace(config, dataset=args.dataset)
    if args.deepspeed_include is not None:
        config = dataclass_replace(config, deepspeed_include=args.deepspeed_include)

    command = build_sft_command(config)
    if args.dry_run:
        print(shlex.join(command))
        return 0

    Path(config.save_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    return 0


def dataclass_replace(config: SFTConfig, **changes: object) -> SFTConfig:
    """Small local replacement helper to keep the config immutable by default."""
    values = config.__dict__.copy()
    values.update(changes)
    return SFTConfig(**values)


if __name__ == "__main__":
    raise SystemExit(main())
