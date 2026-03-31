# Evaluation Assets

This repository keeps only the orchestration layer for benchmark evaluation.
Third-party benchmark code and datasets must be downloaded separately.

## Expected checkouts

- `dependencies/benchmarks/reward-bench`
  Source: <https://github.com/allenai/reward-bench>
- `dependencies/benchmarks/RM-Bench`
  Source: <https://github.com/THU-KEG/RM-Bench>
- `dependencies/benchmarks/RMB-Reward-Model-Benchmark`
  Source: <https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark>

You can override these paths with:

- `FINRAG_REWARD_BENCH_ROOT`
- `FINRAG_RM_BENCH_ROOT`
- `FINRAG_RMB_ROOT`

## Run

```bash
bash scripts/eval/run_eval.sh \
  --model your-model-path \
  --model-save-name your-model-name \
  --device 0,1,2,3 \
  --num-gpus 4
```

Use `--dry-run` first if you want to inspect the generated commands.
