#!/bin/bash
set -euo pipefail

python3 -m src.training.sft_trainer \
  --model-name-or-path "${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}" \
  --save-path "${SAVE_PATH:-experiments/round2/distill_qwen2_5_7b}" \
  --deepspeed-include "${DEEPSPEED_INCLUDE:-localhost:0,1,2,3}" \
  "$@"
