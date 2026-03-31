#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-distilled_model_dir}"
TRAIN_PER_GPU="${TRAIN_PER_GPU:-1}"
FORWARD_PER_GPU="${FORWARD_PER_GPU:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-RM-R1-Qwen2.5-Instruct-14B-RLVR-after-Distill-LR1.0e-6}"
SAVE_NAME="${SAVE_NAME:-RM-R1-Qwen2.5-Instruct-14B-RLVR-after-Distill-LR1.0e-6}"

source "$(dirname "$0")/train_rlvr_local.sh"
