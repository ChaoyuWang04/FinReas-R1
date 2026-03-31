#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
TRAIN_PER_GPU="${TRAIN_PER_GPU:-1}"
FORWARD_PER_GPU="${FORWARD_PER_GPU:-1}"
N_NODES="${N_NODES:-4}"
LR="${LR:-8.0e-7}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-RM-R1-Dpsk-Distilled-32B-LR8.0e-7}"
SAVE_NAME="${SAVE_NAME:-RM-R1-Dpsk-Distilled-32B-LR8.0e-7}"

source "$(dirname "$0")/train_rlvr_slurm.sh"
