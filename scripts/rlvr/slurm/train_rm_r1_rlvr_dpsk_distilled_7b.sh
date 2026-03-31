#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
TRAIN_PER_GPU="${TRAIN_PER_GPU:-4}"
FORWARD_PER_GPU="${FORWARD_PER_GPU:-4}"
N_NODES="${N_NODES:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-RM-R1-Dpsk-Distilled-7B-LR1.0e-6}"
SAVE_NAME="${SAVE_NAME:-RM-R1-Dpsk-Distilled-7B-LR1.0e-6}"

source "$(dirname "$0")/train_rlvr_slurm.sh"
