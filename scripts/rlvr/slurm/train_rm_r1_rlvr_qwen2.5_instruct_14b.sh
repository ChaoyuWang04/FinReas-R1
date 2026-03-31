#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-distilled_model_dir}"
TRAIN_PER_GPU="${TRAIN_PER_GPU:-1}"
FORWARD_PER_GPU="${FORWARD_PER_GPU:-1}"
N_NODES="${N_NODES:-2}"
LR="${LR:-7.0e-7}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-RM-R1-Qwen2.5-Instruct-14B-RLVR-after-Distill-LR7.0e-7}"
SAVE_NAME="${SAVE_NAME:-RM-R1-Qwen2.5-Instruct-14B-RLVR-after-Distill-LR7.0e-7}"

source "$(dirname "$0")/train_rlvr_slurm.sh"
