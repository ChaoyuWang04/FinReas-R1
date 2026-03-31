#!/bin/bash
set -euo pipefail

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
else
  DRY_RUN="${DRY_RUN:-0}"
fi

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VERL_PPO_LOGGING_LEVEL="${VERL_PPO_LOGGING_LEVEL:-INFO}"
export PYTHONUNBUFFERED=1

N_GPU="${N_GPU:-8}"
N_NODES="${N_NODES:-1}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
LR="${LR:-1.0e-6}"
GPU_MEM_UTILIZATION="${GPU_MEM_UTILIZATION:-0.5}"
TOTAL_EPISODES="${TOTAL_EPISODES:-1}"
SAVE_EVERY_STEP="${SAVE_EVERY_STEP:-100}"
TRAIN_BS="${TRAIN_BS:-1024}"
PPO_MINI_BS="${PPO_MINI_BS:-128}"
WARMUP_STYLE="${WARMUP_STYLE:-constant}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
TRAIN_PER_GPU="${TRAIN_PER_GPU:-4}"
FORWARD_PER_GPU="${FORWARD_PER_GPU:-4}"
PROJECT_NAME="${PROJECT_NAME:-FinRAG-GRPO}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rlvr-slurm}"
SAVE_NAME="${SAVE_NAME:-rlvr-slurm}"
SAVE_META_DIR="${SAVE_META_DIR:-experiments/round3_rl}"
REWARD_PATH="${REWARD_PATH:-src/reward/base_reward.py}"
REWARD_FUNC_NAME="${REWARD_FUNC_NAME:-lm_as_judge_match}"
TRAIN_TASK="${TRAIN_TASK:-data/processed/train_with_sys.jsonl}"
EVAL_TASK="${EVAL_TASK:-data/processed/test_with_sys.jsonl}"

MAX_NUM_BATCHED_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

PYTHON_CMD=(
  python3 -m src.training.rl_trainer
  data.train_files="${TRAIN_TASK}"
  data.val_files="${EVAL_TASK}"
  data.prompt_key=context_messages
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.train_batch_size="${TRAIN_BS}"
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.actor.optim.lr="${LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BS}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${TRAIN_PER_GPU}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTILIZATION}"
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${FORWARD_PER_GPU}"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${FORWARD_PER_GPU}"
  custom_reward_function.path="${REWARD_PATH}"
  custom_reward_function.name="${REWARD_FUNC_NAME}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.total_epochs="${TOTAL_EPISODES}"
  trainer.save_freq="${SAVE_EVERY_STEP}"
  trainer.test_freq=-1
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.n_gpus_per_node="${N_GPU}"
  trainer.nnodes="${N_NODES}"
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.optim.warmup_style="${WARMUP_STYLE}"
  trainer.default_local_dir="${SAVE_META_DIR}/${SAVE_NAME}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${PYTHON_CMD[@]}"
  printf '\n'
  exit 0
fi

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
fi

port=6379
ip_head="${head_node_ip}:${port}"
export ip_head

srun --nodes=1 --ntasks=1 -w "$head_node" ray stop || true
sleep 5
srun --nodes=1 --ntasks=1 -w "$head_node" \
  ray start --head --node-ip-address="$head_node_ip" --port="$port" \
  --num-cpus 128 --num-gpus "${N_GPU}" --block &
sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w "$node_i" ray stop || true
  sleep 5
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    ray start --address "$ip_head" --num-cpus 128 --num-gpus "${N_GPU}" --block &
  sleep 5
done

"${PYTHON_CMD[@]}"
