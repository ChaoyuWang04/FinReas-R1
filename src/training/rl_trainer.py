# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python entrypoint for the GRPO / RLVR training pipeline.

This is the main orchestration layer for policy training:
1. Hydra loads ``src/training/config/rl_7b.yaml`` or CLI overrides.
2. Ray is initialized and a worker-side ``main_task`` is launched.
3. Tokenizer, reward function, worker roles, and resource pools are assembled.
4. ``RubricRMRayPPOTrainer`` starts the actual verl-driven GRPO loop.

Input:
- processed JSONL data from ``data/processed/*_with_sys.jsonl``
- a base model path / HF identifier
- reward function path, usually ``src/reward/base_reward.py``

Output:
- checkpoint directories under ``trainer.default_local_dir``
- training logs and intermediate validation results
"""
from __future__ import annotations

import os

import hydra
import ray

from src.training.internal.ray_trainer import RubricRMRayPPOTrainer


def get_custom_reward_fn(config):
    """
    Load the custom reward callback referenced in the Hydra config.

    The reward function is kept separate from the trainer so experiments can
    swap reward logic without changing the control flow code in this module.
    """
    import importlib.util
    import os

    # 从配置中获取奖励函数所在的路径和函数名
    reward_fn_config = config.get('custom_reward_function') or {}
    file_path = reward_fn_config.get('path')
    if not file_path:
        # 如果没有配置自定义奖励函数，则返回 None
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"奖励函数文件 '{file_path}' 未找到。",
        )

    # 使用 importlib 动态地从文件路径加载 Python 模块
    spec = importlib.util.spec_from_file_location('custom_module', file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"从 '{file_path}' 加载模块时出错: {e}")

    function_name = reward_fn_config.get('name')

    # 检查函数是否存在于加载的模块中
    if not hasattr(module, function_name):
        raise AttributeError(
            f"奖励函数 '{function_name}' 在 '{file_path}' 中未找到。",
        )

    print(
        f"使用自定义奖励函数 '{function_name}' from '{file_path}'",
    )

    # 返回加载到的函数对象
    return getattr(module, function_name)


# 使用 @hydra.main 装饰器，这是 Hydra 库的核心功能。
# 它会自动读取配置文件，并将解析后的配置对象作为 `config` 参数传给 `main` 函数。
# config_path: 配置文件所在的目录（相对于本文件）。
# config_name: 主配置文件的名称（不含 .yaml 后缀）。
@hydra.main(config_path='config', config_name='rl_7b', version_base=None)
def main(config):
    """Hydra entrypoint that forwards the resolved config to ``run_ppo``."""
    run_ppo(config)


def run_ppo(config) -> None:
    """Prepare Ray runtime state and submit the real training task."""
    # 设置环境变量，用于解决 SGLang 与 Ray 设备隔离的潜在冲突
    os.environ['ENSURE_CUDA_VISIBLE_DEVICES'] = os.environ.get(
        'CUDA_VISIBLE_DEVICES', '',
    )
    # 检查 Ray 是否已经初始化
    if not ray.is_initialized():
        # 如果没有，则初始化一个本地 Ray 集群。
        # 这通常用于直接在命令行运行此脚本的场景。
        # 如果是通过 shell 脚本启动的 ray start，则此部分不会执行。
        ray.init(
            runtime_env={
                'env_vars': {
                    'TOKENIZERS_PARALLELISM': 'true', # 禁用 Hugging Face Tokenizer 的并行处理，避免与 Ray 冲突。
                    'NCCL_DEBUG': 'WARN',             # 设置 NCCL（NVIDIA 分布式通信库）的日志级别。
                    'VLLM_LOGGING_LEVEL': 'WARN',     # 设置 vLLM 的日志级别。
                },
            },
        )
    
    # 将核心训练任务 `main_task` 作为一个远程任务提交给 Ray 执行。
    # `ray.get()` 会阻塞，直到该任务完成。
    ray.get(main_task.remote(config))


# 使用 @ray.remote 装饰器，将 `main_task` 函数定义为一个可以在 Ray Worker 进程中执行的远程任务。
# num_cpus=1: 指定该任务需要 1 个 CPU 核心。
@ray.remote(num_cpus=1)
def main_task(config):
    """
    Worker-side training bootstrap executed inside Ray.

    Keeping this logic inside a Ray task isolates the runtime that owns model
    loading, worker creation, and the actual ``trainer.fit()`` call.
    """
    from verl.utils.fs import copy_to_local
    from pprint import pprint
    from omegaconf import OmegaConf

    # 打印最终解析后的配置，方便调试。
    # resolve=True 会计算配置中的变量引用，如 ${...}
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # 如果模型路径是远程路径（如 HDFS），则下载到本地
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # --- 1. 初始化 Tokenizer 和 Processor ---
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer = hf_tokenizer(local_path)
    # Processor 主要用于多模态模型，如果不是多模态则为 None
    processor = hf_processor(local_path, use_fast=True)

    # --- 2. 根据配置选择并定义 Worker 类 ---
    # ``strategy`` 决定底层并行后端。当前项目默认走 FSDP，Megatron 分支更多
    # 是保留与 verl 的兼容接口。
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        # 使用 PyTorch FSDP (Fully Sharded Data Parallel) 后端
        from src.training.internal.fsdp_workers import ActorRolloutRefWorker
        from verl.workers.fsdp_workers import CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        # 使用 Megatron-LM 后端
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    # --- 3. 定义角色与 Worker 的映射关系 ---
    # 这些角色对应 GRPO 训练里的三类核心计算：
    # - ActorRollout: 生成候选回答并计算策略梯度
    # - Critic: 估计 value / advantage 相关量
    # - RefPolicy: 提供 KL 约束的参考策略
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker), # Actor 和 Rollout 功能合并在一个 Worker 中
        Role.Critic: ray.remote(CriticWorker),               # Critic（价值网络）
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),   # Reference Model（参考模型，用于计算 KL 散度）
    }

    # --- 4. 定义资源池与角色分配 ---
    # 当前默认把所有 GPU 放进一个全局池，适合单实验独占机器的训练方式。
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    # 将所有角色都映射到这个全局资源池。
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # --- 5. 如果启用了模型作为奖励，则添加 RewardModel 角色 ---
    # 当前仓库默认关闭 reward model，主要使用 rule-based reward function。
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # --- 6. 初始化奖励管理器与函数 ---
    reward_manager_name = config.reward_model.get('reward_manager', 'naive')
    if reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
    else:
        raise NotImplementedError

    # 默认 reward path 指向 ``src/reward/base_reward.py``，也可以通过 CLI 覆盖。
    compute_score = get_custom_reward_fn(config)
    # 创建奖励函数实例，它会封装自定义的 `compute_score` 函数
    reward_fn = reward_manager_cls(
        tokenizer=tokenizer, num_examine=1, compute_score=compute_score,
    )

    # 验证阶段沿用同一套 reward 逻辑，保证训练与验证协议一致。
    val_reward_fn = reward_fn

    # --- 7. 初始化资源管理器和核心训练器 ---
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping,
    )

    # 实例化核心训练器 `RubricRMRayPPOTrainer`
    trainer = RubricRMRayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )

    # --- 8. 启动训练 ---
    trainer.init_workers() # 初始化所有 Ray Worker 进程
    trainer.fit()          # 开始 PPO 训练循环

# Python 标准的程序入口
if __name__ == '__main__':
    main()
