# Architecture

## Goal

Split project-owned research code from upstream dependencies so the repository can evolve around your workflow instead of preserving the original `rm_r1/` layout.

## Decisions

1. `src/` is the only long-term project code surface.
2. `dependencies/` holds external frameworks that may be replaced or re-pinned independently.
3. `data/` stores raw and processed datasets outside the code tree.
4. `experiments/` stores logs, notebooks, and one-off outputs outside the code tree.
5. Thin shell wrappers live under `scripts/`, while Python entry points live under `src/`.

## Migration Map

- `generate_customer_service_data.py` -> `src/data/generate_customer_service_data.py`
- `merge_and_split_dataset.py` -> `src/data/split.py`
- `rm_r1/dataset/mix_data/*jsonl` -> `data/raw/` and `data/processed/`
- `rm_r1/dataset/mix_data/preprocess_data.py` -> `src/model/prompt_template.py`
- `rm_r1/verl/trainer/main_ppo.py` -> `src/training/rl_trainer.py`
- `rm_r1/verl/trainer/ppo/ray_trainer.py` -> `src/training/internal/ray_trainer.py`
- `rm_r1/verl/workers/fsdp_workers.py` -> `src/training/internal/fsdp_workers.py`
- `rm_r1/verl/utils/dataset/rl_dataset.py` -> `src/data/rl_dataset.py`
- `rm_r1/verl/utils/reward_score/lm_as_judge.py` -> `src/reward/base_reward.py`
- `demo/demo.py` -> `src/model/inference.py`
- `outputs/` -> `experiments/round1/outputs/`

## Known Gaps

- `dependencies/verl` is still a partial local dependency snapshot rather than a clean upstream clone.
- Third-party benchmark code is intentionally external and must be downloaded separately.
- `.gitignore` has not been redesigned yet.
