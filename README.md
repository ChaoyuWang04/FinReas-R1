<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ChaoyuWang04/FinRAG-GRPO">
    <img src="images/logo.jpg" alt="Logo" width="100" height="100">
  </a>

<h3 align="center">FinReas-R1</h3>

<p align="center">
  A GRPO training pipeline for reasoning reward modeling on Chinese customer-service preference data.
  <br /><br />
  | <a href="https://huggingface.co/datasets/SamWang0405/FinRAG-GRPO">🤗 HuggingFace Dataset</a> |
  <a href="https://github.com/ChaoyuWang04/FinRAG-GRPO/issues/new?labels=bug">Report Bug</a> |
  <a href="https://github.com/ChaoyuWang04/FinRAG-GRPO/issues/new?labels=enhancement">Request Feature</a> |
</p>

</div>


<!-- ABOUT THE PROJECT -->
## About The Project

FinRAG-GRPO is an open-source training pipeline for **Reasoning Reward Models (ReasRM)** built around **GRPO** fine-tuning. Instead of directly predicting a scalar reward, the model is trained to first produce an explicit judging process and then output a final pairwise preference between two candidate responses.

This repository currently focuses on **Chinese customer-service preference modeling**:

| Component | Description |
|--------|-------|
| Synthetic data generation | Multi-threaded generation of customer-service A/B preference data |
| Data format | JSONL with `context_messages` and `winner` |
| Task style | Pairwise preference judgment for customer-service answers |
| Training method | GRPO / PPO-style RL training with veRL + Ray + vLLM |
| Reward function | Rule-based match on the final `<answer>[[A/B]]</answer>` tag |
| Inference demo | Hugging Face model loading and single-example evaluation |
| Data files included | Raw shards, merged train/test, and `_with_sys` variants |

The current workflow in this repo is:

1. Generate synthetic customer-service preference samples.
2. Merge and split them into train/test JSONL files.
3. Optionally inject a Chinese system prompt for rubric-style judging.
4. Train a reasoning reward model with GRPO.
5. Export the FSDP checkpoint and run local inference.


### Built With

[![Python][Python-badge]][Python-url]
[![Ray][Ray-badge]][Ray-url]
[![vLLM][vLLM-badge]][vLLM-url]
[![Transformers][Transformers-badge]][Transformers-url]
[![HuggingFace][HuggingFace-badge]][HuggingFace-url]



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

- Python 3.11 recommended
- Conda
- CUDA-capable GPUs for training
- [veRL](https://github.com/volcengine/verl) on a pinned commit
- [vLLM](https://github.com/vllm-project/vllm) on a pinned commit
- `flash-attn==2.7.2.post1` recommended for faster training

For the exact environment notes used in this repo, see `setup.sh`.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ChaoyuWang04/FinRAG-GRPO.git
   cd FinRAG-GRPO
   ```

2. Create a Python environment
   ```sh
   conda create -n rm-r1-1 python=3.11 -y
   conda activate rm-r1-1
   ```

3. Install baseline dependencies
   ```sh
   pip install -r requirements.txt
   ```

4. Install veRL at the pinned commit
   ```sh
   git clone https://github.com/volcengine/verl dependencies/verl
   cd dependencies/verl
   git checkout e49fb572bf85a8f0ef7124c898f509bd6d9832a1
   pip install -e .
   cd ../..
   ```

5. Install vLLM at the pinned commit
   ```sh
   git clone https://github.com/vllm-project/vllm.git dependencies/vllm
   cd dependencies/vllm
   git checkout ed6e9075d31e32c8548b480a47d1ffb77da1f54c
   git cherry-pick caac5c2e597b1780c3df54a537c34e6061c32cff
   export VLLM_COMMIT=ed6e9075d31e32c8548b480a47d1ffb77da1f54c
   export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/ed6e9075d31e32c8548b480a47d1ffb77da1f54c/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
   VLLM_USE_PRECOMPILED=1 pip install --editable .
   cd ../..
   ```

6. Install flash-attention
   ```sh
   pip install flash-attn==2.7.2.post1 --no-build-isolation
   ```

7. Verify project structure
   ```
   FinRAG-GRPO/
   ├── data/
   │   ├── raw/
   │   ├── processed/
   │   └── reasoning_chains/
   ├── demo/
   │   ├── convert_fsdp_to_hf.py
   │   ├── demo.ipynb
   │   └── demo.py
   ├── docs/
   │   ├── architecture.md
   │   ├── evaluation.md
   │   └── note.md
   ├── images/
   │   └── logo.jpg
   ├── scripts/
   │   ├── distill/
   │   ├── eval/
   │   └── rlvr/
   ├── src/
   │   ├── data/
   │   ├── eval/
   │   ├── model/
   │   ├── reward/
   │   └── training/
   ├── README.md
   ├── LICENSE
   ├── requirements.txt
   └── setup.sh
   ```



<!-- USAGE EXAMPLES -->
## Usage

The current pipeline can be understood as four stages:

**Stage 1 - Generate synthetic customer-service preference data**

`src.data.generate_customer_service_data` creates pairwise A/B samples for Chinese e-commerce customer-service scenarios.  
Note: this script imports `call_llm` from `src.model.llm`, so you need to configure your API credentials before running it.

```sh
python -m src.data.generate_customer_service_data
# Output: data/raw/customer_service_dataset.jsonl
```

**Stage 2 - Merge and split dataset shards**

Use the project data utilities to merge the raw JSONL shards, shuffle them with a fixed seed, and write the final training and test sets.

```sh
python -m src.data.split
# Output:
#   data/processed/train.jsonl
#   data/processed/test.jsonl
```

**Stage 3 - Inject the system prompt**

`src.model.prompt_template` prepends the Chinese judging prompt to each sample and produces `_with_sys` variants for training and evaluation.

```sh
python -m src.model.prompt_template
# Output:
#   data/processed/train_with_sys.jsonl
#   data/processed/test_with_sys.jsonl
```

**Stage 4 - Launch GRPO training**

The local training entrypoint is:

```sh
bash ./scripts/rlvr/local/train_rm_r1_rlvr_dpsk_distilled_7b.sh
```

The script configures:

- Ray startup and teardown
- model path and save path
- GRPO batch sizes and token limits
- custom reward loading from `src/reward/base_reward.py`
- local JSONL train/validation files

Before running it, you will likely want to update the environment variables or defaults inside the script, such as:

- `MODEL_PATH`
- `SAVE_META_DIR`
- `TRAIN_TASK`
- `EVAL_TASK`

**Optional - Run SFT / distillation**

```sh
bash ./scripts/distill/local/distill_qwen2.5-7b-instruct.sh --dry-run
```

**Optional - Convert FSDP checkpoints to a Hugging Face model**

```sh
python demo/convert_fsdp_to_hf.py
```

**Optional - Run the local inference demo**

```sh
python demo/demo.py
```

**Optional - Run the evaluation harness**

```sh
bash ./scripts/eval/run_eval.sh --model your-model --model-save-name your-model-name --dry-run
```

The evaluation orchestration is project-owned, while benchmark code and datasets stay external. See `docs/evaluation.md` for the expected checkout paths.


<!-- MARKDOWN LINKS & IMAGES -->
[Python-badge]: https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[Ray-badge]: https://img.shields.io/badge/Ray-Distributed-028CF0?style=for-the-badge
[Ray-url]: https://www.ray.io/
[vLLM-badge]: https://img.shields.io/badge/vLLM-Inference-FF6B6B?style=for-the-badge
[vLLM-url]: https://github.com/vllm-project/vllm
[Transformers-badge]: https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?style=for-the-badge
[Transformers-url]: https://github.com/huggingface/transformers
[HuggingFace-badge]: https://img.shields.io/badge/HuggingFace-Models-FFCC4D?style=for-the-badge&logo=huggingface&logoColor=black
[HuggingFace-url]: https://huggingface.co/
