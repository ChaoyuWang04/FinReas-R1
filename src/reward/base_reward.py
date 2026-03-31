"""Rule-based reward used by the current GRPO training loop.

This file is intentionally small because it acts as a pluggable reward function
loaded by ``src.training.rl_trainer``. The trainer passes model generations and
ground-truth labels here; this module returns a scalar reward consumed by verl's
PPO/GRPO optimization loop.

Current behavior:
- inspect only the tail of the generated text
- reward ``+1`` if the final tag matches the labeled winner
- otherwise return ``-1``
"""

def answer_reward(solution_str: str, answer: str) ->float:
    """Map the generated final answer tag to a scalar reward signal."""
    pred = solution_str[-80:]

    if answer == 'model_a':
        if '<answer>[[A]]</answer>' in pred and '<answer>[[B]]</answer>' not in pred:
            return 1.0
        else:
            return -1.0
    elif answer == 'model_b':
        if '<answer>[[B]]</answer>' in pred and '<answer>[[A]]</answer>' not in pred:
            return 1.0
        else:
            return -1.0
    else:
        raise NotImplementedError('Check your dataset label!')

def lm_as_judge_match(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
):
    """Adapter with the signature expected by verl reward managers."""
    r = answer_reward(solution_str, ground_truth)
    
    return r 
