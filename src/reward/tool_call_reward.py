"""Placeholder for future business-specific reward functions.

The repository currently trains with the simpler tag-matching reward in
``src.reward.base_reward``. This module marks the intended extension point for
future rewards that may depend on tool execution traces, structured business
rules, or multi-signal scoring.
"""


def tool_call_reward(*args, **kwargs) -> float:
    """Placeholder signature matching the reward-function extension style."""
    raise NotImplementedError(
        "tool_call_reward is a scaffold. Replace it with the ad-campaign tool-calling reward logic."
    )
