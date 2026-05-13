"""Custom reward function for math_500 / amc_23 eval sets.

Plumb via verl config:
    data.custom_reward_function.path=/workspace/rl_finetuning/rewards.py
    data.custom_reward_function.name=compute_score
"""

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import math_reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source in ("math_500", "amc_23"):
        return math_reward.compute_score(solution_str, ground_truth)
    return default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
