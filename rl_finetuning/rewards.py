"""Custom reward function for our experiments.

Wires via verl config:
    data.custom_reward_function.path=/workspace/rl_finetuning/rewards.py
    data.custom_reward_function.name=compute_score

Handles:
- openai/gsm8k:  accepts either `\\boxed{N}` (Qwen-Math native format) or
                 `#### N` (canonical GSM8K format) anywhere in the last 500 chars.
- math_500, amc_23:  delegates to verl's math_reward (LaTeX-aware via sympy).
- everything else:  delegates to verl's built-in default_compute_score.
"""

import re

from verl.utils.reward_score import default_compute_score, math_reward


_GSM8K_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_GSM8K_HASH = re.compile(r"####\s*(\-?[0-9\.,]+)")


def _extract_gsm8k_answer(text: str) -> str | None:
    tail = text[-500:]
    # \boxed{...} — take the LAST one (final answer)
    m = list(_GSM8K_BOXED.finditer(tail))
    if m:
        return m[-1].group(1).strip().replace(",", "").replace("$", "")
    m = list(_GSM8K_HASH.finditer(tail))
    if m:
        return m[-1].group(1).strip().replace(",", "").replace("$", "")
    return None


def _gsm8k_score(solution_str: str, ground_truth: str) -> float:
    pred = _extract_gsm8k_answer(solution_str)
    if pred is None:
        return 0.0
    if pred == str(ground_truth).strip():
        return 1.0
    try:
        return 1.0 if float(pred) == float(ground_truth) else 0.0
    except (ValueError, TypeError):
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source == "openai/gsm8k":
        return _gsm8k_score(solution_str, ground_truth)
    if data_source in ("math_500", "amc_23"):
        return math_reward.compute_score(solution_str, ground_truth)
    return default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
