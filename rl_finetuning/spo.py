"""SPO (Segment Policy Optimization) advantage estimator.

Registered with verl's adv-estimator registry at import time so that
`algorithm.adv_estimator=spo` dispatches here.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("spo")
def compute_spo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config=None,
    old_log_probs: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Segment-weighted outcome advantage.

    Group-normalizes scalar rewards GRPO-style, then distributes each response's
    normalized reward across `n_segs` segments with linearly increasing weights.
    Optionally zeros advantages on tokens whose `old_log_prob` exceeds
    `prob_mask_threshold` (paper Eq. 3, requires old_log_probs to be wired
    through ray_trainer dispatch).
    """
    cutpoint_interval = 5
    prob_mask_threshold = 0.9
    if config is not None:
        cutpoint_interval = int(getattr(config, "spo_cutpoint_interval", cutpoint_interval) or cutpoint_interval)
        prob_mask_threshold = float(getattr(config, "spo_prob_mask_threshold", prob_mask_threshold) or prob_mask_threshold)

    scores = token_level_rewards.sum(dim=-1)
    bsz = scores.shape[0]
    with torch.no_grad():
        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
        normalized_scores = scores.clone()
        for i in range(bsz):
            normalized_scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        advantages = torch.zeros_like(token_level_rewards)
        for i in range(bsz):
            resp_len = int(response_mask[i].sum().item())
            if resp_len == 0:
                continue
            norm_reward = normalized_scores[i].item()
            cutpoints = list(range(cutpoint_interval - 1, resp_len, cutpoint_interval))
            if len(cutpoints) == 0 or cutpoints[-1] != resp_len - 1:
                cutpoints.append(resp_len - 1)
            n_segs = len(cutpoints)
            weights = torch.arange(1, n_segs + 1, dtype=torch.float)
            weights = weights / weights.sum()
            prev_cp = 0
            for k, cp in enumerate(cutpoints):
                advantages[i, prev_cp:cp + 1] = norm_reward * weights[k].item() * n_segs
                prev_cp = cp + 1
            if old_log_probs is not None:
                probs = torch.exp(old_log_probs[i, :resp_len])
                advantages[i, :resp_len][probs > prob_mask_threshold] = 0.0
        advantages = advantages * response_mask
    return advantages, advantages
