"""SPO (Segment Policy Optimization) advantage estimator.

Registered with verl's adv-estimator registry at import time so that
`algorithm.adv_estimator=spo` dispatches here.

Two execution paths:

1. **Paper-faithful SPO-chain** (MC values present on the batch):
   `spo_mc.compute_and_store_mc_values` writes `spo_segment_values [B, T+2]`
   and `spo_segment_cutpoints [B, T+2]` before this estimator runs. We:
     - Center per row by V(s_0) so V_centered(s_0)=0 and V_centered(s_T)=R-V(s_0).
     - Group-normalize by the group's std of R so the per-response advantage
       sum has GRPO-comparable variance.
     - Compute segment advantages A_t = V_centered_norm(s_{cp_t}) - V_centered_norm(s_{cp_{t-1}})
       and broadcast over the tokens of segment t.
     - Optionally zero tokens where the old policy was already very confident
       (paper Eq. 3: p_old > threshold). Wired via the trainer monkey-patch
       passing `old_log_probs`.

2. **GRPO-style fallback** (MC values absent): group-normalize outcome rewards
   and broadcast uniformly over tokens — equivalent to vanilla GRPO.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est


def _group_normalize_scores(scores: torch.Tensor, index: np.ndarray, eps: float = 1e-6):
    """Returns (mean_per_sample, std_per_sample), each [B]."""
    bsz = scores.shape[0]
    by_idx: dict = defaultdict(list)
    for i in range(bsz):
        by_idx[index[i]].append(scores[i])
    g_mean: dict = {}
    g_std: dict = {}
    for k, vs in by_idx.items():
        if len(vs) == 1:
            g_mean[k] = torch.tensor(0.0, device=scores.device)
            g_std[k] = torch.tensor(1.0, device=scores.device)
        else:
            t = torch.stack(vs)
            g_mean[k] = t.mean()
            g_std[k] = t.std()
    mean_per_sample = torch.stack([g_mean[index[i]] for i in range(bsz)])
    std_per_sample = torch.stack([g_std[index[i]] for i in range(bsz)])
    return mean_per_sample, std_per_sample + eps


@register_adv_est("spo")
def compute_spo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Optional[np.ndarray] = None,
    config=None,
    spo_segment_values: Optional[torch.Tensor] = None,
    spo_segment_cutpoints: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    **kwargs,
):
    """SPO segment advantage.

    Returns:
        advantages, returns: both [B, response_max_length] FloatTensors.
    """
    prob_mask_threshold = 0.9
    if config is not None:
        v = getattr(config, "spo_prob_mask_threshold", None)
        if v is not None:
            prob_mask_threshold = float(v)

    bsz, resp_max_len = token_level_rewards.shape
    advantages = torch.zeros_like(token_level_rewards)

    have_mc = spo_segment_values is not None and spo_segment_cutpoints is not None

    if have_mc:
        # Paper-faithful SPO-chain.
        # spo_segment_values[i] = [V(s_0), V(s_cp1), ..., V(s_cpT), R_i]
        # spo_segment_cutpoints[i] = [0, cp1, ..., cpT, resp_len]
        V = spo_segment_values.float()
        cps = spo_segment_cutpoints

        # Center each row by V(s_0). After this, V_c[:,0]=0 and V_c[:,-1] = R - V(s_0).
        V_c = V - V[:, :1]

        # Group-normalize by R's std for variance reduction (GRPO-comparable scale).
        if index is not None:
            R = V[:, -1]
            _, std_per_sample = _group_normalize_scores(R, index)
            V_norm = V_c / std_per_sample.unsqueeze(-1)
        else:
            V_norm = V_c

        # Segment advantages: A_t = V_norm(s_{cp_t}) - V_norm(s_{cp_{t-1}}) for segment t in [0..T]
        # We broadcast A_t over tokens [cps[t], cps[t+1]).
        for i in range(bsz):
            cp_row = cps[i].tolist()
            vn_row = V_norm[i].tolist()
            for j in range(len(cp_row) - 1):
                start = int(cp_row[j])
                end = int(cp_row[j + 1])
                if end <= start:
                    continue
                end = min(end, resp_max_len)
                if start >= resp_max_len:
                    break
                advantages[i, start:end] = vn_row[j + 1] - vn_row[j]

        # Probability masking (paper Eq. 3).
        if old_log_probs is not None:
            probs = torch.exp(old_log_probs)
            mask = (probs > prob_mask_threshold) & response_mask.bool()
            advantages = advantages.masked_fill(mask, 0.0)
    else:
        # GRPO-style fallback (MC values absent).
        scores = token_level_rewards.sum(dim=-1)
        if index is None:
            index = np.arange(bsz)
        mean_ps, std_ps = _group_normalize_scores(scores, index)
        norm_scores = (scores - mean_ps) / std_ps
        for i in range(bsz):
            resp_len = int(response_mask[i].sum().item())
            if resp_len == 0:
                continue
            advantages[i, :resp_len] = norm_scores[i]

    advantages = advantages * response_mask
    return advantages, advantages
