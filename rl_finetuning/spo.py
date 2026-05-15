"""SPO (Segment Policy Optimization) advantage estimator.

Registered with verl's adv-estimator registry at import time so that
`algorithm.adv_estimator=spo` dispatches here.

Two execution paths:

1. **Paper-faithful SPO-chain** (default when MC values are available):
   The trainer's `compute_advantage` wrapper (installed in `spo_trainer.py`)
   calls `spo_mc.compute_and_store_mc_values` first, which writes
   `spo_segment_values [B, T+2]` and `spo_segment_cutpoints [B, T+2]` onto the
   batch. This estimator reads them and emits segment advantages
   A_t = V(s_{cp_t}) - V(s_{cp_{t-1}}) broadcast over the tokens of segment t.
   The first slot is V(s_0) = group baseline; the last is the actual outcome
   reward, so the telescoping sum recovers the per-response advantage and
   intermediate V values give per-segment credit.

   Probability masking (paper Eq. 3): tokens whose old policy probability
   exceeds `prob_mask_threshold` get zeroed advantage. This requires the
   trainer wrapper to pass `old_log_probs` through as a kwarg.

2. **GRPO-style fallback** (if the MC fields are absent on the batch — e.g.
   the trainer wrapper was not installed): group-normalize outcome rewards
   and broadcast uniformly over tokens. Equivalent to GRPO.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est


def _group_normalize(scores: torch.Tensor, index: np.ndarray, eps: float = 1e-6) -> torch.Tensor:
    """Group-normalize scalar scores by index (GRPO-style)."""
    bsz = scores.shape[0]
    by_idx: dict = defaultdict(list)
    for i in range(bsz):
        by_idx[index[i]].append(scores[i])
    mean: dict = {}
    std: dict = {}
    for k, vs in by_idx.items():
        if len(vs) == 1:
            mean[k] = torch.tensor(0.0, device=scores.device)
            std[k] = torch.tensor(1.0, device=scores.device)
        else:
            t = torch.stack(vs)
            mean[k] = t.mean()
            std[k] = t.std()
    out = scores.clone()
    for i in range(bsz):
        out[i] = (scores[i] - mean[index[i]]) / (std[index[i]] + eps)
    return out


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
        prob_mask_threshold = float(
            getattr(config, "spo_prob_mask_threshold", prob_mask_threshold) or prob_mask_threshold
        )

    bsz, resp_max_len = token_level_rewards.shape
    advantages = torch.zeros_like(token_level_rewards)

    have_mc = spo_segment_values is not None and spo_segment_cutpoints is not None

    if have_mc:
        # Paper-faithful SPO-chain.
        # spo_segment_values[i, :] = [V(s_0), V(s_cp1), ..., V(s_cpT), R_i]
        # spo_segment_cutpoints[i, :] = [0, cp1, ..., cpT, resp_len]
        # All indexing in token coordinates within the response (0..resp_max_len-1).

        # Optional: group-normalize the segment values across the prompt group so the
        # scale matches GRPO. We normalize the final R column (since V(s_0) is the
        # group baseline; normalizing it would zero it out) and then form differences.
        if index is not None:
            r_col = spo_segment_values[:, -1]
            r_norm = _group_normalize(r_col, index)
            scale = (r_norm.abs().mean() + 1e-6) / (r_col.abs().mean() + 1e-6)
            seg_vals = spo_segment_values.clone()
            # Center each row by V(s_0) and scale so the final value matches normalized R.
            seg_vals = seg_vals - seg_vals[:, :1]
            # Scale uniformly within a row so seg_vals[:, -1] equals r_norm - 0 = r_norm.
            denom = seg_vals[:, -1].abs() + 1e-6
            row_scale = (r_norm.abs() + 1e-6) / denom
            row_sign = torch.sign(r_norm) * torch.sign(seg_vals[:, -1] + 1e-12)
            row_scale = row_scale * row_sign
            seg_vals = seg_vals * row_scale.unsqueeze(-1)
        else:
            seg_vals = spo_segment_values - spo_segment_values[:, :1]

        for i in range(bsz):
            cps = spo_segment_cutpoints[i].tolist()
            vs = seg_vals[i].tolist()
            for j in range(len(cps) - 1):
                start = int(cps[j])
                end = int(cps[j + 1])
                if end <= start:
                    continue
                end = min(end, resp_max_len)
                if start >= resp_max_len:
                    break
                seg_adv = vs[j + 1] - vs[j]
                advantages[i, start:end] = seg_adv

        # Probability masking: zero tokens where old policy was already confident.
        if old_log_probs is not None:
            probs = torch.exp(old_log_probs)
            mask = (probs > prob_mask_threshold) & response_mask.bool()
            advantages = advantages.masked_fill(mask, 0.0)
    else:
        # GRPO-style fallback.
        scores = token_level_rewards.sum(dim=-1)
        if index is None:
            index = np.arange(bsz)
        norm_scores = _group_normalize(scores, index)
        for i in range(bsz):
            resp_len = int(response_mask[i].sum().item())
            if resp_len == 0:
                continue
            advantages[i, :resp_len] = norm_scores[i]

    advantages = advantages * response_mask
    return advantages, advantages
