"""Monkey-patches that wire SPO-chain MC value estimation into verl's PPO trainer.

Why monkey-patch instead of subclass: verl's `RayPPOTrainer.fit()` is one
multi-hundred-line method with no extension hooks. We need to:
 (a) Hold a reference to the current trainer instance so the patched
     `compute_advantage` can drive MC mini-rollouts via `actor_rollout_wg` /
     `async_rollout_manager`.
 (b) Insert MC value estimation between rollout and advantage computation.
 (c) Pass `old_log_probs`, `spo_segment_values`, `spo_segment_cutpoints` into
     the SPO advantage estimator (the stock dispatcher doesn't pass these).

Call `install()` from `train.py` before invoking verl's `main()`.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from verl.trainer.ppo import ray_trainer as _vt
from verl.trainer.ppo import core_algos as _ca

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_TRAINER_REF = None  # set by patched fit(), read by patched compute_advantage
_ORIG_FIT = None
_ORIG_COMPUTE_ADV = None


def _patched_fit(self, *args, **kwargs):
    global _TRAINER_REF
    _TRAINER_REF = self
    try:
        return _ORIG_FIT(self, *args, **kwargs)
    finally:
        _TRAINER_REF = None


def _patched_compute_advantage(
    data,
    adv_estimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
):
    """Wraps verl's compute_advantage to inject SPO MC value estimation
    and pass extra kwargs (V values + old_log_probs) into the SPO estimator.
    """
    est_name = adv_estimator.value if hasattr(adv_estimator, "value") else str(adv_estimator)

    if est_name != "spo":
        return _ORIG_COMPUTE_ADV(
            data,
            adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    # --- SPO branch: do MC value estimation, then call estimator with extras ---
    import torch
    import numpy as np

    if "response_mask" not in data.batch.keys():
        from verl.trainer.ppo.ray_trainer import compute_response_mask
        data.batch["response_mask"] = compute_response_mask(data)

    n_cutpoints = int(os.environ.get("SPO_N_CUTPOINTS", "2"))
    k_continuations = int(os.environ.get("SPO_K_CONTINUATIONS", "4"))
    enable_mc = os.environ.get("SPO_ENABLE_MC", "1") != "0"

    if enable_mc and _TRAINER_REF is not None:
        try:
            from rl_finetuning.spo_mc import compute_and_store_mc_values
            compute_and_store_mc_values(
                data,
                _TRAINER_REF,
                n_cutpoints=n_cutpoints,
                k_continuations=k_continuations,
            )
        except Exception as e:
            logger.warning(f"SPO MC value estimation failed; falling back to GRPO-style: {e}")
            import traceback
            traceback.print_exc()

    est_fn = _ca.get_adv_estimator_fn(adv_estimator)
    adv_kwargs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "config": config,
    }
    if "uid" in data.non_tensor_batch:
        adv_kwargs["index"] = data.non_tensor_batch["uid"]
    if "spo_segment_values" in data.batch:
        adv_kwargs["spo_segment_values"] = data.batch["spo_segment_values"]
    if "spo_segment_cutpoints" in data.batch:
        adv_kwargs["spo_segment_cutpoints"] = data.batch["spo_segment_cutpoints"]
    if "old_log_probs" in data.batch:
        adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]

    advantages, returns = est_fn(**adv_kwargs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def install() -> None:
    """Idempotently install the SPO trainer patches."""
    global _ORIG_FIT, _ORIG_COMPUTE_ADV
    if _ORIG_FIT is None:
        _ORIG_FIT = _vt.RayPPOTrainer.fit
        _vt.RayPPOTrainer.fit = _patched_fit
    if _ORIG_COMPUTE_ADV is None:
        _ORIG_COMPUTE_ADV = _vt.compute_advantage
        _vt.compute_advantage = _patched_compute_advantage
    logger.info("SPO trainer patches installed.")
