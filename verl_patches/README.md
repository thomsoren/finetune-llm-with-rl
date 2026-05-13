# verl patches

These files override their upstream counterparts in [verl-project/verl](https://github.com/verl-project/verl). After cloning verl, copy each file into the matching path:

```
verl_patches/trainer/ppo/core_algos.py        → verl/verl/trainer/ppo/core_algos.py
verl_patches/utils/optim/soap.py              → verl/verl/utils/optim/soap.py
verl_patches/utils/reward_score/__init__.py   → verl/verl/utils/reward_score/__init__.py
```

## What each patch does

**`trainer/ppo/core_algos.py`** — adds `compute_spo_outcome_advantage`, registered via `@register_adv_est("spo")`. Group-normalizes scalar reward (GRPO-style), then distributes across response segments with linearly increasing weights (later tokens get more credit). Optional probability-masking on high-probability tokens if `old_log_probs` is provided.

**`utils/optim/soap.py`** — single-file SOAP optimizer from [nikhilvyas/SOAP](https://github.com/nikhilvyas/SOAP) (Shampoo + Adam in eigenbasis). Importable as `verl.utils.optim.soap.SOAP`. Use it by passing `actor_rollout_ref.actor.optim.optimizer=SOAP` and `optimizer_impl=verl.utils.optim.soap`.

**`utils/reward_score/__init__.py`** — extends `default_compute_score`'s math-style branch to recognize `data_source ∈ {"math_500", "amc_23"}`. Without this, eval on those parquet files raises `NotImplementedError`.

## Verified against verl

verl HEAD as of 2026-05 (verl-0.8.0.dev0). The registry/dispatch pattern these patches use was introduced upstream sometime in 2025 — older verl forks won't have `register_adv_est`.
