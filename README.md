# finetune-llm-rl

Inspiration from: 
https://github.com/AIFrameResearch/SPO - arxiv.org/abs/2505.23564
Deepseek V4 - uses GPRO for RL-Finetuning

SOAP has only been seen used in pre-training as a better alternative of AdamW.

SOAP is meant to handle noisy environments by using double derivative of loss instead of only single derivative to get gradient. 

Thats why I find it intuitive to research how SOAP will perform on CoT reasoning RL finetuning using state of the art RL finetunings methods like GRPO and SPO. 

The aim of this is to prove that SOAP + SPO is a good method to create state of the art large language models. 

3-row RL fine-tuning experiment on Qwen2.5-Math-1.5B comparing GRPO / SPO / SPO+SOAP for math reasoning.

| Row | Algorithm | Optimizer | Claim |
|-----|-----------|-----------|-------|
| 1   | GRPO      | AdamW     | Baseline |
| 2   | SPO       | AdamW     | Segment-level credit improves over GRPO |
| 3   | SPO       | SOAP      | + curvature-aware optimizer (the thesis) |

## Design

Upstream [verl-project/verl](https://github.com/verl-project/verl) is used unmodified. Our additions plug in via verl's existing extension points:

| Extension | verl plugin point | Lives in |
|---|---|---|
| SPO advantage estimator | `register_adv_est("spo")(fn)` registry | `rl_finetuning/spo.py` |
| SOAP optimizer | `actor_rollout_ref.actor.optim.optimizer_impl=<dotted.path>` | `rl_finetuning/soap.py` |
| math_500 / amc_23 reward | `data.custom_reward_function.path/.name` | `rl_finetuning/rewards.py` |

`rl_finetuning/train.py` is a 3-line wrapper that imports the package (which registers SPO as a side-effect) and then hands off to verl's PPO main. No verl files are patched.

## Layout

```
rl_finetuning/
  __init__.py     # imports spo.py — registers SPO at import time
  spo.py          # @register_adv_est("spo") compute_spo_outcome_advantage
  soap.py         # SOAP optimizer (Shampoo + Adam in eigenbasis)
  rewards.py      # custom compute_score for math_500 / amc_23
  train.py        # entry point: `python -m rl_finetuning.train ...`

vastai_setup.sh         # set up Python env, install verl + deps
vastai_clean_setup.sh   # cleaner re-install variant
apply_spo_patch.py      # legacy patcher (no longer required after refactor)

vastai_row1.sh          # Row 1: GRPO + AdamW   (GSM8K)
vastai_row2.sh          # Row 2: SPO  + AdamW   (GSM8K)
vastai_row3.sh          # Row 3: SPO  + SOAP    (GSM8K)
```

## Running

1. Provision a 4×GPU box (validated on 4× RTX 3090 24 GB, CUDA 12.4+).
2. Clone verl into `/workspace/verl` and `pip install -e . --no-deps --no-build-isolation`. Install torch 2.8 / sglang 0.5.2 / vllm 0.11 / transformers 4.56 / flash-attn 2.8 / hydra-core / omegaconf / ray ≥ 2.55 / tensordict ≥ 0.8 / wandb / datasets / accelerate / peft.
3. Download Qwen2.5-Math-1.5B and a math dataset (the row scripts default to GSM8K; preprocess with `python verl/examples/data_preprocess/gsm8k.py`).
4. `export WANDB_API_KEY=... HF_TOKEN=...`
5. `bash vastai_row1.sh` (then `vastai_row2.sh`, `vastai_row3.sh`).

The row scripts set `PYTHONPATH=/workspace` so `rl_finetuning` resolves, and launch via `python -m rl_finetuning.train`.

## Configuration notes

Tuned for 4× 24 GB GPUs:
- `gpu_memory_utilization=0.55`, `tensor_model_parallel_size=1`, `rollout.n=8`
- `data.train_batch_size=32`, `ppo_mini_batch_size=16`
- Dynamic batching (`use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=3072`)
- Ref model offloaded to CPU during rollout
- `save_freq=-1` (no checkpoints — disk on vast.ai instances is tight)

## Caveats

`compute_spo_outcome_advantage` in `rl_finetuning/spo.py` does **GRPO-style scalar reward redistribution across segments**, not the per-segment Monte Carlo value estimation from the SPO paper ([arxiv 2505.23564](https://arxiv.org/abs/2505.23564)). For a faithful SPO comparison, the rollout pipeline would need to do `N=9` MC rollouts from each segment-boundary state and use `Â_k = V̂(s_{t_k+1}) − V̂(s_{t_k})`.
