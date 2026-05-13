# finetune-llm-rl

3-row RL fine-tuning experiment on Qwen2.5-Math-1.5B comparing GRPO / SPO / SPO+SOAP for math reasoning.

| Row | Algorithm | Optimizer | Claim |
|-----|-----------|-----------|-------|
| 1   | GRPO      | AdamW     | Baseline |
| 2   | SPO       | AdamW     | Segment-level credit improves over GRPO |
| 3   | SPO       | SOAP      | + curvature-aware optimizer (the thesis) |

## Layout

```
vastai_setup.sh          # Set up Python env, install verl + deps
vastai_clean_setup.sh    # Cleaner re-install version
apply_spo_patch.py       # Applies SPO patch to a fresh verl clone

vastai_row1.sh           # Launch Row 1: GRPO + AdamW
vastai_row2.sh           # Launch Row 2: SPO  + AdamW
vastai_row3.sh           # Launch Row 3: SPO  + SOAP

verl_patches/            # Patched verl files (drop on top of upstream verl)
  trainer/ppo/core_algos.py      # adds compute_spo_outcome_advantage + registry
  utils/optim/soap.py            # SOAP optimizer (Shampoo + Adam in eigenbasis)
  utils/reward_score/__init__.py # adds math_500/amc_23 to reward dispatch
```

## Running

1. Provision a 4×GPU machine (validated on 4× RTX 3090 24 GB, CUDA 12.4+).
2. Clone [verl-project/verl](https://github.com/verl-project/verl), then copy files from `verl_patches/` over the matching paths in the verl tree.
3. Install verl (`pip install -e . --no-deps --no-build-isolation`) plus torch 2.8 / sglang 0.5.2 / vllm 0.11 / transformers 4.56 / flash-attn 2.8 / hydra-core / omegaconf / ray ≥ 2.55 / tensordict ≥ 0.8 / wandb / datasets / accelerate / peft.
4. Download Qwen2.5-Math-1.5B and your math dataset (DAPO, GSM8K, etc.) into `rl-finetuning/`.
5. Export `WANDB_API_KEY` and `HF_TOKEN`, then `bash vastai_row1.sh` (Row 2/3 similar).

Configuration is tuned for 4× 24 GB GPUs:
- `gpu_memory_utilization=0.55`, `tensor_model_parallel_size=1`, `rollout.n=8`
- Dynamic batching (`use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=3072`)
- Ref model offloaded to CPU during rollout

## Caveats

The SPO implementation in `core_algos.py` does **GRPO-style scalar reward redistribution across segments**, not the per-segment Monte Carlo value estimation from the SPO paper ([arxiv 2505.23564](https://arxiv.org/abs/2505.23564)). For a faithful SPO comparison, additional work is needed in the rollout pipeline.
