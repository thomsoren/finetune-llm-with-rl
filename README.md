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

Upstream [verl-project/verl](https://github.com/verl-project/verl) is the trainer. We add three things via verl's existing plugin points, plus one tiny lazy-load patch that lets Ray actors discover our adv estimator:

| Extension | verl plugin point | Lives in |
|---|---|---|
| SPO advantage estimator | `register_adv_est("spo")(fn)` registry | `rl_finetuning/spo.py` |
| SOAP / Hybrid SOAP+AdamW | `actor_rollout_ref.actor.optim.optimizer_impl=<dotted.path>` | `rl_finetuning/soap.py` |
| GSM8K / math_500 / amc_23 reward | `data.custom_reward_function.path/.name` | `rl_finetuning/rewards.py` |

`rl_finetuning/train.py` imports the package in the driver (which registers SPO), then hands off to verl's PPO main. Ray workers are separate processes, so they also need to load the estimator — that's what `apply_spo_patch.py` is for (see the **Why we patch verl** note below).

## Layout

```
rl_finetuning/
  __init__.py     # imports spo.py — registers SPO at import time
  spo.py          # @register_adv_est("spo") compute_spo_outcome_advantage
  soap.py         # SOAP optimizer + HybridSOAPAdamW (1D/embed → AdamW, rest → SOAP)
  rewards.py      # custom compute_score for gsm8k / math_500 / amc_23
  train.py        # entry point: `python -m rl_finetuning.train ...`

apply_spo_patch.py      # idempotent post-install patch — see "Why we patch verl"

vastai_row1.sh          # Row 1: GRPO + AdamW            (50 steps, GSM8K)
vastai_row2.sh          # Row 2: SPO  + AdamW            (50 steps, GSM8K)
vastai_row3.sh          # Row 3: SPO  + HybridSOAPAdamW  (200 steps, GSM8K) — Phase 1
```

`vastai_setup.sh` and `vastai_clean_setup.sh` are kept for reference but the **current setup path uses verl's own install script** directly (see below).

## Running (current, validated)

1. Provision a 4×GPU box. Validated on 4× RTX 3090 24 GB, CUDA 12.8, torch 2.8.0+cu128.
2. Use the system venv (`source /venv/main/bin/activate` on vast.ai images). Don't `uv venv` a fresh one — that's how we lost an afternoon last time.
3. Clone verl and let its own install script handle vllm / sglang / flash-attn:
   ```bash
   git clone --depth 1 https://github.com/volcengine/verl.git /workspace/verl
   cd /workspace/verl
   USE_MEGATRON=0 USE_SGLANG=1 bash scripts/install_vllm_sglang_mcore.sh
   uv pip install --no-deps -e .
   ```
   verl's script downgrades torch to 2.8 (vllm 0.11 / sglang 0.5.2 pin it) and installs flash-attn 2.8.1. **It also bumps numpy to 2.4.x via opencv-fixer; we have to pull it back:**
   ```bash
   uv pip install 'numpy>=2.0,<2.3'
   ```
   otherwise vllm crashes at engine init with `Numba needs NumPy 2.2 or less`.
4. Apply the verl patch (one liner, idempotent):
   ```bash
   python apply_spo_patch.py
   ```
5. Symlink the package so verl's PYTHONPATH=/workspace finds it:
   ```bash
   ln -sfn $(pwd)/rl_finetuning /workspace/rl_finetuning
   ```
6. Download model + data:
   ```bash
   huggingface-cli download Qwen/Qwen2.5-Math-1.5B --local-dir /workspace/rl-finetuning/models/Qwen2.5-Math-1.5B
   python /workspace/verl/examples/data_preprocess/gsm8k.py --local_save_dir /workspace/rl-finetuning/data/gsm8k
   ```
7. Set env and launch:
   ```bash
   export HF_TOKEN=... WANDB_API_KEY=...
   export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1   # required on consumer GPUs — see below
   bash vastai_row1.sh   # then row2, row3
   ```

## Why we patch verl

verl's advantage-estimator registry is a module-level dict populated by `@register_adv_est` decorators at module-import time. Built-ins (`grpo`, `gae`, `rloo`, …) register themselves when `verl.trainer.ppo.core_algos` is imported. Our SPO lives in `rl_finetuning.spo`; the driver process imports it through `rl_finetuning.train`, but **Ray actors are separate Python processes** that import `verl.trainer.main_ppo` directly and never load our package. So `algorithm.adv_estimator=spo` crashes at the first training step:

```
ValueError: Unknown advantage estimator simply: spo
```

(Row 1 worked because `grpo` is a verl built-in.)

Things that don't work cleanly:

- **A `.pth` file in site-packages** that does `import rl_finetuning` at interpreter startup pulls torch in via the verl chain *before* Ray sets `CUDA_VISIBLE_DEVICES` per worker. torch caches all four GPUs; every rank ends up bound to GPU 0 and NCCL dies with `Duplicate GPU detected : rank 1 and rank 0 both on CUDA device 98000`.
- **Ray `runtime_env`** with `py_modules` — verl owns the `ray.init()` call, so we can't pass it our config without forking verl harder.

What `apply_spo_patch.py` does instead is one tiny edit to `verl/trainer/ppo/core_algos.py`:

```python
if name not in ADV_ESTIMATOR_REGISTRY:
    pkg = os.environ.get("VERL_ADV_EST_USER_PKG")
    if pkg:
        try: __import__(pkg)
        except ImportError: pass
```

It fires *only* on a registry miss, which happens after Ray has bound the worker to its GPU and torch has initialized. The row scripts set `VERL_ADV_EST_USER_PKG=rl_finetuning`. Idempotent; safe to re-run.

## NCCL on consumer GPUs (RTX 3090, no NVLink)

Without NVLink, NCCL's default P2P paths fail with `Cuda failure 217 'peer access is not supported between these two devices'`. All three rows need:

```
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
NCCL_IB_DISABLE=1
```

Row scripts inherit them from the env (they don't export defaults — that's deliberate, so you don't carry them onto NVLink boxes).

## Configuration notes

Tuned for 4× 24 GB GPUs:
- `gpu_memory_utilization=0.55`, `tensor_model_parallel_size=1`, `rollout.n=8`
- `data.train_batch_size=32`, `ppo_mini_batch_size=16`
- Dynamic batching (`use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=3072`)
- Ref model offloaded to CPU during rollout
- `save_freq=-1` (no checkpoints — disk on vast.ai instances is tight)

### Row 3 SOAP config (Phase 1)

`HybridSOAPAdamW` splits params by shape since verl hands us a raw iterator with no names:
- 1D params (norms, biases) → `AdamW(lr, β=(0.9, 0.95))`
- 2D params with `max(shape) > 100_000` (Qwen's 152k vocab catches embeddings + lm_head) → AdamW
- Other 2D (attention, MLP weights) → `SOAP(lr, β=(0.95, 0.95), precondition_frequency=100, max_precond_dim=2048, shampoo_beta=-1)`

`max_precond_dim=2048` gives "one-sided" preconditioning automatically: attention (1536×1536) is fully two-sided; MLP layers (1536×8960) skip the 8960 eigendecomp. 150-step linear warmup, `lr_scheduler_type=constant` (flat after warmup), 200 steps total — this is the Phase 1 single-config run to confirm SOAP trains stably. LR sweep is Phase 2 (`[1e-5, 3e-5, 1e-4]`, then β2 and wd around the winner).

KL coefficient is tightened ~2× vs the AdamW baselines (`kl_loss_coef=5e-5` vs `1e-4`) because SOAP makes the policy move faster.

## Results so far

50-step smoke runs on Qwen2.5-Math-1.5B / GSM8K, 4× RTX 3090:

| Row | val acc step 0 | step 25 | step 50 | training reward (step 50) | notes |
|---|---|---|---|---|---|
| 1 GRPO + AdamW | 0.637 | **0.700** (+6.4) | 0.692 (+5.5) | 0.703 | Peaks at step 25 then dips — slight overfit |
| 2 SPO  + AdamW | 0.635 | 0.656 (+2.1) | **0.685** (+5.1) | 0.535 | Slower start, monotonic climb |
| 3 SPO  + Hybrid SOAP+AdamW | _Phase 1 pending_ | | | | |

Both step-50 val numbers are close (~0.69). The shape difference (GRPO peaks early, SPO catches up) is suggestive but 50 steps is too short to draw conclusions — that's why row 3 is Phase 1 at **200 steps**.

## Caveats

`compute_spo_outcome_advantage` in `rl_finetuning/spo.py` does **GRPO-style scalar reward redistribution across segments**, not the per-segment Monte Carlo value estimation from the SPO paper ([arxiv 2505.23564](https://arxiv.org/abs/2505.23564)). For a faithful SPO comparison, the rollout pipeline would need to do `N=9` MC rollouts from each segment-boundary state and use `Â_k = V̂(s_{t_k+1}) − V̂(s_{t_k})`.

## Configuration notes

Tuned for 4× 24 GB GPUs:
- `gpu_memory_utilization=0.55`, `tensor_model_parallel_size=1`, `rollout.n=8`
- `data.train_batch_size=32`, `ppo_mini_batch_size=16`
- Dynamic batching (`use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=3072`)
- Ref model offloaded to CPU during rollout
- `save_freq=-1` (no checkpoints — disk on vast.ai instances is tight)

## Caveats

`compute_spo_outcome_advantage` in `rl_finetuning/spo.py` does **GRPO-style scalar reward redistribution across segments**, not the per-segment Monte Carlo value estimation from the SPO paper ([arxiv 2505.23564](https://arxiv.org/abs/2505.23564)). For a faithful SPO comparison, the rollout pipeline would need to do `N=9` MC rollouts from each segment-boundary state and use `Â_k = V̂(s_{t_k+1}) − V̂(s_{t_k})`.
