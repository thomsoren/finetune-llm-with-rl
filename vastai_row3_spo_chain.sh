#!/usr/bin/env bash
# Paper-faithful SPO-chain (with Kron optimizer) on GSM8K.
#
# What's different from vastai_row3_kron.sh:
#   - SPO_ENABLE_MC=1 turns on the MC value-estimation hook installed by
#     rl_finetuning.spo_trainer. Each step the trainer now:
#       * picks SPO_N_CUTPOINTS=2 cutpoints per response,
#       * generates SPO_K_CONTINUATIONS=4 vllm continuations from each prefix,
#       * scores them with the custom reward function,
#       * writes V(s_{cp}) tensors onto the batch, and
#       * the SPO advantage estimator emits TD-style segment advantages
#         A_t = V(s_{cp_t}) - V(s_{cp_{t-1}}) instead of GRPO broadcast.
#   - Probability masking (paper Eq. 3) is wired through `old_log_probs`.
#   - Per-step compute is ~30-40% slower than plain Kron due to the
#     MC mini-rollouts (each one is shorter than a full main rollout).
set -uo pipefail
export PYTHONUNBUFFERED=1

: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set}"
export WANDB_PROJECT="${WANDB_PROJECT:-rl-finetuning-thesis}"
export PYTHONPATH="/workspace:${PYTHONPATH:-}"
export VERL_ADV_EST_USER_PKG="${VERL_ADV_EST_USER_PKG:-rl_finetuning}"

# SPO-chain knobs (read by rl_finetuning.spo_trainer._patched_compute_advantage)
export SPO_ENABLE_MC="${SPO_ENABLE_MC:-1}"
export SPO_N_CUTPOINTS="${SPO_N_CUTPOINTS:-2}"
export SPO_K_CONTINUATIONS="${SPO_K_CONTINUATIONS:-4}"

mkdir -p /workspace/rl-finetuning/logs /workspace/rl-finetuning/checkpoints/row3_spo_chain

ray stop --force 2>/dev/null || true

MODEL=/workspace/rl-finetuning/models/Qwen2.5-Math-1.5B
TRAIN=/workspace/rl-finetuning/data/gsm8k/train.parquet
VAL=/workspace/rl-finetuning/data/gsm8k/test.parquet

/venv/main/bin/python -m rl_finetuning.train \
    algorithm.adv_estimator=spo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    data.truncation=error \
    custom_reward_function.path=/workspace/rl_finetuning/rewards.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.optimizer=HybridKronAdamW \
    actor_rollout_ref.actor.optim.optimizer_impl=rl_finetuning.kron \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.betas='[0.9,0.999]' \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.override_optimizer_config='{max_size_triangular:2048,min_ndim_triangular:2,precond_lr:0.1,merge_dims:false}' \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    trainer.project_name=rl-finetuning-thesis \
    trainer.experiment_name=row3_spo_chain_kron_gsm8k \
    trainer.logger='[console,wandb]' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=200 \
    trainer.default_local_dir=/workspace/rl-finetuning/checkpoints/row3_spo_chain \
    2>&1 | tee /workspace/rl-finetuning/logs/row3_spo_chain.log

ray stop --force 2>/dev/null || true
echo "Row 3 SPO-chain done: $(date)"
