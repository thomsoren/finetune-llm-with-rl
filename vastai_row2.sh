#!/usr/bin/env bash
# Row 2: SPO + AdamW (segment-level credit assignment)
set -uo pipefail
export PYTHONUNBUFFERED=1

: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set}"
export WANDB_PROJECT="${WANDB_PROJECT:-rl-finetuning-thesis}"

mkdir -p /workspace/rl-finetuning/logs /workspace/rl-finetuning/checkpoints/row2

ray stop --force 2>/dev/null || true

MODEL=/workspace/rl-finetuning/models/Qwen2.5-Math-1.5B
TRAIN=/workspace/rl-finetuning/data/gsm8k/train.parquet
VAL=/workspace/rl-finetuning/data/gsm8k/test.parquet

/venv/main/bin/python -m verl.trainer.main_ppo \
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
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
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
    trainer.experiment_name=row2_spo_adamw_gsm8k \
    trainer.logger='[console,wandb]' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=25 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=50 \
    trainer.default_local_dir=/workspace/rl-finetuning/checkpoints/row2 \
    2>&1 | tee /workspace/rl-finetuning/logs/row2.log

ray stop --force 2>/dev/null || true
echo "Row 2 done: $(date)"
