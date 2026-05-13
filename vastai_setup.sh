#!/bin/bash
# ============================================================================
# vast.ai Setup — 2x RTX 5090
# ============================================================================
# Run this after SSH into your vast.ai instance.
# Assumes PyTorch + CUDA base image.
# ============================================================================

set -euo pipefail

WORK_DIR="/workspace/rl-finetuning"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

echo "=== RL Fine-Tuning Setup (vast.ai) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

# === 1. Install dependencies ===
echo "[1/6] Installing dependencies..."
pip install --upgrade pip setuptools wheel -q

# Clone simpleRL-reason
if [ ! -d "simpleRL-reason" ]; then
    git clone --branch v1 --depth 1 https://github.com/hkust-nlp/simpleRL-reason.git
fi
cd simpleRL-reason
pip install -e . -q 2>&1 | tail -3

pip install -q \
    "vllm<=0.6.3" \
    "transformers<4.48" \
    "ray[default]>=2.10.0" \
    "hydra-core>=1.3.0" \
    "omegaconf>=2.3.0" \
    "wandb" \
    "datasets" \
    "huggingface_hub" \
    "math-verify[antlr4_11_0]>=0.6.0" \
    "sympy" \
    "pandas" \
    "pyarrow" \
    "flash-attn" --no-build-isolation 2>&1 | tail -5

cd "${WORK_DIR}"

# === 2. Download model ===
echo "[2/6] Downloading Qwen2.5-Math-1.5B..."
if [ ! -f "models/Qwen2.5-Math-1.5B/config.json" ]; then
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-Math-1.5B', local_dir='${WORK_DIR}/models/Qwen2.5-Math-1.5B')
"
else
    echo "  Already downloaded"
fi

# === 3. Download datasets ===
echo "[3/6] Downloading datasets..."
mkdir -p data
python -c "
from datasets import load_dataset
import os

data_dir = '${WORK_DIR}/data'

# Training data (already in verl format)
if not os.path.exists(f'{data_dir}/dapo_math_17k_train.parquet'):
    print('  Downloading DAPO-Math-17k...')
    ds = load_dataset('BytedTsinghua-SIA/DAPO-Math-17k', split='train')
    ds.to_parquet(f'{data_dir}/dapo_math_17k_train.parquet')
    print(f'  Saved: {len(ds)} rows')

# Eval datasets
PROMPT_TEMPLATE = 'Solve the following math problem step by step. The last line of your response should be of the form Answer: \$Answer (without quotes) where \$Answer is the answer to the problem.\n\n{question}\n\nRemember to put your answer on its own line after \"Answer:\".'

import pandas as pd
for name, hf_path in [('aime_2024', 'HuggingFaceH4/aime_2024'), ('math_500', 'HuggingFaceH4/MATH-500'), ('amc_23', 'AI-MO/aimo-validation-amc')]:
    if os.path.exists(f'{data_dir}/{name}.parquet'):
        continue
    print(f'  Downloading {name}...')
    for split in ['test', 'train']:
        try:
            ds = load_dataset(hf_path, split=split)
            break
        except: continue
    records = []
    for item in ds:
        q = item.get('problem', item.get('question', ''))
        a = item.get('answer', item.get('solution', ''))
        records.append({'data_source': name, 'prompt': [{'role': 'user', 'content': PROMPT_TEMPLATE.format(question=q)}], 'reward_model': {'style': 'rule-lighteval/MATH_v2', 'ground_truth': str(a)}, 'extra_info': {'index': item.get('unique_id', item.get('id', ''))}})
    pd.DataFrame(records).to_parquet(f'{data_dir}/{name}.parquet', index=False)
    print(f'  Saved: {len(records)} rows')
print('  Done')
"

# === 4. Apply SOAP optimizer patch ===
echo "[4/6] Applying SOAP optimizer..."
REPO="${WORK_DIR}/simpleRL-reason"
mkdir -p "${REPO}/verl/utils/optim"
touch "${REPO}/verl/utils/optim/__init__.py"

# Download SOAP single-file
if [ ! -f "${REPO}/verl/utils/optim/soap.py" ]; then
    curl -sL "https://raw.githubusercontent.com/nikhilvyas/SOAP/main/soap.py" -o "${REPO}/verl/utils/optim/soap.py"
    # Patch for bf16 compatibility: force fp32 in project/project_back
    python << 'SOAPFIX'
fpath = "/workspace/rl-finetuning/simpleRL-reason/verl/utils/optim/soap.py"
with open(fpath, 'r') as f:
    content = f.read()

# Fix project method: cast to float for tensordot
old_project = "    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):\n        \"\"\"\n        Projects the gradient to the eigenbases of the preconditioner.\n        \"\"\"\n        original_shape = grad.shape"
new_project = "    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):\n        \"\"\"\n        Projects the gradient to the eigenbases of the preconditioner.\n        All projection math in fp32 for numerical stability.\n        \"\"\"\n        original_shape = grad.shape\n        original_dtype = grad.dtype\n        grad = grad.float()"

if old_project in content:
    content = content.replace(old_project, new_project)

# Fix return in project: cast back
old_proj_return = "        return grad\n        \n    def update_preconditioner"
new_proj_return = "        return grad.to(original_dtype)\n        \n    def update_preconditioner"
if old_proj_return in content:
    content = content.replace(old_proj_return, new_proj_return)

# Fix project_back similarly
old_pb = "    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):\n        \"\"\"\n        Projects the gradient back to the original space.\n        \"\"\"\n        original_shape = grad.shape"
new_pb = "    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):\n        \"\"\"\n        Projects the gradient back to the original space.\n        All projection math in fp32 for stability.\n        \"\"\"\n        original_shape = grad.shape\n        original_dtype = grad.dtype\n        grad = grad.float()"
if old_pb in content:
    content = content.replace(old_pb, new_pb)

# Fix return in project_back
old_pb_return = "        return grad\n        \n\n    def get_orthogonal_matrix"
new_pb_return = "        return grad.to(original_dtype)\n        \n\n    def get_orthogonal_matrix"
if old_pb_return in content:
    content = content.replace(old_pb_return, new_pb_return)

# Fix update_preconditioner: cast grad to float for outer products
content = content.replace(
    "                state['GG'][idx].lerp_(outer_product, 1-state['shampoo_beta'])",
    "                state['GG'][idx].lerp_(outer_product.float(), 1-state['shampoo_beta'])"
)

# Fix merge_dims default to True and precondition_frequency to 20
content = content.replace("precondition_frequency: int=10,", "precondition_frequency: int=20,")
content = content.replace("merge_dims: bool = False,", "merge_dims: bool = True,")

with open(fpath, 'w') as f:
    f.write(content)
print("  SOAP patched for bf16 compatibility")
SOAPFIX
fi

# === 5. Patch fsdp_workers.py + core_algos.py + ray_trainer.py ===
echo "[5/6] Patching verl for SOAP + SPO..."

python << 'PATCH_ALL'
from collections import defaultdict
import re

REPO = "/workspace/rl-finetuning/simpleRL-reason"

# --- Patch fsdp_workers.py: add SOAP optimizer dispatch ---
fpath = f"{REPO}/verl/workers/fsdp_workers.py"
with open(fpath, 'r') as f:
    content = f.read()

if 'optim_name' not in content:
    # Find the AdamW optimizer line
    pattern = r'(actor_optimizer = optim\.AdamW\(actor_module_fsdp\.parameters\(\),\s*\n\s*lr=optim_config\.lr,\s*\n\s*betas=optim_config\.get\([^)]+\),\s*\n\s*weight_decay=optim_config\.get\([^)]+\)\))'

    replacement = """# Optimizer dispatch (supports AdamW and SOAP)
            optim_name = optim_config.get('name', 'adamw')
            if optim_name == 'adamw':
                actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                              lr=optim_config.lr,
                                              betas=optim_config.get('betas', (0.9, 0.999)),
                                              weight_decay=optim_config.get('weight_decay', 1e-2))
            elif optim_name == 'soap':
                from verl.utils.optim.soap import SOAP
                actor_optimizer = SOAP(actor_module_fsdp.parameters(),
                                       lr=optim_config.lr,
                                       betas=optim_config.get('betas', (0.95, 0.95)),
                                       weight_decay=optim_config.get('weight_decay', 0.01),
                                       precondition_frequency=optim_config.get('precondition_frequency', 20),
                                       max_precond_dim=optim_config.get('max_precond_dim', 10000),
                                       merge_dims=optim_config.get('merge_dims', True))
            else:
                raise ValueError(f'Unknown optimizer: {optim_name}')"""

    new_content = re.sub(pattern, replacement, content, count=1)
    if new_content != content:
        with open(fpath, 'w') as f:
            f.write(new_content)
        print("  Patched fsdp_workers.py")
    else:
        print("  WARNING: Could not auto-patch fsdp_workers.py (regex mismatch)")
else:
    print("  fsdp_workers.py already patched")

# --- Patch core_algos.py: add SPO advantage function ---
fpath = f"{REPO}/verl/trainer/ppo/core_algos.py"
with open(fpath, 'r') as f:
    content = f.read()

if 'compute_spo_outcome_advantage' not in content:
    spo_fn = '''

def compute_spo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index,
                                   log_probs: torch.Tensor = None,
                                   cutpoint_interval: int = 5,
                                   prob_mask_threshold: float = 0.9,
                                   epsilon: float = 1e-6):
    """SPO: Segment-level credit assignment. Later segments get more credit."""
    response_length = token_level_rewards.shape[-1]
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
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score for index: {idx}")

        normalized_scores = scores.clone()
        for i in range(bsz):
            normalized_scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        advantages = torch.zeros_like(token_level_rewards)
        for i in range(bsz):
            resp_len = int(eos_mask[i].sum().item())
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
                seg_adv = norm_reward * weights[k].item() * n_segs
                advantages[i, prev_cp:cp + 1] = seg_adv
                prev_cp = cp + 1
            if log_probs is not None:
                probs = torch.exp(log_probs[i, :resp_len])
                high_prob = probs > prob_mask_threshold
                advantages[i, :resp_len][high_prob] = 0.0
        advantages = advantages * eos_mask
    return advantages, advantages

'''
    insert_at = content.find("def compute_rewards(")
    if insert_at > 0:
        content = content[:insert_at] + spo_fn + "\n" + content[insert_at:]
        with open(fpath, 'w') as f:
            f.write(content)
        print("  Patched core_algos.py with SPO")
    else:
        print("  WARNING: Could not find insertion point in core_algos.py")
else:
    print("  core_algos.py already has SPO")

# --- Patch ray_trainer.py: register SPO estimator ---
fpath = f"{REPO}/verl/trainer/ppo/ray_trainer.py"
with open(fpath, 'r') as f:
    content = f.read()

if "adv_estimator == 'spo'" not in content:
    # Add to __init__
    content = content.replace(
        "elif self.config.algorithm.adv_estimator == 'grpo':\n            self.use_critic = False\n        else:\n            raise NotImplementedError",
        "elif self.config.algorithm.adv_estimator == 'grpo':\n            self.use_critic = False\n        elif self.config.algorithm.adv_estimator == 'spo':\n            self.use_critic = False\n        else:\n            raise NotImplementedError"
    )

    # Add advantage dispatch
    old_dispatch = """        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError"""

    new_dispatch = """        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'spo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        log_probs = data.batch.get('old_log_probs', None)
        advantages, returns = core_algos.compute_spo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
            log_probs=log_probs)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError"""

    content = content.replace(old_dispatch, new_dispatch)

    with open(fpath, 'w') as f:
        f.write(content)
    print("  Patched ray_trainer.py with SPO")
else:
    print("  ray_trainer.py already has SPO")

# Verify all compile
import py_compile
for f in ['verl/workers/fsdp_workers.py', 'verl/trainer/ppo/core_algos.py', 'verl/trainer/ppo/ray_trainer.py']:
    try:
        py_compile.compile(f"{REPO}/{f}", doraise=True)
        print(f"  {f}: OK")
    except Exception as e:
        print(f"  {f}: COMPILE ERROR - {e}")
PATCH_ALL

# === 6. Setup wandb ===
echo "[6/6] Setting up wandb..."
export WANDB_API_KEY="${WANDB_API_KEY:-}"
if [ -n "${WANDB_API_KEY}" ]; then
    python -c "import wandb; wandb.login(); print('  wandb: OK')"
else
    echo "  Set WANDB_API_KEY before training: export WANDB_API_KEY=your_key"
fi

echo ""
echo "=== SETUP COMPLETE ==="
echo ""
echo "To train:"
echo "  cd ${WORK_DIR}/simpleRL-reason"
echo "  bash /workspace/rl-finetuning/run_row1.sh"
echo "  bash /workspace/rl-finetuning/run_row2.sh"
echo "  bash /workspace/rl-finetuning/run_row3.sh"
