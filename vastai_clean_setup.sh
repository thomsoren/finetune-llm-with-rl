#!/bin/bash
# ============================================================================
# CLEAN vast.ai Setup v2 — using uv + Python 3.10 venv
# ============================================================================
set -euo pipefail

echo "=== Clean RL Fine-Tuning Setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Disk: $(df -h /workspace | tail -1 | awk '{print $4}') free"
echo ""

# === Step 1: Python 3.10 venv via uv ===
echo "[1/7] Setting up Python 3.10 via uv..."
if [ ! -d "/workspace/rl-venv" ]; then
    uv venv /workspace/rl-venv --python 3.10
fi
source /workspace/rl-venv/bin/activate
echo "  Python: $(python --version)"

# === Step 2: Install PyTorch ===
echo "[2/7] Installing PyTorch..."
uv pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3
python -c "import torch; print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.is_available()}')"

# === Step 3: Clone and install verl ===
echo "[3/7] Installing verl..."
cd /workspace
if [ ! -d "verl" ]; then
    git clone --depth 1 https://github.com/volcengine/verl.git
fi
cd verl

# Install verl deps via their script
chmod +x ./scripts/install_vllm_sglang_mcore.sh
bash ./scripts/install_vllm_sglang_mcore.sh 2>&1 | tail -10

# Install verl itself
uv pip install --no-deps -e . 2>&1 | tail -3

# Extra deps
uv pip install wandb datasets huggingface_hub "math-verify[antlr4_11_0]>=0.6.0" sympy pandas pyarrow accelerate peft tensordict dill codetiming word2number 2>&1 | tail -3

echo "  verl installed"

# === Step 4: Download model ===
echo "[4/7] Downloading Qwen2.5-Math-1.5B..."
mkdir -p /workspace/rl-finetuning/models
if [ ! -f "/workspace/rl-finetuning/models/Qwen2.5-Math-1.5B/config.json" ]; then
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-Math-1.5B',
    local_dir='/workspace/rl-finetuning/models/Qwen2.5-Math-1.5B',
    token='${HF_TOKEN}' if '${HF_TOKEN}' else None)
print('  Model downloaded')
"
else
    echo "  Already exists"
fi

# === Step 5: Download datasets ===
echo "[5/7] Downloading datasets..."
mkdir -p /workspace/rl-finetuning/data
python /workspace/download_data.py 2>/dev/null || python << 'DLDATA'
from datasets import load_dataset
import pandas as pd, os
data_dir = '/workspace/rl-finetuning/data'
if not os.path.exists(f'{data_dir}/dapo_math_17k_train.parquet'):
    ds = load_dataset('BytedTsinghua-SIA/DAPO-Math-17k', split='train')
    ds.to_parquet(f'{data_dir}/dapo_math_17k_train.parquet')
    print(f'  Train: {len(ds)} rows')
else: print('  Train: exists')
for name, path in [('aime_2024','HuggingFaceH4/aime_2024'),('math_500','HuggingFaceH4/MATH-500'),('amc_23','AI-MO/aimo-validation-amc')]:
    if os.path.exists(f'{data_dir}/{name}.parquet'): print(f'  {name}: exists'); continue
    for split in ['test','train']:
        try: d = load_dataset(path, split=split); break
        except: continue
    records = [{'data_source':name,'prompt':[{'role':'user','content':item.get('problem',item.get('question',''))}],'reward_model':{'style':'rule','ground_truth':str(item.get('answer',''))},'extra_info':{'index':''}} for item in d]
    pd.DataFrame(records).to_parquet(f'{data_dir}/{name}.parquet',index=False)
    print(f'  {name}: {len(records)} rows')
DLDATA

# === Step 6: Apply all patches ===
echo "[6/7] Applying SOAP + SPO patches..."
python /workspace/apply_patches.py 2>/dev/null || python << 'PATCHES'
from collections import defaultdict
import re, os
REPO = "/workspace/verl"
os.makedirs(f"{REPO}/verl/utils/optim", exist_ok=True)
open(f"{REPO}/verl/utils/optim/__init__.py", 'w').close()
os.system(f'curl -sL "https://raw.githubusercontent.com/nikhilvyas/SOAP/main/soap.py" -o "{REPO}/verl/utils/optim/soap.py"')
print("  SOAP: downloaded")
with open(f"{REPO}/verl/workers/fsdp_workers.py") as f: c = f.read()
if 'optim_name' not in c:
    old = "            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),\n                                          lr=optim_config.lr,\n                                          betas=optim_config.get('betas', (0.9, 0.999)),\n                                          weight_decay=optim_config.get('weight_decay', 1e-2))"
    new = ("            optim_name = optim_config.get('name', 'adamw')\n"
           "            if optim_name == 'adamw':\n"
           "                actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),\n"
           "                                              lr=optim_config.lr,\n"
           "                                              betas=optim_config.get('betas', (0.9, 0.999)),\n"
           "                                              weight_decay=optim_config.get('weight_decay', 1e-2))\n"
           "            elif optim_name == 'soap':\n"
           "                from verl.utils.optim.soap import SOAP\n"
           "                actor_optimizer = SOAP(actor_module_fsdp.parameters(),\n"
           "                                       lr=optim_config.lr,\n"
           "                                       betas=optim_config.get('betas', (0.95, 0.95)),\n"
           "                                       weight_decay=optim_config.get('weight_decay', 0.01),\n"
           "                                       precondition_frequency=optim_config.get('precondition_frequency', 20),\n"
           "                                       max_precond_dim=optim_config.get('max_precond_dim', 10000),\n"
           "                                       merge_dims=optim_config.get('merge_dims', True))\n"
           "            else:\n"
           "                raise ValueError(f'Unknown optimizer: {optim_name}')")
    if old in c:
        c = c.replace(old, new)
        with open(f"{REPO}/verl/workers/fsdp_workers.py",'w') as f: f.write(c)
        print("  fsdp_workers: patched")
    else: print("  fsdp_workers: exact match not found")
else: print("  fsdp_workers: already done")
import torch
with open(f"{REPO}/verl/trainer/ppo/core_algos.py") as f: c = f.read()
if 'compute_spo_outcome_advantage' not in c:
    spo = ("\n\ndef compute_spo_outcome_advantage(token_level_rewards, eos_mask, index,"
           " log_probs=None, cutpoint_interval=5, prob_mask_threshold=0.9, epsilon=1e-6):\n"
           "    response_length = token_level_rewards.shape[-1]\n"
           "    scores = token_level_rewards.sum(dim=-1)\n"
           "    bsz = scores.shape[0]\n"
           "    with torch.no_grad():\n"
           "        id2score = defaultdict(list)\n"
           "        id2mean = {}\n"
           "        id2std = {}\n"
           "        for i in range(bsz):\n"
           "            id2score[index[i]].append(scores[i])\n"
           "        for idx in id2score:\n"
           "            if len(id2score[idx]) == 1:\n"
           "                id2mean[idx] = torch.tensor(0.0)\n"
           "                id2std[idx] = torch.tensor(1.0)\n"
           "            elif len(id2score[idx]) > 1:\n"
           "                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))\n"
           "                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))\n"
           "        normalized_scores = scores.clone()\n"
           "        for i in range(bsz):\n"
           "            normalized_scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)\n"
           "        advantages = torch.zeros_like(token_level_rewards)\n"
           "        for i in range(bsz):\n"
           "            resp_len = int(eos_mask[i].sum().item())\n"
           "            if resp_len == 0:\n"
           "                continue\n"
           "            norm_reward = normalized_scores[i].item()\n"
           "            cutpoints = list(range(cutpoint_interval - 1, resp_len, cutpoint_interval))\n"
           "            if len(cutpoints) == 0 or cutpoints[-1] != resp_len - 1:\n"
           "                cutpoints.append(resp_len - 1)\n"
           "            n_segs = len(cutpoints)\n"
           "            weights = torch.arange(1, n_segs + 1, dtype=torch.float)\n"
           "            weights = weights / weights.sum()\n"
           "            prev_cp = 0\n"
           "            for k, cp in enumerate(cutpoints):\n"
           "                advantages[i, prev_cp:cp + 1] = norm_reward * weights[k].item() * n_segs\n"
           "                prev_cp = cp + 1\n"
           "            if log_probs is not None:\n"
           "                probs = torch.exp(log_probs[i, :resp_len])\n"
           "                advantages[i, :resp_len][probs > prob_mask_threshold] = 0.0\n"
           "        advantages = advantages * eos_mask\n"
           "    return advantages, advantages\n\n")
    pos = c.find("def compute_rewards(")
    if pos > 0:
        c = c[:pos] + spo + c[pos:]
        with open(f"{REPO}/verl/trainer/ppo/core_algos.py",'w') as f: f.write(c)
        print("  core_algos: SPO added")
else: print("  core_algos: already done")
with open(f"{REPO}/verl/trainer/ppo/ray_trainer.py") as f: c = f.read()
if "adv_estimator == 'spo'" not in c:
    c = c.replace(
        "elif self.config.algorithm.adv_estimator == 'grpo':\n            self.use_critic = False\n        else:\n            raise NotImplementedError",
        "elif self.config.algorithm.adv_estimator == 'grpo':\n            self.use_critic = False\n        elif self.config.algorithm.adv_estimator == 'spo':\n            self.use_critic = False\n        else:\n            raise NotImplementedError")
    old_d = "        data.batch['advantages'] = advantages\n        data.batch['returns'] = returns\n    else:\n        raise NotImplementedError"
    new_d = ("        data.batch['advantages'] = advantages\n"
             "        data.batch['returns'] = returns\n"
             "    elif adv_estimator == 'spo':\n"
             "        token_level_rewards = data.batch['token_level_rewards']\n"
             "        index = data.non_tensor_batch['uid']\n"
             "        responses = data.batch['responses']\n"
             "        response_length = responses.size(-1)\n"
             "        attention_mask = data.batch['attention_mask']\n"
             "        response_mask = attention_mask[:, -response_length:]\n"
             "        advantages, returns = core_algos.compute_spo_outcome_advantage(\n"
             "            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index)\n"
             "        data.batch['advantages'] = advantages\n"
             "        data.batch['returns'] = returns\n"
             "    else:\n"
             "        raise NotImplementedError")
    c = c.replace(old_d, new_d)
    with open(f"{REPO}/verl/trainer/ppo/ray_trainer.py",'w') as f: f.write(c)
    print("  ray_trainer: SPO added")
else: print("  ray_trainer: already done")
import py_compile
for f in ['verl/workers/fsdp_workers.py','verl/trainer/ppo/core_algos.py','verl/trainer/ppo/ray_trainer.py']:
    try: py_compile.compile(f"{REPO}/{f}",doraise=True); print(f"  {f}: OK")
    except Exception as e: print(f"  {f}: ERROR - {e}")
PATCHES

# === Step 7: Verify ===
echo "[7/7] Final verification..."
if [ -n "${WANDB_API_KEY:-}" ]; then
    python -c "import wandb; wandb.login(); print('  wandb: OK')"
fi
python -c "
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/workspace/rl-finetuning/models/Qwen2.5-Math-1.5B', torch_dtype=torch.bfloat16).cuda()
print(f'  Model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B on {torch.cuda.get_device_name(0)}')
del model; torch.cuda.empty_cache()
"
mkdir -p /workspace/rl-finetuning/logs /workspace/rl-finetuning/checkpoints
echo ""
echo "=== SETUP COMPLETE ==="
echo "Run: bash /workspace/vastai_row1.sh"
