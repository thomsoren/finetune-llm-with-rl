"""Apply SPO patches to upstream verl (core_algos.py + ray_trainer.py only)."""
from collections import defaultdict
import os

REPO = "/workspace/verl"

# --- SPO in core_algos.py ---
with open(f"{REPO}/verl/trainer/ppo/core_algos.py") as f:
    c = f.read()

if 'compute_spo_outcome_advantage' not in c:
    import torch
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
        with open(f"{REPO}/verl/trainer/ppo/core_algos.py", 'w') as f:
            f.write(c)
        print("core_algos: SPO added")
    else:
        print("core_algos: ERROR - insertion point not found")
else:
    print("core_algos: already done")

# --- SPO in ray_trainer.py ---
with open(f"{REPO}/verl/trainer/ppo/ray_trainer.py") as f:
    c = f.read()

if "adv_estimator == 'spo'" not in c:
    # Add to __init__
    c = c.replace(
        "elif self.config.algorithm.adv_estimator == 'grpo':\n            self.use_critic = False\n        else:\n            raise NotImplementedError",
        "elif self.config.algorithm.adv_estimator == 'grpo':\n            self.use_critic = False\n        elif self.config.algorithm.adv_estimator == 'spo':\n            self.use_critic = False\n        else:\n            raise NotImplementedError"
    )

    # Add advantage dispatch
    old_d = "        data.batch['advantages'] = advantages\n        data.batch['returns'] = returns\n    else:\n        raise NotImplementedError"
    new_d = (
        "        data.batch['advantages'] = advantages\n"
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
        "        raise NotImplementedError"
    )
    c = c.replace(old_d, new_d)

    with open(f"{REPO}/verl/trainer/ppo/ray_trainer.py", 'w') as f:
        f.write(c)
    print("ray_trainer: SPO added")
else:
    print("ray_trainer: already done")

# Verify
import py_compile
for f in ['verl/trainer/ppo/core_algos.py', 'verl/trainer/ppo/ray_trainer.py']:
    try:
        py_compile.compile(f"{REPO}/{f}", doraise=True)
        print(f"  {f}: OK")
    except Exception as e:
        print(f"  {f}: ERROR - {e}")
