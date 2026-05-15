"""SPO-chain Monte Carlo value estimation.

After the main rollouts finish, this module:
 1. Picks T cutpoints inside each completed response.
 2. For each (sample, cutpoint) pair, asks vllm to generate K continuations
    starting from `prompt + response[:cp]`, calling `LLMServerClient.generate`
    directly (not the agent_loop manager). Going direct avoids the manager's
    `_postprocess`, which pads prompts to `rollout_config.prompt_length` and
    would truncate our extended prefixes.
 3. Decodes each continuation and scores it via the custom reward function
    (the same one configured in `data.custom_reward_function`).
 4. Aggregates K continuations per cutpoint into V(s_{cp}) and writes a
    `[B, T+2]` tensor onto the batch alongside cutpoint positions. Index 0 is
    V(s_0) = group mean reward (value of the empty prefix); the last index is
    R_i (actual outcome reward of the response).

The SPO advantage estimator (`rl_finetuning.spo`) then reads these values and
computes proper TD-style segment advantages A_t = V(s_{cp_t}) - V(s_{cp_{t-1}}).
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import time
from typing import Any, Callable, Optional
from uuid import uuid4

import numpy as np
import torch

from verl import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_REWARD_FN: Optional[Callable] = None
_TOKENIZER = None


def _pick_cutpoints(resp_len: int, n_cutpoints: int) -> list[int]:
    """Return n_cutpoints evenly-spaced positions strictly inside (0, resp_len)."""
    if resp_len <= 1 or n_cutpoints <= 0:
        return []
    spacing = resp_len / (n_cutpoints + 1)
    cps: list[int] = []
    for i in range(n_cutpoints):
        cp = int(round((i + 1) * spacing))
        cp = max(1, min(resp_len - 1, cp))
        if not cps or cp > cps[-1]:
            cps.append(cp)
    return cps


def _extract_prompt_token_ids(batch: DataProto, idx: int) -> list[int]:
    """Pull the unpadded original prompt token ids for sample idx (left-padded convention)."""
    prompts = batch.batch["prompts"][idx]  # [prompt_max_length]
    attn = batch.batch["attention_mask"][idx]  # [prompt_max_length + response_max_length]
    prompt_max_len = prompts.size(0)
    prompt_mask = attn[:prompt_max_len]
    return prompts[prompt_mask.bool()].tolist()


def _extract_response_token_ids(batch: DataProto, idx: int) -> list[int]:
    responses = batch.batch["responses"][idx]
    resp_mask = batch.batch["response_mask"][idx]
    return responses[resp_mask.bool()].tolist()


def _load_reward_fn(trainer) -> Callable:
    """Load the custom_reward_function from config. Cached after first call."""
    global _REWARD_FN
    if _REWARD_FN is not None:
        return _REWARD_FN
    # The reward fn is configured at config.reward.custom_reward_function (path, name).
    cfg = trainer.config
    rcfg = cfg.get("reward", {})
    crf = rcfg.get("custom_reward_function") if rcfg else None
    if crf is None:
        crf = cfg.get("custom_reward_function")
    path = crf.get("path") if crf else None
    name = crf.get("name") if crf else None
    if not path or not name:
        # Fall back to rl_finetuning.rewards.compute_score.
        from rl_finetuning import rewards
        _REWARD_FN = rewards.compute_score
        return _REWARD_FN
    spec = importlib.util.spec_from_file_location("_spo_reward_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _REWARD_FN = getattr(mod, name)
    return _REWARD_FN


def _get_tokenizer(trainer):
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    _TOKENIZER = trainer.tokenizer
    return _TOKENIZER


async def _gen_one(client, prompt_ids: list[int], sampling_params: dict, k: int):
    """K parallel continuations from one prompt. Returns list[TokenOutput]."""
    tasks = [
        client.generate(
            request_id=uuid4().hex,
            prompt_ids=list(prompt_ids),
            sampling_params=sampling_params,
        )
        for _ in range(k)
    ]
    return await asyncio.gather(*tasks)


async def _gen_all(client, mc_prompts: list[list[int]], sampling_params_per_prompt: list[dict], k: int):
    """For each prompt, generate K continuations in parallel.

    Returns: list of list[TokenOutput], outer indexed by prompt.
    """
    tasks = [
        _gen_one(client, mc_prompts[i], sampling_params_per_prompt[i], k)
        for i in range(len(mc_prompts))
    ]
    return await asyncio.gather(*tasks)


def _run_async(coro):
    """Drive an async coroutine from sync code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside an event loop (rare for trainer driver, but be safe).
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def compute_and_store_mc_values(
    main_batch: DataProto,
    trainer,
    *,
    n_cutpoints: int = 2,
    k_continuations: int = 4,
) -> None:
    """Run MC rollouts and write segment values + cutpoint positions onto main_batch.

    Writes:
      - `spo_segment_values`: FloatTensor [B, T+2]
            [0]   = V(s_0) (group mean reward, same across the group)
            [1:T] = MC-estimated V(s_{cp_t}) (mean reward of K continuations)
            [T+1] = R_i (actual outcome reward of the response)
      - `spo_segment_cutpoints`: LongTensor [B, T+2]
            [0]=0, [1:T]=interior cutpoints, [T+1]=response_length.
            For samples with fewer than T real cutpoints, padded slots collapse
            to response_length so adjacent-diff segment advantages are zero.
    """
    response_mask = main_batch.batch["response_mask"]
    bsz, response_max_length = response_mask.shape
    actual_resp_len = response_mask.sum(dim=-1).long()

    cutpoints_per_sample: list[list[int]] = [
        _pick_cutpoints(int(actual_resp_len[i].item()), n_cutpoints) for i in range(bsz)
    ]

    # Flatten: one entry per (sample_i, cutpoint_pos).
    mc_prompts: list[list[int]] = []
    mc_meta: list[tuple[int, int]] = []  # (sample_i, cutpoint_pos)
    mc_sampling_params: list[dict] = []
    mc_extras: list[dict] = []  # for scoring later

    rollout_cfg = trainer.config.actor_rollout_ref.rollout
    base_sampling = dict(
        temperature=float(rollout_cfg.temperature),
        top_p=float(rollout_cfg.top_p),
        top_k=int(rollout_cfg.get("top_k", -1)),
        repetition_penalty=1.0,
    )

    data_source = main_batch.non_tensor_batch.get("data_source")
    reward_model = main_batch.non_tensor_batch.get("reward_model")
    extra_info = main_batch.non_tensor_batch.get("extra_info")

    for i in range(bsz):
        prompt_ids = _extract_prompt_token_ids(main_batch, i)
        response_ids = _extract_response_token_ids(main_batch, i)
        for cp in cutpoints_per_sample[i]:
            prefix_resp = response_ids[:cp]
            mc_prompts.append(prompt_ids + prefix_resp)
            mc_meta.append((i, cp))
            remaining = max(1, response_max_length - cp)
            sp = dict(base_sampling)
            sp["max_tokens"] = remaining
            mc_sampling_params.append(sp)
            mc_extras.append({
                "prefix_resp": prefix_resp,
                "data_source": data_source[i] if data_source is not None else None,
                "reward_model": reward_model[i] if reward_model is not None else None,
                "extra_info": extra_info[i] if extra_info is not None else None,
            })

    # Container for V values keyed by (sample, cutpoint_pos).
    v_at_cp: dict[tuple[int, int], float] = {}

    if mc_prompts:
        t_mc_start = time.time()
        client = trainer.llm_server_manager.get_client()

        trainer.checkpoint_manager.wake_up_replicas()
        try:
            outputs_per_prompt = _run_async(
                _gen_all(client, mc_prompts, mc_sampling_params, k_continuations)
            )
        finally:
            trainer.checkpoint_manager.sleep_replicas()

        # Score each continuation and aggregate K per (sample, cutpoint).
        reward_fn = _load_reward_fn(trainer)
        tokenizer = _get_tokenizer(trainer)

        n_scored = 0
        total_reward = 0.0
        for prompt_idx, outputs in enumerate(outputs_per_prompt):
            sample_i, cp = mc_meta[prompt_idx]
            ex = mc_extras[prompt_idx]
            rm_info = ex["reward_model"] or {}
            ground_truth = rm_info.get("ground_truth") if isinstance(rm_info, dict) else None
            rewards: list[float] = []
            for out in outputs:
                gen_ids = list(out.token_ids)
                # Score against the continuation alone -- matches what the agent_loop
                # scoring pipeline does for the main rollouts (it sees response_ids
                # only, not prompt+response).
                try:
                    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    r = reward_fn(
                        ex["data_source"], text, ground_truth, extra_info=ex["extra_info"]
                    )
                    rewards.append(float(r))
                except Exception as e:
                    logger.warning(f"MC scoring failed (prompt {prompt_idx}): {e}")
                    rewards.append(0.0)
            mean_r = float(np.mean(rewards)) if rewards else 0.0
            v_at_cp[(sample_i, cp)] = mean_r
            n_scored += len(rewards)
            total_reward += float(np.sum(rewards))

        elapsed = time.time() - t_mc_start
        avg = total_reward / max(n_scored, 1)
        logger.info(
            f"SPO MC: {len(mc_prompts)} prefixes x {k_continuations} cont = {n_scored} rollouts, "
            f"avg V={avg:.3f}, took {elapsed:.1f}s"
        )

    # V(s_0) per response: group-mean outcome reward (per-prompt baseline).
    token_level_rewards = main_batch.batch.get("token_level_rewards")
    if token_level_rewards is None:
        token_level_rewards = main_batch.batch.get("token_level_scores")
    outcome_rewards = token_level_rewards.sum(dim=-1).float().cpu().numpy()

    uid = main_batch.non_tensor_batch.get("uid")
    if uid is None:
        group_means = np.full(bsz, float(outcome_rewards.mean()), dtype=np.float32)
    else:
        uid_arr = np.asarray(uid)
        unique_uids, inv = np.unique(uid_arr, return_inverse=True)
        group_sum = np.zeros(len(unique_uids), dtype=np.float64)
        group_count = np.zeros(len(unique_uids), dtype=np.int64)
        np.add.at(group_sum, inv, outcome_rewards)
        np.add.at(group_count, inv, 1)
        group_mean_per_group = group_sum / np.maximum(group_count, 1)
        group_means = group_mean_per_group[inv].astype(np.float32)

    T = n_cutpoints
    seg_values = torch.zeros(bsz, T + 2, dtype=torch.float32)
    seg_cps = torch.zeros(bsz, T + 2, dtype=torch.long)
    for i in range(bsz):
        cps = cutpoints_per_sample[i]
        rl = int(actual_resp_len[i].item())
        seg_cps[i, 0] = 0
        seg_values[i, 0] = float(group_means[i])
        last_v = float(group_means[i])
        for j in range(T):
            if j < len(cps):
                cp = cps[j]
                seg_cps[i, j + 1] = cp
                last_v = v_at_cp.get((i, cp), last_v)
                seg_values[i, j + 1] = last_v
            else:
                seg_cps[i, j + 1] = rl
                seg_values[i, j + 1] = last_v
        seg_cps[i, T + 1] = rl
        seg_values[i, T + 1] = float(outcome_rewards[i])

    main_batch.batch["spo_segment_values"] = seg_values
    main_batch.batch["spo_segment_cutpoints"] = seg_cps
