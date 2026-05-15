"""SPO-chain Monte Carlo value estimation.

After the main rollouts finish, this module:
 1. Picks T cutpoints inside each completed response.
 2. Builds a batch of mini-rollouts that continue from each (prompt + response[:cp])
    prefix, using the `spo_mc_agent` agent loop to bypass chat templating.
 3. Submits the mini-rollouts back through `async_rollout_manager.generate_sequences`.
 4. Scores each continuation with the same custom reward function the main rollouts
    use, treating the prefix's text + continuation text as the full solution string.
 5. Aggregates K continuations per cutpoint into V(s_{cp}) and writes a `[B, T+2]`
    tensor onto the batch alongside the cutpoint positions, where index 0 is V(s_0)
    (group mean reward — value of the empty prefix) and the last index is the
    actual outcome reward R_i.

The SPO advantage estimator (`rl_finetuning.spo`) then reads these values and
computes proper TD-style segment advantages A_t = V(s_{cp_t}) - V(s_{cp_{t-1}}).
"""

import logging
import os
import time
from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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
    """Pull the unpadded original prompt token ids for sample idx."""
    # `prompts` is the prompt-only tensor (left-padded). attention_mask covers
    # prompts+responses; the prompt portion is the first prompt_max_length cols.
    prompts = batch.batch["prompts"][idx]  # [prompt_max_length]
    attn = batch.batch["attention_mask"][idx]  # [prompt_max_length + response_max_length]
    prompt_max_len = prompts.size(0)
    prompt_mask = attn[:prompt_max_len]
    keep = prompts[prompt_mask.bool()].tolist()
    return keep


def _extract_response_token_ids(batch: DataProto, idx: int) -> list[int]:
    responses = batch.batch["responses"][idx]  # [response_max_length]
    resp_mask = batch.batch["response_mask"][idx]  # [response_max_length]
    return responses[resp_mask.bool()].tolist()


def _build_mc_dataproto(
    main_batch: DataProto,
    cutpoints_per_sample: list[list[int]],
    k_continuations: int,
    response_max_length: int,
) -> tuple[DataProto, list[tuple[int, int, int]]]:
    """Construct a DataProto of MC mini-rollouts.

    Returns:
        mc_batch: DataProto with one row per (sample_i, cutpoint_j, continuation_k)
        index: list of (sample_i, cutpoint_j, k) tuples for each row, for de-aliasing.
    """
    from tensordict import TensorDict

    rows_mc_prompt_ids: list[list[int]] = []
    rows_mc_response_length: list[int] = []
    rows_agent_name: list[str] = []
    rows_data_source: list[Any] = []
    rows_reward_model: list[Any] = []
    rows_extra_info: list[Any] = []
    rows_uid: list[str] = []
    rows_prefix_response_ids: list[list[int]] = []
    index: list[tuple[int, int, int]] = []

    bsz = main_batch.batch["prompts"].size(0)
    data_source = main_batch.non_tensor_batch.get("data_source")
    reward_model = main_batch.non_tensor_batch.get("reward_model")
    extra_info = main_batch.non_tensor_batch.get("extra_info")
    uids = main_batch.non_tensor_batch.get("uid")

    for i in range(bsz):
        prompt_ids = _extract_prompt_token_ids(main_batch, i)
        response_ids = _extract_response_token_ids(main_batch, i)
        for j, cp in enumerate(cutpoints_per_sample[i]):
            prefix_resp = response_ids[:cp]
            mc_prompt = prompt_ids + prefix_resp
            remaining = max(1, response_max_length - cp)
            for k in range(k_continuations):
                rows_mc_prompt_ids.append(mc_prompt)
                rows_mc_response_length.append(remaining)
                rows_agent_name.append("spo_mc_agent")
                rows_data_source.append(data_source[i] if data_source is not None else None)
                rows_reward_model.append(reward_model[i] if reward_model is not None else None)
                rows_extra_info.append(extra_info[i] if extra_info is not None else None)
                rows_uid.append(f"{uids[i] if uids is not None else i}__mc_{j}_{k}")
                rows_prefix_response_ids.append(prefix_resp)
                index.append((i, j, k))

    n_rows = len(rows_mc_prompt_ids)
    if n_rows == 0:
        return None, index

    # Pad mc_prompt_ids to a uniform length for the tensor part of DataProto.
    # The actual prompt sent to vllm is taken from non_tensor_batch["mc_prompt_ids"],
    # but DataProto.batch must contain something; we use input_ids/attention_mask
    # placeholders so the framework's bookkeeping works.
    pad_id = 0
    max_prompt_len = max(len(p) for p in rows_mc_prompt_ids)
    input_ids = torch.full((n_rows, max_prompt_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n_rows, max_prompt_len), dtype=torch.long)
    for r, p in enumerate(rows_mc_prompt_ids):
        L = len(p)
        # left-pad to match verl convention
        input_ids[r, max_prompt_len - L:] = torch.tensor(p, dtype=torch.long)
        attention_mask[r, max_prompt_len - L:] = 1
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0) * attention_mask

    td = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=n_rows,
    )
    non_tensor = {
        "mc_prompt_ids": np.array(rows_mc_prompt_ids, dtype=object),
        "mc_response_length": np.array(rows_mc_response_length, dtype=np.int64),
        "agent_name": np.array(rows_agent_name, dtype=object),
        "data_source": np.array(rows_data_source, dtype=object),
        "reward_model": np.array(rows_reward_model, dtype=object),
        "extra_info": np.array(rows_extra_info, dtype=object),
        "uid": np.array(rows_uid, dtype=object),
        # store the prefix tokens so we can reconstruct full text for scoring
        "_prefix_response_ids": np.array(rows_prefix_response_ids, dtype=object),
    }
    mc_batch = DataProto.from_dict(tensors=td, non_tensors=non_tensor)
    mc_batch.meta_info = {
        "temperature": main_batch.meta_info.get("temperature", 1.0),
        "global_steps": main_batch.meta_info.get("global_steps", 0),
    }
    return mc_batch, index


def _extract_mc_rewards(mc_output: DataProto) -> np.ndarray:
    """The agent_loop pipeline already scores each generated response and stores
    the result in `rm_scores` (per-token, with the scalar reward placed on the last
    response token). We just sum to get the per-response scalar.

    Note: rm_scores reflects the reward of the *continuation alone*, not the
    full (prefix + continuation). For math problems whose reward is "does the
    final \\boxed{} match ground truth", that's exactly what we want for
    V(s_{cp}): the value of starting from prefix s_{cp} is the probability the
    policy completes the rest correctly.
    """
    rm_scores = mc_output.batch.get("rm_scores")
    if rm_scores is None:
        # No automatic scoring happened — return zeros.
        n = mc_output.batch["responses"].size(0)
        return np.zeros(n, dtype=np.float32)
    return rm_scores.sum(dim=-1).float().cpu().numpy()


def compute_and_store_mc_values(
    main_batch: DataProto,
    trainer,
    *,
    n_cutpoints: int = 2,
    k_continuations: int = 4,
) -> None:
    """Run MC rollouts and write segment values + cutpoint positions onto main_batch.

    Writes the following onto `main_batch.batch`:
      - `spo_segment_values`: FloatTensor [B, T+2]
            Index 0 = V(s_0) (group mean reward, same across the group).
            Indices 1..T = MC-estimated V(s_{cp_t}) (mean of K continuations).
            Index T+1   = R_i (actual outcome reward of the response).
      - `spo_segment_cutpoints`: LongTensor [B, T+2]
            Index 0 = 0, indices 1..T = interior cutpoints, index T+1 = response length.
            Padded with response_length for samples that had fewer real cutpoints.
    """
    response_mask = main_batch.batch["response_mask"]  # [B, response_max_length]
    response_max_length = response_mask.size(1)
    bsz = response_mask.size(0)
    actual_resp_len = response_mask.sum(dim=-1).long()  # [B]

    # Pick cutpoints per sample. Sample with len<=1 gets no interior cutpoints.
    cutpoints_per_sample = [
        _pick_cutpoints(int(actual_resp_len[i].item()), n_cutpoints) for i in range(bsz)
    ]

    # Build MC batch (one row per sample × cutpoint × continuation).
    mc_batch, mc_index = _build_mc_dataproto(
        main_batch, cutpoints_per_sample, k_continuations, response_max_length
    )

    # Containers for V values keyed by (sample, cutpoint_pos).
    v_at_cp: dict[tuple[int, int], list[float]] = {}

    if mc_batch is not None:
        t_mc_start = time.time()
        # Wake replicas, generate, sleep again.
        trainer.checkpoint_manager.wake_up_replicas()
        try:
            mc_output: DataProto = trainer.async_rollout_manager.generate_sequences(mc_batch)
        finally:
            trainer.checkpoint_manager.sleep_replicas()

        # Agent loop already scored each continuation via its built-in scoring
        # pipeline (rm_scores is populated). Just read those.
        mc_rewards = _extract_mc_rewards(mc_output)

        # Aggregate K continuations per (sample, cutpoint).
        bucket: dict[tuple[int, int], list[float]] = {}
        for r, (i, j, k) in enumerate(mc_index):
            bucket.setdefault((i, j), []).append(float(mc_rewards[r]))
        for (i, j), vs in bucket.items():
            cp_pos = cutpoints_per_sample[i][j]
            v_at_cp[(i, cp_pos)] = float(np.mean(vs))

        logger.info(
            f"SPO MC: {len(mc_index)} continuations across {bsz} responses, "
            f"avg V={float(np.mean(mc_rewards)):.3f}, took {time.time() - t_mc_start:.1f}s"
        )

    # Compute V(s_0) per response. Use the group mean reward (per-prompt baseline).
    # Group is identified by `uid` in non_tensor_batch (this is the prompt UID, shared
    # by the n rollouts from that prompt).
    token_level_rewards = main_batch.batch.get("token_level_rewards")
    if token_level_rewards is None:
        token_level_rewards = main_batch.batch.get("token_level_scores")
    outcome_rewards = token_level_rewards.sum(dim=-1).float().cpu().numpy()  # [B]

    uid = main_batch.non_tensor_batch.get("uid")
    if uid is None:
        # No grouping info; use global mean as baseline (degenerate case).
        group_means = np.full(bsz, float(outcome_rewards.mean()), dtype=np.float32)
    else:
        # Average outcome_rewards per uid group.
        uid_arr = np.asarray(uid)
        unique_uids, inv = np.unique(uid_arr, return_inverse=True)
        group_sum = np.zeros(len(unique_uids), dtype=np.float64)
        group_count = np.zeros(len(unique_uids), dtype=np.int64)
        np.add.at(group_sum, inv, outcome_rewards)
        np.add.at(group_count, inv, 1)
        group_mean_per_group = group_sum / np.maximum(group_count, 1)
        group_means = group_mean_per_group[inv].astype(np.float32)

    # Build [B, T+2] tensors. We use T = n_cutpoints fixed; for samples with fewer
    # actual interior cutpoints, we pad with response_length and copy V from the last
    # known cutpoint (so adjacent diffs are zero for the padded slots).
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
                # padded slot: collapse to the response end
                seg_cps[i, j + 1] = rl
                seg_values[i, j + 1] = last_v
        seg_cps[i, T + 1] = rl
        seg_values[i, T + 1] = float(outcome_rewards[i])

    main_batch.batch["spo_segment_values"] = seg_values
    main_batch.batch["spo_segment_cutpoints"] = seg_cps
