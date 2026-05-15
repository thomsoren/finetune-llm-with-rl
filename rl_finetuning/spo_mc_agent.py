"""SPO Monte Carlo continuation agent loop.

Registered as agent_name "spo_mc_agent" so that verl's AgentLoopManager can
dispatch MC mini-rollouts through it. Unlike single_turn_agent_loop, this loop
bypasses chat templating: it takes pre-tokenized `mc_prompt_ids` (original
prompt + first `cp` response tokens) directly from kwargs and asks vllm to
continue generating from there.

For Qwen-style chat templates this works because the original prompt already
ends inside the assistant turn (no trailing `<|im_end|>` until the model emits
it), so concatenating `prompt_ids + response_ids[:cp]` produces a valid
continuation point.
"""

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("spo_mc_agent")
class SpoMcAgentLoop(AgentLoopBase):
    """SPO MC continuation: skip chat templating, generate from raw prefix."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        prompt_ids = list(kwargs["mc_prompt_ids"])
        mc_response_length = int(kwargs.get("mc_response_length", self.response_length))

        sp = dict(sampling_params)
        sp["max_tokens"] = mc_response_length

        metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sp,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        response_ids = output.token_ids[:mc_response_length]
        response_mask = [1] * len(response_ids)

        result = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=output.log_probs[:mc_response_length] if output.log_probs else None,
            routed_experts=None,
            multi_modal_data={},
            mm_processor_kwargs={},
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields or {},
        )
        result.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return result
