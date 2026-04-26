"""LangChain callback that accumulates Bedrock token usage by model_id.

Sync handler so it works for both `llm.invoke(...)` (judge) and
`graph.astream_events(...)` (agent) call paths."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


@dataclass
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    calls: int = 0


@dataclass
class ToolCall:
    tool_name: str
    result_tokens_est: int
    result_chars: int


class UsageCollector(BaseCallbackHandler):
    def __init__(self) -> None:
        self.by_model: dict[str, ModelUsage] = defaultdict(ModelUsage)
        self._run_models: dict[UUID, str] = {}
        self.tool_calls: list[ToolCall] = []
        self._tool_starts: dict[UUID, str] = {}

    def on_chat_model_start(
        self, serialized: dict, messages, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._run_models[run_id] = _extract_model_id(serialized, kwargs)

    def on_llm_start(
        self, serialized: dict, prompts, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._run_models[run_id] = _extract_model_id(serialized, kwargs)

    def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, **kwargs: Any
    ) -> None:
        model_id = self._run_models.pop(run_id, "unknown")
        for generations in response.generations:
            for gen in generations:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    self._record(model_id, msg)

    def on_tool_start(
        self, serialized: dict, input_str: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        name = (serialized or {}).get("name") or "unknown"
        self._tool_starts[run_id] = name

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        name = self._tool_starts.pop(run_id, "unknown")
        s = output if isinstance(output, str) else str(output)
        # Rough estimate; tiktoken would be more accurate but adds a dep.
        # ~4 chars per token is the conventional approximation.
        self.tool_calls.append(
            ToolCall(
                tool_name=name,
                result_tokens_est=len(s) // 4,
                result_chars=len(s),
            )
        )

    def _record(self, model_id: str, msg: BaseMessage) -> None:
        meta = getattr(msg, "usage_metadata", None) or {}
        if not meta:
            return
        details = meta.get("input_token_details") or {}
        u = self.by_model[model_id]
        u.input_tokens += meta.get("input_tokens", 0)
        u.output_tokens += meta.get("output_tokens", 0)
        u.cache_read_tokens += details.get("cache_read", 0)
        u.cache_creation_tokens += details.get("cache_creation", 0)
        u.calls += 1


def _extract_model_id(serialized: dict | None, kwargs: dict) -> str:
    if serialized:
        kw = serialized.get("kwargs") or {}
        for key in ("model_id", "model"):
            if kw.get(key):
                return kw[key]
    invocation = kwargs.get("invocation_params") or {}
    for key in ("model_id", "model"):
        if invocation.get(key):
            return invocation[key]
    metadata = kwargs.get("metadata") or {}
    return metadata.get("ls_model_name") or metadata.get("model_id") or "unknown"
