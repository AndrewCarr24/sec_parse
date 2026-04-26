from typing import Literal

from langchain_core.messages import AIMessage
from loguru import logger

from src.application.orchestrator.workflow.state import AgentState

MAX_TOOL_CALLS_PER_TURN = 16


def route_by_intent(state: AgentState) -> Literal["cache_check", "simple_response"]:
    """Route router output: rag_query -> cache_check, otherwise -> simple_response."""
    intent = state.get("intent", "rag_query")
    if intent == "rag_query":
        return "cache_check"
    return "simple_response"


def route_after_cache(state: AgentState) -> Literal["agent"]:
    """Always route to agent after cache check.
    On a hit the cached answer is injected as context for the agent to evaluate.
    On a miss the agent proceeds with normal RAG."""
    return "agent"


def should_continue(state: AgentState) -> Literal["tools", "finalize", "end"]:
    """ReAct loop decision:
    - last message has tool_calls and budget remains → run tools
    - last message has tool_calls but budget exhausted → finalize (force text answer)
    - last message is a plain AI text response → end
    """
    messages = state.get("messages", [])
    tool_call_count = state.get("tool_call_count", 0)

    if not messages:
        return "end"

    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        if tool_call_count >= MAX_TOOL_CALLS_PER_TURN:
            logger.warning(
                f"Tool call limit ({MAX_TOOL_CALLS_PER_TURN}) reached, routing to finalize"
            )
            return "finalize"
        return "tools"
    return "end"
