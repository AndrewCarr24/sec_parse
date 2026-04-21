"""LangGraph workflow: router + ReAct agent with RAG tool + memory post-hook."""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from src.application.orchestrator.workflow.edges import (
    route_by_intent,
    should_continue,
)
from src.application.orchestrator.workflow.nodes import (
    agent_node,
    cache_check_node,
    finalize_node,
    memory_post_hook,
    router_node,
    simple_response_node,
)
from src.application.orchestrator.workflow.state import AgentState
from src.application.orchestrator.workflow.tools import get_tools
from src.config import settings

_graph_instance = None


def create_graph(force_recreate: bool = False):
    """
    Build the agent graph.

        START -> router_node -> [intent?]
                                  ├── rag_query  -> agent_node <-> tool_node
                                  └── simple/off -> simple_response_node
                                                           │
                                                  memory_post_hook -> END
    """
    global _graph_instance
    if _graph_instance is not None and not force_recreate:
        return _graph_instance

    logger.info("Creating RAG agent graph (router + ReAct + memory)")

    builder = StateGraph(AgentState)
    builder.add_node("router_node", router_node)
    builder.add_node("cache_check_node", cache_check_node)
    builder.add_node("agent_node", agent_node)
    builder.add_node("simple_response_node", simple_response_node)
    builder.add_node("finalize_node", finalize_node)
    builder.add_node("tool_node", ToolNode(get_tools()))
    builder.add_node("memory_post_hook", memory_post_hook)

    builder.add_edge(START, "router_node")
    builder.add_conditional_edges(
        "router_node",
        route_by_intent,
        {"cache_check": "cache_check_node", "simple_response": "simple_response_node"},
    )
    builder.add_edge("cache_check_node", "agent_node")
    builder.add_conditional_edges(
        "agent_node",
        should_continue,
        {
            "tools": "tool_node",
            "finalize": "finalize_node",
            "end": "memory_post_hook",
        },
    )
    builder.add_edge("tool_node", "agent_node")
    builder.add_edge("finalize_node", "memory_post_hook")
    builder.add_edge("simple_response_node", "memory_post_hook")
    builder.add_edge("memory_post_hook", END)

    if settings.MEMORY_ID:
        from src.infrastructure.memory import ShortTermMemory
        checkpointer = ShortTermMemory().get_memory()
        _graph_instance = builder.compile(checkpointer=checkpointer)
    else:
        logger.warning("MEMORY_ID unset — running without checkpointer or memory hook")
        _graph_instance = builder.compile()
    logger.info("RAG agent graph compiled")
    return _graph_instance


def reset_graph() -> None:
    global _graph_instance
    _graph_instance = None
