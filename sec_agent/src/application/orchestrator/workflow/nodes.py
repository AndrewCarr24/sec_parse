"""LangGraph nodes: router, cache check, agent (ReAct), simple response, memory post-hook."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger

from src.application.orchestrator.workflow.chains import (
    get_agent_chain,
    get_finalize_chain,
    get_router_chain,
    get_simple_response_chain,
    with_cache_on_last,
)
from src.application.orchestrator.workflow.state import AgentState, IntentType
from src.config import settings
from src.infrastructure.model import extract_text_content


async def router_node(state: AgentState, config: RunnableConfig) -> dict:
    """Classify intent and store it on state."""
    messages = list(state["messages"])
    chain = get_router_chain()
    response = await chain.ainvoke({"messages": messages}, config)
    text = extract_text_content(response.content).strip().lower()

    if "rag_query" in text:
        intent: IntentType = "rag_query"
    elif "off_topic" in text:
        intent = "off_topic"
    elif "simple" in text:
        intent = "simple"
    else:
        logger.warning(f"Unclear intent: {text!r}, defaulting to rag_query")
        intent = "rag_query"

    logger.info(f"Router classified intent: {intent}")
    return {"intent": intent}


async def cache_check_node(state: AgentState, config: RunnableConfig) -> dict:
    """No-op placeholder. The old embedding-based answer cache lived in
    the retired `rag_app` package; we kept the node in the graph so the
    topology stays stable while the facts-DB tool path matures."""
    return {"cache_hit": False}


async def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """Run the ReAct agent for rag_query intents."""
    messages = list(state["messages"])
    tool_call_count = state.get("tool_call_count", 0)

    configurable = config.get("configurable", {})
    customer_name = configurable.get("customer_name", "Guest")

    chain = get_agent_chain(customer_name=customer_name)
    response = await chain.ainvoke(
        {"messages": with_cache_on_last(messages)}, config
    )

    has_tool_calls = bool(getattr(response, "tool_calls", None))
    new_count = tool_call_count + (len(response.tool_calls) if has_tool_calls else 0)
    logger.debug(
        f"agent_node: has_tool_calls={has_tool_calls}, tool_call_count={new_count}"
    )
    return {"messages": response, "tool_call_count": new_count}


async def finalize_node(state: AgentState, config: RunnableConfig) -> dict:
    """Force a text answer after the ReAct tool budget is exhausted.

    Collapses the tool-call/tool-result message pairs into plain
    HumanMessages so Bedrock doesn't require a toolConfig, then asks
    the model (without tools) to synthesize a final answer.
    """
    from langchain_core.messages import ToolMessage

    raw_messages = list(state["messages"])
    condensed = []
    for msg in raw_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            continue
        if isinstance(msg, ToolMessage):
            condensed.append(HumanMessage(
                content=f"[Tool result for '{msg.name}']\n{msg.content}"
            ))
            continue
        condensed.append(msg)

    logger.debug(f"finalize_node: condensed {len(raw_messages)} msgs → {len(condensed)}")

    configurable = config.get("configurable", {})
    customer_name = configurable.get("customer_name", "Guest")

    chain = get_finalize_chain(customer_name=customer_name)
    response = await chain.ainvoke({"messages": condensed}, config)
    logger.info("finalize_node: produced fallback answer")
    return {"messages": response}


async def simple_response_node(state: AgentState, config: RunnableConfig) -> dict:
    """Handle greetings/thanks/off-topic without tools."""
    messages = list(state["messages"])
    configurable = config.get("configurable", {})
    customer_name = configurable.get("customer_name", "Guest")

    chain = get_simple_response_chain(customer_name=customer_name)
    response = await chain.ainvoke({"messages": messages}, config)
    return {"messages": response}


async def memory_post_hook(state: AgentState, config: RunnableConfig) -> dict:
    """Save the user/agent turn to AgentCore Memory strategies."""
    if not settings.MEMORY_ID:
        return {}

    from src.infrastructure.memory import get_memory_instance

    configurable = config.get("configurable", {})
    actor_id = configurable.get("actor_id", "user:default")
    session_id = configurable.get("thread_id", "default_session")

    messages = state.get("messages", [])
    user_input = ""
    agent_response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not agent_response:
            if msg.content and not msg.tool_calls:
                agent_response = extract_text_content(msg.content)
        elif isinstance(msg, HumanMessage) and not user_input:
            user_input = extract_text_content(msg.content)
        if user_input and agent_response:
            break

    if not user_input or not agent_response:
        logger.debug("memory_post_hook: missing input or response, skipping")
        return {}

    try:
        memory = get_memory_instance()
        result = memory.process_turn(
            actor_id=actor_id,
            session_id=session_id,
            user_input=user_input,
            agent_response=agent_response,
        )
        if not result.get("success"):
            logger.warning(f"memory process_turn error: {result.get('error')}")
    except Exception as e:
        logger.error(f"memory_post_hook failed: {e}")

    return {}
