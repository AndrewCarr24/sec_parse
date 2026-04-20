"""Run the graph and yield only the final-answer tokens to the user."""

import re
from typing import AsyncGenerator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from loguru import logger

from src.application.orchestrator.workflow.graph import create_graph
from src.infrastructure.model import extract_text_content

_RESPONSE_NODES = {"agent_node", "simple_response_node", "finalize_node", "cache_check_node"}


async def get_streaming_response(
    messages: str,
    customer_name: str = "Guest",
    conversation_id: str | None = None,
    callbacks: list | None = None,
) -> AsyncGenerator[str, None]:
    graph = create_graph()
    thread_id = conversation_id or "default-thread"
    actor_id = _sanitize_actor_id(customer_name)

    config: dict = {
        "configurable": {
            "thread_id": thread_id,
            "customer_name": customer_name,
            "actor_id": actor_id,
        }
    }
    if callbacks:
        config["callbacks"] = callbacks
    input_data = {
        "messages": [HumanMessage(content=messages)],
        "customer_name": customer_name,
        "tool_call_count": 0,
    }

    logger.info(
        f"Running RAG graph (thread_id={thread_id}, actor_id={actor_id})"
    )

    current_node: str | None = None
    streamed_any = False
    final_state = None

    try:
        async for event in graph.astream_events(
            input=input_data, config=config, version="v2"
        ):
            event_type = event.get("event")
            event_data = event.get("data", {})
            name = event.get("name", "")

            if event_type == "on_chain_start" and name in _RESPONSE_NODES:
                current_node = name
                logger.debug(f"Streaming: entered response node '{name}'")

            elif event_type == "on_chain_end":
                output = event_data.get("output")
                if output and isinstance(output, dict) and "messages" in output:
                    final_state = output
                if name == current_node:
                    current_node = None

            elif event_type == "on_chat_model_stream":
                if current_node not in _RESPONSE_NODES:
                    continue

                chunk = event_data.get("chunk")
                if not chunk or not isinstance(chunk, AIMessageChunk):
                    continue

                if chunk.tool_calls or chunk.tool_call_chunks:
                    continue

                content = extract_text_content(chunk.content)
                if content:
                    streamed_any = True
                    yield content

        if not streamed_any and final_state:
            logger.warning("No tokens streamed; extracting final answer from state")
            for msg in reversed(final_state.get("messages", [])):
                if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                    text = extract_text_content(msg.content)
                    if text:
                        yield text
                        break
    except Exception as e:
        logger.error(f"Streaming error: {type(e).__name__}: {e}")
        logger.exception("Full traceback:")
        raise


def _sanitize_actor_id(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9\-_ ]", "", name)
    sanitized = sanitized.replace(" ", "-").lower()
    return f"user:{sanitized or 'guest'}"
