"""Streaming at the API boundary.

AgentCore's BedrockAgentCoreApp handles SSE framing, so we just yield
plain text chunks — no extra data:/JSON wrapping needed.
"""

from typing import AsyncGenerator

from loguru import logger

from src.application.orchestrator.streaming import get_streaming_response


async def stream_response(
    user_input: str,
    customer_name: str = "Guest",
    conversation_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield plain text chunks; AgentCore wraps them in SSE."""
    try:
        async for chunk in get_streaming_response(
            messages=user_input,
            customer_name=customer_name,
            conversation_id=conversation_id,
        ):
            if chunk:
                yield chunk
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e) or "Unknown error"
        logger.error(f"stream_response error: {error_type}: {error_msg}")
        logger.exception("Full traceback:")
        yield f"Error: {error_type}: {error_msg}"
