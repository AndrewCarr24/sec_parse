"""Bedrock model loader."""

from typing import Any

from langchain_aws import ChatBedrockConverse

from src.config import settings


def extract_text_content(content: Any) -> str:
    """Extract plain text from a Bedrock model response."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content) if content else ""


def get_model(temperature: float = 0.5, router: bool = False) -> ChatBedrockConverse:
    model_id = settings.ROUTER_MODEL_ID if router else settings.ORCHESTRATOR_MODEL_ID
    return ChatBedrockConverse(
        model=model_id,
        temperature=temperature,
        region_name=settings.AWS_REGION,
    )
