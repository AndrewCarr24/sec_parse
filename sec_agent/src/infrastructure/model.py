"""LLM loader — orchestrator can be Bedrock/Claude or DeepSeek, router/judge
stay on Bedrock Haiku so intent classification and grading remain consistent
across orchestrator experiments."""

from typing import Any

from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.config import settings


def extract_text_content(content: Any) -> str:
    """Extract plain text from a chat-model response."""
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


def _bedrock(model_id: str, temperature: float) -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model=model_id,
        temperature=temperature,
        region_name=settings.AWS_REGION,
    )


def _deepseek_class():
    """Return a ChatDeepSeek subclass that round-trips `reasoning_content`.

    DeepSeek's thinking mode requires each assistant message to carry its
    original `reasoning_content` back on subsequent turns, or the API errors
    with 400: "The reasoning_content in the thinking mode must be passed
    back to the API." langchain_deepseek captures it on inbound into
    `message.additional_kwargs["reasoning_content"]` but does not re-emit
    it on outbound. This subclass adds the round-trip.
    """
    from langchain_deepseek import ChatDeepSeek

    class ChatDeepSeekRoundtrip(ChatDeepSeek):
        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            try:
                if hasattr(input_, "to_messages"):
                    original = input_.to_messages()
                else:
                    original = list(input_) if input_ is not None else []
            except Exception:
                return payload

            # Reasoning_content for each AIMessage in input, in order.
            ai_reasoning = [
                (m.additional_kwargs or {}).get("reasoning_content")
                for m in original
                if isinstance(m, AIMessage)
            ]
            # Inject onto each assistant message in outbound payload in order.
            idx = 0
            for msg in payload.get("messages", []):
                if msg.get("role") != "assistant":
                    continue
                if idx < len(ai_reasoning) and ai_reasoning[idx]:
                    msg["reasoning_content"] = ai_reasoning[idx]
                idx += 1
            return payload

    return ChatDeepSeekRoundtrip


def _deepseek(model_id: str, temperature: float) -> BaseChatModel:
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError(
            "ORCHESTRATOR_PROVIDER=deepseek but DEEPSEEK_API_KEY is not set."
        )
    ChatCls = _deepseek_class()
    return ChatCls(
        model=model_id,
        temperature=temperature,
        api_key=settings.DEEPSEEK_API_KEY,
        api_base=settings.DEEPSEEK_BASE_URL,
    )


def get_model(temperature: float = 0.5, router: bool = False) -> BaseChatModel:
    """Return the chat model for the requested role.

    Router and judge always use Bedrock Haiku. The orchestrator honors
    `settings.ORCHESTRATOR_PROVIDER`.
    """
    if router:
        return _bedrock(settings.ROUTER_MODEL_ID, temperature)
    if settings.ORCHESTRATOR_PROVIDER == "deepseek":
        return _deepseek(settings.DEEPSEEK_MODEL_ID, temperature)
    return _bedrock(settings.ORCHESTRATOR_MODEL_ID, temperature)


def orchestrator_is_bedrock() -> bool:
    """Bedrock-specific features (cachePoint content blocks) are gated on this."""
    return settings.ORCHESTRATOR_PROVIDER == "bedrock"
