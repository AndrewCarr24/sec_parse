"""Bedrock model pricing (USD per million tokens).

Update when models or prices change. Source:
https://aws.amazon.com/bedrock/pricing/
"""

import re

_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_read": 0.10, "cache_write": 1.25},
}

_VERSION_SUFFIX = re.compile(r"-\d{8}(-v\d+:\d+)?$")
_GEO_PREFIXES = ("us.", "eu.", "au.", "apac.", "global.")


def normalize_model_id(model_id: str) -> str:
    """us.anthropic.claude-sonnet-4-5-20250929-v1:0 -> claude-sonnet-4-5"""
    s = model_id
    for prefix in _GEO_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.startswith("anthropic."):
        s = s[len("anthropic."):]
    return _VERSION_SUFFIX.sub("", s)


def cost_usd(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """USD cost. cache_read_tokens and cache_creation_tokens are assumed to be
    subsets of input_tokens (LangChain's UsageMetadata reports them that way)."""
    rates = _PRICING.get(normalize_model_id(model_id))
    if not rates:
        return 0.0
    plain_input = max(input_tokens - cache_read_tokens - cache_creation_tokens, 0)
    return (
        plain_input * rates["input"]
        + cache_read_tokens * rates["cache_read"]
        + cache_creation_tokens * rates["cache_write"]
        + output_tokens * rates["output"]
    ) / 1_000_000
