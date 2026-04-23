"""Haiku-based question-conditional compressor for tool outputs.

Gated by `settings.COMPRESS_TOOL_OUTPUTS`. When disabled (the default)
or on any error, `compress_tool_output` returns `raw` unchanged, so
the feature is removable by flipping one env flag or by deleting this
module plus the three tool callsites in `workflow/tools.py`.

Small outputs bypass Haiku entirely — the compression hop is only
worth it when there's meaningful bulk to drop.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.config import settings
from src.infrastructure.model import extract_text_content, get_model


# Don't bother compressing results smaller than this; the Haiku hop
# costs more than it saves on tiny payloads.
_MIN_CHARS = 500


_SYSTEM = """\
You are a precision filter for an LLM agent's tool output.

You will receive:
  1. USER QUESTION — the question the agent is answering.
  2. TOOL — the tool that produced the result.
  3. TOOL RESULT — the raw tool output (JSON or prose).

Re-emit ONLY the parts of the tool result the agent needs to answer
the user question. Preserve exact numbers, labels, concept names,
dates, and identifiers verbatim. Drop rows, fields, or sentences
that cannot conceivably be cited in the final answer.

Rules:
- Keep numeric values as-is. No rounding, no reformatting.
- Keep identifiers (concept, label, source_table_title) verbatim.
- For structured JSON input, re-emit a smaller JSON with the same
  shape, dropping irrelevant entries. Do not invent keys.
- For prose input, keep only relevant sentences; do not paraphrase.
- If the result is already concise and fully relevant, return it
  unchanged.
- Output ONLY the compressed tool result. No preamble, no commentary.
"""


def compress_tool_output(
    tool_name: str,
    question: str,
    raw: str,
    config: dict[str, Any] | None = None,
) -> str:
    if not settings.COMPRESS_TOOL_OUTPUTS:
        return raw
    if not raw or not question:
        return raw
    if len(raw) < _MIN_CHARS:
        return raw
    try:
        model = get_model(temperature=0.0, router=True)  # Haiku
        resp = model.invoke(
            [
                ("system", _SYSTEM),
                (
                    "user",
                    f"USER QUESTION:\n{question}\n\n"
                    f"TOOL: {tool_name}\n"
                    f"TOOL RESULT:\n{raw}",
                ),
            ],
            config=config,  # propagates callbacks so usage is tallied
        )
        out = extract_text_content(resp.content).strip()
        if not out:
            return raw
        logger.debug(
            f"compressor[{tool_name}]: {len(raw):,} -> {len(out):,} chars"
        )
        return out
    except Exception as e:
        logger.warning(
            f"compressor[{tool_name}] failed: {e}; returning raw"
        )
        return raw
