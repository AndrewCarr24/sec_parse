"""Lazy singleton access to the dsRAG KnowledgeBase.

Pointed at the KB built by `data_pipeline_dsrag/build_kb.py`. We set up
environment variables so dsRAG's OpenAI client (used if any dsRAG
component calls the LLM at query time — e.g. query expansion) routes to
DeepSeek's OpenAI-compatible endpoint, matching the rest of our stack.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger


_REPO_ROOT = Path(__file__).resolve().parents[3]
DSRAG_STORE_DIR = _REPO_ROOT / "data" / "dsrag_store"
DSRAG_KB_ID = "act_10q_2024_09_30"
# The pipeline code registers BedrockTitanEmbedding / FlashRankReranker
# as Embedding/Reranker subclasses at import time — dsRAG's from_dict
# deserialization needs those classes registered before KB load.
_PIPELINE_DIR = _REPO_ROOT / "data_pipeline_dsrag"


_kb = None


def _ensure_imports_registered() -> None:
    if str(_PIPELINE_DIR) not in sys.path:
        sys.path.insert(0, str(_PIPELINE_DIR))
    import bedrock_embedding  # noqa: F401 — registers subclass with dsRAG
    import flashrank_reranker  # noqa: F401


def _configure_deepseek_as_openai() -> None:
    """dsRAG's AutoContext/semantic-sectioning LLM routes through its
    OpenAI client; point it at DeepSeek if we're configured for that."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return
    # Don't stomp a real OPENAI_API_KEY if the caller set one explicitly.
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault(
        "DSRAG_OPENAI_BASE_URL",
        os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )


def get_kb():
    global _kb
    if _kb is not None:
        return _kb
    _ensure_imports_registered()
    _configure_deepseek_as_openai()
    from dsrag.knowledge_base import KnowledgeBase
    from dsrag.reranker import NoReranker

    logger.info(f"Loading dsRAG KB '{DSRAG_KB_ID}' from {DSRAG_STORE_DIR}")
    _kb = KnowledgeBase(
        DSRAG_KB_ID,
        storage_directory=str(DSRAG_STORE_DIR),
        exists_ok=True,
    )
    # Swap the persisted reranker (FlashRank CPU cross-encoder) for
    # NoReranker — FlashRank is too slow on CPU for multi-query reranks.
    # The reranker is a query-time component, so we don't need to rebuild
    # embeddings or chunks to change it.
    _kb.reranker = NoReranker(ignore_absolute_relevance=True)
    logger.info("dsRAG KB reranker set to NoReranker for this process")
    return _kb
