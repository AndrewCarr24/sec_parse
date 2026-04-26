"""Lazy singleton access to the dsRAG KnowledgeBase.

Pointed at the KB built by `data_pipeline_dsrag/build_kb.py`. We set up
environment variables so dsRAG's OpenAI client (used if any dsRAG
component calls the LLM at query time — e.g. query expansion) routes to
DeepSeek's OpenAI-compatible endpoint, matching the rest of our stack.

Also exposes `get_search_queries`, our own implementation of dsRAG's
auto-query helper. The upstream `dsrag.auto_query.get_search_queries`
is marked legacy in dsRAG's source and is hardcoded to Claude Sonnet 3.5
via the Anthropic API; we replicate the (very small) logic here so it
routes through our DeepSeek client instead. The `AUTO_QUERY_GUIDANCE`
prompt is taken from dsRAG's published FinanceBench eval script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

from loguru import logger
from pydantic import BaseModel


_REPO_ROOT = Path(__file__).resolve().parents[3]
DSRAG_STORE_DIR = _REPO_ROOT / "data" / "dsrag_store"
# Single multi-document KB. Filings are distinguished by their `doc_id`
# metadata (TICKER_FORM_PERIOD, matching data/parsed/<stem>.md). The agent
# passes `doc_id` as a metadata filter when scoping retrieval to one filing.
DSRAG_KB_ID = "filings_kb"
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


# ---------------------------------------------------------------------------
# Query expansion (auto-query)
# ---------------------------------------------------------------------------

# Domain guidance taken verbatim from dsRAG's published FinanceBench eval
# script (tests/financebench/run_eval.py in the dsRAG repo).
AUTO_QUERY_GUIDANCE = """
The knowledge base contains SEC filings for publicly traded companies, like 10-Ks, 10-Qs, and 8-Ks. Keep this in mind when generating search queries. The things you search for should be things that are likely to be found in these documents.

When deciding what to search for, first consider the pieces of information that will be needed to answer the question. Then, consider what to search for to find those pieces of information. For example, if the question asks what the change in revenue was from 2019 to 2020, you would want to search for the 2019 and 2020 revenue numbers in two separate search queries, since those are the two separate pieces of information needed. You should also think about where you are most likely to find the information you're looking for. If you're looking for assets and liabilities, you may want to search for the balance sheet, for example.
""".strip()


_AUTO_QUERY_SYSTEM = """\
You are a query generation system. Please generate one or more search queries (up to a maximum of {max_queries}) based on the provided user input. DO NOT generate the answer, just queries.

Each of the queries you generate will be used to search a knowledge base for information that can be used to respond to the user input. Make sure each query is specific enough to return relevant information. If multiple pieces of information would be useful, you should generate multiple queries, one for each specific piece of information needed.

{auto_query_guidance}"""


class _Queries(BaseModel):
    queries: List[str]


_auto_query_client = None


def _get_auto_query_client():
    """Cached instructor client routed at DeepSeek's OpenAI-compatible API.

    Uses `deepseek-chat` (not v4-flash) because instructor's default Mode.TOOLS
    sends a forced `tool_choice` that v4-flash's reasoner backend rejects.
    """
    global _auto_query_client
    if _auto_query_client is not None:
        return _auto_query_client
    _configure_deepseek_as_openai()
    import instructor
    from openai import OpenAI

    oa = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("DSRAG_OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=60.0,
    )
    _auto_query_client = instructor.from_openai(oa, mode=instructor.Mode.TOOLS)
    return _auto_query_client


def get_search_queries(
    user_input: str,
    auto_query_guidance: str = AUTO_QUERY_GUIDANCE,
    max_queries: int = 6,
) -> List[str]:
    """Decompose a user question into a small set of independent KB search
    queries via a single LLM call. Mirrors dsRAG's deprecated upstream
    helper but routes through DeepSeek's `deepseek-chat` instead of the
    Anthropic-locked default."""
    client = _get_auto_query_client()
    resp = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=400,
        temperature=0.2,
        response_model=_Queries,
        messages=[
            {
                "role": "system",
                "content": _AUTO_QUERY_SYSTEM.format(
                    max_queries=max_queries,
                    auto_query_guidance=auto_query_guidance,
                ),
            },
            {"role": "user", "content": user_input},
        ],
    )
    return resp.queries[:max_queries]
