"""Build / extend the dsRAG KnowledgeBase with SEC filings under
data/parsed/. Each filing is ingested as its own dsRAG document — the
filename stem (`TICKER_FORM_PERIOD`) becomes the `doc_id`, which the agent
later passes as a metadata filter to scope retrieval to one filing.

Usage:
    # ingest every *.md under data/parsed/
    python data_pipeline_dsrag/build_kb.py

    # ingest only specific filings (identified by doc_id stem)
    python data_pipeline_dsrag/build_kb.py \\
        ACT_10-Q_2024-09-30 \\
        AMD_10-K_2022-12-31 \\
        BA_10-K_2022-12-31

Inputs:
    data/parsed/*.md                         (Docling-rendered markdown)

Outputs:
    data/dsrag_store/<kb_id>/...             (dsRAG persists here)

Stack:
    Embedding:       BedrockTitanEmbedding (Titan v2)
    Reranker:        FlashRankReranker (persisted config; swapped to
                     NoReranker at query time by src/infrastructure/dsrag_kb.py)
    AutoContext LLM: OpenAIChatAPI pointed at DeepSeek's OpenAI-compatible
                     endpoint (keeps the full stack on DeepSeek).
    VectorDB:        ChromaDB.

Re-runs are idempotent: documents whose doc_id is already in the KB are
skipped (dsRAG's chunk DB tracks doc_ids already ingested).

Requires DEEPSEEK_API_KEY.

Usage (from repo root):
    . .env && python data_pipeline_dsrag/build_kb.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make our custom subclasses registerable before constructing the KB.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bedrock_embedding import BedrockTitanEmbedding  # noqa: E402
from flashrank_reranker import FlashRankReranker  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[1]
PARSED_DIR = _REPO_ROOT / "data" / "parsed"
STORE_DIR = _REPO_ROOT / "data" / "dsrag_store"

KB_ID = "filings_kb"


def _configure_deepseek_as_openai() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set.", file=sys.stderr)
        sys.exit(2)
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["DSRAG_OPENAI_BASE_URL"] = os.environ.get(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
    )


def _already_indexed_doc_ids(kb) -> set[str]:
    """Return the set of doc_ids already present in the KB's chunk DB."""
    try:
        # dsRAG's chunk DB tracks doc_ids via the per-doc chunk index it maintains.
        return set(kb.chunk_db.get_all_doc_ids())
    except Exception:
        return set()


def build(allowlist: list[str] | None = None) -> None:
    _configure_deepseek_as_openai()

    # Delay dsrag imports until after env is configured; KnowledgeBase pulls
    # in auto_context which constructs LLM clients at module load.
    from dsrag.knowledge_base import KnowledgeBase
    from dsrag.llm import OpenAIChatAPI

    STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Use `deepseek-chat` (non-reasoning V3 backend) for AutoContext + semantic
    # sectioning. dsRAG uses `instructor` with default Mode.TOOLS, which sends
    # `tool_choice={"type":"function","function":{"name":"X"}}` — forced
    # function-calling. DeepSeek's reasoner backend (which v4-flash routes to)
    # rejects this with 400 "deepseek-reasoner does not support this
    # tool_choice", causing 100% of semantic-sectioning windows to fall back
    # to generic "Window N-M" titles. `deepseek-chat` is on a non-reasoner
    # backend and accepts forced tool_choice cleanly. AutoContext is an
    # indexing-time call, not per-query, so the cost difference is one-time.
    kb = KnowledgeBase(
        kb_id=KB_ID,
        embedding_model=BedrockTitanEmbedding(),
        reranker=FlashRankReranker(),
        auto_context_model=OpenAIChatAPI(
            model="deepseek-chat",
            temperature=0.2,
            max_tokens=2000,
        ),
        storage_directory=str(STORE_DIR),
        exists_ok=True,
    )

    indexed = _already_indexed_doc_ids(kb)
    all_md = sorted(PARSED_DIR.glob("*.md"))
    if allowlist:
        wanted = set(allowlist)
        md_files = [m for m in all_md if m.stem in wanted]
        missing = wanted - {m.stem for m in md_files}
        if missing:
            raise SystemExit(
                f"No parsed markdown for doc_id(s): {sorted(missing)}. "
                f"Expected data/parsed/<doc_id>.md"
            )
    else:
        md_files = all_md
    if not md_files:
        raise SystemExit(f"No markdown files to ingest under {PARSED_DIR}")

    print(
        f"Building KB id={KB_ID!r} at {STORE_DIR}\n"
        f"  to ingest: {len(md_files)} markdown file(s)\n"
        f"  already indexed: {len(indexed)} ({sorted(indexed)})"
    )

    for md in md_files:
        doc_id = md.stem  # e.g. ACT_10-Q_2024-09-30
        if doc_id in indexed:
            print(f"  skip (already indexed): {doc_id}")
            continue
        text = md.read_text()
        print(f"  ingesting: {doc_id} ({len(text):,} chars)")
        kb.add_document(
            doc_id=doc_id,
            text=text,
            semantic_sectioning_config={
                "use_semantic_sectioning": True,
                "llm_provider": "openai",
                # See comment above on auto_context_model — same reason.
                "model": "deepseek-chat",
            },
            auto_context_config={
                "use_generated_title": True,
                "get_document_summary": True,
                "get_section_summaries": True,
            },
        )
    print(f"Done. KB now holds: {sorted(_already_indexed_doc_ids(kb))}")


if __name__ == "__main__":
    allowlist = sys.argv[1:] or None
    build(allowlist=allowlist)
