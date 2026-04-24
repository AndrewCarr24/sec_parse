"""Build a dsRAG KnowledgeBase from the ACT 10-Q filing.

Inputs:
    data/parsed/ACT_10-Q_2024-09-30.md       (Docling-rendered markdown)

Outputs:
    data/dsrag_store/<kb_id>/ ...            (dsRAG persists here)

Stack:
    Embedding:         BedrockTitanEmbedding (our Titan v2 adapter)
    Reranker:          FlashRankReranker (local cross-encoder)
    AutoContext LLM:   OpenAIChatAPI pointed at DeepSeek's OpenAI-compatible
                       endpoint (matches the agent's orchestrator stack).
    VectorDB:          ChromaDB (matches our existing vector store backend).

Requires DEEPSEEK_API_KEY. Sets DSRAG_OPENAI_BASE_URL and re-aliases the
DeepSeek key as OPENAI_API_KEY in-process so dsRAG's OpenAI client routes
to DeepSeek.

Usage (from repo root):
    . .env && python data_pipeline_dsrag/build_kb.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make our custom subclasses registerable before building the KB.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bedrock_embedding import BedrockTitanEmbedding  # noqa: E402
from flashrank_reranker import FlashRankReranker  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[1]
PARSED_DIR = _REPO_ROOT / "data" / "parsed"
STORE_DIR = _REPO_ROOT / "data" / "dsrag_store"

DOC_FILENAME = "ACT_10-Q_2024-09-30.md"
KB_ID = "act_10q_2024_09_30"


def _configure_deepseek_as_openai() -> None:
    """Route dsRAG's OpenAI client at DeepSeek's OpenAI-compatible endpoint."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set.", file=sys.stderr)
        sys.exit(2)
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["DSRAG_OPENAI_BASE_URL"] = os.environ.get(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
    )


def build() -> None:
    _configure_deepseek_as_openai()

    # Delay dsrag imports until after env is configured; KnowledgeBase pulls
    # in auto_context which constructs LLM clients at module load.
    from dsrag.knowledge_base import KnowledgeBase
    from dsrag.llm import OpenAIChatAPI

    md_path = PARSED_DIR / DOC_FILENAME
    if not md_path.exists():
        raise FileNotFoundError(md_path)
    text = md_path.read_text()

    STORE_DIR.mkdir(parents=True, exist_ok=True)

    kb = KnowledgeBase(
        kb_id=KB_ID,
        embedding_model=BedrockTitanEmbedding(),
        reranker=FlashRankReranker(),
        auto_context_model=OpenAIChatAPI(
            model="deepseek-v4-flash",
            temperature=0.2,
            max_tokens=2000,
        ),
        storage_directory=str(STORE_DIR),
        exists_ok=True,
    )

    print(f"Adding document '{DOC_FILENAME}' ({len(text):,} chars)...")
    kb.add_document(
        doc_id=DOC_FILENAME,
        text=text,
        # Skip the VLM file-parsing step — we already have Docling markdown.
        semantic_sectioning_config={
            "use_semantic_sectioning": True,
            "llm_provider": "openai",  # routed to DeepSeek via DSRAG_OPENAI_BASE_URL
            "model": "deepseek-v4-flash",
        },
        auto_context_config={
            "use_generated_title": True,
            "get_document_summary": True,
            "get_section_summaries": True,
        },
    )
    print(f"Done. KB id={KB_ID!r} stored under {STORE_DIR}")


if __name__ == "__main__":
    build()
