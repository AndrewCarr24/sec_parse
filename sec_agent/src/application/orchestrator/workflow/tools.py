"""Tools available to the agent."""

import json
from typing import Annotated, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from loguru import logger

from src.config import settings
from src.infrastructure.compressor import compress_tool_output
from src.infrastructure.financials_db import get_connection


# Hard cap to keep rows returned from arbitrary agent SQL from blowing up
# the model's context. Agents can paginate/aggregate within SQL instead.
_MAX_ROWS = 100


def _maybe_compress(
    tool_name: str, raw: str, config: RunnableConfig | None
) -> str:
    """Feature-flagged pass-through (see src/infrastructure/compressor.py)."""
    configurable = (config or {}).get("configurable") or {}
    question = configurable.get("user_question") or ""
    return compress_tool_output(tool_name, question, raw, config=config)


@tool
def search_concepts(
    keyword: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """
    Discover facts matching a keyword across concept, label, and
    source_table_title. Use this first to find the right filter for
    `query_financials`.

    For extracted breakdown tables (e.g. NIW by FICO, IIF by LTV) the
    *metric name* lives in source_table_title while `label` holds the
    *breakdown bucket*. Keyword search spans both, so "NIW" and "FICO"
    both surface the same table from different angles.

    Args:
        keyword: Case-insensitive substring to match.

    Returns:
        JSON list of {source, concept, label, source_tables,
        sample_unit, fact_count}. `source_tables` is the list of every
        distinct source_table_title this label appears under — so if a
        metric is split across multiple tables (e.g. one per period),
        they all show up here rather than being hidden by aggregation.
    """
    logger.info(f"search_concepts invoked: {keyword!r}")
    # Fresh cursor per call: DuckDB connections share a single pending-query
    # state, so concurrent tool calls on the shared singleton race. Cursors
    # have independent query state.
    con = get_connection().cursor()
    rows = con.execute(
        """
        SELECT
            source,
            concept,
            label,
            ARRAY_AGG(source_table_title) AS titles_raw,
            ANY_VALUE(unit) AS sample_unit,
            COUNT(*) AS fact_count
        FROM facts
        WHERE concept ILIKE '%' || ? || '%'
           OR label ILIKE '%' || ? || '%'
           OR source_table_title ILIKE '%' || ? || '%'
        GROUP BY source, concept, label
        ORDER BY fact_count DESC
        LIMIT 50
        """,
        [keyword, keyword, keyword],
    ).fetchall()
    results = [
        {
            "source": r[0],
            "concept": r[1],
            "label": r[2],
            "source_tables": sorted({t for t in (r[3] or []) if t}),
            "sample_unit": r[4],
            "fact_count": r[5],
        }
        for r in rows
    ]
    raw = json.dumps(results, indent=2, default=str)
    return _maybe_compress("search_concepts", raw, config)


@tool
def query_financials(
    sql: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """
    Run a read-only SQL query against the `facts` table (DuckDB).

    Schema:
        facts(
            filing_id TEXT,          -- e.g. "ACT_10-Q_2024-09-30"
            ticker TEXT,
            form TEXT,               -- "10-Q" / "10-K"
            fiscal_period_end DATE,
            source TEXT,             -- 'xbrl' or 'extracted'
            concept TEXT,            -- e.g. "us-gaap:Revenues" (may be NULL for extracted)
            label TEXT,              -- human label from table row (extracted only)
            value DOUBLE,
            raw_value TEXT,          -- original string (preserves "$249,055" etc.)
            unit TEXT,               -- 'USD', 'shares', 'USD_per_share', 'pure'
            period_type TEXT,        -- 'instant' or 'duration'
            period_start DATE,
            period_end DATE,
            dimensions TEXT,         -- JSON breakdown (e.g. LTV band); NULL or '{}' = total
            source_table_title TEXT  -- extracted-only: source MD&A table heading
        )

    Tips:
        - Always scope to one filing (or a chosen set) with
          `WHERE ticker = '...' AND fiscal_period_end = DATE '...'`.
          Available filings are listed in the system prompt's filings_catalog.
        - Select only the columns you need; avoid SELECT *. Include a LIMIT
          that matches the result size you expect.
        - `fiscal_period_end` picks the filing; `period_end` picks the fact's
          own period within that filing (a 10-Q carries 3- and 9-month facts
          plus prior-year comparatives).
        - Balance-sheet values have period_type='instant'; P&L values 'duration'.
        - Prefer `source='xbrl'` for GAAP line items; use 'extracted' for
          MD&A operational metrics (IIF, NIW, persistency, delinquencies by LTV, etc.).

    Args:
        sql: A SELECT statement. Only SELECTs are allowed.

    Returns:
        JSON with `columns` and `rows` (max 100 rows).
    """
    logger.info(f"query_financials invoked: {sql!r}")
    stripped = sql.strip().rstrip(";").strip()
    lowered = stripped.lower()
    if not lowered.startswith(("select", "with")):
        return json.dumps({"error": "Only SELECT/WITH queries are allowed."})

    # Block obvious mutation keywords. DuckDB connection is in-memory so the
    # blast radius is small, but defence-in-depth is cheap.
    forbidden = (" insert ", " update ", " delete ", " drop ", " create ", " attach ", " copy ")
    padded = f" {lowered} "
    if any(k in padded for k in forbidden):
        return json.dumps({"error": "Mutating statements are not allowed."})

    con = get_connection().cursor()
    try:
        cur = con.execute(stripped)
    except Exception as e:
        logger.warning(f"query_financials SQL error: {e}")
        return json.dumps({"error": str(e)})

    columns = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchmany(_MAX_ROWS)
    truncated = len(rows) == _MAX_ROWS and len(cur.fetchmany(1)) > 0

    raw = json.dumps(
        {
            "columns": columns,
            "rows": [[_jsonable(v) for v in row] for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
        },
        indent=2,
        default=str,
    )
    return _maybe_compress("query_financials", raw, config)


def _jsonable(v):
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


# Per-thread set of chunk IDs already returned to the agent. Same thread_id
# across multiple search_narrative calls means the same agent run — we skip
# duplicate chunks and backfill with fresh ones so the model doesn't pay to
# re-read the same paragraph.
_SEEN_CHUNKS: dict[str, set[str]] = {}


@tool
def search_narrative(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """
    Semantic search over the narrative (prose) sections of SEC filings —
    MD&A business discussion, risk factors, strategic commentary, M&A and
    regulatory narrative. Use this for questions about management's views,
    strategy, outlook, risks, or any context NOT expressed as a number in
    a table.

    Do NOT use this for numeric lookups — those live in the `facts` table,
    accessed via `search_concepts` / `query_financials`. For hybrid
    questions (e.g. "what does management say about the decline in NIW?"),
    use this alongside the SQL tools.

    Narrative retrieval is currently single-corpus and not scoped by
    filing — the top-k chunks may come from any indexed filing. Cross-check
    the returned `company`/`period_label` metadata against what the user asked.

    Within a single agent run, chunks already returned by earlier calls are
    excluded and replaced by the next-best fresh chunks, so repeated calls
    with similar queries yield new content rather than re-sending the same
    paragraphs.

    Args:
        query: Natural-language query.

    Returns:
        JSON list of {company, filing_type, period_label, source, text}.
    """
    from src.infrastructure.narrative_search import search

    logger.info(f"search_narrative invoked: {query!r}")
    thread_id = ((config or {}).get("configurable") or {}).get("thread_id", "_default")
    seen = _SEEN_CHUNKS.setdefault(thread_id, set())
    try:
        results = search(query, top_k=4, exclude_ids=seen)
    except Exception as e:
        logger.warning(f"search_narrative failed: {e}")
        return json.dumps({"error": str(e)})
    for r in results:
        seen.add(r["id"])
    # Drop the internal id from what we send to the agent — it's only for dedup.
    payload = [{k: v for k, v in r.items() if k != "id"} for r in results]
    raw = json.dumps(payload, indent=2, default=str)
    return _maybe_compress("search_narrative", raw, config)


@tool
async def memory_retrieval_tool(
    query: str,
    memory_types: list[Literal["preferences", "facts", "summaries"]],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """
    Retrieve the user's stored preferences, facts, or session summaries.
    Use before answering to personalize results.

    Args:
        query: Search query for semantic matching.
        memory_types: Which memory types to retrieve.

    Returns:
        JSON with memories organized by type.
    """
    configurable = (config or {}).get("configurable", {})
    actor_id = configurable.get("actor_id", "user:default")
    session_id = configurable.get("thread_id", "default_session")

    try:
        from src.infrastructure.memory import get_memory_instance
        memory = get_memory_instance()
        retrieved = memory.retrieve_specific_memories(
            query=query,
            actor_id=actor_id,
            session_id=session_id,
            memory_types=memory_types,
            top_k=5,
        )
        formatted = {
            k: [item.get("content", str(item)) for item in v]
            for k, v in retrieved.items()
        }
        return json.dumps(formatted, indent=2)
    except Exception as e:
        logger.error(f"memory_retrieval_tool failed: {e}")
        return json.dumps({"error": str(e)})


@tool
def dsrag_kb(queries: list[str]) -> str:
    """
    Semantic search over the loaded SEC filing via a dsRAG knowledge base.
    Returns the most relevant multi-chunk *segments* (contiguous sections
    identified by dsRAG's Relevant Segment Extraction) for one or more
    queries. Segments include an AutoContext header describing the
    document and surrounding section, so you can tell where in the filing
    each excerpt comes from.

    Use this as your sole retrieval tool. Pass 1-3 complementary queries
    per turn — the KB merges and reranks results across them.

    Args:
        queries: 1-3 natural-language search queries capturing distinct
            facets of the user's question.

    Returns:
        JSON list of {score, doc_id, content} segments, highest score first.
    """
    from src.infrastructure.dsrag_kb import get_kb

    logger.info(f"dsrag_kb invoked: {queries!r}")
    kb = get_kb()
    try:
        results = kb.query(queries) if len(queries) > 1 else kb.query(queries)
    except Exception as e:
        logger.warning(f"dsrag_kb query failed: {e}")
        return json.dumps({"error": str(e)})
    payload = [
        {
            "score": round(float(r.get("score", 0.0) or 0.0), 3),
            "doc_id": r.get("doc_id", ""),
            "content": (r.get("content") or r.get("text") or "")[:6000],
        }
        for r in results
    ]
    return json.dumps(payload, indent=2, default=str)


def get_tools() -> list:
    if settings.USE_DSRAG_ONLY:
        # Single-tool mode — only dsrag_kb. Memory retrieval still
        # offered if configured, since it's orthogonal to document RAG.
        tools = [dsrag_kb]
        if settings.MEMORY_ID:
            tools.append(memory_retrieval_tool)
        return tools
    tools = [search_concepts, query_financials, search_narrative]
    if settings.MEMORY_ID:
        tools.append(memory_retrieval_tool)
    return tools
