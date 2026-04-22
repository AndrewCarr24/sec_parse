"""In-memory DuckDB exposing XBRL + extracted facts as a unified `facts` table.

Each filing's pipeline emits two CSVs under
    data/extracted_facts_and_tables/{TICKER}/{FILING_TYPE}_{YYYY-MM-DD}/
        xbrl_facts.csv       — machine-tagged facts from the iXBRL instance
        extracted_facts.csv  — LLM-extracted facts from untagged HTML tables

Both are folded into one long-format `facts` table, unioned across every
filing in the tree, so the agent can query with one SQL surface and filter
by (ticker, filing_period_end) to scope to a single filing.
"""

from pathlib import Path

import duckdb
from loguru import logger

from src.infrastructure.catalog import FACTS_ROOT, list_filings


_conn: duckdb.DuckDBPyConnection | None = None


# iXBRL uses namespaced measure strings; normalize to short labels that
# both humans and the agent can filter on without remembering prefixes.
_UNIT_ALIASES = {
    "iso4217:USD": "USD",
    "xbrli:shares": "shares",
    "iso4217:USD / xbrli:shares": "USD_per_share",
    "xbrli:pure": "pure",
    "iso4217:USD / iso4217:USD": "pure",
}


def _build_connection() -> duckdb.DuckDBPyConnection:
    filings = list_filings()
    if not filings:
        raise FileNotFoundError(
            f"No filings found under {FACTS_ROOT}. "
            "Run data_pipeline/ixbrl_parser.py + table_extractor.py first."
        )

    xbrl_glob = str(FACTS_ROOT / "**" / "xbrl_facts.csv")
    extracted_glob = str(FACTS_ROOT / "**" / "extracted_facts.csv")

    con = duckdb.connect(":memory:")

    unit_case = "CASE unit " + " ".join(
        f"WHEN '{k}' THEN '{v}'" for k, v in _UNIT_ALIASES.items()
    ) + " ELSE unit END"

    # read_csv_auto with glob reads every matching file; union_by_name handles
    # minor schema drift between xbrl and extracted CSVs. We keep the two
    # sources in separate CTEs because their column shapes differ.
    con.execute(f"""
        CREATE TABLE facts AS
        WITH xbrl_raw AS (
            SELECT * FROM read_csv_auto(
                '{xbrl_glob}',
                header=True,
                all_varchar=True,
                union_by_name=True,
                filename=False
            )
        ),
        extracted_raw AS (
            SELECT * FROM read_csv_auto(
                '{extracted_glob}',
                header=True,
                all_varchar=True,
                union_by_name=True,
                filename=False
            )
        ),
        xbrl_norm AS (
            SELECT
                (ticker || '_' || filing_type || '_' || filing_period_end) AS filing_id,
                ticker                               AS ticker,
                filing_type                          AS form,
                TRY_CAST(filing_period_end AS DATE)  AS fiscal_period_end,
                'xbrl'                               AS source,
                concept                              AS concept,
                NULL                                 AS label,
                TRY_CAST(value AS DOUBLE)            AS value,
                value                                AS raw_value,
                {unit_case}                          AS unit,
                period_type                          AS period_type,
                TRY_CAST(period_start AS DATE)       AS period_start,
                -- XBRL uses exclusive end-dates: instants stored as day-after
                -- (2024-09-30 → 2024-10-01) and durations end at midnight of
                -- the day after the last included day. Subtract 1 day in both
                -- cases so dates read as humans expect (Sept 30, not Oct 1).
                (TRY_CAST(period_end AS DATE) - INTERVAL 1 DAY)::DATE
                                                     AS period_end,
                NULLIF(dimensions, '')               AS dimensions,
                NULL                                 AS source_table_title
            FROM xbrl_raw
            WHERE is_nil = 'False'
              -- Drop *TextBlock concepts: their raw_value is the full HTML of
              -- the underlying table (with inline CSS). One such row can be
              -- 50–250KB and a concept-less query can return dozens, which
              -- blows up the agent's context. The prose content is covered by
              -- the narrative store (data/narrative_store/) and any numbers
              -- are covered by the per-element XBRL facts that sit alongside.
              AND concept NOT ILIKE '%TextBlock'
        ),
        extracted_norm AS (
            SELECT
                (ticker || '_' || filing_type || '_' || filing_period_end) AS filing_id,
                ticker                               AS ticker,
                filing_type                          AS form,
                TRY_CAST(filing_period_end AS DATE)  AS fiscal_period_end,
                'extracted'                          AS source,
                NULLIF(concept, '')                  AS concept,
                label                                AS label,
                TRY_CAST(value AS DOUBLE)            AS value,
                raw_display                          AS raw_value,
                unit                                 AS unit,
                period_type                          AS period_type,
                TRY_CAST(period_start AS DATE)       AS period_start,
                TRY_CAST(period_end AS DATE)         AS period_end,
                NULLIF(dimensions, '')               AS dimensions,
                source_title                         AS source_table_title
            FROM extracted_raw
        )
        SELECT * FROM xbrl_norm
        UNION ALL
        SELECT * FROM extracted_norm
    """)

    row_count = con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    filing_ids = [f"{f['ticker']}_{f['filing_type']}_{f['period_end']}" for f in filings]
    logger.info(
        f"DuckDB facts table loaded: {row_count} rows across {len(filings)} filings: "
        f"{', '.join(filing_ids)}"
    )
    return con


def get_connection() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        _conn = _build_connection()
    return _conn


def reset_connection() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
    _conn = None
