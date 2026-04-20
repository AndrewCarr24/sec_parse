"""In-memory DuckDB exposing XBRL + extracted facts as a unified `facts` table.

The upstream pipeline emits two CSVs per filing in `test_output/`:
    xbrl_facts.csv       — machine-tagged facts from the iXBRL instance
    extracted_facts.csv  — LLM-extracted facts from untagged HTML tables

Both are folded into one long-format `facts` table so the agent can query
across both sources with a single SQL surface.
"""

import json
from pathlib import Path

import duckdb
from loguru import logger

from src.config import settings


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


def _load_metadata(test_output_dir: Path) -> dict:
    meta_path = test_output_dir / "xbrl_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    return json.loads(meta_path.read_text())


def _build_connection() -> duckdb.DuckDBPyConnection:
    test_output_dir = Path(settings.TEST_OUTPUT_DIR)
    meta = _load_metadata(test_output_dir)
    ticker = meta["ticker"]
    form = meta["document_type"]
    period_end = meta["period_end_date"]
    filing_id = f"{ticker}_{form}_{period_end}"

    xbrl_csv = test_output_dir / "xbrl_facts.csv"
    extracted_csv = test_output_dir / "extracted_facts.csv"
    for p in (xbrl_csv, extracted_csv):
        if not p.exists():
            raise FileNotFoundError(p)

    con = duckdb.connect(":memory:")

    # Parameters can't substitute for column literals — interpolate the
    # per-filing constants once here, then register the CSVs.
    unit_case = "CASE unit " + " ".join(
        f"WHEN '{k}' THEN '{v}'" for k, v in _UNIT_ALIASES.items()
    ) + " ELSE unit END"

    con.execute(f"""
        CREATE TABLE facts AS
        WITH xbrl_raw AS (
            SELECT * FROM read_csv_auto('{xbrl_csv.as_posix()}', header=True, all_varchar=True)
        ),
        extracted_raw AS (
            SELECT * FROM read_csv_auto('{extracted_csv.as_posix()}', header=True, all_varchar=True)
        ),
        xbrl_norm AS (
            SELECT
                '{filing_id}'                       AS filing_id,
                '{ticker}'                          AS ticker,
                '{form}'                            AS form,
                DATE '{period_end}'                 AS fiscal_period_end,
                'xbrl'                              AS source,
                concept                             AS concept,
                NULL                                AS label,
                TRY_CAST(value AS DOUBLE)           AS value,
                value                               AS raw_value,
                {unit_case}                         AS unit,
                period_type                         AS period_type,
                TRY_CAST(period_start AS DATE)      AS period_start,
                -- XBRL uses exclusive end-dates: instants stored as day-after
                -- (2024-09-30 → 2024-10-01) and durations end at midnight of
                -- the day after the last included day. Subtract 1 day in both
                -- cases so dates read as humans expect (Sept 30, not Oct 1).
                (TRY_CAST(period_end AS DATE) - INTERVAL 1 DAY)::DATE
                                                     AS period_end,
                NULLIF(dimensions, '')              AS dimensions,
                NULL                                AS source_table_title
            FROM xbrl_raw
            WHERE is_nil = 'False'
        ),
        extracted_norm AS (
            SELECT
                '{filing_id}'                       AS filing_id,
                '{ticker}'                          AS ticker,
                '{form}'                            AS form,
                DATE '{period_end}'                 AS fiscal_period_end,
                'extracted'                         AS source,
                NULLIF(concept, '')                 AS concept,
                label                               AS label,
                TRY_CAST(value AS DOUBLE)           AS value,
                raw_display                         AS raw_value,
                unit                                AS unit,
                period_type                         AS period_type,
                TRY_CAST(period_start AS DATE)      AS period_start,
                TRY_CAST(period_end AS DATE)        AS period_end,
                NULLIF(dimensions, '')              AS dimensions,
                source_title                        AS source_table_title
            FROM extracted_raw
        )
        SELECT * FROM xbrl_norm
        UNION ALL
        SELECT * FROM extracted_norm
    """)

    row_count = con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    logger.info(f"DuckDB facts table loaded: {row_count} rows ({filing_id})")
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
