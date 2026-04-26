"""
Extract financial tables from an SEC iXBRL filing that the filer did NOT
tag with XBRL — primarily MD&A operational tables (insurance-in-force,
delinquency, persistency, etc. for mortgage insurers).

Pipeline (hybrid):
    1. Parse submission, isolate every <table> in the primary document.
    2. Skip tables that already have <ix:*> XBRL tags (captured by
       `ixbrl_parser.py`).
    3. Skip layout-scaffold tables (no numeric content).
    4. For each surviving table, find the nearest preceding heading and
       prior paragraph so the LLM knows what the table is *about*.
    5. Send table + context to Claude; get back structured long-format rows.
    6. Emit rows in a schema compatible with `xbrl_facts.csv` so the agent
       tool can query both sources with one SQL/pandas call.

Outputs (in `output_dir`):
    extracted_facts.csv        — long-format rows, schema mirrors xbrl_facts.csv
                                 with extra `label` + `source="extracted"` columns
    extracted_tables.json      — per-table audit: heading, raw HTML excerpt,
                                 LLM response, final rows
    extraction_log.txt         — per-table disposition

Usage:
    python table_extractor.py <accession_dir> <output_dir> [--dry-run] [--limit N]

Requires DEEPSEEK_API_KEY.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

# Reuse the SGML-submission unpacker and filing-id derivation from the XBRL parser.
sys.path.insert(0, str(Path(__file__).parent))
from ixbrl_parser import default_output_dir, derive_filing_id, extract_submission  # noqa: E402


# Model provider for table extraction. Default is DeepSeek v4 Flash via
# their OpenAI-compatible API (10× cheaper than Haiku, comparable quality
# on structured JSON output). To switch back to Haiku, set MODEL to
# "claude-haiku-4-5-20251001" and change the client/call in _call_llm.
MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MAX_OUTPUT_TOKENS = 20000  # large tables (IIF-by-LTV, delinquency-by-state) need ~10–15K
NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
CONTEXT_CHAR_BUDGET = 1200  # rendered text length of preceding context


# ---------------------------------------------------------------------------
# Stage 1–3: find candidate tables
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    index: int
    heading: str
    preceding_text: str
    html: str
    num_numbers: int
    num_rows: int


def _rendered_text(node: Tag | NavigableString) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)) if hasattr(node, "get_text") else str(node).strip()


def _find_heading(table: Tag) -> tuple[str, str]:
    """Walk backward from a <table> to find the nearest heading-like block
    and the preceding paragraph(s), up to CONTEXT_CHAR_BUDGET chars."""
    heading = ""
    preceding_pieces: list[str] = []
    budget = CONTEXT_CHAR_BUDGET

    for prev in table.find_all_previous(limit=60):
        if not isinstance(prev, Tag):
            continue
        text = _rendered_text(prev)
        if not text:
            continue

        # Treat as "heading" if it's short, bolded, or matches ITEM/PART.
        looks_like_heading = (
            prev.name in ("h1", "h2", "h3", "h4", "h5", "h6")
            or (prev.name in ("b", "strong"))
            or bool(re.match(r"^(ITEM\s+\d+[A-C]?|PART\s+[IVX]+)\b", text, re.IGNORECASE))
            or (prev.name == "span" and "font-weight:700" in (prev.get("style") or ""))
        )
        if looks_like_heading and not heading and len(text) < 200:
            heading = text
            break

        if len(text) < budget:
            preceding_pieces.append(text)
            budget -= len(text)
        if budget <= 0:
            break

    preceding_text = " | ".join(reversed(preceding_pieces))[:CONTEXT_CHAR_BUDGET]
    return heading, preceding_text


def _clean_table_html(table: Tag) -> str:
    """Strip style/class attributes to shrink prompt tokens."""
    clone = BeautifulSoup(str(table), "lxml").find("table")
    if clone is None:
        return str(table)
    for el in clone.find_all(True):
        for attr in list(el.attrs):
            if attr not in ("colspan", "rowspan"):
                del el.attrs[attr]
    return str(clone)


def find_candidates(html: str) -> list[Candidate]:
    soup = BeautifulSoup(html, "lxml")
    candidates: list[Candidate] = []

    for i, table in enumerate(soup.find_all("table")):
        table_text = _rendered_text(table)
        numbers = NUMBER_RE.findall(table_text)

        # Already captured by XBRL (or pure layout scaffold).
        if table.find(lambda t: isinstance(t, Tag) and t.name and t.name.startswith("ix:")):
            continue
        # Minimum numeric density — 3 numbers is a weak but cheap filter.
        if len(numbers) < 3:
            continue

        heading, preceding = _find_heading(table)
        candidates.append(Candidate(
            index=i,
            heading=heading,
            preceding_text=preceding,
            html=_clean_table_html(table),
            num_numbers=len(numbers),
            num_rows=len(table.find_all("tr")),
        ))
    return candidates


# ---------------------------------------------------------------------------
# Stage 5: LLM structuring
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You extract structured data from SEC filing tables. Every \
number in the table becomes one row of output. Normalize scale (thousands, \
millions) using the table's units note. Parentheses mean negative. Respect \
period scoping: column headers tell you whether values are instants \
(balance-sheet dates) or durations (three/nine months ended …). Emit JSON \
only — no prose, no markdown fences."""

USER_TEMPLATE = """\
FILING METADATA
    entity:         {entity_name} ({ticker})
    form:           {form}
    period_end:     {period_end}
    fiscal_period:  {fiscal_period_focus} {fiscal_year_focus}

SECTION HEADING (from the document, verbatim):
    {heading}

PRECEDING NARRATIVE (for context about what this table is):
    {preceding_text}

TABLE HTML:
{table_html}

TASK

Emit JSON with this schema (and NOTHING else):

{{
  "title": "short caption of what this table reports",
  "is_data_table": true | false,
  "skip_reason": "…" | null,
  "units_note": "e.g. 'Amounts in thousands' — null if not stated",
  "rows": [
    {{
      "label": "leftmost-cell row label",
      "raw_display": "exact text as shown, e.g. '(67,225)' or '$249,055'",
      "value": <number, after sign and scale applied>,
      "unit": "USD" | "shares" | "percent" | "ratio" | "count" | "USD_per_share" | "years" | "months" | "other",
      "period_type": "instant" | "duration",
      "period_start": "YYYY-MM-DD" | null,
      "period_end": "YYYY-MM-DD",
      "dimensions": {{}}
    }}
  ]
}}

RULES

- Set is_data_table=false (and rows=[]) for tables-of-contents, page-number \
lists, or tables containing only layout/text.
- When a column header says "Three months ended September 30, 2024": \
period_type="duration", period_start="2024-07-01", period_end="2024-09-30".
- When a column header says "September 30, 2024" alone: period_type="instant", \
period_start=null, period_end="2024-09-30".
- If a table breaks out by segment / book year / FICO band / LTV / geography, \
put the breakdown in `dimensions`, e.g. {{"book_year": "2023"}} or \
{{"segment": "Mortgage Insurance"}}.
- If unit is percent: value is the number as shown (e.g. "2.5%" → value=2.5).
- If unit is USD and units_note says "in thousands", multiply by 1000 \
(e.g. "$249,055 thousand" → value=249055000).
- Never invent values not in the table. If a cell is blank or "—", skip it.
"""


def _call_llm(client, candidate: Candidate, filing_meta: dict) -> dict:
    user = USER_TEMPLATE.format(
        entity_name=filing_meta.get("entity_name") or "",
        ticker=filing_meta.get("ticker") or "",
        form=filing_meta.get("document_type") or "",
        period_end=filing_meta.get("period_end_date") or "",
        fiscal_period_focus=filing_meta.get("fiscal_period_focus") or "",
        fiscal_year_focus=filing_meta.get("fiscal_year_focus") or "",
        heading=candidate.heading or "(no heading found)",
        preceding_text=candidate.preceding_text or "(no preceding text)",
        table_html=candidate.html,
    )
    # DeepSeek's OpenAI-compatible Chat Completions API. `client` is an
    # openai.OpenAI pointed at DEEPSEEK_BASE_URL. After the SDK's own
    # retry budget is exhausted, record the failure and continue so one
    # bad table doesn't poison the whole extraction run — re-run targeted
    # tables afterwards using the log.
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
    except Exception as e:
        return {
            "title": "API_ERROR",
            "is_data_table": False,
            "skip_reason": f"{type(e).__name__}: {e}",
            "units_note": None,
            "rows": [],
        }
    text = (resp.choices[0].message.content or "").strip()
    # Strip accidental code fences.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {
            "title": "PARSE_ERROR",
            "is_data_table": False,
            "skip_reason": f"json parse: {e}",
            "units_note": None,
            "rows": [],
            "_raw": text[:2000],
        }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def extract_filing(
    accession_dir: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> None:
    accession_dir = Path(accession_dir)
    ticker, filing_type, filing_period_end = derive_filing_id(accession_dir)
    if output_dir is None:
        output_dir = default_output_dir(accession_dir)
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Filing: {ticker} {filing_type} {filing_period_end} → {output_dir}")

    filing_meta: dict = {}
    meta_file = output_dir / "xbrl_metadata.json"
    if meta_file.exists():
        filing_meta = json.loads(meta_file.read_text())
    else:
        print(f"NOTE: {meta_file} not found — run ixbrl_parser.py first for full context.")

    with tempfile.TemporaryDirectory() as td:
        primary = extract_submission(accession_dir / "full-submission.txt", Path(td))
        html = primary.read_text(encoding="utf-8", errors="replace")

    candidates = find_candidates(html)
    print(f"Found {len(candidates)} candidate tables (untagged, ≥3 numbers).")
    if limit is not None:
        candidates = candidates[:limit]
        print(f"Processing first {len(candidates)} (--limit).")

    if dry_run:
        print("\n=== DRY RUN — table list ===")
        for c in candidates:
            print(f"[{c.index:>3}] rows={c.num_rows:>3} numbers={c.num_numbers:>4}  "
                  f"heading={c.heading[:70]!r}")
        print(f"\nCost estimate: ~{len(candidates)} LLM calls ({MODEL}).")
        return

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set.", file=sys.stderr)
        sys.exit(2)

    from openai import OpenAI
    # Timeout of 120s is enough for a legitimately long extraction on a
    # big table (IIF-by-state, ~10–15K output tokens) while catching a
    # dead connection fast. The SDK retries transient errors internally,
    # and _call_llm wraps persistent failures so the pipeline finishes
    # even when individual tables fail.
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=DEEPSEEK_BASE_URL,
        timeout=120.0,
        max_retries=3,
    )

    audit: list[dict] = []
    all_rows: list[dict] = []
    log_lines: list[str] = []

    for n, cand in enumerate(candidates, 1):
        print(f"[{n}/{len(candidates)}] table #{cand.index} "
              f"({cand.num_rows} rows, {cand.num_numbers} numbers) "
              f"heading={cand.heading[:60]!r}")
        result = _call_llm(client, cand, filing_meta)

        is_data = bool(result.get("is_data_table"))
        rows = result.get("rows") or []
        skip_reason = result.get("skip_reason")
        log_lines.append(
            f"#{cand.index}: "
            f"{'DATA' if is_data else 'SKIP'} "
            f"rows={len(rows)} "
            f"title={result.get('title','')[:60]!r} "
            f"{'skip_reason='+repr(skip_reason) if skip_reason else ''}"
        )
        audit.append({
            "table_index": cand.index,
            "heading": cand.heading,
            "preceding_text": cand.preceding_text[:400],
            "html_excerpt": cand.html[:1500],
            "llm_response": result,
        })

        if is_data:
            for r in rows:
                all_rows.append({
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "filing_period_end": filing_period_end,
                    "source": "extracted",
                    "source_table_index": cand.index,
                    "source_heading": cand.heading,
                    "source_title": result.get("title"),
                    "concept": None,
                    "label": r.get("label"),
                    "raw_display": r.get("raw_display"),
                    "value": r.get("value"),
                    "unit": r.get("unit"),
                    "period_type": r.get("period_type"),
                    "period_start": r.get("period_start"),
                    "period_end": r.get("period_end"),
                    "dimensions": json.dumps(r.get("dimensions") or {}),
                })

    # Emit outputs.
    facts_path = output_dir / "extracted_facts.csv"
    fieldnames = [
        "ticker", "filing_type", "filing_period_end",
        "source", "source_table_index", "source_heading", "source_title",
        "concept", "label", "raw_display", "value", "unit",
        "period_type", "period_start", "period_end", "dimensions",
    ]
    with facts_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    (output_dir / "extracted_tables.json").write_text(json.dumps(audit, indent=2))
    (output_dir / "extraction_log.txt").write_text("\n".join(log_lines) + "\n")

    print(f"\nWrote {len(all_rows)} rows from {sum(1 for a in audit if a['llm_response'].get('is_data_table'))}/{len(audit)} data tables → {facts_path}")
    print(f"Per-table audit → {output_dir / 'extracted_tables.json'}")
    print(f"Log → {output_dir / 'extraction_log.txt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accession_dir")
    ap.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Defaults to data/extracted_facts_and_tables/<TICKER>/<FORM>_<YYYY-MM-DD>/.",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="List candidate tables; do not call the LLM.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N tables (for cost-controlled testing).")
    args = ap.parse_args()
    out = Path(args.output_dir) if args.output_dir else None
    extract_filing(Path(args.accession_dir), out,
                   dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
