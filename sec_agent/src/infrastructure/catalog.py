"""Catalog of available filings under data/extracted_facts_and_tables/.

The catalog is derived from the directory layout:
    data/extracted_facts_and_tables/{TICKER}/{FILING_TYPE}_{YYYY-MM-DD}/

`list_filings()` returns dicts with both machine-friendly (period_end) and
human-friendly (period_label) forms. `format_for_prompt()` renders them as
a compact block suitable for the agent's system prompt — so the agent can
pick the right ticker+period for a user question without calling a tool.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
FACTS_ROOT = _REPO_ROOT / "data" / "extracted_facts_and_tables"

_FILING_DIR_RE = re.compile(r"^(?P<form>10-K|10-Q|10-K/A|10-Q/A)_(?P<period>\d{4}-\d{2}-\d{2})$")

TICKER_TO_COMPANY = {
    "ACT": "Enact Holdings",
    "RDN": "Radian",
    "NMIH": "NMI Holdings",
    "ESNT": "Essent",
    "MTG": "MGIC",
    "ACGL": "Arch Capital",
}


def _period_label(filing_type: str, period_end: str) -> str:
    year = period_end[:4]
    month = int(period_end[5:7])
    form = filing_type.split("/", 1)[0].upper()
    if form == "10-K":
        return f"FY {year}"
    if form == "10-Q":
        quarter = (month - 1) // 3 + 1
        return f"Q{quarter} {year}"
    return period_end


def list_filings() -> list[dict]:
    """Scan FACTS_ROOT and return one dict per filing directory.

    A directory is considered a filing if it contains xbrl_facts.csv — this
    keeps half-finished or empty subfolders out of the catalog.
    """
    if not FACTS_ROOT.exists():
        return []
    out: list[dict] = []
    for ticker_dir in sorted(FACTS_ROOT.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        for filing_dir in sorted(ticker_dir.iterdir()):
            if not filing_dir.is_dir():
                continue
            match = _FILING_DIR_RE.match(filing_dir.name)
            if not match:
                continue
            if not (filing_dir / "xbrl_facts.csv").exists():
                continue
            form = match.group("form")
            period_end = match.group("period")
            out.append(
                {
                    "ticker": ticker,
                    "company": TICKER_TO_COMPANY.get(ticker, ticker),
                    "filing_type": form,
                    "period_end": period_end,
                    "period_label": _period_label(form, period_end),
                    "path": str(filing_dir),
                }
            )
    return out


def format_for_prompt() -> str:
    """Render the catalog as a compact table for the system prompt.

    Returns 'No filings indexed yet.' when empty so the prompt block
    remains well-formed even before the first filing is processed.
    """
    filings = list_filings()
    if not filings:
        return "No filings indexed yet."
    lines = ["ticker | company | form | period_label | period_end (filter on this)"]
    for f in filings:
        lines.append(
            f"{f['ticker']} | {f['company']} | {f['filing_type']} | "
            f"{f['period_label']} | {f['period_end']}"
        )
    return "\n".join(lines)
