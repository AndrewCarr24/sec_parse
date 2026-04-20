"""
Parse an SEC iXBRL filing with Arelle and emit a long-format facts table.

Input:  accession directory containing `full-submission.txt` (the SGML-wrapped
        EDGAR submission) and `primary-document.html` (the iXBRL instance).

Arelle needs the entity extension schema (e.g. `act-20240930.xsd`) and its
linkbases to resolve concept QNames. Those files live inside
`full-submission.txt`, not on disk alongside the HTML — so we unpack the
submission into a temp directory and point Arelle at the extracted primary
document.

Outputs (in `output_dir`), all prefixed `xbrl_` to distinguish from the
sibling `table_extractor.py` outputs (prefix `extracted_`):
    xbrl_facts.csv       — one row per reported XBRL fact
    xbrl_contexts.json   — period + dimensional context definitions
    xbrl_units.json      — unit definitions (USD, shares, USD/share, …)
    xbrl_metadata.json   — filing-level DEI (CIK, form, period, entity, …)
"""

import csv
import json
import re
import sys
import tempfile
from pathlib import Path

from arelle import Cntlr


SUBMISSION_FILE = "full-submission.txt"

# Only unpack files Arelle might need to resolve the instance. Exhibits
# (graphics, PDFs) are uuencoded inside the SGML wrapper and would corrupt
# if written as text — we don't need them anyway.
UNPACK_EXTENSIONS = {".htm", ".html", ".xml", ".xsd"}


def extract_submission(submission_txt: Path, out_dir: Path) -> Path:
    """Unpack SEC full-submission.txt; return path of the primary iXBRL doc."""
    raw = submission_txt.read_text(encoding="utf-8", errors="ignore")
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_re = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL)
    fname_re = re.compile(r"<FILENAME>([^\n<]+)", re.IGNORECASE)
    text_re = re.compile(r"<TEXT>\s*(.*?)\s*</TEXT>", re.DOTALL | re.IGNORECASE)

    primary: Path | None = None
    for doc in doc_re.finditer(raw):
        body = doc.group(1)
        fname_match = fname_re.search(body)
        text_match = text_re.search(body)
        if not fname_match or not text_match:
            continue
        fname = fname_match.group(1).strip()
        if Path(fname).suffix.lower() not in UNPACK_EXTENSIONS:
            continue
        (out_dir / fname).write_text(text_match.group(1), encoding="utf-8", errors="replace")
        if primary is None:
            primary = out_dir / fname

    if primary is None:
        raise RuntimeError(f"No parseable DOCUMENT sections found in {submission_txt}")
    return primary


def _period_fields(ctx) -> tuple[str, str | None, str | None]:
    """Return (period_type, period_start, period_end) for a context."""
    if ctx is None:
        return "unknown", None, None
    if ctx.isInstantPeriod:
        return "instant", None, ctx.instantDatetime.date().isoformat()
    if ctx.isStartEndPeriod:
        return (
            "duration",
            ctx.startDatetime.date().isoformat(),
            ctx.endDatetime.date().isoformat(),
        )
    return "forever", None, None


def _dimensions(ctx) -> dict[str, str | None]:
    if ctx is None:
        return {}
    out: dict[str, str | None] = {}
    for dim in ctx.qnameDims.values():
        if dim.isTyped:
            out[str(dim.dimensionQname)] = (
                dim.typedMember.stringValue if dim.typedMember is not None else None
            )
        else:
            out[str(dim.dimensionQname)] = str(dim.memberQname)
    return out


def _unit_string(unit) -> str:
    """Render a unit as 'iso4217:USD' or 'iso4217:USD / xbrli:shares'."""
    if unit is None:
        return ""
    measures = unit.measures  # (numerator_list, denominator_list)
    num = " * ".join(str(m) for m in measures[0])
    den = " * ".join(str(m) for m in measures[1])
    return f"{num} / {den}" if den else num


def parse_filing(accession_dir: Path, output_dir: Path) -> None:
    accession_dir = Path(accession_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    submission = accession_dir / SUBMISSION_FILE
    if not submission.exists():
        raise FileNotFoundError(submission)

    with tempfile.TemporaryDirectory() as td:
        primary = extract_submission(submission, Path(td))
        print(f"Extracted primary document: {primary.name}")

        cntlr = Cntlr.Cntlr(logFileName="logToPrint")
        try:
            model = cntlr.modelManager.load(str(primary))
            if model is None:
                raise RuntimeError(f"Arelle failed to load {primary}")

            # --- DEI / filing-level metadata ---
            dei: dict[str, str] = {}
            for fact in model.facts:
                qname = str(fact.qname)
                if qname.startswith("dei:") and not fact.isNil:
                    dei[qname.split(":", 1)[1]] = str(fact.value)

            metadata = {
                "source_submission": str(submission),
                "primary_document": primary.name,
                "entity_cik": dei.get("EntityCentralIndexKey"),
                "entity_name": dei.get("EntityRegistrantName"),
                "ticker": dei.get("TradingSymbol"),
                "document_type": dei.get("DocumentType"),
                "period_end_date": dei.get("DocumentPeriodEndDate"),
                "fiscal_year_focus": dei.get("DocumentFiscalYearFocus"),
                "fiscal_period_focus": dei.get("DocumentFiscalPeriodFocus"),
                "amendment_flag": dei.get("AmendmentFlag"),
                "dei_facts": dei,
            }
            (output_dir / "xbrl_metadata.json").write_text(json.dumps(metadata, indent=2))

            # --- Contexts ---
            contexts: dict[str, dict] = {}
            for ctx_id, ctx in model.contexts.items():
                period_type, period_start, period_end = _period_fields(ctx)
                contexts[ctx_id] = {
                    "entity_identifier": (
                        ctx.entityIdentifier[1] if ctx.entityIdentifier else None
                    ),
                    "period_type": period_type,
                    "period_start": period_start,
                    "period_end": period_end,
                    "dimensions": _dimensions(ctx),
                }
            (output_dir / "xbrl_contexts.json").write_text(json.dumps(contexts, indent=2))

            # --- Units ---
            units = {uid: _unit_string(u) for uid, u in model.units.items()}
            (output_dir / "xbrl_units.json").write_text(json.dumps(units, indent=2))

            # --- Facts (long format) ---
            rows = []
            for fact in model.facts:
                ctx = fact.context
                period_type, period_start, period_end = _period_fields(ctx)
                dims = _dimensions(ctx)
                rows.append({
                    "concept": str(fact.qname),
                    "value": None if fact.isNil else fact.value,
                    "is_nil": fact.isNil,
                    "decimals": fact.decimals,
                    "unit_id": fact.unitID or "",
                    "unit": _unit_string(fact.unit) if fact.unit is not None else "",
                    "context_id": fact.contextID,
                    "period_type": period_type,
                    "period_start": period_start,
                    "period_end": period_end,
                    "dimensions": json.dumps(dims) if dims else "",
                })

            facts_csv = output_dir / "xbrl_facts.csv"
            with facts_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            print(f"Wrote {len(rows)} facts → {facts_csv}")
            print(f"Wrote {len(contexts)} contexts → {output_dir / 'xbrl_contexts.json'}")
            print(f"Wrote {len(units)} units → {output_dir / 'xbrl_units.json'}")
            print(f"Wrote metadata → {output_dir / 'xbrl_metadata.json'}")

            if model.errors:
                print(f"\nArelle reported {len(model.errors)} load-time issues "
                      f"(facts still extracted; inspect if needed).")
        finally:
            cntlr.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ixbrl_parser.py <accession_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)
    parse_filing(Path(sys.argv[1]), Path(sys.argv[2]))
