import os
import re
from pathlib import Path

from docling.document_converter import DocumentConverter


def _clean_docling_table_row(line: str) -> str | None:
    """Clean a Docling markdown table row by deduplicating repeated cells
    and removing empty cells.

    Docling represents merged table cells by repeating the value across
    every spanned column, e.g.:
        | Total assets | Total assets | Total assets | $ | 6,535,136 |  |  |
    becomes:
        | Total assets | $ | 6,535,136 |

    Returns None for rows that are entirely empty or just separators.
    """
    if not line.strip().startswith('|'):
        return line

    # Table separator row (| - | - | - |)
    if re.match(r'^(\|\s*-\s*)+\|?\s*$', line.strip()):
        return None

    cells = line.split('|')
    # First and last elements are empty strings from leading/trailing pipes
    if len(cells) < 3:
        return line

    inner = cells[1:-1]
    stripped_cells = [c.strip() for c in inner]

    # Drop empty cells and deduplicate consecutive identical values
    deduped = []
    prev = None
    for cell in stripped_cells:
        if not cell:
            continue
        if cell == prev:
            continue
        deduped.append(cell)
        prev = cell

    if not deduped:
        return None

    return '| ' + ' | '.join(deduped) + ' |'


def _remove_page_headers(line: str) -> bool:
    """Return True if the line is a Docling page header/footer to strip."""
    return bool(re.match(r'^.{5,60} \| \d+$', line.strip()))


def _postprocess_docling(md_content: str) -> str:
    """Post-process Docling markdown:
    1. Inject proper markdown headings (ITEM N, PART N, bold titles)
    2. Remove page headers/footers
    3. Clean table rows: deduplicate repeated cells, drop empty rows
    """
    lines = md_content.split('\n')
    result = []

    for line in lines:
        stripped = line.strip()

        if _remove_page_headers(stripped):
            continue

        if re.match(r'^ITEM\s+\d+[A-C]?\.?\s', stripped, re.IGNORECASE):
            result.append(f'\n## {stripped}')
            continue

        if re.match(r'^PART\s+[IVX]+\b', stripped, re.IGNORECASE):
            result.append(f'\n## {stripped}')
            continue

        bold_match = re.match(r'^\*\*(.+)\*\*$', stripped)
        if bold_match and len(stripped) < 120:
            title = bold_match.group(1)
            result.append(f'\n### {title}')
            continue

        if stripped.startswith('|'):
            cleaned = _clean_docling_table_row(stripped)
            if cleaned is not None:
                result.append(cleaned)
            continue

        result.append(line)

    return '\n'.join(result)


def _extract_period_of_report(accession_dir: Path) -> str | None:
    """
    Read the SEC-HEADER from full-submission.txt in an accession folder and
    return the period of report as YYYY-MM-DD, or None if not found.

    The header line looks like: `CONFORMED PERIOD OF REPORT:	20240930`
    """
    submission_file = accession_dir / "full-submission.txt"
    if not submission_file.exists():
        return None

    try:
        # Only scan the header — the file can be huge (embedded filings) and
        # the header appears in the first ~50 lines.
        with open(submission_file, "r", encoding="utf-8", errors="ignore") as f:
            head = "".join(next(f) for _ in range(200))
    except StopIteration:
        head = ""
    except Exception:
        return None

    match = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", head)
    if not match:
        return None

    raw = match.group(1)
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def parse_with_docling(input_paths: list[str], output_dir: str):
    """
    Parse documents (HTML, PDF, etc.) with Docling and export them to Markdown.

    Output filename format: ``{ticker}_{form_type}_{period}.md`` where period
    is the filing's "period of report" (YYYY-MM-DD). Including the period in
    the filename prevents multiple 10-Qs from the same company from
    overwriting each other and lets the indexer tag chunks with fiscal period.
    """
    os.makedirs(output_dir, exist_ok=True)

    converter = DocumentConverter()

    for path in input_paths:
        file_path = Path(path)
        if not file_path.exists():
            print(f"File not found: {path}")
            continue

        # Derive the output filename BEFORE running Docling so we can skip
        # files that have already been parsed — Docling conversion is slow
        # and we want `parse` to be idempotent / cheap to re-run.
        try:
            parts = file_path.parts
            base_idx = parts.index("sec-edgar-filings")
            ticker = parts[base_idx + 1]
            form_type = parts[base_idx + 2]

            # Supports both old ({ticker}/{form}/{accession}/) and
            # new ({ticker}/{form}/{period}/{accession}/) layouts.
            next_part = parts[base_idx + 3]
            if re.match(r"\d{4}-\d{2}-\d{2}$", next_part):
                period = next_part
                accession = parts[base_idx + 4]
                accession_dir = Path(*parts[: base_idx + 5])
            else:
                accession = next_part
                accession_dir = Path(*parts[: base_idx + 4])
                period = _extract_period_of_report(accession_dir)
                if period is None:
                    period = accession
                    print(
                        f"  WARN: could not extract period of report for "
                        f"{ticker} {form_type} {accession}; using accession id"
                    )
            out_filename = f"{ticker}_{form_type}_{period}.md"
        except (ValueError, IndexError):
            out_filename = file_path.with_suffix(".md").name

        out_path = os.path.join(output_dir, out_filename)
        if os.path.exists(out_path):
            print(f"Skipping (already parsed): {out_filename}")
            continue

        print(f"Parsing document: {file_path.name}")
        try:
            result = converter.convert(path)
            md_content = result.document.export_to_markdown(compact_tables=True)
            md_content = _postprocess_docling(md_content)

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            print(f"Successfully exported Markdown to {out_path}")

        except Exception as e:
            print(f"Error parsing {file_path.name}: {e}")


def get_all_sec_html_files(raw_dir: str) -> list[str]:
    """Find all downloaded SEC EDGAR HTML files (skips the full-submission txt)."""
    html_files = []
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.endswith(".html") or file.endswith(".htm") or file == "primary_doc.xml":
                html_files.append(os.path.join(root, file))
    return html_files


if __name__ == "__main__":
    rag_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(rag_app_dir, "data", "raw", "sec-edgar-filings")
    parsed_dir = os.path.join(rag_app_dir, "data", "parsed")

    print(f"Searching for SEC documents in {raw_dir}...")
    files_to_parse = get_all_sec_html_files(raw_dir)
    print(f"Found {len(files_to_parse)} documents to parse.")

    if len(files_to_parse) > 0:
        parse_with_docling(files_to_parse, parsed_dir)
