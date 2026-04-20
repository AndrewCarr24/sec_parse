import os
import re
import shutil
from pathlib import Path

from sec_edgar_downloader import Downloader


def _extract_period_from_submission(accession_dir: Path) -> str | None:
    """Read CONFORMED PERIOD OF REPORT from full-submission.txt → YYYY-MM-DD."""
    submission_file = accession_dir / "full-submission.txt"
    if not submission_file.exists():
        return None
    try:
        with open(submission_file, "r", encoding="utf-8", errors="ignore") as f:
            head = "".join(next(f) for _ in range(200))
    except (StopIteration, Exception):
        return None
    match = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", head)
    if not match:
        return None
    raw = match.group(1)
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def _reorganize_by_period(filings_dir: str):
    """Reorganize raw filings from {ticker}/{form}/{accession}/
    to {ticker}/{form}/{period}/{accession}/ so periods are visible
    in the folder structure.

    Skips directories that are already nested under a period folder
    (YYYY-MM-DD pattern).
    """
    filings_path = Path(filings_dir)
    if not filings_path.exists():
        return

    for ticker_dir in filings_path.iterdir():
        if not ticker_dir.is_dir():
            continue
        for form_dir in ticker_dir.iterdir():
            if not form_dir.is_dir():
                continue
            for child in list(form_dir.iterdir()):
                if not child.is_dir():
                    continue
                # Skip if already under a period folder (YYYY-MM-DD)
                if re.match(r"\d{4}-\d{2}-\d{2}$", child.name):
                    continue
                accession_name = child.name
                period = _extract_period_from_submission(child)
                if period is None:
                    print(f"  WARN: no period for {ticker_dir.name}/{form_dir.name}/{accession_name}, skipping reorg")
                    continue
                period_dir = form_dir / period
                period_dir.mkdir(exist_ok=True)
                dest = period_dir / accession_name
                if dest.exists():
                    continue
                shutil.move(str(child), str(dest))
                print(f"  Moved {ticker_dir.name}/{form_dir.name}/{accession_name} → …/{period}/{accession_name}")


def fetch_sec_filings(
    tickers: list[str],
    download_folder: str,
    form_type: str = "10-K",
    limit: int = 1,
):
    """
    Downloads SEC filings for a list of tickers into the specified folder,
    then reorganizes by period: {ticker}/{form}/{period}/{accession}/.
    """
    print(f"Initializing SEC Edgar Downloader in {download_folder}...")

    dl = Downloader("AI_Eval_Tool", "eval_user@example.com", download_folder)

    for ticker in tickers:
        print(f"Fetching {limit} recent {form_type} filings for {ticker}...")
        try:
            dl.get(form_type, ticker, limit=limit, download_details=True)
            print(f"Successfully downloaded {ticker} {form_type}.")
        except Exception as e:
            print(f"Failed to download {ticker} {form_type}: {e}")

    filings_dir = os.path.join(download_folder, "sec-edgar-filings")
    print("Reorganizing raw filings by period...")
    _reorganize_by_period(filings_dir)


if __name__ == "__main__":
    mi_tickers = ["ACT", "ESNT", "MTG", "RDN", "ACGL", "NMIH"]

    rag_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(rag_app_dir, "data", "raw")
    print(f"Targeting download directory: {target_dir}")

    # Annual: most recent 10-K per company.
    fetch_sec_filings(mi_tickers, target_dir, form_type="10-K", limit=1)

    # Quarterly: most recent 4 10-Qs covers roughly the last year of reporting.
    # (A fiscal year has three 10-Qs + one 10-K, so 4 is a full trailing year.)
    fetch_sec_filings(mi_tickers, target_dir, form_type="10-Q", limit=4)
