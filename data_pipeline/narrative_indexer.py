"""Index the narrative (non-tabular) portions of SEC filings into Chroma.

MVP scope: only data/parsed/ACT_10-Q_2024-09-30.md. Markdown tables are
stripped before chunking so the vector store holds prose only — tabular
data is already structured in test_output/xbrl_facts.csv +
test_output/extracted_facts.csv and served via the agent's SQL tools.

Embeddings: Amazon Titan v2 via Bedrock (no local torch/sentence-transformers
required). Requires AWS creds with Bedrock access.

Usage:
    python data_pipeline/narrative_indexer.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import boto3
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


_REPO_ROOT = Path(__file__).resolve().parents[1]
PARSED_DIR = _REPO_ROOT / "data" / "parsed"
STORE_DIR = _REPO_ROOT / "data" / "narrative_store"
COLLECTION_NAME = "narrative_titan_v2"
TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"

TICKER_TO_COMPANY = {
    "ACT": "Enact",
    "RDN": "Radian",
    "NMIH": "NMI Holdings",
    "ESNT": "Essent",
    "MTG": "MGIC",
    "ACGL": "Arch Capital",
}

INCLUDE_FILINGS = {"ACT_10-Q_2024-09-30.md"}


def _strip_markdown_tables(text: str) -> str:
    """Collapse runs of `|`-delimited rows into a single `[table omitted]` marker."""
    out: list[str] = []
    in_table = False
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            if not in_table:
                out.append("[table omitted]")
                in_table = True
            continue
        in_table = False
        out.append(line)
    return "\n".join(out)


def _period_label(filing_type: str, period: str) -> str:
    if not period or len(period) < 7 or period[4] != "-":
        return period or "unknown period"
    year = period[:4]
    month = int(period[5:7])
    ft = filing_type.upper()
    if ft.startswith("10-K"):
        return f"FY {year}"
    if ft.startswith("10-Q"):
        quarter = (month - 1) // 3 + 1
        return f"Q{quarter} {year}"
    return period


def _chunk(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator=True,
    )
    return [c for c in splitter.split_text(text) if c.strip()]


class _TitanEmbedder:
    def __init__(self) -> None:
        region = os.environ.get("AWS_REGION", "us-east-1")
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            resp = self._client.invoke_model(
                modelId=TITAN_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": t}),
            )
            body = json.loads(resp["body"].read())
            out.append(body["embedding"])
        return out


def index() -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(STORE_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    embedder = _TitanEmbedder()
    total = 0

    for filename in sorted(os.listdir(PARSED_DIR)):
        if filename not in INCLUDE_FILINGS:
            continue
        print(f"Indexing {filename}...")
        text = (PARSED_DIR / filename).read_text()
        narrative = _strip_markdown_tables(text)
        chunks = _chunk(narrative)

        stem = filename.replace(".md", "")
        parts = stem.split("_")
        ticker = parts[0]
        filing_type = parts[1] if len(parts) > 1 else "unknown"
        period = parts[2] if len(parts) > 2 else ""
        company = TICKER_TO_COMPANY.get(ticker, ticker)
        plabel = _period_label(filing_type, period)

        print(f"  {len(chunks)} chunks — embedding via Titan v2...")
        vectors = embedder.encode(chunks)
        ids = [f"{stem}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": filename,
                "ticker": ticker,
                "company": company,
                "filing_type": filing_type,
                "period": period,
                "period_label": plabel,
            }
            for _ in chunks
        ]
        collection.add(
            ids=ids, embeddings=vectors, documents=chunks, metadatas=metadatas
        )
        total += len(chunks)

    print(f"Done. {total} chunks written to {STORE_DIR}")


if __name__ == "__main__":
    index()
