"""Semantic search over narrative chunks indexed by data_pipeline/narrative_indexer.py.

Lazily opens the Chroma collection on first query and embeds queries via
Bedrock Titan v2 to match the indexer's embedding space.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

import boto3
import chromadb


_REPO_ROOT = Path(__file__).resolve().parents[3]
STORE_DIR = _REPO_ROOT / "data" / "narrative_store"
COLLECTION_NAME = "narrative_titan_v2"
TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"

_collection = None
_bedrock = None
_lock = Lock()


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    with _lock:
        if _collection is None:
            client = chromadb.PersistentClient(path=str(STORE_DIR))
            _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def _get_bedrock():
    global _bedrock
    if _bedrock is not None:
        return _bedrock
    with _lock:
        if _bedrock is None:
            region = os.environ.get("AWS_REGION", "us-east-1")
            _bedrock = boto3.client("bedrock-runtime", region_name=region)
    return _bedrock


def _embed(text: str) -> list[float]:
    resp = _get_bedrock().invoke_model(
        modelId=TITAN_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text}),
    )
    body = json.loads(resp["body"].read())
    return body["embedding"]


def search(
    query: str,
    top_k: int = 4,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    """Semantic search over indexed narrative chunks.

    When `exclude_ids` is provided we over-fetch (3× top_k) and filter out
    already-seen chunk IDs, preserving `top_k` fresh results — so repeated
    calls within one agent run don't re-send the same chunks.
    """
    collection = _get_collection()
    vec = _embed(query)
    exclude = exclude_ids or set()
    n_fetch = top_k * 3 if exclude else top_k
    res = collection.query(
        query_embeddings=[vec],
        n_results=n_fetch,
        include=["documents", "metadatas"],
    )
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    out: list[dict] = []
    for cid, d, m in zip(ids, docs, metas):
        if cid in exclude:
            continue
        out.append({
            "id": cid,
            "company": m.get("company", "unknown"),
            "filing_type": m.get("filing_type", ""),
            "period_label": m.get("period_label", ""),
            "source": m.get("source", ""),
            "text": d,
        })
        if len(out) >= top_k:
            break
    return out
