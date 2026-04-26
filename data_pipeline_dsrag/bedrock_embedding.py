"""Bedrock Titan v2 embedding adapter for dsRAG.

Matches the embedding space used by our existing narrative_search store
so dsRAG's KB is built on the same vectors we've been evaluating.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import boto3

from dsrag.database.vector.types import Vector
from dsrag.embedding import Embedding


TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"
TITAN_DIM = 1024


class BedrockTitanEmbedding(Embedding):
    """Amazon Bedrock Titan v2 via boto3. Single-call per text (Titan's
    runtime API doesn't batch), so embedding throughput is bounded by
    Bedrock's request rate, not ours."""

    def __init__(self, dimension: Optional[int] = None) -> None:
        super().__init__(dimension=dimension or TITAN_DIM)
        region = os.environ.get("AWS_REGION", "us-east-1")
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def get_embeddings(
        self, text: list[str] | str, input_type: Optional[str] = None
    ) -> list[Vector] | Vector:
        is_single = isinstance(text, str)
        texts = [text] if is_single else list(text)
        out: list[Vector] = []
        for t in texts:
            resp = self._client.invoke_model(
                modelId=TITAN_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": t}),
            )
            body = json.loads(resp["body"].read())
            out.append(body["embedding"])
        return out[0] if is_single else out

    def to_dict(self):
        # dimension is serialized by the parent; no other config to persist.
        return super().to_dict()
