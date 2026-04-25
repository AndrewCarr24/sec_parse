"""Probe whether Q13 (tax rate) and Q14 (gross margin) failures are retrieval
issues or agent-harness issues.

For each question:
  1. Call kb.query([verbatim_question]) with no rewriting, scoped to the
     Boeing 2022 10-K via doc_id filter.
  2. Print segment scores and content; flag whether key literal values
     (FinanceBench ground truth) appear in the retrieved text.
  3. Send segments + question to a single Flash call (no agent loop) and
     print the resulting answer for comparison.

Run from sec_agent/:
    set -a && . ../.env && set +a
    python scripts/probe_q13_q14.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Register dsRAG subclasses
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "data_pipeline_dsrag"))
import bedrock_embedding  # noqa: F401, E402
import flashrank_reranker  # noqa: F401, E402

# Route dsRAG's OpenAI client at DeepSeek
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))
os.environ.setdefault("DSRAG_OPENAI_BASE_URL", "https://api.deepseek.com/v1")

from dsrag.knowledge_base import KnowledgeBase  # noqa: E402
from dsrag.reranker import NoReranker  # noqa: E402
from openai import OpenAI  # noqa: E402


KB_DIR = _REPO_ROOT / "data" / "dsrag_store"
DOC_ID = "BA_10-K_2022-12-31"

QUESTIONS = [
    {
        "id": "Q13",
        "question": (
            "How does Boeing's effective tax rate in FY2022 compare to FY2021?"
        ),
        "expected": (
            "Effective tax rate in FY2022 was 0.62%, compared to -14.76% in FY2021."
        ),
        "key_literals": [
            ("-14.76", "FY21 effective tax rate (signed)"),
            ("(14.76)", "FY21 effective tax rate (parenthesis-negative)"),
            ("14.76", "FY21 ETR magnitude (any sign)"),
            ("0.62", "FY22 effective tax rate"),
            ("tax benefit", "narrative framing of negative ETR"),
        ],
    },
    {
        "id": "Q14",
        "question": (
            "Does Boeing have an improving gross margin profile as of FY2022? "
            "If gross margin is not a useful metric for a company like this, "
            "then state that and explain why."
        ),
        "expected": (
            "Yes. Gross profit improved from $3,017M in FY2021 to $3,502M in "
            "FY2022. Gross margin % improved from 4.8% in FY2021 to 5.3% in FY2022."
        ),
        "key_literals": [
            ("3,017", "FY21 gross profit"),
            ("3,502", "FY22 gross profit"),
            ("4.8", "FY21 gross margin %"),
            ("5.3", "FY22 gross margin %"),
            ("gross margin", "phrase"),
            ("gross profit", "phrase"),
        ],
    },
]


def _load_kb():
    kb = KnowledgeBase(
        "filings_kb",
        storage_directory=str(KB_DIR),
        exists_ok=True,
    )
    # Match the production runtime — the persisted reranker is FlashRank,
    # but our agent process swaps to NoReranker at load (see dsrag_kb.py).
    kb.reranker = NoReranker(ignore_absolute_relevance=True)
    return kb


def _scan_for_literals(text: str, literals: list[tuple[str, str]]) -> list[tuple[str, str, bool]]:
    """For each literal, return (literal, label, found)."""
    out = []
    lower = text.lower()
    for lit, label in literals:
        out.append((lit, label, lit.lower() in lower))
    return out


def _generate_clean_answer(question: str, segments: list[dict]) -> tuple[str, dict]:
    """One-shot Flash generation against the retrieved segments."""
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
        timeout=120.0,
    )
    context = "\n\n---\n\n".join(
        f"[score={s.get('score', 0):.3f} doc={s.get('doc_id', '')}]\n{(s.get('content') or s.get('text') or '')[:5000]}"
        for s in segments
    )
    system = (
        "You are answering questions about SEC 10-K filings. "
        "Ground every numeric claim in the provided context. "
        "If the context is insufficient, say so explicitly. "
        "Cite figures as they appear; preserve sign and units."
    )
    user = f"QUESTION:\n{question}\n\nCONTEXT (retrieved segments):\n{context}"
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        temperature=0.1,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    msg = resp.choices[0].message.content or ""
    usage = resp.usage.model_dump() if resp.usage else {}
    return msg, usage


def main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set; source .env first")

    kb = _load_kb()
    metadata_filter = {"field": "doc_id", "operator": "equals", "value": DOC_ID}

    for q in QUESTIONS:
        print("=" * 78)
        print(f"{q['id']}: {q['question']}")
        print(f"EXPECTED: {q['expected']}")
        print("=" * 78)

        # Step 1: Verbatim retrieval
        segments = kb.query([q["question"]], metadata_filter=metadata_filter)
        print(f"\nRetrieved {len(segments)} segments (verbatim query, doc_id={DOC_ID}):")

        # Step 2: Scan each segment for key literals
        all_text = ""
        for i, seg in enumerate(segments, 1):
            content = seg.get("content") or seg.get("text") or ""
            all_text += "\n" + content
            score = seg.get("score", 0.0)
            print(f"\n  --- segment {i}  score={score:.3f}  {len(content):,} chars ---")
            # Print the first ~400 chars and the last ~200 chars (to get header + trailer)
            print("  HEAD:", content[:400].replace("\n", " ")[:400])
            if len(content) > 600:
                print("  TAIL:", content[-300:].replace("\n", " ")[:300])

        # Step 3: Across-segment literal hits
        print("\nKey-literal scan across all retrieved segments:")
        hits = _scan_for_literals(all_text, q["key_literals"])
        for lit, label, found in hits:
            mark = "✓" if found else "✗"
            print(f"  {mark}  {lit!r:<14}  {label}")

        # Step 4: Find concrete co-occurrences of the key facts
        # For Q13, look for any line containing "(14.76)" or "-14.76"
        # For Q14, look for the gross-margin numbers near "gross"
        print("\nLine-level evidence (lines from any segment containing key literal):")
        any_hit = False
        for lit, _label in q["key_literals"]:
            for line in all_text.splitlines():
                if lit.lower() in line.lower():
                    print(f"  [{lit!r}]: {line.strip()[:200]}")
                    any_hit = True
                    break
        if not any_hit:
            print("  (no direct line matches)")

        # Step 5: Clean one-shot generation against retrieved segments
        print("\n--- One-shot Flash answer (no agent loop) ---")
        answer, usage = _generate_clean_answer(q["question"], segments)
        print(answer)
        print(f"\n[generation usage: {usage}]")
        print()


if __name__ == "__main__":
    main()
