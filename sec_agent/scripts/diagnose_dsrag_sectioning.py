"""Diagnose why dsRAG semantic sectioning sometimes fails on DeepSeek.

Mimics dsRAG's exact code path (same `instructor` client, same Pydantic
schema, same system prompt) but captures the raw HTTP request and
response for each window via an httpx event hook. Then prints a side-by-side
diff of a successful window vs a failed window so we can see exactly what
differs in the wire payload.

Run from sec_agent/:
    set -a && . ../.env && set +a
    python scripts/diagnose_dsrag_sectioning.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import instructor
from openai import OpenAI

# Use dsRAG's actual schema + system prompt so we're not approximating.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".venv/lib/python3.13/site-packages"))
from dsrag.dsparse.sectioning_and_chunking.semantic_sectioning import (  # noqa: E402
    StructuredDocument,
    SYSTEM_PROMPT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PARSED_MD = REPO_ROOT / "data" / "parsed" / "ACT_10-Q_2024-09-30.md"
WINDOW_CHARS = 20000  # matches dsRAG's default max_characters_per_window


# ---------- HTTP capture ----------

_captured: list[dict] = []


def _on_request(request: httpx.Request) -> None:
    _captured.append({
        "kind": "request",
        "url": str(request.url),
        "method": request.method,
        "body": request.content.decode("utf-8", errors="replace") if request.content else None,
    })


def _on_response(response: httpx.Response) -> None:
    response.read()
    _captured.append({
        "kind": "response",
        "status_code": response.status_code,
        "body": response.content.decode("utf-8", errors="replace") if response.content else None,
    })


def _build_client():
    http_client = httpx.Client(
        event_hooks={"request": [_on_request], "response": [_on_response]},
        timeout=120.0,
    )
    oa = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
        http_client=http_client,
    )
    return instructor.from_openai(oa)


# ---------- Window prep ----------

def make_windows(text: str) -> list[tuple[int, int, str]]:
    """Split text into roughly-WINDOW_CHARS windows, returning
    (line_start_global, line_end_global, numbered_text) tuples that
    mirror dsRAG's `get_document_text_for_window` output."""
    lines = text.splitlines()
    out = []
    char_budget = 0
    line_start = 0
    for i in range(len(lines)):
        char_budget += len(lines[i]) + 1
        if char_budget >= 0.9 * WINDOW_CHARS or i == len(lines) - 1:
            numbered = "\n".join(f"[{j}] {lines[j]}" for j in range(line_start, i + 1))
            out.append((line_start, i, numbered))
            line_start = i + 1
            char_budget = 0
    return out


# ---------- Main ----------

def main():
    if not PARSED_MD.exists():
        sys.exit(f"missing: {PARSED_MD}")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set; source .env first")

    text = PARSED_MD.read_text()
    windows = make_windows(text)
    print(f"Document: {PARSED_MD.name} ({len(text):,} chars, {len(text.splitlines())} lines)")
    print(f"Windows: {len(windows)} (target ~{WINDOW_CHARS} chars each)\n")

    client = _build_client()

    by_window = []
    for idx, (l_start, l_end, numbered) in enumerate(windows):
        _captured.clear()
        try:
            result = client.chat.completions.create(
                model="deepseek-v4-flash",
                response_model=StructuredDocument,
                max_tokens=4000,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(start_line=l_start)},
                    {"role": "user", "content": numbered},
                ],
            )
            outcome = ("success", len(result.sections) if result.sections else 0, None)
        except Exception as e:
            outcome = ("fail", 0, str(e)[:300])

        # `_captured` should hold pairs of (request, response) — usually
        # one pair, but instructor may issue extra calls if it tries to
        # repair a malformed response.
        first_request = next((c for c in _captured if c["kind"] == "request"), None)
        first_response = next((c for c in _captured if c["kind"] == "response"), None)

        by_window.append({
            "window_idx": idx,
            "line_range": (l_start, l_end),
            "char_count": len(numbered),
            "outcome": outcome[0],
            "n_sections": outcome[1],
            "error": outcome[2],
            "request_body": first_request["body"] if first_request else None,
            "response_status": first_response["status_code"] if first_response else None,
            "response_body": first_response["body"] if first_response else None,
        })

        print(
            f"Window {idx:>2}  lines [{l_start:>4}-{l_end:<4}]  "
            f"{len(numbered):>6} chars  outcome={outcome[0]:<7}  "
            + (f"sections={outcome[1]}" if outcome[0] == "success" else f"err={outcome[2][:70]}")
        )

    # Pick one success and one failure for a side-by-side diff
    success = next((w for w in by_window if w["outcome"] == "success"), None)
    failure = next((w for w in by_window if w["outcome"] == "fail"), None)

    if not (success and failure):
        print("\nCouldn't get one of each — all windows had the same outcome.")
        return

    print("\n" + "=" * 78)
    print(f"DIFFING WINDOW {success['window_idx']} (success) vs WINDOW {failure['window_idx']} (fail)")
    print("=" * 78)

    s_req = json.loads(success["request_body"])
    f_req = json.loads(failure["request_body"])

    # Compare top-level keys present in each request
    print("\nTop-level keys in each request body:")
    print(f"  success: {sorted(s_req.keys())}")
    print(f"  fail:    {sorted(f_req.keys())}")

    print("\nKey-by-key value comparison (success vs fail):")
    for k in sorted(set(s_req.keys()) | set(f_req.keys())):
        sv = s_req.get(k, "<missing>")
        fv = f_req.get(k, "<missing>")
        if k == "messages":
            print(f"  {k}: success has {len(sv)} msgs, fail has {len(fv)} msgs")
            for i, (sm, fm) in enumerate(zip(sv, fv)):
                same_role = sm.get("role") == fm.get("role")
                same_len = len(sm.get("content", "")) == len(fm.get("content", ""))
                print(
                    f"    msg[{i}] role={sm.get('role')} "
                    f"(success {len(sm.get('content',''))} chars / "
                    f"fail {len(fm.get('content',''))} chars; "
                    f"role-match={same_role}, len-match={same_len})"
                )
            continue
        if isinstance(sv, (dict, list)) or isinstance(fv, (dict, list)):
            print(f"  {k}: success={json.dumps(sv)[:200]!r}")
            print(f"  {k}: fail   ={json.dumps(fv)[:200]!r}")
        else:
            same = "✓" if sv == fv else "✗"
            print(f"  {k}: success={sv!r:<40}  fail={fv!r}  {same}")

    print("\n--- failure response body ---")
    print(failure["response_body"])
    print("\n--- success response body (truncated) ---")
    print(success["response_body"][:2000] + ("..." if len(success["response_body"]) > 2000 else ""))


if __name__ == "__main__":
    main()
