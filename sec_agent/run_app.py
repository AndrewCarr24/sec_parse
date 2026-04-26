"""Single-query runner for the sec_agent.

Usage:
    cd sec_agent
    python run_app.py "What was ACT's revenue in Q3 2024?"
    python run_app.py --mode tools "..."        # override retrieval stack

The default retrieval stack comes from settings.USE_DSRAG_ONLY (dsRAG).
Streams the agent's final answer to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make `src.*` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import settings  # noqa: E402


def _resolve_mode(mode: str | None) -> str:
    """Return the resolved mode string and mutate settings accordingly."""
    if mode is None:
        mode = "dsrag" if settings.USE_DSRAG_ONLY else "tools"
    if mode == "dsrag":
        settings.USE_DSRAG_ONLY = True
        from src.infrastructure.dsrag_kb import DSRAG_STORE_DIR
        if not DSRAG_STORE_DIR.exists():
            raise SystemExit(
                f"--mode dsrag requires the KB at {DSRAG_STORE_DIR}. "
                "Build it first with: python data_pipeline_dsrag/build_kb.py"
            )
    else:
        settings.USE_DSRAG_ONLY = False
    return mode


async def run(query: str, mode: str | None) -> None:
    mode = _resolve_mode(mode)
    # Import AFTER settings are finalized so chain/tool construction sees
    # the right flag.
    from src.application.orchestrator.streaming import get_streaming_response

    print(f"[mode: {mode}]\n")
    async for chunk in get_streaming_response(messages=query, customer_name="User"):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Natural-language question for the agent.")
    parser.add_argument(
        "--mode",
        choices=["dsrag", "tools"],
        default=None,
        help=(
            "Retrieval stack: 'dsrag' (single dsrag_kb tool) or 'tools' "
            "(search_concepts + query_financials + search_narrative). "
            "Defaults to settings.USE_DSRAG_ONLY."
        ),
    )
    args = parser.parse_args()
    asyncio.run(run(args.query, args.mode))
