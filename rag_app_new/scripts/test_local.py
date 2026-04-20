"""Invoke the RAG agent graph locally without the AgentCore runtime.

Usage:
    cd rag_app_new
    python scripts/test_local.py "What was MTG's loss ratio last quarter?"

Requires AWS credentials with Bedrock access. MEMORY_ID can be left unset.
"""

import asyncio
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # rag_app_new/ (for `src.*` imports)
sys.path.insert(0, str(_HERE.parents[2]))  # repo root (for `rag_app` import)

from src.application.orchestrator.streaming import get_streaming_response  # noqa: E402


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "Hi"
    print(f"Q: {query}\n\nA: ", end="", flush=True)
    async for chunk in get_streaming_response(
        messages=query,
        customer_name="Andrew",
        conversation_id="local-test",
    ):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
