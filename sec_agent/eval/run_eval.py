"""Run the agent over a CSV of (question, expected_answer) pairs and grade.

Usage:
    cd sec_agent
    python eval/run_eval.py                              # questions.csv, config default mode
    python eval/run_eval.py eval/other.csv               # custom input CSV
    python eval/run_eval.py --mode tools                 # force three-tool stack
    python eval/run_eval.py --mode dsrag                 # force single dsrag_kb tool
    python eval/run_eval.py --mode tools eval/other.csv  # combine

Modes (default is `dsrag` per settings.USE_DSRAG_ONLY):
    dsrag  — single dsrag_kb tool over the dsRAG KnowledgeBase
             (requires data/dsrag_store/ built by data_pipeline_dsrag/build_kb.py).
    tools  — search_concepts + query_financials + search_narrative.

Outputs:
    eval/results/<input_stem>_<ts>.csv       # per-question results
    eval/logs.json                           # append-only run log
"""

import asyncio
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))  # eval/
sys.path.insert(0, str(_HERE.parents[1]))  # sec_agent/

from langchain_aws import ChatBedrockConverse  # noqa: E402
from loguru import logger  # noqa: E402

from pricing import cost_usd  # noqa: E402
from usage import UsageCollector  # noqa: E402

from src.application.orchestrator.streaming import get_streaming_response  # noqa: E402
from src.config import settings  # noqa: E402
from src.infrastructure.model import extract_text_content  # noqa: E402


EVAL_DIR = _HERE.parent
RESULTS_DIR = EVAL_DIR / "results"
LOG_FILE = EVAL_DIR / "logs.json"

JUDGE_SYSTEM = (
    "You grade whether an assistant's answer is substantively correct given a "
    "question and a verified expected answer. Numeric values within a 1% "
    "rounding tolerance count as correct. Extra context is fine as long as "
    "the core figures/direction match. A partially-correct answer (some "
    "values right, some wrong) is INCORRECT. Reply with ONE line: "
    "'CORRECT: <=20 word reason' or 'INCORRECT: <=20 word reason'."
)


async def run_agent(question: str, run_id: str, collector: UsageCollector) -> str:
    chunks: list[str] = []
    async for chunk in get_streaming_response(
        messages=question,
        customer_name="Evaluator",
        conversation_id=f"eval-{run_id}",
        callbacks=[collector],
    ):
        chunks.append(chunk)
    return "".join(chunks).strip()


def judge(
    question: str, expected: str, actual: str, collector: UsageCollector
) -> tuple[bool, str]:
    llm = ChatBedrockConverse(
        model_id=settings.ROUTER_MODEL_ID,
        region_name=settings.AWS_REGION,
        temperature=0,
    )
    prompt = (
        f"Question:\n{question}\n\n"
        f"Expected answer:\n{expected}\n\n"
        f"Agent answer:\n{actual}"
    )
    resp = llm.invoke(
        [("system", JUDGE_SYSTEM), ("user", prompt)],
        config={"callbacks": [collector]},
    )
    text = extract_text_content(resp.content).strip()
    first_line = text.splitlines()[0] if text else ""
    verdict, _, rationale = first_line.partition(":")
    correct = verdict.strip().upper() == "CORRECT"
    return correct, rationale.strip() or first_line


def _append_log(entry: dict) -> None:
    existing = json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else []
    existing.append(entry)
    LOG_FILE.write_text(json.dumps(existing, indent=2))


def _resolve_mode(mode: str | None) -> str:
    """Resolve CLI --mode into a canonical mode string ('dsrag' or 'tools').

    A value of None falls back to the config default (settings.USE_DSRAG_ONLY).
    Mutates the settings singleton so get_tools() and _build_agent_system()
    see the right value at call time.
    """
    if mode is None:
        mode = "dsrag" if settings.USE_DSRAG_ONLY else "tools"
    if mode == "dsrag":
        settings.USE_DSRAG_ONLY = True
        # Pre-flight: bail loudly if the KB hasn't been built yet.
        from src.infrastructure.dsrag_kb import DSRAG_STORE_DIR
        if not DSRAG_STORE_DIR.exists():
            raise SystemExit(
                f"--mode dsrag requires the KB at {DSRAG_STORE_DIR}. "
                "Build it first with: python data_pipeline_dsrag/build_kb.py"
            )
    else:
        settings.USE_DSRAG_ONLY = False
    return mode


async def main(csv_path: Path, mode: str | None) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"{csv_path} has no rows")

    mode = _resolve_mode(mode)
    logger.info(f"Retrieval mode: {mode}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    collector = UsageCollector()

    def _total_cost() -> float:
        return sum(
            cost_usd(m, u.input_tokens, u.output_tokens,
                     u.cache_read_tokens, u.cache_creation_tokens)
            for m, u in collector.by_model.items()
        )

    results = []
    run_start = time.perf_counter()
    for i, row in enumerate(rows, 1):
        q = row["question"]
        expected = row["expected_answer"]
        logger.info(f"[{i}/{len(rows)}] {q}")
        c0 = _total_cost()
        tc0 = len(collector.tool_calls)
        t0 = time.perf_counter()
        try:
            actual = await run_agent(q, run_id=f"{ts}-{i}", collector=collector)
        except Exception as e:
            logger.error(f"agent error on row {i}: {e}")
            actual = f"[agent error: {type(e).__name__}: {e}]"
        agent_seconds = time.perf_counter() - t0
        c1 = _total_cost()
        per_q_tools = collector.tool_calls[tc0:]
        correct, rationale = judge(q, expected, actual, collector)
        c2 = _total_cost()
        tools_summary = [
            {"tool": t.tool_name, "tokens_est": t.result_tokens_est}
            for t in per_q_tools
        ]
        total_tool_tokens = sum(t.result_tokens_est for t in per_q_tools)
        logger.info(
            f"  -> {'OK' if correct else 'MISS'} | ${c1 - c0:.4f} agent / "
            f"${c2 - c1:.4f} judge | {len(per_q_tools)} tool calls, "
            f"{total_tool_tokens:,} tool-result tokens | "
            f"{agent_seconds:.1f}s | {rationale}"
        )
        results.append({
            "question": q,
            "expected_answer": expected,
            "agent_answer": actual,
            "correct": correct,
            "rationale": rationale,
            "agent_cost_usd": round(c1 - c0, 6),
            "judge_cost_usd": round(c2 - c1, 6),
            "tool_calls": json.dumps(tools_summary),
            "tool_result_tokens_est": total_tool_tokens,
            "agent_seconds": round(agent_seconds, 2),
            "retrieval_mode": mode,
        })
    run_seconds = time.perf_counter() - run_start

    out_csv = RESULTS_DIR / f"{csv_path.stem}_{ts}.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    n_correct = sum(r["correct"] for r in results)
    accuracy = n_correct / len(results)

    usage_summary = {
        model_id: {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_tokens": u.cache_read_tokens,
            "cache_creation_tokens": u.cache_creation_tokens,
            "calls": u.calls,
            "cost_usd": round(
                cost_usd(
                    model_id,
                    u.input_tokens,
                    u.output_tokens,
                    u.cache_read_tokens,
                    u.cache_creation_tokens,
                ),
                6,
            ),
        }
        for model_id, u in collector.by_model.items()
    }
    total_cost = round(sum(m["cost_usd"] for m in usage_summary.values()), 6)

    _append_log({
        "input_df": csv_path.name,
        "agent_ts": ts,
        "retrieval_mode": mode,
        "accuracy": round(accuracy, 4),
        "n": len(results),
        "n_correct": n_correct,
        "results_file": str(out_csv.relative_to(EVAL_DIR)),
        "orchestrator_provider": settings.ORCHESTRATOR_PROVIDER,
        "orchestrator_model": (
            settings.DEEPSEEK_MODEL_ID
            if settings.ORCHESTRATOR_PROVIDER == "deepseek"
            else settings.ORCHESTRATOR_MODEL_ID
        ),
        "compress_tool_outputs": settings.COMPRESS_TOOL_OUTPUTS,
        "usage": usage_summary,
        "total_cost_usd": total_cost,
        "run_seconds": round(run_seconds, 2),
    })

    print(f"\nMode:     {mode}")
    print(f"Accuracy: {accuracy:.1%} ({n_correct}/{len(results)})")
    print(f"Results:  {out_csv}")
    print(f"Log:      {LOG_FILE}")
    print(f"Cost:     ${total_cost:.4f}")
    print(f"Time:     {run_seconds:.1f}s  (compressor={'on' if settings.COMPRESS_TOOL_OUTPUTS else 'off'})")
    for mid, m in usage_summary.items():
        print(
            f"  {mid}: {m['input_tokens']:,} in / {m['output_tokens']:,} out "
            f"({m['calls']} calls) ${m['cost_usd']:.4f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(EVAL_DIR / "questions.csv"),
        help="Input CSV (default: eval/questions.csv).",
    )
    parser.add_argument(
        "--mode",
        choices=["dsrag", "tools"],
        default=None,
        help=(
            "Retrieval stack: 'dsrag' (single dsrag_kb tool) or 'tools' "
            "(search_concepts + query_financials + search_narrative). "
            "Defaults to the value of settings.USE_DSRAG_ONLY."
        ),
    )
    args = parser.parse_args()
    asyncio.run(main(Path(args.csv_path), args.mode))
