"""Run the agent over a CSV of (question, expected_answer) pairs and grade.

Usage:
    cd rag_app_new
    python eval/run_eval.py                  # uses eval/questions.csv
    python eval/run_eval.py eval/other.csv   # custom input CSV

Outputs:
    eval/results/<input_stem>_<ts>.csv       # per-question results
    eval/logs.json                           # append-only run log (input_df, agent_ts, accuracy)
"""

import asyncio
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))  # eval/
sys.path.insert(0, str(_HERE.parents[1]))  # rag_app_new/

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


async def main(csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"{csv_path} has no rows")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    collector = UsageCollector()

    results = []
    for i, row in enumerate(rows, 1):
        q = row["question"]
        expected = row["expected_answer"]
        logger.info(f"[{i}/{len(rows)}] {q}")
        try:
            actual = await run_agent(q, run_id=f"{ts}-{i}", collector=collector)
        except Exception as e:
            logger.error(f"agent error on row {i}: {e}")
            actual = f"[agent error: {type(e).__name__}: {e}]"
        correct, rationale = judge(q, expected, actual, collector)
        logger.info(f"  -> {'OK' if correct else 'MISS'} | {rationale}")
        results.append({
            "question": q,
            "expected_answer": expected,
            "agent_answer": actual,
            "correct": correct,
            "rationale": rationale,
        })

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
        "accuracy": round(accuracy, 4),
        "n": len(results),
        "n_correct": n_correct,
        "results_file": str(out_csv.relative_to(EVAL_DIR)),
        "orchestrator_model": settings.ORCHESTRATOR_MODEL_ID,
        "usage": usage_summary,
        "total_cost_usd": total_cost,
    })

    print(f"\nAccuracy: {accuracy:.1%} ({n_correct}/{len(results)})")
    print(f"Results:  {out_csv}")
    print(f"Log:      {LOG_FILE}")
    print(f"Cost:     ${total_cost:.4f}")
    for mid, m in usage_summary.items():
        print(
            f"  {mid}: {m['input_tokens']:,} in / {m['output_tokens']:,} out "
            f"({m['calls']} calls) ${m['cost_usd']:.4f}"
        )


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else EVAL_DIR / "questions.csv"
    asyncio.run(main(path))
