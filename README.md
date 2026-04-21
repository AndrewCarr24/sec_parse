# parse_sec

SEC 10-K / 10-Q parser and LangGraph-based RAG agent for a small set of US
mortgage insurance companies (ACT, ESNT, MTG, RDN, ACGL, NMIH). The pipeline
downloads filings from EDGAR, extracts both structured (iXBRL) and
unstructured (MD&A tables) facts into CSV, and serves them to an agent that
answers financial questions via SQL over DuckDB.

## Structure

```
parse_sec/
├── data_pipeline/              # fetch + parse SEC filings
│   ├── fetchers.py             # download filings from EDGAR
│   ├── parsers.py              # HTML → markdown via Docling
│   ├── ixbrl_parser.py         # iXBRL → xbrl_facts.csv (Arelle)
│   └── table_extractor.py      # MD&A tables → extracted_facts.csv (LLM)
├── data/
│   ├── raw/                    # EDGAR downloads (gitignored)
│   └── parsed/                 # docling-rendered markdown of each filing
├── test_output/                # facts CSVs + XBRL metadata (DB input)
└── sec_agent/                 # LangGraph agent
    ├── src/
    │   ├── application/orchestrator/   # router → agent → tools graph
    │   ├── domain/prompts.py
    │   └── infrastructure/financials_db.py  # DuckDB loader
    ├── scripts/test_local.py   # one-shot interactive query
    ├── eval/                   # eval harness
    │   ├── questions.csv       # gold Q/A pairs
    │   ├── run_eval.py         # run + grade + log
    │   ├── pricing.py          # Bedrock $/MTok table
    │   └── usage.py            # LangChain callback: tokens by model
    └── infra/                  # CDK stacks (unused for local runs)
```

## Prerequisites

- Python 3.11+ (recommended: [uv](https://github.com/astral-sh/uv))
- AWS credentials with Bedrock access to Claude Sonnet 4.6 + Haiku 4.5
  (default region `us-east-1`). The code relies on the boto3 credential
  chain — env vars, `~/.aws/credentials`, or SSO all work.

### Setup

```bash
# data pipeline deps (top-level)
uv venv && source .venv/bin/activate
uv pip install sec-edgar-downloader docling arelle-release

# agent deps
cd sec_agent
uv venv && source .venv/bin/activate
uv pip install -e .
```

Optional `.env` in `sec_agent/` to override defaults:

```
ORCHESTRATOR_MODEL_ID=us.anthropic.claude-sonnet-4-6
ROUTER_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
AWS_REGION=us-east-1
```

## Data pipeline

The pipeline is four stages; each script is runnable standalone.

### 1. Fetch filings from EDGAR

```bash
python data_pipeline/fetchers.py
```

Downloads the most recent 10-K + 4 most recent 10-Qs per ticker to
`data/raw/sec-edgar-filings/`, then reorganizes into
`{ticker}/{form}/{period}/{accession}/`.

### 2. Parse HTML → markdown

```bash
python data_pipeline/parsers.py
```

Walks `data/raw/sec-edgar-filings/`, runs each primary HTML doc through
Docling, post-processes (merged-cell dedup, heading injection, page-header
stripping), and writes `data/parsed/{TICKER}_{FORM}_{PERIOD}.md`. Idempotent —
skips any filing already parsed.

### 3. Extract iXBRL facts

```bash
python data_pipeline/ixbrl_parser.py <accession_dir> <output_dir>
# e.g.
python data_pipeline/ixbrl_parser.py \
    data/raw/sec-edgar-filings/ACT/10-Q/2024-09-30/0001324404-24-000017 \
    test_output
```

Unpacks the SGML submission into a temp dir, points Arelle at the primary
iXBRL doc, and writes to `<output_dir>/`:

- `xbrl_facts.csv` — one row per reported fact (concept, value, unit, period, dimensions)
- `xbrl_contexts.json` — period + dimension definitions
- `xbrl_units.json` — unit definitions
- `xbrl_metadata.json` — filing-level DEI (CIK, form, period, entity)

### 4. Extract MD&A / narrative tables

```bash
python data_pipeline/table_extractor.py <accession_dir> <output_dir> [--dry-run] [--limit N]
```

LLM-driven extraction of tables that aren't iXBRL-tagged (operational metrics
like NIW by FICO, IIF by LTV, etc.). Writes to `<output_dir>/`:

- `extracted_facts.csv` — one row per (table, row, column) value
- `extracted_tables.json` — per-table audit: LLM classification + schema
- `extraction_log.txt` — run log

The agent's DuckDB loader reads both `xbrl_facts.csv` and
`extracted_facts.csv` from `test_output/` and folds them into a single
`facts` table.

## Running the agent

```bash
cd sec_agent
python scripts/test_local.py "What was ACT's revenue in Q3 2024?"
```

Streams the answer to stdout. On first invocation the DuckDB loader reads
the CSVs in `test_output/`, normalizes units/period-ends, and builds the
`facts` table in-memory.

The agent is a LangGraph ReAct loop:

1. **Router** (Haiku) — classifies intent (`rag_query` / `simple` / `off_topic`)
2. **Agent** (Sonnet 4.6, prompt-cached) — calls `search_concepts` and
   `query_financials` tools against DuckDB until it has enough context
3. **Finalize** — if the 12-call tool budget is hit, produce a best-effort
   answer from tool results already gathered

## Running the eval

```bash
cd sec_agent
python eval/run_eval.py                  # uses eval/questions.csv
python eval/run_eval.py eval/other.csv   # custom input
```

For each `(question, expected_answer)` row:

1. Run the agent, capture the final answer.
2. Judge with Haiku — binary correct/incorrect with a ≤20-word rationale.
3. Accumulate tokens by model via a LangChain callback
   ([eval/usage.py](sec_agent/eval/usage.py)) and price them with
   [eval/pricing.py](sec_agent/eval/pricing.py).

Outputs:

- `eval/results/{input_stem}_{utc_ts}.csv` — per-question Q / expected /
  agent answer / correct / rationale
- `eval/logs.json` — append-only log of runs: accuracy, n, n_correct,
  orchestrator model, per-model token usage (input/output/cache_read/
  cache_creation/calls/cost), total USD cost
- stdout — accuracy, cost summary, per-model breakdown
