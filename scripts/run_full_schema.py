"""Approach (a): full schema (tables + 3 example rows) in the prompt.

Resumable: per-question results are appended to results/raw/<run_name>.jsonl, and questions
already in that file are skipped. The summary is rebuilt from the whole file at the end.

Usage:
    uv run python scripts/run_full_schema.py --model ministral-8b-2512 --slice dev50 --rpm 2
    uv run python scripts/run_full_schema.py --model ministral-8b-2512 --slice all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from mistralai.client import Mistral

from text2sql_eval.bird import db_path, load_gold_cache, load_questions
from text2sql_eval.prompting import build_messages, chat, extract_sql
from text2sql_eval.schema import render_schema
from text2sql_eval.scorer import compare, execute_sql

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data/raw/mini_dev_sqlite.jsonl"
DB_ROOT = ROOT / "data/raw/minidev/MINIDEV/dev_databases"
GOLD = ROOT / "data/cache/gold_results.json"
SLICES = ROOT / "slices"
RAW_OUT = ROOT / "results/raw"
SUMMARY_OUT = ROOT / "results/summary"

# $ per 1M tokens (input, output), docs.mistral.ai/inference/pricing, checked 2026-11-08
PRICES = {
    "ministral-8b-2512": (0.15, 0.15),
    "mistral-small-2603": (0.15, 0.60),
    "mistral-large-4": (0.70, 2.10),
}

logger = logging.getLogger("run_full_schema")


def summarize(raw_path: Path, model: str, slice_name: str) -> dict[str, object]:
    """Build the run summary from all per-question records in a raw results file.

    Args:
        raw_path: The run's JSONL file.
        model: Model name (for pricing).
        slice_name: Slice name.

    Returns:
        Summary with accuracy under both rules, failures, tokens and cost.
    """
    recs = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    n = len(recs)
    tok_in = sum(r["prompt_tokens"] for r in recs)
    tok_out = sum(r["completion_tokens"] for r in recs)
    p_in, p_out = PRICES[model]
    return {
        "run": raw_path.stem,
        "approach": "full_schema",
        "model": model,
        "slice": slice_name,
        "n": n,
        "ex_bird": round(sum(r["bird"] for r in recs) / n, 4),
        "ex_strict": round(sum(r["strict"] for r in recs) / n, 4),
        "failed_to_run": sum(r["pred_error"] is not None for r in recs),
        "prompt_tokens": tok_in,
        "completion_tokens": tok_out,
        "cost_usd": round(tok_in / 1e6 * p_in + tok_out / 1e6 * p_out, 4),
    }


def main() -> None:
    """Run approach (a) on a slice, append per-question results, write a summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(PRICES))
    parser.add_argument("--slice", default="dev50", help="'all' or a file name in slices/")
    parser.add_argument("--n-rows", type=int, default=3)
    parser.add_argument("--rpm", type=float, default=0, help="max requests per minute (0 = no cap)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv(ROOT / ".env")
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    questions = load_questions(JSONL)
    if args.slice != "all":
        ids = set(json.loads((SLICES / f"{args.slice}.json").read_text())["question_ids"])
        questions = [q for q in questions if q.question_id in ids]
    gold = load_gold_cache(GOLD)
    schemas: dict[str, str] = {}

    RAW_OUT.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_OUT / f"full_schema_{args.model}_{args.slice}.jsonl"
    done: set[int] = set()
    if raw_path.exists():
        done = {json.loads(x)["question_id"] for x in raw_path.read_text().splitlines()}
        logger.info("resuming: %d questions already done", len(done))
    todo = [q for q in questions if q.question_id not in done]
    min_gap = 60.0 / args.rpm if args.rpm else 0.0
    last_call = 0.0

    with raw_path.open("a", encoding="utf-8") as out:
        for i, q in enumerate(todo, start=1):
            db = db_path(DB_ROOT, q.db_id)
            if q.db_id not in schemas:
                schemas[q.db_id] = render_schema(db, n_rows=args.n_rows)
            messages = build_messages(schemas[q.db_id], q.question, q.evidence)

            wait = last_call + min_gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last_call = time.monotonic()
            reply = chat(client, args.model, messages)

            sql = extract_sql(reply.text)
            g = gold[q.question_id]
            pred = execute_sql(db, sql, timeout_s=g.pred_timeout_s)
            score = compare(pred, g.as_exec_result(), q.sql)
            record = {
                "question_id": q.question_id,
                "db_id": q.db_id,
                "difficulty": q.difficulty,
                "pred_sql": sql,
                "pred_error": pred.error,
                "pred_rows": len(pred.rows) if pred.rows is not None else None,
                "bird": score.bird,
                "strict": score.strict,
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            logger.info(
                "%d/%d q%d bird=%s strict=%s%s",
                i,
                len(todo),
                q.question_id,
                score.bird,
                score.strict,
                f" error={pred.error[:60]}" if pred.error else "",
            )

    summary = summarize(raw_path, args.model, args.slice)
    SUMMARY_OUT.mkdir(parents=True, exist_ok=True)
    (SUMMARY_OUT / f"{raw_path.stem}.json").write_text(json.dumps(summary, indent=1) + "\n")
    logger.info("summary: %s", summary)


if __name__ == "__main__":
    main()
