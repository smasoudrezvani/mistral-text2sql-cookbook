"""Export a fixed random sample of wrong answers for manual error labelling (no API calls).

For each sampled question it prints the question, hint, gold SQL, predicted SQL and the first
rows of both results, so a human (or Claude) can label the reason for the error.

Usage:
    uv run python scripts/export_errors.py --run retry_mistral-large-4_all --n 50 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from text2sql_eval.bird import db_path, load_gold_cache, load_questions
from text2sql_eval.scorer import execute_sql

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data/raw/mini_dev_sqlite.jsonl"
DB_ROOT = ROOT / "data/raw/minidev/MINIDEV/dev_databases"
GOLD = ROOT / "data/cache/gold_results.json"
RAW = ROOT / "results/raw"
OUT = ROOT / "data/error_review"


def _rows(rows: list[tuple[object, ...]] | None, k: int = 4) -> str:
    if rows is None:
        return "(none)"
    shown = [tuple(str(v)[:30] for v in r) for r in rows[:k]]
    more = f" ... ({len(rows)} rows)" if len(rows) > k else f" ({len(rows)} rows)"
    return f"{shown}{more}"


def main() -> None:
    """Write data/error_review/<run>_sample<n>.md with one block per sampled wrong answer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="raw results file name without .jsonl")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    qs = {q.question_id: q for q in load_questions(JSONL)}
    gold = load_gold_cache(GOLD)
    recs = [json.loads(x) for x in (RAW / f"{args.run}.jsonl").read_text().splitlines()]
    wrong = sorted((r for r in recs if not r["bird"]), key=lambda r: r["question_id"])
    sample = sorted(
        random.Random(args.seed).sample(wrong, min(args.n, len(wrong))),
        key=lambda r: r["question_id"],
    )

    blocks = []
    for r in sample:
        q = qs[r["question_id"]]
        pred = execute_sql(db_path(DB_ROOT, q.db_id), r["pred_sql"])
        blocks.append(
            f"## q{q.question_id} ({q.db_id}, {q.difficulty})\n"
            f"Q: {q.question}\nHINT: {q.evidence or '(none)'}\n"
            f"GOLD: {q.sql}\nPRED: {r['pred_sql']}\n"
            f"GOLD ROWS: {_rows(gold[q.question_id].rows)}\n"
            f"PRED ROWS: {_rows(pred.rows) if pred.ok else 'ERROR ' + str(pred.error)}\n"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.run}_sample{args.n}.md"
    out.write_text(
        f"# {len(sample)} of {len(wrong)} wrong answers in {args.run} (seed {args.seed})\n\n"
        + "\n".join(blocks),
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
