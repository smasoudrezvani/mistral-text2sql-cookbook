"""Show one question in full: hint, gold SQL and the model's SQL, both run on the real database.

No API calls: the model's SQL comes from a saved run in results/raw/.

Usage:
    uv run python scripts/show_case.py 390
    uv run python scripts/show_case.py 438 --run retry_mistral-large-4_all --rows 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2sql_eval.bird import db_path, load_questions
from text2sql_eval.scorer import compare, execute_sql

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data/raw/mini_dev_sqlite.jsonl"
DB_ROOT = ROOT / "data/raw/minidev/MINIDEV/dev_databases"
RAW = ROOT / "results/raw"


def _print_rows(rows: list[tuple[object, ...]] | None, error: str | None, limit: int) -> None:
    if rows is None:
        print(f"  (query failed: {error})")
        return
    for r in rows[:limit]:
        print("  " + " | ".join("NULL" if v is None else str(v) for v in r))
    if len(rows) > limit:
        print(f"  ... {len(rows)} rows in total")
    elif not rows:
        print("  (0 rows)")


def main() -> None:
    """Print the case."""
    parser = argparse.ArgumentParser()
    parser.add_argument("qid", type=int, help="question id, e.g. 390")
    parser.add_argument("--run", default="retry_mistral-large-4_all", help="results/raw file name")
    parser.add_argument("--rows", type=int, default=5, help="rows to show per result")
    args = parser.parse_args()

    q = next(x for x in load_questions(JSONL) if x.question_id == args.qid)
    rec = next(
        json.loads(line)
        for line in (RAW / f"{args.run}.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["question_id"] == args.qid
    )
    db = db_path(DB_ROOT, q.db_id)
    gold = execute_sql(db, q.sql, timeout_s=300, max_rows=None)
    pred = execute_sql(db, rec["pred_sql"], timeout_s=300, max_rows=None)
    score = compare(pred, gold, q.sql) if gold.ok else None

    print(f"=== q{q.question_id}  database: {q.db_id}  ({db.relative_to(ROOT)})")
    print(f"\nQUESTION:\n  {q.question}\n\nHINT:\n  {q.evidence or '(none)'}")
    print(f"\nGOLD SQL:\n  {q.sql}\n\nGOLD RESULT:")
    _print_rows(gold.rows, gold.error, args.rows)
    print(f"\nMODEL SQL ({args.run}):\n  " + rec["pred_sql"].replace("\n", "\n  "))
    print("\nMODEL RESULT:")
    _print_rows(pred.rows, pred.error, args.rows)
    if score is not None:
        print(
            f"\nSCORE: BIRD rule {'correct' if score.bird else 'wrong'}, "
            f"strict rule {'correct' if score.strict else 'wrong'}"
        )


if __name__ == "__main__":
    main()
