"""Recompute table-based fields in saved results (no API calls) and rebuild the summaries.

Needed once after tables_in_sql switched to the sqlglot parser (the old word-matching version
counted ``ORDER BY`` as the table ``order`` and column names as tables). Recomputes:
``gold_tables`` / ``table_recall`` (retrieval runs) and ``used_distractor`` (distractor runs).

Usage:
    uv run python scripts/rescore_tables.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from text2sql_eval.bird import db_path, load_questions
from text2sql_eval.distractors import load_manifest
from text2sql_eval.retrieval import tables_in_sql
from text2sql_eval.schema import table_names

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import (  # noqa: E402
    DB_ROOT,
    DISTRACT,
    JSONL,
    RAW_OUT,
    SUMMARY_OUT,
    summarize,
)

logger = logging.getLogger("rescore_tables")


def main() -> None:
    """Rewrite table fields in every results/raw file and regenerate its summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gold_sql = {q.question_id: q.sql for q in load_questions(JSONL)}
    for raw in sorted(RAW_OUT.glob("*.jsonl")):
        distract = raw.stem.endswith("_distract")
        recs = [json.loads(x) for x in raw.read_text(encoding="utf-8").splitlines()]
        for r in recs:
            extra = [DISTRACT / f"{r['db_id']}.sqlite"] if distract else None
            if "tables" in r:
                names = table_names(db_path(DB_ROOT, r["db_id"]), extra)
                gold = tables_in_sql(gold_sql[r["question_id"]], names)
                r["gold_tables"] = sorted(gold)
                r["table_recall"] = gold <= set(r["tables"])
            if distract:
                fakes = list(load_manifest(DISTRACT / f"{r['db_id']}.sqlite"))
                r["used_distractor"] = bool(tables_in_sql(r["pred_sql"], fakes))
        raw.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")

        stem = raw.stem.removesuffix("_distract")
        approach = next(a for a in ("full_schema", "retrieval", "retry") if stem.startswith(a))
        model, slice_name = stem[len(approach) + 1 :].rsplit("_", 1)
        summary = summarize(raw, approach, model, slice_name)
        (SUMMARY_OUT / f"{raw.stem}.json").write_text(json.dumps(summary, indent=1) + "\n")
        logger.info(
            "%s: recall=%s used_distractor=%s",
            raw.stem,
            summary.get("table_recall"),
            summary.get("used_distractor"),
        )


if __name__ == "__main__":
    main()
