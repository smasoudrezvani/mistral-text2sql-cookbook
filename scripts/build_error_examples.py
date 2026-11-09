# ruff: noqa: E501  (long lines are markdown table text)
"""Write docs/error_examples.md: one real failure per error category, plus possible fixes.

Every SQL query and result row is taken from the saved run and the databases (no API calls).

Usage:
    uv run python scripts/build_error_examples.py
"""

from __future__ import annotations

import json
from pathlib import Path

from text2sql_eval.bird import db_path, load_gold_cache, load_questions
from text2sql_eval.scorer import execute_sql

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data/raw/mini_dev_sqlite.jsonl"
DB_ROOT = ROOT / "data/raw/minidev/MINIDEV/dev_databases"
GOLD = ROOT / "data/cache/gold_results.json"
RUN = "retry_mistral-large-4_all"
LABELS = ROOT / f"results/error_labels/{RUN}_sample50.json"
OUT = ROOT / "docs/error_examples.md"

# category -> (title, chosen question id, plain-language explanation)
EXAMPLES = [
    (
        "dirty_data",
        "Dirty data",
        348,
        'In this database a weight of 0 means "unknown". The model averaged the zeros in as real '
        "weights, so the average dropped from 103.6 to 82.14 kg.",
    ),
    (
        "wrong_filter",
        "Wrong filter (misread question)",
        390,
        '"Patients in the thrombosis registry" means everyone in this database. The model read it '
        'as "patients who had thrombosis" and added a filter, so the answer became 100%.',
    ),
    (
        "wrong_column",
        "Wrong column",
        145,
        '"Average number of up-votes cast" is stored in users.UpVotes. The model counted rows in the '
        "votes table instead, where the user id is almost always empty, so it got 0.",
    ),
    (
        "wrong_filter_value",
        "Wrong filter value",
        423,
        "The model wrote '-' for a negative anti-SSA result, but this column stores the word "
        "'negative', so negatives were counted as abnormal and the count doubled.",
    ),
    (
        "wrong_aggregation",
        "Wrong aggregation",
        293,
        "One driver appears twice in the results table for this race. COUNT(*) counts rows; the "
        "question asks for drivers, so COUNT(DISTINCT driverId) is needed.",
    ),
    (
        "format_only",
        "Format only (the answer is actually right)",
        86,
        "Same year and count, but the model returned the year as the number 2015 and the gold as the "
        "text '2015'. Exact matching (ours and BIRD's) treats them as different.",
    ),
    (
        "wrong_join",
        "Wrong join",
        25,
        "An INNER JOIN to the frpm table silently dropped schools that have no frpm row, and the "
        "district type was taken from that table instead of schools.DOCType.",
    ),
    (
        "ambiguous",
        "Ambiguous question",
        438,
        '"Whose liver enzymes stayed in the normal range" can mean "never above normal" (the '
        "model's reading) or just \"use their normal-range tests\" (the gold's reading).",
    ),
]

SOLUTIONS = """\
## How to fix each kind of error

| Error | Fix | How |
|---|---|---|
| Dirty data (21/50) | **Add BIRD's column descriptions to the schema** | BIRD's description CSVs already say e.g. *"If the weight_kg is NULL or 0, it means the weight of the superhero is missing"* and *"Thrombosis: 0 = negative, 1-3 = degree"*. Today the prompt only has CREATE TABLE + 3 example rows. |
| Dirty data | **Data profile per column** (automatic, no LLM) | For each column compute: share of NULL / 0 / '' / '-', numbers stored as text, values that look like lists ("a, b"). Add one short note per risky column, e.g. `weight_kg: 31% are 0 (likely missing)`. |
| Dirty data | **System-prompt rules** | "Treat 0, '' and '-' in measurement columns as missing unless the question says otherwise. CAST text numbers before MIN/MAX/comparisons. Fields may hold comma-separated lists." |
| Wrong filter (misread) | **System-prompt rule + self-check** | "The database itself is the cohort/registry; don't add filters the question doesn't ask for." Ask the model to list each condition it used and point to the words in the question that require it. |
| Wrong column | **Column descriptions** (same as above) | Descriptions say which column holds what (e.g. `users.UpVotes` = up-votes cast). |
| Wrong filter value | **Show the distinct values of small columns** | For columns with <= 20 distinct values, list them in the schema (`Thrombosis: 0, 1, 2, 3`; `SSA: 'negative', '0', '1', '4', ...`). Value retrieval (extra 2) only helps when the words match. |
| Wrong aggregation | **System-prompt rule + self-check** | "When counting entities (drivers, patients), use COUNT(DISTINCT id); check whether joins create duplicates." |
| Format only | **System-prompt rule** and/or a lenient scorer as a footnote | "Return exactly the requested columns; don't combine or re-type values." A lenient comparison would add only 7 of 500 for Large 4 (c). |
| Wrong join | **System-prompt rule** | "Use LEFT JOIN when the question says to include all rows (with zeros)." |
| Ambiguous | Not fixable by prompting | Report it; ideally the question should be clarified. |

The first three rows target 21 of the 50 labelled errors and need no extra model call, so they
are the best first experiment.
"""


def _rows(rows: list[tuple[object, ...]] | None, k: int = 5) -> str:
    if rows is None:
        return "(query failed)"
    lines = [" | ".join("NULL" if v is None else str(v) for v in r) for r in rows[:k]]
    more = f"\n... ({len(rows)} rows in total)" if len(rows) > k else ""
    return "\n".join(lines) + more


def main() -> None:
    """Build the markdown file."""
    qs = {q.question_id: q for q in load_questions(JSONL)}
    gold = load_gold_cache(GOLD)
    recs = {
        r["question_id"]: r
        for r in map(json.loads, (ROOT / f"results/raw/{RUN}.jsonl").read_text().splitlines())
    }
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    counts: dict[str, int] = {}
    for v in labels.values():
        counts[v["category"]] = counts.get(v["category"], 0) + 1

    parts = [
        "# Why text-to-SQL answers fail: one real example per error type\n",
        "System: Mistral Large 4, approach (c) execute-and-retry, BIRD mini-dev (2026 release), "
        "500 questions, 65.0% correct. 50 of the 175 wrong answers were labelled by hand "
        f"(`results/error_labels/{RUN}_sample50.json`).\n",
        "| Category | Count (of 50) |\n|---|---|",
        *[f"| {title} | {counts.get(cat, 0)} |" for cat, title, _, _ in EXAMPLES],
        "",
    ]
    for cat, title, qid, why in EXAMPLES:
        q, r = qs[qid], recs[qid]
        assert labels[str(qid)]["category"] == cat, (qid, cat)
        pred = execute_sql(db_path(DB_ROOT, q.db_id), r["pred_sql"])
        parts += [
            f"## {title} ({counts.get(cat, 0)} of 50)\n",
            f"**Question** (q{qid}, database `{q.db_id}`): {q.question}\n",
            f"**Hint:** {q.evidence or '(none)'}\n",
            f"**What went wrong:** {why}\n",
            "**Model's SQL:**\n```sql\n" + r["pred_sql"].strip() + "\n```\n",
            "**Model's answer:**\n```\n" + _rows(pred.rows) + "\n```\n",
            "**Gold SQL:**\n```sql\n" + q.sql.strip() + "\n```\n",
            "**Gold answer:**\n```\n" + _rows(gold[qid].rows) + "\n```\n",
        ]
    parts.append(SOLUTIONS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
