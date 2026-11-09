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
        '"Patients in the thrombosis registry" means everyone in this database (it is a thrombosis '
        "registry; the database is even called `thrombosis_prediction`). The model saw the word "
        '"thrombosis", saw a `Thrombosis` column, and connected them: it added '
        "`AND e.Thrombosis = 1`, so it only counted patients who actually had thrombosis and the "
        "answer became 100% instead of 35.9%. Language models tend to use every important word of "
        "the question somewhere in the SQL, even words that only describe the database. The same "
        "mistake happened 3 times in the 50 labelled failures (q390, q430, q439), always with "
        '"thrombosis registry/cohort". A possible fix is a prompt rule: words that describe the '
        "database or the whole cohort are not filters.",
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
        'The question says "the three whose liver enzymes **stayed** in the normal range" and then '
        '"take their highest GPT value **among their laboratory tests that fall in the normal '
        "range**\". This can be read in two ways. **Reading A** (the model's, and most people's): "
        '"stayed in the normal range" means the GPT was *never* 60 or above, so any patient with an '
        "abnormal test is excluded. **Reading B** (the gold's): nobody is excluded; just ignore the "
        "abnormal tests and rank patients by their highest normal one. The table below shows why the "
        "answers differ: the gold's top patient had 30 abnormal GPT tests (up to 227), which hardly "
        '"stayed in the normal range". So the question is ambiguous, and the gold is arguably '
        "questionable.",
    ),
]

SOLUTIONS = """\
## How to fix each kind of error

| Error | Fix | How |
|---|---|---|
| Dirty data (21/50) | **Add BIRD's column descriptions to the schema** | BIRD's description CSVs already say e.g. *"If the weight_kg is NULL or 0, it means the weight of the superhero is missing"* and *"Thrombosis: 0 = negative, 1-3 = degree"*. The plain prompt only had CREATE TABLE + 3 example rows. |
| Dirty data | **Data profile per column** (automatic, no LLM) | For each column compute: share of NULL / 0 / '' / '-', numbers stored as text, values that look like lists ("a, b"). Add one short note per risky column, e.g. `weight_kg: 31% are 0 (likely missing)`. |
| Dirty data | **System-prompt rules** | "Treat 0, '' and '-' in measurement columns as missing unless the question says otherwise. CAST text numbers before MIN/MAX/comparisons. Fields may hold comma-separated lists." |
| Wrong filter (misread) | **System-prompt rule + self-check** | "The database itself is the cohort/registry; don't add filters the question doesn't ask for." Ask the model to list each condition it used and point to the words in the question that require it. |
| Wrong column | **Column descriptions** (same as above) | Descriptions say which column holds what (e.g. `users.UpVotes` = up-votes cast). |
| Wrong filter value | **Show the distinct values of small columns** | For columns with <= 20 distinct values, list them in the schema (`Thrombosis: 0, 1, 2, 3`; `SSA: 'negative', '0', '1', '4', ...`). Value retrieval (extra 2) only helps when the words match. |
| Wrong aggregation | **System-prompt rule + self-check** | "When counting entities (drivers, patients), use COUNT(DISTINCT id); check whether joins create duplicates." |
| Format only | **System-prompt rule** and/or a lenient scorer as a footnote | "Return exactly the requested columns; don't combine or re-type values." A lenient comparison would add only 7 of 500 for Large 4 (c). |
| Wrong join | **System-prompt rule** | "Use LEFT JOIN when the question says to include all rows (with zeros)." |
| Ambiguous | Not fixable by prompting | Report it; ideally the question should be clarified. |

**We tested the first two rows together as "column notes"** (BIRD descriptions + an automatic
data profile, `--notes` in `scripts/run_eval.py`), with BIRD's hints still in the prompt:

| Model | Plain prompt | + column notes |
|---|---|---|
| Ministral 3 8B (500 questions) | 10.6% | 16.4% |
| Mistral Small 4 (500 questions) | 18.6% | 30.8% |
| Mistral Large 4 (200-question subset) | 61.5% | 72.0% |

All three gains are statistically significant (paired McNemar test, p <= 0.01). The prompt rules
for misread filters and aggregations were not tested.

To inspect any question yourself (free, no API calls):
`uv run python scripts/show_case.py <question_id>`
"""


Q438_TABLE_SQL = """
SELECT p.ID, p.Birthday,
       MAX(CASE WHEN l.GPT < 60 THEN l.GPT END) AS highest_normal_gpt,
       MAX(l.GPT) AS highest_gpt_ever,
       SUM(l.GPT >= 60) AS abnormal_gpt_tests
FROM Patient p JOIN Laboratory l ON l.ID = p.ID
WHERE p.ID IN (SELECT ID FROM Laboratory WHERE CAST(RNP AS REAL) >= 16)
GROUP BY p.ID
HAVING highest_normal_gpt IS NOT NULL
ORDER BY highest_normal_gpt DESC, p.ID
LIMIT 6
"""


def q438_table(db: Path) -> str:
    """Markdown table showing the two readings of q438, computed from the database."""
    rows = execute_sql(db, Q438_TABLE_SQL).rows or []
    lines = [
        "**The two readings, with the real data** (patients with RNP >= 16, ranked by their highest "
        "*normal* GPT test; generated from the database):\n",
        "| Patient | Birthday | Highest normal GPT | Highest GPT ever | Abnormal GPT tests | Reading A keeps? |",
        "|---|---|---|---|---|---|",
    ]
    for pid, bday, normal, ever, abnormal in rows:
        keep = "yes" if abnormal == 0 else "no"
        lines.append(f"| {pid} | {bday} | {normal} | {ever} | {abnormal} | {keep} |")
    lines.append(
        "\nReading B (gold) takes the first three rows. Reading A (model) skips every patient with "
        "an abnormal test, so it takes 1958-10-23, 1969-11-30 and the next patient with none.\n"
    )
    return "\n".join(lines)


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
        "500 questions, 65.0% correct. 50 of the 175 wrong answers were labelled one by one "
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
        if qid == 438:
            parts.append(q438_table(db_path(DB_ROOT, q.db_id)))
    parts.append(SOLUTIONS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
