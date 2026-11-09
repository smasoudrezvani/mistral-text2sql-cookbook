# ruff: noqa: E501  (long lines are notebook markdown text)
"""Build notebooks/text2sql_mistral_bird.ipynb, the cookbook notebook.

The code cells are copied from the tested package modules (so the notebook can't drift from the
tests), and the full-results tables are generated from the committed results/summary files.

Usage:
    uv run python scripts/build_notebook.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src/text2sql_eval"
SUMMARY = ROOT / "results/summary"
LABELS = ROOT / "results/error_labels/retry_mistral-large-4_all_sample50.json"
OUT = ROOT / "notebooks/text2sql_mistral_bird.ipynb"
REPO = "https://github.com/smasoudrezvani/mistral-text2sql-cookbook"

DEMO_IDS = [10, 18, 24, 25, 35, 37, 339, 348, 350, 356, 361, 363, 368, 374, 375, 382, 387, 391,
            404, 406, 408, 413, 414, 420, 427, 433, 438, 442, 444, 454]  # fmt: skip
DEMO_DBS = ["california_schools", "superhero", "thrombosis_prediction"]


def module_code(name: str) -> str:
    """Return a package module's source without its docstring and package-internal imports."""
    src = (PKG / f"{name}.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    drop: list[tuple[int, int]] = []
    for i, node in enumerate(tree.body):
        is_doc = i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        is_pkg = isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "text2sql_eval"
        )
        is_tc_import = (
            isinstance(node, ast.ImportFrom)
            and node.module == "typing"
            and [a.name for a in node.names] == ["TYPE_CHECKING"]
        )
        is_tc_block = isinstance(node, ast.If) and getattr(node.test, "id", "") == "TYPE_CHECKING"
        if is_doc or is_future or is_pkg or is_tc_import or is_tc_block:
            drop.append((node.lineno, node.end_lineno or node.lineno))
    lines = src.splitlines()
    keep = [ln for no, ln in enumerate(lines, start=1) if not any(a <= no <= b for a, b in drop)]
    text = "\n".join(keep).strip()
    text = text.replace("from typing import TYPE_CHECKING, ", "from typing import ")
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    return f"# --- copied from src/text2sql_eval/{name}.py in the companion repo ---\n{text}\n"


def summary(run: str) -> dict:
    """Load one committed run summary."""
    return json.loads((SUMMARY / f"{run}.json").read_text(encoding="utf-8"))


def pct(run: str) -> str:
    """Execution accuracy (BIRD rule) of a run as a percentage string."""
    return f"{100 * summary(run)['ex_bird']:.1f}%"


def full_results_md() -> str:
    """Markdown tables of the full runs, generated from results/summary."""
    m8, sm, lg = "ministral-8b-2512", "mistral-small-2603", "mistral-large-4"

    def sub200(run: str) -> str:
        sub = set(json.loads((ROOT / "slices/sub200.json").read_text())["question_ids"])
        recs = [json.loads(x) for x in (ROOT / f"results/raw/{run}.jsonl").read_text().splitlines()]
        recs = [r for r in recs if r["question_id"] in sub]
        return f"{100 * sum(r['bird'] for r in recs) / len(recs):.1f}%"

    rows = [
        (
            "(a) full schema",
            pct(f"full_schema_{m8}_all"),
            pct(f"full_schema_{sm}_all"),
            pct(f"full_schema_{lg}_all"),
            sub200(f"full_schema_{lg}_all"),
        ),
        (
            "(b) schema retrieval",
            pct(f"retrieval_{m8}_all"),
            pct(f"retrieval_{sm}_all"),
            pct(f"retrieval_{lg}_all"),
            sub200(f"retrieval_{lg}_all"),
        ),
        (
            "(c) execute and retry",
            pct(f"retry_{m8}_all"),
            pct(f"retry_{sm}_all"),
            pct(f"retry_{lg}_all"),
            sub200(f"retry_{lg}_all"),
        ),
        (
            "(a) + value retrieval",
            pct(f"full_schema_{m8}_all_values"),
            pct(f"full_schema_{sm}_all_values"),
            "–",
            pct(f"full_schema_{lg}_sub200_values"),
        ),
        ("majority vote, 5 samples", pct(f"vote_{m8}_all"), pct(f"vote_{sm}_all"), "–", "–"),
        (
            "**(a) + column notes**",
            f"**{pct(f'full_schema_{m8}_all_notes')}**",
            f"**{pct(f'full_schema_{sm}_all_notes')}**",
            "–",
            f"**{pct(f'full_schema_{lg}_sub200_notes')}**",
        ),
        (
            "**(c) + column notes**",
            f"**{pct(f'retry_{m8}_all_notes')}**",
            f"**{pct(f'retry_{sm}_all_notes')}**",
            "–",
            "–",
        ),
    ]
    table = "| Approach | Ministral 3 8B (500 q) | Mistral Small 4 (500 q) | Mistral Large 4 (500 q) | Mistral Large 4 (200 q subset) |\n|---|---|---|---|---|\n"
    table += "\n".join("| " + " | ".join(r) + " |" for r in rows)

    distract = (
        "| Model | (a) without → with 30 distractor tables | (b) without → with |\n|---|---|---|\n"
        f"| Ministral 3 8B | {pct(f'full_schema_{m8}_all')} → {pct(f'full_schema_{m8}_all_distract')} | {pct(f'retrieval_{m8}_all')} → {pct(f'retrieval_{m8}_all_distract')} |\n"
        f"| Mistral Small 4 | {pct(f'full_schema_{sm}_all')} → {pct(f'full_schema_{sm}_all_distract')} | {pct(f'retrieval_{sm}_all')} → {pct(f'retrieval_{sm}_all_distract')} |\n"
        f"| Mistral Large 4 (200 q) | {sub200(f'full_schema_{lg}_all')} → {pct(f'full_schema_{lg}_sub200_distract')} | {sub200(f'retrieval_{lg}_all')} → {pct(f'retrieval_{lg}_sub200_distract')} |"
    )
    return table, distract


def error_table_md() -> str:
    """Markdown table of the labelled error categories."""
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    counts: dict[str, int] = {}
    for v in labels.values():
        counts[v["category"]] = counts.get(v["category"], 0) + 1
    names = {
        "dirty_data": "Dirty data (0 or '-' meaning missing, numbers stored as text, lists in one field)",
        "wrong_filter": "Wrong filter (question misread, extra condition added)",
        "wrong_column": "Wrong column",
        "wrong_filter_value": "Wrong filter value ('-' instead of the stored 'negative')",
        "wrong_aggregation": "Wrong aggregation (COUNT instead of COUNT DISTINCT)",
        "format_only": "Format only (answer right, e.g. 2015 vs '2015')",
        "wrong_join": "Wrong join",
        "ambiguous": "Ambiguous question",
    }
    rows = [f"| {names[k]} | {counts[k]} |" for k in names if k in counts]
    return "| Error type | Count (of 50) |\n|---|---|\n" + "\n".join(rows)


def build() -> nbformat.NotebookNode:
    """Assemble the notebook cells."""
    full, distract = full_results_md()
    m8, sm = "ministral-8b-2512", "mistral-small-2603"
    c = []
    md = lambda s: c.append(new_markdown_cell(s.strip()))  # noqa: E731
    code = lambda s: c.append(new_code_cell(s.strip()))  # noqa: E731

    md(f"""
# Text-to-SQL with Mistral models: measuring execution accuracy on BIRD

*Author: Masoud Rezvaninejad ([@smasoudrezvani](https://github.com/smasoudrezvani)), Independent*

Turning a question like *"What is the average weight of heroes as smart as Abomination?"* into SQL
looks easy, until the database has real-world quirks: a weight of `0` that means "unknown", numbers
stored as text, or several diagnoses packed into one field.

In this notebook you will:
1. Load real, messy databases from the [BIRD](https://bird-bench.github.io/) benchmark (mini-dev, 2026 release).
2. Build an **execution-accuracy scorer**: run the model's SQL and the correct SQL, and compare the results.
3. Try three ways of prompting a Mistral model: (a) the plain schema, (a) + **column notes**, and (c) an **execute-and-retry** loop.
4. See results for Ministral 3 8B, Mistral Small 4 and Mistral Large 4 on all 500 questions, and what *didn't* help.

The headline: adding short **column notes** (BIRD's column descriptions plus an automatic check for data traps) to the prompt raised Mistral Small 4 from **{pct(f"full_schema_{sm}_all")} to {pct(f"full_schema_{sm}_all_notes")}** on 500 questions, with no extra model calls.

All code, tests and per-question results are in the companion repo: {REPO}
""")
    md("""
## Setup

We need the Mistral SDK and `remotezip`, which lets us download only the few databases we need from the 800 MB BIRD archive.
""")
    code("%pip install -q mistralai==3.7.0 remotezip==0.12.6")
    code("""
import os
from getpass import getpass

from mistralai.client import Mistral

if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass("Mistral API key: ")
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

MODEL = "mistral-small-2603"  # also try "ministral-8b-2512" or "mistral-large-4" (slower: it reasons first)
""")
    md("""
## 1. The data: BIRD mini-dev

BIRD has 500 hand-checked mini-dev questions over 11 real SQLite databases. Each record has a question, a short **hint** ("evidence") written by an expert, and the correct (**gold**) SQL. The questions come from Hugging Face; the databases from the official archive. To keep this notebook quick and cheap we use **30 fixed questions** from three databases (two of them full of messy values).
""")
    code(f"""
import json
import urllib.request
from pathlib import Path

from remotezip import RemoteZip

QUESTIONS_URL = "https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/main/data/mini_dev_sqlite.jsonl"
DB_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
DEMO_DBS = {DEMO_DBS!r}
DEMO_IDS = {DEMO_IDS!r}

with urllib.request.urlopen(QUESTIONS_URL) as f:
    questions = [json.loads(line) for line in f.read().decode("utf-8").splitlines() if line.strip()]
demo = [q for q in questions if q["question_id"] in DEMO_IDS]

DATA = Path("bird")
with RemoteZip(DB_ZIP_URL) as z:  # downloads only these databases (~19 MB), not the whole archive
    for name in z.namelist():
        if any(f"/dev_databases/{{db}}/" in name for db in DEMO_DBS) and not name.endswith("/"):
            z.extract(name, DATA)
DB_ROOT = DATA / "minidev/MINIDEV/dev_databases"


def db_file(db_id: str) -> Path:
    return DB_ROOT / db_id / f"{{db_id}}.sqlite"


print(len(questions), "questions in mini-dev;", len(demo), "in this demo")
print(json.dumps(demo[7], indent=1))
""")
    md("""
## 2. The scorer: execution accuracy

We don't compare SQL text (two different queries can both be right). We **run** the model's query and the gold query and compare the rows:

- **BIRD rule** (the official one): the two results are equal as *sets* (row order and duplicates ignored).
- **Strict rule**: duplicates count, and order counts when the gold query has an outer `ORDER BY`.

The database is opened **read-only** (a model's `DELETE` can't damage it), and a timeout really stops a runaway query.
""")
    code(module_code("scorer"))
    code("""
GOLD = {q["question_id"]: execute_sql(db_file(q["db_id"]), q["SQL"], timeout_s=120) for q in demo}
assert all(g.ok for g in GOLD.values())
print("gold answers ready for", len(GOLD), "questions")
""")
    md("""
## 3. The prompt: schema, example rows, hint, question

The model sees each table's `CREATE TABLE` statement and 3 example rows, then the hint and the question, and replies with one SQL query in a code block. Some Mistral models (like Large 4) reason before answering; we keep only the answer text.
""")
    code(module_code("schema"))
    code(module_code("prompting"))
    code("""
def evaluate(label: str, use_notes: bool = False, retry: bool = False) -> dict[int, bool]:
    \"\"\"Answer every demo question, score it, and print the execution accuracy.\"\"\"
    correct = {}
    for q in demo:
        db = db_file(q["db_id"])
        schema = render_schema(db, notes=NOTES.get(q["db_id"]) if use_notes else None)
        messages = build_messages(schema, q["question"], q["evidence"])
        if retry:
            pred = solve_with_retry(client, MODEL, messages, db, timeout_s=30).result
        else:
            pred = execute_sql(db, extract_sql(chat(client, MODEL, messages).text), timeout_s=30)
        correct[q["question_id"]] = compare(pred, GOLD[q["question_id"]], q["SQL"]).bird
    n = sum(correct.values())
    print(f"{label}: {n}/{len(demo)} correct ({n / len(demo):.0%})")
    return correct


NOTES: dict = {}
results = {"(a) plain schema": evaluate("(a) plain schema")}
""")
    md("""
## 4. Column notes: tell the model what the data means

Most failures we saw were **dirty data**, not bad SQL. BIRD ships a description file per table that already warns about some traps, e.g. for `superhero.weight_kg`: *"If the weight_kg is NULL or 0, it means the weight of the superhero is missing"*. Our plain prompt never showed it.

**Column notes** add, per column:
1. BIRD's description, and
2. an automatic **profile** that looks at the actual values and writes an instruction only when it finds a trap: many zeros, placeholders like `''` or `'-'`, numbers stored as text, numbers mixed with words, codes with leading zeros, comma-separated lists, and the full list of values for small columns.
""")
    code(module_code("notes"))
    code("""
NOTES = {db: build_notes(db_file(db), DB_ROOT / db / "database_description") for db in DEMO_DBS}
print("note for superhero.weight_kg:\\n ", NOTES["superhero"]["superhero"]["weight_kg"])
print("note for Laboratory.SSA:\\n ", NOTES["thrombosis_prediction"]["Laboratory"]["SSA"])
""")
    code('results["(a) + column notes"] = evaluate("(a) + column notes", use_notes=True)')
    md("""
## 5. Execute and retry

The model writes SQL, we run it, and if it fails (an error, a timeout, or zero rows) we send that feedback back and ask for a fix, up to 3 attempts. This fixes SQL that doesn't run; it rarely fixes SQL that runs but answers the wrong question.
""")
    code(module_code("agent"))
    code(
        'results["(c) retry + column notes"] = evaluate("(c) retry + column notes", use_notes=True, retry=True)'
    )
    md("""
## 6. Results

**Demo (30 questions).** With only 30 questions, a single question is 3 points, and the same prompt can give different SQL on a rerun even at temperature 0, so treat these numbers as an illustration.
""")
    code("""
for label, res in results.items():
    print(f"{label:28s} {sum(res.values()):2d}/{len(res)}")
base, best = results["(a) plain schema"], results["(c) retry + column notes"]
print("\\nwrong -> right:", [q for q in base if best[q] and not base[q]])
print("right -> wrong:", [q for q in base if base[q] and not best[q]])
""")
    md(f"""
**Full runs (all 500 questions, BIRD rule).** These come from the companion repo, where every run is reproducible with one command and every per-question answer is saved. Large 4 reasons before answering and costs about 15x more per question than Small 4, so some of its extras were run on a fixed 200-question subset.

{full}

What we learned:
- **Column notes are the biggest win** for every model (+5.8 points for Ministral 3 8B, +12.2 for Small 4, +10.5 for Large 4 on the 200-question subset; all statistically significant in a paired McNemar test). They help even though BIRD's per-question hints are already in the prompt: the hints explain the *question*, the notes explain the *data*.
- **Execute-and-retry** fixes SQL that fails to run, and stacks with column notes (Ministral 3 8B: {pct(f"full_schema_{m8}_all")} → {pct(f"retry_{m8}_all_notes")}).
- **Value retrieval** (putting stored values that match words in the question into the prompt) and **majority voting** gave no clear gain here; for Large 4, the 5 voted samples agreed on most questions.
- **Schema retrieval** (showing only relevant tables) changes nothing on BIRD's small databases, but it protects accuracy when the database is large. With 30 unrelated "distractor" tables added, the plain prompt loses accuracy and retrieval does not:

{distract}
""")
    md(f"""
## 7. Why do the remaining answers fail?

We read 50 random failures of Mistral Large 4 with execute-and-retry (before column notes):

{error_table_md()}

One real example per type, with the model's SQL and the gold SQL, is in [docs/error_examples.md]({REPO}/blob/main/docs/error_examples.md). Dirty data was the largest group, which is what led us to column notes.

## Takeaways

1. Measure with **execution accuracy** on real, messy data; a demo database hides these problems.
2. Before adding agents or more model calls, **give the model the data's meaning**: column descriptions plus a cheap automatic profile.
3. Add an **execute-and-retry** loop for SQL that doesn't run.
4. Use **schema retrieval** only when the schema is large.

## References

- BIRD: Li et al., 2023, [arXiv 2305.03111](https://arxiv.org/abs/2305.03111)
- Synthetic column descriptions: Wretblad et al., 2024, [arXiv 2408.04691](https://arxiv.org/abs/2408.04691)
- Automatic metadata extraction (profiling): Shkapenyuk et al., 2025, [arXiv 2505.19988](https://arxiv.org/abs/2505.19988)
- CHESS (value and context retrieval): Talaei et al., 2024, [arXiv 2405.16755](https://arxiv.org/abs/2405.16755)
- Self-consistency (majority voting): Wang et al., 2022, [arXiv 2203.11171](https://arxiv.org/abs/2203.11171)
""")
    nb = new_notebook(cells=c)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def main() -> None:
    """Write the notebook file."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build(), OUT)
    print(OUT)


if __name__ == "__main__":
    main()
