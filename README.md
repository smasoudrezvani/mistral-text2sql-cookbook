# mistral-text2sql-cookbook

Text-to-SQL with Mistral models, measured by **execution accuracy** on the
[BIRD mini-dev](https://github.com/bird-bench/mini_dev) benchmark (2026 release, SQLite, 500 questions).

The notebook [`notebooks/text2sql_mistral_bird.ipynb`](notebooks/text2sql_mistral_bird.ipynb) is a
short, Colab-runnable walkthrough. This repo holds the tested code behind it and every run.

## Results (all 500 questions, BIRD's official execution-accuracy rule)

| Approach | Ministral 3 8B | Mistral Small 4 | Mistral Large 4 | Large 4 (200-question subset) |
|---|---|---|---|---|
| (a) full schema + 3 example rows + hint | 10.6% | 18.6% | 60.2% | 61.5% |
| (b) schema retrieval (embeddings, top 5 + linked tables) | 10.2% | 20.4% | 60.8% | 61.5% |
| (c) execute and retry (max 3 attempts) | 14.0% | 20.6% | 65.0% | 66.0% |
| (a) + value retrieval | 10.2% | 18.4% | – | 60.5% |
| majority vote, 5 samples | 10.6% | 21.0% | – | – |
| **(a) + column notes** | **16.4%** | **30.8%** | – | **72.0%** |
| **(c) + column notes** | **21.0%** | **32.8%** | – | – |

Models: `ministral-8b-2512`, `mistral-small-2603`, `mistral-large-4` (default reasoning), temperature 0.
Large 4 reasons before answering and costs ~15x more per question, so its extras ran on a fixed
200-question subset (`slices/sub200.json`).

**Column notes** = BIRD's column descriptions + an automatic profile of each column's values that
flags data traps (0 or '-' meaning missing, numbers stored as text, codes with leading zeros,
comma-separated lists, the full value list of small columns). They help every model, even with
BIRD's per-question hints already in the prompt (paired McNemar test: p <= 0.01 for all three).

More findings:
- With 30 unrelated **distractor tables** per database, (a) loses 3-6 points; (b) schema retrieval holds.
- **Value retrieval** and **majority voting** gave no clear gain.
- Of 50 hand-labelled failures of Large 4 + retry, **21 were dirty data**
  ([docs/error_examples.md](docs/error_examples.md)).
- Literature context: [docs/literature_review.md](docs/literature_review.md).

Every number above is in [`results/summary/`](results/summary/) and can be reproduced with the
commands below. The run-to-run noise of the API is a few points on these sets (the same prompt
at temperature 0 does not always return the same SQL), so compare runs question by question.

## Reproduce

```bash
uv sync
cp .env.example .env                     # add your MISTRAL_API_KEY
uv run pytest                            # 88 tests, no API calls

uv run python scripts/download_data.py   # questions + 11 SQLite databases (~1.4 GB)
uv run python scripts/build_gold_cache.py
uv run python scripts/build_notes.py     # column notes (free)
uv run python scripts/build_value_index.py
uv run python scripts/build_distractors.py

# one run = one approach x one model x one slice (resumable; results/raw + results/summary)
uv run python scripts/run_eval.py --approach full_schema --model mistral-small-2603 --slice all --notes
uv run python scripts/run_eval.py --approach retry --model ministral-8b-2512 --slice all --notes --workers 8
```

Options of `run_eval.py`: `--approach full_schema|retrieval|retry|vote`, `--model`,
`--slice all|dev50|sub200`, `--notes`, `--values`, `--distractors`, `--workers N`.

## Layout

```
src/text2sql_eval/
  scorer.py       execution accuracy: run SQL read-only with a timeout; BIRD and strict rules
  bird.py         load questions; run and cache gold results
  schema.py       schema as prompt text (CREATE TABLE + example rows [+ column notes])
  prompting.py    prompt, Mistral chat call (handles reasoning chunks), SQL extraction
  retrieval.py    (b) table retrieval with mistral-embed + foreign-key neighbours
  agent.py        (c) execute-and-retry loop
  notes.py        column notes: BIRD descriptions + data-trap profile
  values.py       value retrieval (word index of stored values)
  voting.py       majority vote on result rows
  distractors.py  distractor tables borrowed from other BIRD databases
scripts/          data download, caches, run_eval.py, analysis and notebook builders
tests/            pytest suite (no API calls)
results/          summary/ per run (committed); raw/ per question (not committed)
docs/             error examples, literature review
notebooks/        the cookbook notebook (built by scripts/build_notebook.py)
```

## License

Code: MIT. BIRD data: see the [BIRD license](https://bird-bench.github.io/) (not redistributed here).
