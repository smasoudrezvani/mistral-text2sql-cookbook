# mistral-text2sql-cookbook

Text-to-SQL with Mistral models, measured by **execution accuracy** on the
[BIRD mini-dev](https://github.com/bird-bench/mini_dev) benchmark (SQLite).

Work in progress. Three approaches are compared on the same questions:

1. full schema in the prompt
2. schema retrieval (only relevant tables)
3. execute-and-retry agent loop

## Setup

```bash
uv sync
cp .env.example .env   # then add your MISTRAL_API_KEY
uv run pytest
```
