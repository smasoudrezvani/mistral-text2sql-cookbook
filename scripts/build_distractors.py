"""Build one distractor-table file per BIRD database (tables borrowed from the other databases).

Usage:
    uv run python scripts/build_distractors.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from text2sql_eval.bird import db_path
from text2sql_eval.distractors import build_distractor_db

ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = ROOT / "data/raw/minidev/MINIDEV/dev_databases"
OUT = ROOT / "data/cache/distractors"


def main() -> None:
    """Write data/cache/distractors/<db_id>.sqlite (+ .json manifest) for every database."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dbs = {p.name: db_path(DB_ROOT, p.name) for p in sorted(DB_ROOT.iterdir()) if p.is_dir()}
    for db_id, path in dbs.items():
        others = {k: v for k, v in dbs.items() if k != db_id}
        build_distractor_db(path, others, OUT / f"{db_id}.sqlite", n_tables=30, seed=0)


if __name__ == "__main__":
    main()
