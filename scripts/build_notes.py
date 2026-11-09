"""Build column notes (BIRD descriptions + data profile) for every BIRD database.

Usage:
    uv run python scripts/build_notes.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from text2sql_eval.bird import db_path
from text2sql_eval.notes import build_notes, save_notes

ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = ROOT / "data/raw/minidev/MINIDEV/dev_databases"
OUT = ROOT / "data/cache/notes"

logger = logging.getLogger("build_notes")


def main() -> None:
    """Write data/cache/notes/<db_id>.json for every database."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for d in sorted(p for p in DB_ROOT.iterdir() if p.is_dir()):
        start = time.monotonic()
        notes = build_notes(db_path(DB_ROOT, d.name), d / "database_description")
        save_notes(notes, OUT / f"{d.name}.json")
        n = sum(len(v) for v in notes.values())
        logger.info("%-25s %3d column notes (%.0f s)", d.name, n, time.monotonic() - start)


if __name__ == "__main__":
    main()
