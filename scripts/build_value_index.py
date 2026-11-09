"""Build the value index (short text values by word) for every BIRD database.

Usage:
    uv run python scripts/build_value_index.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from text2sql_eval.bird import db_path
from text2sql_eval.values import ValueIndex

ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = ROOT / "data/raw/minidev/MINIDEV/dev_databases"
OUT = ROOT / "data/cache/values"

logger = logging.getLogger("build_value_index")


def main() -> None:
    """Write data/cache/values/<db_id>.json for every database."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for d in sorted(p for p in DB_ROOT.iterdir() if p.is_dir()):
        start = time.monotonic()
        idx = ValueIndex.from_db(db_path(DB_ROOT, d.name))
        idx.save(OUT / f"{d.name}.json")
        cols = len({(t, c) for t, c, _ in idx.entries})
        logger.info(
            "%-25s %7d values in %3d columns (%.0f s)",
            d.name,
            len(idx.entries),
            cols,
            time.monotonic() - start,
        )


if __name__ == "__main__":
    main()
