"""Run all BIRD mini-dev gold queries once and save the results and the 50-question dev slice.

Usage:
    uv run python scripts/build_gold_cache.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from text2sql_eval.bird import build_gold_cache, load_questions, sample_ids, save_gold_cache

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data/raw/mini_dev_sqlite.jsonl"
DB_ROOT = ROOT / "data/raw/minidev/MINIDEV/dev_databases"
GOLD_OUT = ROOT / "data/cache/gold_results.json"
DEV_SLICE_OUT = ROOT / "slices/dev50.json"

logger = logging.getLogger("build_gold_cache")


def main() -> None:
    """Build the gold cache and the dev slice."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    questions = load_questions(JSONL)
    logger.info("loaded %d questions", len(questions))

    start = time.monotonic()
    cache = build_gold_cache(questions, DB_ROOT)
    save_gold_cache(cache, GOLD_OUT)
    slow = sorted(((g.seconds, qid) for qid, g in cache.items()), reverse=True)[:3]
    logger.info(
        "saved %d gold results in %.0f s; slowest (s, id): %s",
        len(cache),
        time.monotonic() - start,
        [(round(s, 1), q) for s, q in slow],
    )

    ids = sample_ids(questions, 50, seed=0)
    DEV_SLICE_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEV_SLICE_OUT.write_text(json.dumps({"seed": 0, "question_ids": ids}, indent=1) + "\n")
    logger.info("dev slice: %s", DEV_SLICE_OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
