"""Download BIRD mini-dev: the 2026 questions (Hugging Face) and the SQLite databases.

Only the SQLite files and description CSVs are extracted from the 800 MB archive (remote,
partial download), not the MySQL/PostgreSQL dumps.

Usage:
    uv run python scripts/download_data.py                 # all 11 databases (~1.4 GB)
    uv run python scripts/download_data.py --dbs superhero # just one database
"""

from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path

from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
QUESTIONS_URL = (
    "https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/main/data/mini_dev_sqlite.jsonl"
)
DB_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
PREFIX = "minidev/MINIDEV/dev_databases/"

logger = logging.getLogger("download_data")


def main() -> None:
    """Download questions and (selected) databases into data/raw/."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbs", nargs="*", help="database ids to fetch (default: all)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "mini_dev_sqlite.jsonl"
    urllib.request.urlretrieve(QUESTIONS_URL, out)
    logger.info("questions -> %s", out.relative_to(ROOT))

    with RemoteZip(DB_ZIP_URL) as z:
        names = [
            n
            for n in z.namelist()
            if n.startswith(PREFIX)
            and n.lower().endswith((".sqlite", ".csv"))
            and (not args.dbs or n[len(PREFIX) :].split("/")[0] in args.dbs)
        ]
        for i, n in enumerate(names, start=1):
            z.extract(n, RAW)
            if i % 10 == 0 or i == len(names):
                logger.info("extracted %d / %d files", i, len(names))
    logger.info("databases -> %s", (RAW / "minidev/MINIDEV/dev_databases").relative_to(ROOT))


if __name__ == "__main__":
    main()
