"""Distractor tables: make a small BIRD database look like a big warehouse.

For each target database we copy real tables from the *other* BIRD databases into a separate
SQLite file (a few hundred rows each). At run time that file is ATTACHed read-only next to the
real database. SQLite resolves unqualified table names in the main database first, so the gold
SQL is unaffected, while a model that picks a distractor table gets SQL that runs but returns a
wrong answer, just like picking the wrong table in a real warehouse.
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_N_TABLES = 30
DEFAULT_MAX_ROWS = 500


def _ro(path: Path) -> str:
    return f"{Path(path).resolve().as_uri()}?mode=ro"


def _user_tables(con: sqlite3.Connection, schema: str = "main") -> dict[str, str]:
    rows = con.execute(
        f"SELECT name, sql FROM {schema}.sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL"
    ).fetchall()
    return dict(rows)


def build_distractor_db(
    target_db: Path,
    source_dbs: dict[str, Path],
    out_path: Path,
    n_tables: int = DEFAULT_N_TABLES,
    max_rows: int = DEFAULT_MAX_ROWS,
    seed: int = 0,
) -> dict[str, str]:
    """Copy random tables from other databases into one distractor file.

    Tables whose name already exists in the target (case-insensitive) are skipped, so a
    distractor can never shadow a real table.

    Args:
        target_db: The real database the distractors are for.
        source_dbs: Other databases to borrow from, ``{db_id: path}`` (target excluded).
        out_path: Where to write the distractor ``.sqlite`` file (overwritten).
        n_tables: How many distractor tables to copy.
        max_rows: Rows copied per table.
        seed: Random seed (same seed, same tables).

    Returns:
        Manifest ``{distractor_table: source_db_id}``, also saved next to out_path as JSON.
    """
    con = sqlite3.connect(_ro(target_db), uri=True)
    taken = {t.lower() for t in _user_tables(con)}
    con.close()

    candidates: list[tuple[str, str]] = []
    for db_id, path in sorted(source_dbs.items()):
        con = sqlite3.connect(_ro(path), uri=True)
        candidates += [(db_id, t) for t in sorted(_user_tables(con))]
        con.close()
    random.Random(seed).shuffle(candidates)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    # uri=True so the read-only "file:...?mode=ro" sources can be ATTACHed below
    out = sqlite3.connect(f"{out_path.resolve().as_uri()}?mode=rwc", uri=True)
    manifest: dict[str, str] = {}
    try:
        for db_id, table in candidates:
            if len(manifest) >= n_tables:
                break
            if table.lower() in taken:
                continue
            out.execute("ATTACH DATABASE ? AS src", (_ro(source_dbs[db_id]),))
            try:
                ddl = _user_tables(out, "src")[table]
                out.execute(ddl)
                quoted = '"' + table.replace('"', '""') + '"'
                out.execute(
                    f"INSERT INTO main.{quoted} SELECT * FROM src.{quoted} LIMIT ?", (max_rows,)
                )
                out.commit()
            finally:
                out.execute("DETACH DATABASE src")
            manifest[table] = db_id
            taken.add(table.lower())
    finally:
        out.close()
    out_path.with_suffix(".json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    logger.info("wrote %d distractor tables to %s", len(manifest), out_path.name)
    return manifest


def load_manifest(distractor_db: Path) -> dict[str, str]:
    """Read the ``{table: source_db_id}`` manifest saved by build_distractor_db."""
    return json.loads(Path(distractor_db).with_suffix(".json").read_text(encoding="utf-8"))
