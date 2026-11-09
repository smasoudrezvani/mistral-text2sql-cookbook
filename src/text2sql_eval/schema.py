"""Render a SQLite database schema as prompt text: CREATE TABLE statements plus example rows."""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

MAX_VALUE_CHARS = 40


def _short(value: object) -> str:
    """Format one cell value for the prompt, cutting long text."""
    if value is None:
        return "NULL"
    text = str(value).replace("\n", " ")
    return text if len(text) <= MAX_VALUE_CHARS else text[: MAX_VALUE_CHARS - 3] + "..."


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)


def _tables_with_ddl(db_path: Path) -> list[tuple[str, str]]:
    con = _connect(db_path)
    try:
        return con.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL"
        ).fetchall()
    finally:
        con.close()


def table_names(db_path: Path, extra_dbs: list[Path] | None = None) -> list[str]:
    """List user tables in a SQLite database (and any extra databases).

    Args:
        db_path: Path to the ``.sqlite`` file.
        extra_dbs: Extra databases whose tables are listed after the main ones.

    Returns:
        Table names in creation order (internal ``sqlite_*`` tables excluded).
    """
    names = [n for n, _ in _tables_with_ddl(db_path)]
    for extra in extra_dbs or []:
        names += [n for n, _ in _tables_with_ddl(extra)]
    return names


def render_schema(
    db_path: Path,
    n_rows: int = 3,
    tables: list[str] | None = None,
    extra_dbs: list[Path] | None = None,
    shuffle_seed: int | None = None,
    notes: dict[str, dict[str, str]] | None = None,
) -> str:
    """Describe tables as CREATE TABLE statements, each followed by a few example rows.

    Args:
        db_path: Path to the ``.sqlite`` file.
        n_rows: Example rows per table (0 = none).
        tables: Only these tables (default: all).
        extra_dbs: Extra databases (e.g. distractor tables) rendered as if they were part of
            the main database.
        shuffle_seed: If given, shuffle the table order with this seed, so extra tables are
            mixed in with the real ones instead of all coming last.
        notes: Optional column notes ``{table: {column: note}}`` added after each table.

    Returns:
        Schema text for the prompt.
    """
    entries = [(name, ddl, Path(db_path)) for name, ddl in _tables_with_ddl(db_path)]
    for extra in extra_dbs or []:
        entries += [(name, ddl, Path(extra)) for name, ddl in _tables_with_ddl(extra)]
    if tables is not None:  # keep the caller's order (e.g. retrieval rank)
        by_name = {e[0]: e for e in entries}
        entries = [by_name[t] for t in tables if t in by_name]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(entries)

    parts = []
    cons: dict[Path, sqlite3.Connection] = {}
    try:
        for name, ddl, src in entries:
            block = ddl.strip()
            if n_rows > 0:
                con = cons.setdefault(src, _connect(src))
                quoted = '"' + name.replace('"', '""') + '"'
                cur = con.execute(f"SELECT * FROM {quoted} LIMIT ?", (n_rows,))
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
                lines = ["\t".join(cols)] + ["\t".join(_short(v) for v in r) for r in rows]
                block += f"\n/* {len(rows)} example rows from {name}:\n" + "\n".join(lines) + "\n*/"
            if notes and notes.get(name):
                lines = [f"- {c}: {n}" for c, n in notes[name].items()]
                block += f"\n/* Column notes for {name}:\n" + "\n".join(lines) + "\n*/"
            parts.append(block)
    finally:
        for con in cons.values():
            con.close()
    return "\n\n".join(parts)
