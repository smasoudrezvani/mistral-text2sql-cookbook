"""Render a SQLite database schema as prompt text: CREATE TABLE statements plus example rows."""

from __future__ import annotations

import sqlite3
from pathlib import Path

MAX_VALUE_CHARS = 40


def _short(value: object) -> str:
    """Format one cell value for the prompt, cutting long text."""
    if value is None:
        return "NULL"
    text = str(value).replace("\n", " ")
    return text if len(text) <= MAX_VALUE_CHARS else text[: MAX_VALUE_CHARS - 3] + "..."


def table_names(db_path: Path) -> list[str]:
    """List user tables in a SQLite database.

    Args:
        db_path: Path to the ``.sqlite`` file.

    Returns:
        Table names in creation order (internal ``sqlite_*`` tables excluded).
    """
    con = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def render_schema(db_path: Path, n_rows: int = 3, tables: list[str] | None = None) -> str:
    """Describe tables as CREATE TABLE statements, each followed by a few example rows.

    Args:
        db_path: Path to the ``.sqlite`` file.
        n_rows: Example rows per table (0 = none).
        tables: Only these tables (default: all).

    Returns:
        Schema text for the prompt.
    """
    con = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        ddl = dict(
            con.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL"
            ).fetchall()
        )
        wanted = list(ddl) if tables is None else [t for t in tables if t in ddl]
        parts = []
        for name in wanted:
            block = ddl[name].strip()
            if n_rows > 0:
                quoted = '"' + name.replace('"', '""') + '"'
                cur = con.execute(f"SELECT * FROM {quoted} LIMIT ?", (n_rows,))
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
                lines = ["\t".join(cols)] + ["\t".join(_short(v) for v in r) for r in rows]
                block += f"\n/* {len(rows)} example rows from {name}:\n" + "\n".join(lines) + "\n*/"
            parts.append(block)
    finally:
        con.close()
    return "\n\n".join(parts)
