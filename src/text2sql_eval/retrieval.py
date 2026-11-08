"""Approach (b): schema retrieval. Pick the tables a question needs before writing SQL.

Each table is described as text (name, columns, BIRD column descriptions) and embedded with
``mistral-embed``. For a question we keep the ``k`` most similar tables, then add tables linked
to them by a foreign key (one hop), so join tables are not lost.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

import sqlglot
from sqlglot import exp

if TYPE_CHECKING:
    from mistralai.client import Mistral

logger = logging.getLogger(__name__)

EMBED_MODEL = "mistral-embed-2312"
MAX_VALUE_DESC_CHARS = 150


def _read_text(path: Path) -> str:
    """Read a description CSV; a few BIRD files are not UTF-8."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)


def table_docs(db_path: Path, desc_dir: Path | None = None) -> dict[str, str]:
    """Describe each table as one paragraph of text for embedding.

    Args:
        db_path: Path to the ``.sqlite`` file.
        desc_dir: BIRD ``database_description`` folder (one CSV per table), or None.

    Returns:
        Mapping from table name to its description text.
    """
    descs: dict[str, dict[str, str]] = {}
    if desc_dir is not None and Path(desc_dir).is_dir():
        for p in Path(desc_dir).glob("*.csv"):
            cols: dict[str, str] = {}
            for row in csv.DictReader(io.StringIO(_read_text(p))):
                row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
                name = row.get("original_column_name", "")
                parts = [row.get("column_name", ""), row.get("column_description", "")]
                value = row.get("value_description", "")
                if value:
                    parts.append("values: " + " ".join(value.split())[:MAX_VALUE_DESC_CHARS])
                text = "; ".join(x for x in dict.fromkeys(parts) if x)
                if name:
                    cols[name.lower()] = text
            descs[p.stem.lower()] = cols

    con = _connect(db_path)
    try:
        names = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        docs = {}
        for t in names:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
            tdesc = descs.get(t.lower(), {})
            col_texts = [f"{c} ({tdesc[c.lower()]})" if tdesc.get(c.lower()) else c for c in cols]
            docs[t] = f"Table {t}. Columns: " + ", ".join(col_texts)
    finally:
        con.close()
    return docs


def foreign_key_neighbors(db_path: Path) -> dict[str, set[str]]:
    """Find tables linked by a foreign key, in both directions.

    Args:
        db_path: Path to the ``.sqlite`` file.

    Returns:
        Mapping from table name to the set of tables it links to or is linked from.
    """
    con = _connect(db_path)
    try:
        names = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        lower = {n.lower(): n for n in names}
        nbrs: dict[str, set[str]] = {n: set() for n in names}
        for t in names:
            for fk in con.execute(f'PRAGMA foreign_key_list("{t}")'):
                other = lower.get(str(fk[2]).lower())
                if other and other != t:
                    nbrs[t].add(other)
                    nbrs[other].add(t)
    finally:
        con.close()
    return nbrs


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def select_tables(
    question_vec: list[float],
    table_vecs: dict[str, list[float]],
    neighbors: dict[str, set[str]],
    k: int = 5,
) -> list[str]:
    """Pick the top-k most similar tables, plus their foreign-key neighbours.

    Args:
        question_vec: Embedding of the question (and hint).
        table_vecs: Embedding of each table's description.
        neighbors: Output of foreign_key_neighbors.
        k: How many tables to keep by similarity.

    Returns:
        Selected table names: the top-k by similarity first, then added neighbours.
    """
    ranked = sorted(table_vecs, key=lambda t: cosine(question_vec, table_vecs[t]), reverse=True)
    top = ranked[:k]
    extra = sorted({n for t in top for n in neighbors.get(t, set())} - set(top))
    return top + extra


_FROM_JOIN = re.compile(
    r'\b(?:FROM|JOIN)\s+(?:`([^`]+)`|"([^"]+)"|\[([^\]]+)\]|([A-Za-z_][A-Za-z0-9_]*))',
    re.IGNORECASE,
)


def tables_in_sql(sql: str, table_names: list[str]) -> set[str]:
    """Find which known tables a SQL query reads from.

    Uses the sqlglot parser, so keywords (``ORDER BY``), column names and CTE names are not
    mistaken for tables. If the SQL can't be parsed, falls back to names right after FROM/JOIN.

    Args:
        sql: A SQL query.
        table_names: All tables in the database.

    Returns:
        Table names (as spelled in table_names) that the query reads from.
    """
    by_lower = {t.lower(): t for t in table_names}
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
        found = {t.name.lower() for t in tree.find_all(exp.Table)} - ctes
    except Exception:  # unparseable model output: be conservative
        found = {next(g for g in m.groups() if g).lower() for m in _FROM_JOIN.finditer(sql or "")}
    return {by_lower[n] for n in found if n in by_lower}


def embed(
    client: Mistral, texts: list[str], batch_size: int = 32, retries: int = 6
) -> list[list[float]]:
    """Embed texts with mistral-embed, in batches, with retry on transient errors.

    Args:
        client: A ``mistralai.client.Mistral`` instance.
        texts: Texts to embed.
        batch_size: Texts per request.
        retries: Attempts per batch.

    Returns:
        One vector per text, in order.

    Raises:
        RuntimeError: If a batch fails every attempt.
    """
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(retries):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, inputs=batch)
                out.extend(d.embedding for d in resp.data)
                break
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                wait = 60 if status == 429 else 2**attempt
                logger.warning("embed failed (%.80s); retry %d in %ds", e, attempt + 1, wait)
                time.sleep(wait)
        else:
            raise RuntimeError(f"embedding batch {i // batch_size} failed after {retries} tries")
    return out
