"""Column notes: BIRD column descriptions plus an automatic data profile, as prompt text.

For every column we combine:
1. BIRD's description (``database_description/<table>.csv``): what the column means, and its
   value description (e.g. "If the weight_kg is NULL or 0, it means ... missing").
2. A profile computed with SQL on a sample of rows, kept only when it finds a trap, and written
   as an instruction: many zeros, placeholder values ('', '-', 'None'), numbers stored as text,
   numbers mixed with words, codes with leading zeros, comma-separated lists, and the full list of
   values for columns with few distinct values.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import sqlite3
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_ROWS = 20_000
MAX_LISTED_VALUES = 12
PLACEHOLDERS = ("", "-", "--", "none", "null", "n/a", "na", "unknown", "?")
_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_DESC_MAX = 160
_VALUE_DESC_MAX = 260


def _read_csv(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _clean(text: str, limit: int) -> str:
    text = re.sub(r"(?i)commonsense evidence:\s*", "", text or "")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def read_descriptions(desc_dir: Path) -> dict[str, dict[str, str]]:
    """Read BIRD column descriptions.

    Args:
        desc_dir: A ``database_description`` folder (one CSV per table).

    Returns:
        ``{table_lower: {column_lower: text}}``; empty text is left out.
    """
    out: dict[str, dict[str, str]] = {}
    for p in sorted(Path(desc_dir).glob("*.csv")):
        cols: dict[str, str] = {}
        for row in csv.DictReader(io.StringIO(_read_csv(p))):
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            name = row.get("original_column_name", "")
            desc = _clean(row.get("column_description", ""), _DESC_MAX)
            value = _clean(row.get("value_description", ""), _VALUE_DESC_MAX)
            text = desc + (f" Values: {value}" if value else "")
            if name and text.strip():
                cols[name.lower()] = text.strip()
        out[p.stem.lower()] = cols
    return out


def profile_column(values: list[object]) -> list[str]:
    """Describe data traps in a sample of column values, as short instructions.

    Args:
        values: Sampled values of one column (may include None).

    Returns:
        Zero or more short notes.
    """
    n = len(values)
    if n == 0:
        return []
    notes: list[str] = []
    non_null = [v for v in values if v is not None]
    if not non_null:
        return ["all sampled values are NULL"]

    numbers = [v for v in non_null if isinstance(v, (int, float))]
    texts = [v for v in non_null if isinstance(v, str)]

    zeros = sum(1 for v in numbers if v == 0)
    if numbers and zeros / n >= 0.05:
        notes.append(f"{zeros / n:.0%} of values are 0 (0 may mean missing; see description)")

    if texts:
        stripped = [t.strip() for t in texts]
        ph = Counter(s for s in stripped if s.lower() in PLACEHOLDERS)
        for val, c in ph.most_common(3):
            share = f"{c / n:.0%}" if c / n >= 0.01 else "<1%"
            notes.append(f"placeholder {val!r} in {share} of rows: treat as missing")
        real = [s for s in stripped if s.lower() not in PLACEHOLDERS]
        numeric = [s for s in real if _NUMBER.match(s)]
        if real and len(numeric) / len(real) >= 0.9:
            if any(len(s) > 1 and s.startswith("0") and s.isdigit() for s in numeric):
                notes.append("digit codes with leading zeros, stored as text: compare as text")
            else:
                notes.append("numbers stored as text: CAST(... AS REAL) before MIN/MAX/<,>")
        elif real and 0.2 <= len(numeric) / len(real) < 0.9:
            words = Counter(s for s in real if not _NUMBER.match(s)).most_common(3)
            shown = ", ".join(repr(w) for w, _ in words)
            notes.append(f"mixes numbers and words (e.g. {shown}): CAST only the numeric ones")
        listy = sum(1 for s in real if "," in s and len(s) <= 120)
        if real and listy / len(real) >= 0.05:
            notes.append("some values are comma-separated lists: match items with LIKE, not =")

    distinct = Counter(non_null)
    if 1 < len(distinct) <= MAX_LISTED_VALUES and len(non_null) >= 20:
        shown = ", ".join(repr(v) for v, _ in sorted(distinct.items(), key=lambda kv: -kv[1]))
        notes.append(f"all values: {shown}")
    return notes


def build_notes(db_path: Path, desc_dir: Path | None) -> dict[str, dict[str, str]]:
    """Build a note per column: BIRD description + profile instructions.

    Args:
        db_path: Path to the ``.sqlite`` file.
        desc_dir: BIRD ``database_description`` folder, or None.

    Returns:
        ``{table: {column: note}}`` for columns that have something to say.
    """
    descs = read_descriptions(desc_dir) if desc_dir and Path(desc_dir).is_dir() else {}
    con = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    out: dict[str, dict[str, str]] = {}
    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for t in tables:
            qt = '"' + t.replace('"', '""') + '"'
            cur = con.execute(f"SELECT * FROM {qt} LIMIT ?", (SAMPLE_ROWS,))
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            tdesc = descs.get(t.lower(), {})
            notes_t: dict[str, str] = {}
            for i, c in enumerate(cols):
                parts = []
                if tdesc.get(c.lower()):
                    parts.append(tdesc[c.lower()])
                prof = profile_column([r[i] for r in rows])
                if prof:
                    parts.append("[data] " + "; ".join(prof))
                if parts:
                    notes_t[c] = " ".join(parts)
            if notes_t:
                out[t] = notes_t
    finally:
        con.close()
    return out


def format_notes(table: str, notes: dict[str, str]) -> str:
    """Render one table's notes as a SQL comment block for the prompt."""
    if not notes:
        return ""
    lines = [f"- {c}: {n}" for c, n in notes.items()]
    return f"/* Column notes for {table}:\n" + "\n".join(lines) + "\n*/"


def save_notes(notes: dict[str, dict[str, str]], path: Path) -> None:
    """Write notes to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(notes, indent=1, ensure_ascii=False), encoding="utf-8")


def load_notes(path: Path) -> dict[str, dict[str, str]]:
    """Read notes written by save_notes."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
