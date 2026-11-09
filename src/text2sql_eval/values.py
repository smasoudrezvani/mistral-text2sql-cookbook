"""Value retrieval: find stored values that match words in the question, before writing SQL.

The model sees column names but has to guess the values it filters on ("Alameda County" vs the
stored 'Alameda'). We index every short text value in the database by its words, and for each
question look up the values that contain the question's word groups (1-3 words). The best
matches go into the prompt with their exact spelling.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_VALUE_CHARS = 60
MAX_DISTINCT_PER_COLUMN = 20_000
MAX_CANDIDATES_PER_NGRAM = 5_000  # skip word groups that match too many values (e.g. "school")

STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or not is are was were be been this that these "
    "those it its as which who whom whose what when where how many much each all any some per "
    "than then there their they them do does did has have had i me my we our you your please "
    "list give show find number count total average name names value values".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    """Lowercase word tokens (letters and digits)."""
    return _WORD.findall(text.lower())


@dataclass(frozen=True)
class ValueHit:
    """One stored value that matches part of the question.

    Attributes:
        table: Table name.
        column: Column name.
        value: The stored value, exact spelling.
        score: Share of the value's words covered by the question words (0-1].
    """

    table: str
    column: str
    value: str
    score: float


class ValueIndex:
    """Word index over the short text values of one database."""

    def __init__(self, entries: list[tuple[str, str, str]]) -> None:
        """Build the index.

        Args:
            entries: ``(table, column, value)`` triples.
        """
        self.entries = entries
        self._words: list[frozenset[str]] = [frozenset(words(v)) for _, _, v in entries]
        self._inverted: dict[str, set[int]] = defaultdict(set)
        for i, ws in enumerate(self._words):
            for w in ws:
                self._inverted[w].add(i)

    @classmethod
    def from_db(
        cls,
        db_path: Path,
        max_value_chars: int = MAX_VALUE_CHARS,
        max_distinct: int = MAX_DISTINCT_PER_COLUMN,
    ) -> ValueIndex:
        """Collect distinct short text values from every column.

        Args:
            db_path: Path to the ``.sqlite`` file.
            max_value_chars: Ignore longer values (free text).
            max_distinct: Skip columns with more distinct short values than this (IDs, text).

        Returns:
            The index.
        """
        con = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
        entries: list[tuple[str, str, str]] = []
        try:
            tables = [
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            ]
            for t in tables:
                qt = '"' + t.replace('"', '""') + '"'
                for col in [r[1] for r in con.execute(f"PRAGMA table_info({qt})")]:
                    qc = '"' + col.replace('"', '""') + '"'
                    vals = con.execute(
                        f"SELECT DISTINCT {qc} FROM {qt} WHERE typeof({qc}) = 'text' "
                        f"AND length({qc}) BETWEEN 1 AND ? LIMIT ?",
                        (max_value_chars, max_distinct + 1),
                    ).fetchall()
                    if len(vals) > max_distinct:
                        logger.debug("skip %s.%s (> %d distinct values)", t, col, max_distinct)
                        continue
                    entries += [(t, col, v[0]) for v in vals if words(v[0])]
        finally:
            con.close()
        return cls(entries)

    def save(self, path: Path) -> None:
        """Write the entries to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.entries), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ValueIndex:
        """Read an index written by save."""
        return cls([tuple(e) for e in json.loads(Path(path).read_text(encoding="utf-8"))])

    def search(
        self,
        text: str,
        max_hits: int = 15,
        per_column: int = 3,
        min_score: float = 0.5,
        max_n: int = 3,
    ) -> list[ValueHit]:
        """Find stored values that contain a word group (1-3 words) from the text.

        A value scores higher when the word group covers more of it, so 'Alameda' found by
        "alameda" (score 1.0) beats 'Alameda Unified School District' (score 0.25, dropped).

        Args:
            text: Question plus hint.
            max_hits: Most values returned overall.
            per_column: Most values returned per column.
            min_score: Minimum share of the value's words covered by the word group.
            max_n: Longest word group tried.

        Returns:
            Best matches, highest score first.
        """
        toks = words(text)
        best: dict[int, float] = {}
        for n in range(1, max_n + 1):
            for i in range(len(toks) - n + 1):
                gram = toks[i : i + n]
                if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                    continue
                if n == 1 and len(gram[0]) < 3:
                    continue
                sets = [self._inverted.get(w) for w in set(gram)]
                if not all(sets):
                    continue
                cand = set.intersection(*sorted(sets, key=len))
                if len(cand) > MAX_CANDIDATES_PER_NGRAM:
                    continue
                gset = set(gram)
                for idx in cand:
                    score = len(gset) / len(self._words[idx])
                    if score >= min_score and score > best.get(idx, 0.0):
                        best[idx] = score
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], self.entries[kv[0]]))
        hits: list[ValueHit] = []
        per_col: dict[tuple[str, str], int] = defaultdict(int)
        for idx, score in ranked:
            t, c, v = self.entries[idx]
            if per_col[(t, c)] >= per_column:
                continue
            per_col[(t, c)] += 1
            hits.append(ValueHit(t, c, v, round(min(score, 1.0), 3)))
            if len(hits) >= max_hits:
                break
        return hits


def format_hits(hits: list[ValueHit]) -> str:
    """Render value hits as a prompt section ('' if there are none).

    Args:
        hits: Output of ValueIndex.search.

    Returns:
        Prompt text listing ``table.column = 'value'`` lines.
    """
    if not hits:
        return ""
    lines = [f"- {h.table}.`{h.column}` = '{h.value.replace(chr(39), chr(39) * 2)}'" for h in hits]
    return (
        "Values in the database that match words in the question "
        "(use this exact spelling if you filter on them):\n" + "\n".join(lines)
    )
