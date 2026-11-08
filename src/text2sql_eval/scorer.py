"""Execution-accuracy scorer for text-to-SQL on SQLite.

Runs a predicted query and compares its result with the gold query's result.
Two rules are reported side by side:

* ``bird``: BIRD's official rule, ``set(pred) == set(gold)`` (ignores order and duplicates).
* ``strict``: duplicates count, and row order counts when the outer gold query has ORDER BY.

Differences from BIRD's ``evaluation_ex.py``: the database is opened read-only, a timeout really
stops the query (SQLite progress handler), and the error message is kept.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Row = tuple[object, ...]
Mode = Literal["bird", "strict"]

DEFAULT_TIMEOUT_S = 30.0  # same as BIRD
DEFAULT_MAX_ROWS = 200_000  # guard against huge results on a small-RAM machine
_PROGRESS_EVERY_N_STEPS = 10_000


@dataclass(frozen=True)
class ExecResult:
    """Outcome of running one SQL query.

    Attributes:
        rows: Result rows, or None if the query failed or timed out.
        error: SQLite error message, or None.
        timed_out: True if the query was stopped by the timeout.
        seconds: Wall-clock run time.
    """

    rows: list[Row] | None
    error: str | None = None
    timed_out: bool = False
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """True if the query ran to completion and returned rows (possibly zero)."""
        return self.rows is not None and self.error is None and not self.timed_out


@dataclass(frozen=True)
class Score:
    """Correctness of one prediction under both rules.

    Attributes:
        bird: Correct under BIRD's set rule.
        strict: Correct under the strict rule.
    """

    bird: bool
    strict: bool


def execute_sql(
    db_path: Path,
    sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int | None = DEFAULT_MAX_ROWS,
    attach: list[Path] | None = None,
) -> ExecResult:
    """Run one SQL query on a SQLite database, read-only, with a timeout.

    Args:
        db_path: Path to the ``.sqlite`` file.
        sql: A single SQL statement.
        timeout_s: Stop the query after this many seconds.
        max_rows: Fail if the result has more rows than this (None = no limit).
        attach: Extra databases to ATTACH read-only (e.g. distractor tables). Unqualified
            table names resolve in the main database first.

    Returns:
        An ExecResult with rows, or with the error / timeout recorded.
    """
    start = time.monotonic()
    deadline = start + timeout_s
    stopped = False

    def _check_deadline() -> int:
        nonlocal stopped
        if time.monotonic() > deadline:
            stopped = True
            return 1  # non-zero tells SQLite to abort the query
        return 0

    try:
        con = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as e:
        return ExecResult(rows=None, error=str(e), seconds=time.monotonic() - start)

    try:
        for i, extra in enumerate(attach or []):
            con.execute(
                f"ATTACH DATABASE ? AS extra{i}", (f"{Path(extra).resolve().as_uri()}?mode=ro",)
            )
        con.set_progress_handler(_check_deadline, _PROGRESS_EVERY_N_STEPS)
        cur = con.execute(sql)
        rows: list[Row] = []
        while batch := cur.fetchmany(1_000):
            rows.extend(batch)
            if max_rows is not None and len(rows) > max_rows:
                return ExecResult(
                    rows=None,
                    error=f"too many rows (> {max_rows})",
                    seconds=time.monotonic() - start,
                )
        return ExecResult(rows=rows, seconds=time.monotonic() - start)
    except (sqlite3.Error, sqlite3.Warning) as e:
        elapsed = time.monotonic() - start
        if stopped:
            logger.debug("Query timed out after %.1f s", elapsed)
            return ExecResult(rows=None, error="timeout", timed_out=True, seconds=elapsed)
        return ExecResult(rows=None, error=str(e), seconds=elapsed)
    finally:
        con.close()


def results_match(
    pred_rows: list[Row], gold_rows: list[Row], mode: Mode = "bird", ordered: bool = False
) -> bool:
    """Compare two result sets.

    Args:
        pred_rows: Rows returned by the predicted query.
        gold_rows: Rows returned by the gold query.
        mode: ``"bird"`` (set equality) or ``"strict"`` (multiset, or list if ordered).
        ordered: Whether row order matters (only used in strict mode).

    Returns:
        True if the results count as equal under the chosen rule.

    Raises:
        ValueError: If mode is unknown.
    """
    if mode == "bird":
        return set(pred_rows) == set(gold_rows)
    if mode == "strict":
        if ordered:
            return list(pred_rows) == list(gold_rows)
        return Counter(pred_rows) == Counter(gold_rows)
    raise ValueError(f"unknown mode: {mode!r}")


_ORDER_BY = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def has_outer_order_by(sql: str) -> bool:
    """Check whether the outermost query has ORDER BY.

    ORDER BY inside parentheses (subqueries, window functions), string literals, quoted
    identifiers and comments is ignored.

    Args:
        sql: A SQL statement.

    Returns:
        True if ORDER BY appears at parenthesis depth 0.
    """
    out: list[str] = []
    depth = 0
    i, n = 0, len(sql)
    closers = {"'": "'", '"': '"', "`": "`", "[": "]"}
    while i < n:
        ch = sql[i]
        if ch in closers:  # skip a string literal or quoted identifier
            end = sql.find(closers[ch], i + 1)
            i = n if end == -1 else end + 1
            out.append(" ")
            continue
        if sql.startswith("--", i):  # line comment
            end = sql.find("\n", i)
            i = n if end == -1 else end + 1
            out.append(" ")
            continue
        if sql.startswith("/*", i):  # block comment
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            out.append(" ")
            continue
        if ch == "(":
            depth += 1
            out.append(" ")
        elif ch == ")":
            depth = max(0, depth - 1)
            out.append(" ")
        else:
            out.append(ch if depth == 0 else " ")
        i += 1
    return bool(_ORDER_BY.search("".join(out)))


def compare(pred: ExecResult, gold: ExecResult, gold_sql: str) -> Score:
    """Score one prediction against an already-executed gold query.

    Args:
        pred: Result of running the predicted SQL.
        gold: Result of running the gold SQL (must have succeeded).
        gold_sql: The gold SQL text (used to decide whether order matters).

    Returns:
        Score under both the BIRD and the strict rule. A failed prediction is wrong.

    Raises:
        ValueError: If the gold query did not run successfully.
    """
    if not gold.ok:
        raise ValueError(f"gold query failed: {gold.error}")
    if not pred.ok:
        return Score(bird=False, strict=False)
    assert pred.rows is not None and gold.rows is not None
    ordered = has_outer_order_by(gold_sql)
    return Score(
        bird=results_match(pred.rows, gold.rows, mode="bird"),
        strict=results_match(pred.rows, gold.rows, mode="strict", ordered=ordered),
    )
