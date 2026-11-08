"""Load BIRD mini-dev questions and cache the gold query results.

Gold queries are run once (with a generous timeout) and their rows are saved to JSON, so every
model and approach is scored against the same stored answers.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from text2sql_eval.scorer import DEFAULT_TIMEOUT_S, ExecResult, execute_sql

logger = logging.getLogger(__name__)

GOLD_TIMEOUT_S = 300.0
PRED_TIMEOUT_FACTOR = 3.0


@dataclass(frozen=True)
class Question:
    """One BIRD mini-dev record.

    Attributes:
        question_id: Stable id (0-499).
        db_id: Database name, e.g. ``california_schools``.
        question: Natural-language question.
        evidence: Expert hint (may be empty).
        sql: Gold SQL.
        difficulty: ``simple``, ``moderate`` or ``challenging``.
    """

    question_id: int
    db_id: str
    question: str
    evidence: str
    sql: str
    difficulty: str


def load_questions(jsonl_path: Path) -> list[Question]:
    """Read mini-dev questions from the Hugging Face JSONL file.

    Args:
        jsonl_path: Path to ``mini_dev_sqlite.jsonl``.

    Returns:
        Questions in file order.
    """
    questions = []
    with Path(jsonl_path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            questions.append(
                Question(
                    question_id=int(r["question_id"]),
                    db_id=r["db_id"],
                    question=r["question"],
                    evidence=r.get("evidence") or "",
                    sql=r["SQL"],
                    difficulty=r["difficulty"],
                )
            )
    return questions


def db_path(db_root: Path, db_id: str) -> Path:
    """Return the SQLite file for a database id.

    Args:
        db_root: The ``dev_databases`` folder.
        db_id: Database name.

    Returns:
        ``db_root/db_id/db_id.sqlite``.
    """
    return Path(db_root) / db_id / f"{db_id}.sqlite"


def sample_ids(questions: list[Question], n: int, seed: int = 0) -> list[int]:
    """Pick a fixed random subset of question ids.

    Args:
        questions: All questions.
        n: How many to pick.
        seed: Random seed (same seed, same subset).

    Returns:
        Sorted question ids.
    """
    ids = [q.question_id for q in questions]
    return sorted(random.Random(seed).sample(ids, n))


@dataclass(frozen=True)
class Gold:
    """Stored result of one gold query.

    Attributes:
        rows: Result rows.
        seconds: Time the gold query took.
    """

    rows: list[tuple[object, ...]]
    seconds: float

    @property
    def pred_timeout_s(self) -> float:
        """Timeout for predictions on this question: max(30 s, 3 x gold time)."""
        return max(DEFAULT_TIMEOUT_S, PRED_TIMEOUT_FACTOR * self.seconds)

    def as_exec_result(self) -> ExecResult:
        """Return the gold result in the scorer's format."""
        return ExecResult(rows=self.rows, seconds=self.seconds)


def build_gold_cache(
    questions: list[Question], db_root: Path, timeout_s: float = GOLD_TIMEOUT_S
) -> dict[int, Gold]:
    """Run every gold query once.

    Args:
        questions: Questions to run.
        db_root: The ``dev_databases`` folder.
        timeout_s: Timeout per gold query.

    Returns:
        Mapping from question id to its gold result.

    Raises:
        RuntimeError: If any gold query fails (the benchmark itself would be broken).
    """
    cache: dict[int, Gold] = {}
    failed: list[tuple[int, str | None]] = []
    for i, q in enumerate(questions, start=1):
        res = execute_sql(db_path(db_root, q.db_id), q.sql, timeout_s=timeout_s, max_rows=None)
        if not res.ok:
            failed.append((q.question_id, res.error))
            continue
        assert res.rows is not None
        cache[q.question_id] = Gold(rows=res.rows, seconds=res.seconds)
        if i % 100 == 0:
            logger.info("gold queries run: %d / %d", i, len(questions))
    if failed:
        raise RuntimeError(f"gold queries failed: {failed}")
    return cache


def save_gold_cache(cache: dict[int, Gold], path: Path) -> None:
    """Write the gold cache to JSON.

    Args:
        cache: Output of build_gold_cache.
        path: Target JSON file.
    """
    payload = {
        str(qid): {"rows": [list(r) for r in g.rows], "seconds": g.seconds}
        for qid, g in sorted(cache.items())
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_gold_cache(path: Path) -> dict[int, Gold]:
    """Read a gold cache written by save_gold_cache.

    Args:
        path: JSON file.

    Returns:
        Mapping from question id to gold result, with rows as tuples again.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        int(qid): Gold(rows=[tuple(r) for r in v["rows"]], seconds=v["seconds"])
        for qid, v in payload.items()
    }
