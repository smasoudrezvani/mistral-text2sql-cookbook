"""Majority voting: sample several SQL queries, run them all, keep the most common result.

We vote on the *result* (the rows), not the SQL text: two different queries that return the
same rows count as the same answer. Queries that fail or time out get no vote.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from text2sql_eval.scorer import ExecResult


@dataclass(frozen=True)
class Vote:
    """Outcome of a vote.

    Attributes:
        winner: Index of the chosen sample (the first sample in the biggest group), or 0 if
            no sample ran successfully.
        votes: Size of the winning group (0 if nothing ran).
        n_ok: Samples that ran successfully.
        n_groups: Distinct results among the successful samples.
    """

    winner: int
    votes: int
    n_ok: int
    n_groups: int


def result_key(rows: list[tuple[object, ...]]) -> frozenset[tuple[object, ...]]:
    """Group key for a result: its set of rows (order and duplicates ignored, like BIRD)."""
    return frozenset(rows)


def majority_vote(results: list[ExecResult]) -> Vote:
    """Pick the sample whose result is shared by the most samples.

    Ties go to the group that appeared first (so with all-different results we keep sample 0).

    Args:
        results: Execution result of each sample, in sample order.

    Returns:
        The winning sample and vote counts.
    """
    groups: dict[frozenset[tuple[object, ...]], list[int]] = defaultdict(list)
    for i, r in enumerate(results):
        if r.ok and r.rows is not None:
            groups[result_key(r.rows)].append(i)
    if not groups:
        return Vote(winner=0, votes=0, n_ok=0, n_groups=0)
    best = max(groups.values(), key=lambda idxs: (len(idxs), -idxs[0]))
    return Vote(
        winner=best[0],
        votes=len(best),
        n_ok=sum(len(v) for v in groups.values()),
        n_groups=len(groups),
    )
