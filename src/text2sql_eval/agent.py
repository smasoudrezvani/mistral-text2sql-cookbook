"""Approach (c): execute and retry.

The model writes SQL, we run it, and if it fails (error, timeout, or zero rows) we send that
feedback back and ask for a corrected query, up to ``max_attempts`` times in total.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from text2sql_eval.prompting import chat, extract_sql
from text2sql_eval.scorer import ExecResult, execute_sql

if TYPE_CHECKING:
    from mistralai.client import Mistral

NO_SQL = "no SQL found in the reply"
NO_SQL_FEEDBACK = (
    "Your reply did not contain a SQL query (it may have run out of length while reasoning). "
    "Reason briefly, then reply with one SQLite query in a ```sql code block."
)


def feedback_for(result: ExecResult) -> str | None:
    """Turn an execution result into feedback for the model, or None if it looks fine.

    Args:
        result: Result of running the model's SQL.

    Returns:
        A short message describing the problem, or None if the query returned rows.
    """
    if result.timed_out:
        return (
            "The query did not finish within the time limit. It is probably missing a join "
            "condition or scanning too much. Rewrite it to be correct and efficient."
        )
    if result.error is not None:
        return f"The query failed with this SQLite error: {result.error}\nFix the query."
    if result.rows is not None and len(result.rows) == 0:
        return (
            "The query ran but returned 0 rows. Check the filter values (exact spelling and "
            "format in the example rows), the join conditions, and the tables used. If you are "
            "sure 0 rows is correct, repeat the same query."
        )
    return None


@dataclass
class AgentResult:
    """Outcome of the execute-and-retry loop.

    Attributes:
        sql: Final SQL.
        result: Execution result of the final SQL.
        attempts: Number of model calls made.
        first_sql: SQL from the first attempt.
        first_result: Execution result of the first attempt.
        feedback: Feedback messages sent, in order.
        prompt_tokens: Total input tokens over all calls.
        completion_tokens: Total output tokens over all calls.
    """

    sql: str
    result: ExecResult
    attempts: int
    first_sql: str
    first_result: ExecResult
    feedback: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def solve_with_retry(
    client: Mistral,
    model: str,
    messages: list[dict[str, str]],
    db_path: Path,
    timeout_s: float,
    max_attempts: int = 3,
    attach: list[Path] | None = None,
) -> AgentResult:
    """Ask for SQL, run it, and retry with execution feedback until it looks fine.

    Args:
        client: Mistral client.
        model: Pinned model name.
        messages: Initial system + user messages (same as approach a).
        db_path: SQLite database to run against.
        timeout_s: Timeout per execution.
        max_attempts: Maximum model calls (1 = no retry).
        attach: Extra databases to attach when running SQL (e.g. distractor tables).

    Returns:
        The final SQL and result, plus the first attempt and token totals.
    """
    history = list(messages)
    feedback: list[str] = []
    tok_in = tok_out = 0
    first_sql = ""
    first_result: ExecResult | None = None
    sql = ""
    result = ExecResult(rows=None, error="no attempt")
    for attempt in range(1, max_attempts + 1):
        reply = chat(client, model, history)
        tok_in += reply.prompt_tokens
        tok_out += reply.completion_tokens
        sql = extract_sql(reply.text)
        result = (
            execute_sql(db_path, sql, timeout_s=timeout_s, attach=attach)
            if sql
            else ExecResult(rows=None, error=NO_SQL)
        )
        if first_result is None:
            first_sql, first_result = sql, result
        if not sql:
            note: str | None = NO_SQL_FEEDBACK
        else:
            note = feedback_for(result)
        if note is None or attempt == max_attempts:
            return AgentResult(
                sql=sql,
                result=result,
                attempts=attempt,
                first_sql=first_sql,
                first_result=first_result,
                feedback=feedback,
                prompt_tokens=tok_in,
                completion_tokens=tok_out,
            )
        feedback.append(note)
        history += [
            # the API rejects an empty assistant message (e.g. reply was all reasoning)
            {"role": "assistant", "content": reply.text or "(no answer)"},
            {"role": "user", "content": note},
        ]
    raise AssertionError("unreachable")  # pragma: no cover
