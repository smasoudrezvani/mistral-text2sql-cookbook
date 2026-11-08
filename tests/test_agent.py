import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from text2sql_eval.agent import feedback_for, solve_with_retry
from text2sql_eval.scorer import ExecResult


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "toy.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE users (name TEXT, city TEXT)")
    con.executemany("INSERT INTO users VALUES (?, ?)", [("Ana", "Alameda"), ("Bo", "Oakland")])
    con.commit()
    con.close()
    return path


class ScriptedClient:
    """Returns the given replies in order and records the messages it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.chat = SimpleNamespace(complete=self._complete)

    def _complete(self, **kwargs):
        self.sent.append([dict(m) for m in kwargs["messages"]])
        text = self.replies.pop(0)
        msg = SimpleNamespace(content=text)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


START = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]


def test_feedback_for():
    assert feedback_for(ExecResult(rows=[(1,)])) is None
    assert "0 rows" in feedback_for(ExecResult(rows=[]))
    assert "no such column: x" in feedback_for(ExecResult(rows=None, error="no such column: x"))
    assert "time limit" in feedback_for(ExecResult(rows=None, error="timeout", timed_out=True))


def test_first_try_ok_makes_one_call(db):
    client = ScriptedClient(["```sql\nSELECT name FROM users\n```"])
    r = solve_with_retry(client, "m", START, db, timeout_s=5)
    assert r.attempts == 1
    assert r.result.rows == [("Ana",), ("Bo",)]
    assert r.feedback == []


def test_error_then_fixed(db):
    client = ScriptedClient(
        ["```sql\nSELECT nme FROM users\n```", "```sql\nSELECT name FROM users\n```"]
    )
    r = solve_with_retry(client, "m", START, db, timeout_s=5)
    assert r.attempts == 2
    assert r.first_result.error.startswith("no such column")
    assert r.result.ok
    second_call = client.sent[1]
    assert second_call[-2] == {"role": "assistant", "content": "```sql\nSELECT nme FROM users\n```"}
    assert "no such column" in second_call[-1]["content"]
    assert (r.prompt_tokens, r.completion_tokens) == (200, 20)


def test_empty_result_triggers_retry(db):
    client = ScriptedClient(
        [
            "```sql\nSELECT name FROM users WHERE city = 'Alameda County'\n```",
            "```sql\nSELECT name FROM users WHERE city = 'Alameda'\n```",
        ]
    )
    r = solve_with_retry(client, "m", START, db, timeout_s=5)
    assert r.attempts == 2
    assert r.first_result.rows == []
    assert r.result.rows == [("Ana",)]


def test_gives_up_after_max_attempts(db):
    client = ScriptedClient(["```sql\nSELEC 1\n```"] * 3)
    r = solve_with_retry(client, "m", START, db, timeout_s=5, max_attempts=3)
    assert r.attempts == 3
    assert not r.result.ok
    assert len(r.feedback) == 2


def test_reply_without_sql(db):
    client = ScriptedClient(["I cannot answer that.", "```sql\nSELECT 1\n```"])
    r = solve_with_retry(client, "m", START, db, timeout_s=5)
    assert r.first_result.error is not None
    assert r.result.rows == [(1,)]
    assert "syntax error" in r.feedback[0]  # plain text is tried as SQL


def test_empty_reply_never_sent_back_empty(db):
    client = ScriptedClient(["", "```sql\nSELECT 1\n```"])
    r = solve_with_retry(client, "m", START, db, timeout_s=5)
    assert r.result.rows == [(1,)]
    assert "did not contain a SQL query" in r.feedback[0]
    assistant_turn = client.sent[1][-2]
    assert assistant_turn == {"role": "assistant", "content": "(no answer)"}
