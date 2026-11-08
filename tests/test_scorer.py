import sqlite3
import time
from pathlib import Path

import pytest

from text2sql_eval.scorer import (
    ExecResult,
    compare,
    execute_sql,
    has_outer_order_by,
    results_match,
)

INFINITE = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """Tiny database: users(name, age) with Ana 35, Bo 41, Cas 22."""
    path = tmp_path / "toy.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE users (name TEXT, age INTEGER)")
    con.executemany("INSERT INTO users VALUES (?, ?)", [("Ana", 35), ("Bo", 41), ("Cas", 22)])
    con.commit()
    con.close()
    return path


# ---------- execute_sql ----------


def test_execute_returns_rows(db):
    res = execute_sql(db, "SELECT name FROM users WHERE age > 30 ORDER BY name")
    assert res.ok
    assert res.rows == [("Ana",), ("Bo",)]


def test_syntax_error_is_kept(db):
    res = execute_sql(db, "SELEC name FROM users")
    assert not res.ok
    assert res.rows is None
    assert "syntax error" in res.error


def test_unknown_column_error_message(db):
    res = execute_sql(db, "SELECT salary FROM users")
    assert "no such column" in res.error


def test_database_is_read_only(db):
    res = execute_sql(db, "DELETE FROM users")
    assert not res.ok
    assert "readonly" in res.error
    # rows are still there
    assert execute_sql(db, "SELECT count(*) FROM users").rows == [(3,)]


def test_timeout_stops_the_query(db):
    start = time.monotonic()
    res = execute_sql(db, INFINITE, timeout_s=0.3)
    assert res.timed_out
    assert not res.ok
    assert time.monotonic() - start < 3


def test_too_many_rows(db):
    sql = INFINITE.replace("SELECT count(*) FROM c", "SELECT x FROM c LIMIT 1000")
    res = execute_sql(db, sql, max_rows=10)
    assert not res.ok
    assert "too many rows" in res.error


def test_missing_database_file(tmp_path):
    res = execute_sql(tmp_path / "nope.sqlite", "SELECT 1")
    assert not res.ok
    assert "unable to open" in res.error


def test_two_statements_rejected(db):
    res = execute_sql(db, "SELECT 1; SELECT 2")
    assert not res.ok


# ---------- results_match ----------


def test_order_ignored_when_not_ordered():
    assert results_match([("Bo",), ("Ana",)], [("Ana",), ("Bo",)], mode="bird")
    assert results_match([("Bo",), ("Ana",)], [("Ana",), ("Bo",)], mode="strict")


def test_duplicates_bird_vs_strict():
    pred = [("Bo",), ("Ana",), ("Ana",)]
    gold = [("Ana",), ("Bo",)]
    assert results_match(pred, gold, mode="bird")
    assert not results_match(pred, gold, mode="strict")


def test_order_matters_in_strict_when_ordered():
    pred = [("Bo",), ("Ana",)]
    gold = [("Ana",), ("Bo",)]
    assert results_match(pred, gold, mode="bird", ordered=True)  # BIRD ignores order
    assert not results_match(pred, gold, mode="strict", ordered=True)


def test_int_and_float_equal():
    assert results_match([(42,)], [(42.0,)], mode="strict")


def test_column_order_matters():
    assert not results_match([(35, "Ana")], [("Ana", 35)], mode="bird")


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        results_match([], [], mode="loose")


# ---------- has_outer_order_by ----------


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT a FROM t ORDER BY a", True),
        ("select a from t order  by a desc limit 5", True),
        ("SELECT a FROM t", False),
        ("SELECT a FROM (SELECT a FROM t ORDER BY a)", False),
        ("SELECT a, RANK() OVER (ORDER BY b) FROM t", False),
        ("SELECT a FROM t WHERE c = 'order by'", False),
        ("SELECT a FROM t -- ORDER BY a", False),
        ("SELECT a FROM t /* ORDER BY a */", False),
        ('SELECT "order by" FROM t', False),
        ("SELECT a FROM t UNION SELECT a FROM u ORDER BY a", True),
        ("SELECT a FROM t WHERE b IN (SELECT b FROM u ORDER BY b) ORDER BY a", True),
    ],
)
def test_has_outer_order_by(sql, expected):
    assert has_outer_order_by(sql) is expected


# ---------- compare ----------


def test_compare_ordered_gold():
    gold_sql = "SELECT name FROM users ORDER BY age"
    gold = ExecResult(rows=[("Cas",), ("Ana",), ("Bo",)])
    pred = ExecResult(rows=[("Ana",), ("Bo",), ("Cas",)])
    s = compare(pred, gold, gold_sql)
    assert s.bird and not s.strict


def test_compare_failed_prediction_is_wrong():
    gold = ExecResult(rows=[(1,)])
    s = compare(ExecResult(rows=None, error="syntax error"), gold, "SELECT 1")
    assert not s.bird and not s.strict
    s = compare(ExecResult(rows=None, timed_out=True), gold, "SELECT 1")
    assert not s.bird and not s.strict


def test_compare_rejects_broken_gold():
    with pytest.raises(ValueError):
        compare(ExecResult(rows=[(1,)]), ExecResult(rows=None, error="boom"), "SELECT 1")
