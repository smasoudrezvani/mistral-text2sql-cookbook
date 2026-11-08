import sqlite3
from pathlib import Path

import pytest

from text2sql_eval.distractors import build_distractor_db, load_manifest
from text2sql_eval.schema import render_schema, table_names
from text2sql_eval.scorer import execute_sql


def _make_db(path: Path, script: str) -> Path:
    con = sqlite3.connect(path)
    con.executescript(script)
    con.close()
    return path


@pytest.fixture
def dbs(tmp_path: Path) -> dict[str, Path]:
    target = _make_db(
        tmp_path / "school.sqlite",
        "CREATE TABLE users (name TEXT, age INTEGER);"
        "INSERT INTO users VALUES ('Ana', 35), ('Bo', 41);",
    )
    other1 = _make_db(
        tmp_path / "f1.sqlite",
        "CREATE TABLE races (id INTEGER, track TEXT);"
        "INSERT INTO races VALUES (1, 'Monza'), (2, 'Spa'), (3, 'Imola');"
        "CREATE TABLE users (name TEXT);"  # same name as a real table: must be skipped
        "INSERT INTO users VALUES ('Zed');",
    )
    other2 = _make_db(
        tmp_path / "bank.sqlite",
        "CREATE TABLE loans (id INTEGER, amount REAL);"
        "INSERT INTO loans VALUES (1, 100.0), (2, 250.5);",
    )
    return {"target": target, "f1": other1, "bank": other2}


def test_build_skips_name_clashes_and_caps_rows(dbs, tmp_path):
    out = tmp_path / "distract" / "school.sqlite"
    manifest = build_distractor_db(
        dbs["target"], {"f1": dbs["f1"], "bank": dbs["bank"]}, out, n_tables=10, max_rows=2
    )
    assert manifest == {"races": "f1", "loans": "bank"}
    assert load_manifest(out) == manifest
    con = sqlite3.connect(out)
    assert con.execute("SELECT count(*) FROM races").fetchone() == (2,)  # capped at max_rows
    con.close()


def test_build_is_repeatable_and_respects_n(dbs, tmp_path):
    src = {"f1": dbs["f1"], "bank": dbs["bank"]}
    a = build_distractor_db(dbs["target"], src, tmp_path / "a.sqlite", n_tables=1, seed=3)
    b = build_distractor_db(dbs["target"], src, tmp_path / "b.sqlite", n_tables=1, seed=3)
    assert a == b and len(a) == 1


def test_attached_tables_query_and_real_tables_win(dbs, tmp_path):
    out = tmp_path / "d.sqlite"
    build_distractor_db(dbs["target"], {"f1": dbs["f1"], "bank": dbs["bank"]}, out)
    res = execute_sql(dbs["target"], "SELECT track FROM races ORDER BY id", attach=[out])
    assert res.rows == [("Monza",), ("Spa",), ("Imola",)]
    res = execute_sql(dbs["target"], "SELECT name FROM users ORDER BY name", attach=[out])
    assert res.rows == [("Ana",), ("Bo",)]  # the real table, not f1's 'users'
    assert not execute_sql(dbs["target"], "DELETE FROM races", attach=[out]).ok  # read-only


def test_render_schema_with_extras(dbs, tmp_path):
    out = tmp_path / "d.sqlite"
    build_distractor_db(dbs["target"], {"f1": dbs["f1"], "bank": dbs["bank"]}, out)
    names = table_names(dbs["target"], [out])
    assert names[0] == "users"  # real tables first, then distractors (in seeded order)
    assert sorted(names[1:]) == ["loans", "races"]
    text = render_schema(dbs["target"], extra_dbs=[out])
    for t in ("users", "races", "loans"):
        assert f"CREATE TABLE {t}" in text
    assert "Monza" in text  # example rows read from the distractor file
    shuffled = render_schema(dbs["target"], extra_dbs=[out], shuffle_seed=1)
    assert sorted(shuffled.split("\n\n")) == sorted(text.split("\n\n"))
    only = render_schema(dbs["target"], extra_dbs=[out], tables=["loans"])
    assert "users" not in only and "loans" in only
