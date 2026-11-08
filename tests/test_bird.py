import json
import sqlite3
from pathlib import Path

import pytest

from text2sql_eval.bird import (
    Gold,
    build_gold_cache,
    load_gold_cache,
    load_questions,
    sample_ids,
    save_gold_cache,
)
from text2sql_eval.scorer import compare


@pytest.fixture
def bird_dir(tmp_path: Path) -> tuple[Path, Path]:
    """A fake mini-dev: one database 'toy' and a JSONL file with 3 questions."""
    db_root = tmp_path / "dev_databases"
    (db_root / "toy").mkdir(parents=True)
    con = sqlite3.connect(db_root / "toy" / "toy.sqlite")
    con.execute("CREATE TABLE users (name TEXT, age INTEGER, score REAL)")
    con.executemany(
        "INSERT INTO users VALUES (?, ?, ?)",
        [("Ana", 35, 0.1), ("Bo", 41, None), ("Cas", 22, 2.5)],
    )
    con.commit()
    con.close()
    rows = [
        {
            "question_id": 0,
            "db_id": "toy",
            "question": "Who is over 30?",
            "evidence": "",
            "SQL": "SELECT name FROM users WHERE age > 30",
            "difficulty": "simple",
        },
        {
            "question_id": 1,
            "db_id": "toy",
            "question": "Scores by age?",
            "evidence": "x",
            "SQL": "SELECT name, score FROM users ORDER BY age",
            "difficulty": "moderate",
        },
        {
            "question_id": 2,
            "db_id": "toy",
            "question": "How many?",
            "evidence": None,
            "SQL": "SELECT count(*) FROM users",
            "difficulty": "challenging",
        },
    ]
    jsonl = tmp_path / "mini.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return jsonl, db_root


def test_load_questions(bird_dir):
    jsonl, _ = bird_dir
    qs = load_questions(jsonl)
    assert [q.question_id for q in qs] == [0, 1, 2]
    assert qs[1].sql.startswith("SELECT name, score")
    assert qs[2].evidence == ""  # null evidence becomes empty string


def test_sample_ids_is_fixed():
    class Q:
        def __init__(self, i):
            self.question_id = i

    qs = [Q(i) for i in range(500)]
    a = sample_ids(qs, 50, seed=0)
    assert a == sample_ids(qs, 50, seed=0)
    assert len(set(a)) == 50
    assert a != sample_ids(qs, 50, seed=1)


def test_gold_cache_roundtrip(bird_dir, tmp_path):
    jsonl, db_root = bird_dir
    qs = load_questions(jsonl)
    cache = build_gold_cache(qs, db_root)
    path = tmp_path / "cache" / "gold.json"
    save_gold_cache(cache, path)
    loaded = load_gold_cache(path)
    assert loaded.keys() == cache.keys()
    for qid in cache:
        assert loaded[qid].rows == cache[qid].rows  # tuples, floats and NULLs survive
    # the stored gold still scores itself as correct, including the ordered one
    s = compare(loaded[1].as_exec_result(), loaded[1].as_exec_result(), qs[1].sql)
    assert s.bird and s.strict


def test_broken_gold_raises(bird_dir):
    jsonl, db_root = bird_dir
    qs = load_questions(jsonl)
    broken = [qs[0].__class__(**{**qs[0].__dict__, "sql": "SELECT nope FROM users"})]
    with pytest.raises(RuntimeError, match="no such column"):
        build_gold_cache(broken, db_root)


def test_pred_timeout_rule():
    assert Gold(rows=[], seconds=0.2).pred_timeout_s == 30.0
    assert Gold(rows=[], seconds=28.0).pred_timeout_s == pytest.approx(84.0)
