import sqlite3
from pathlib import Path

import pytest

from text2sql_eval.values import ValueHit, ValueIndex, format_hits, words


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "schools.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE schools (County TEXT, District TEXT, Code INTEGER, Note TEXT);
        INSERT INTO schools VALUES
            ('Alameda', 'Alameda Unified School District', 1, NULL),
            ('Alameda', 'Oakland Unified', 2, 'x'),
            ('Contra Costa', 'Orinda Union', 3, '');
        CREATE TABLE "lang data" ("Language Name" TEXT);
        INSERT INTO "lang data" VALUES ('Chinese Simplified'), ('Chinese Traditional'), ('French');
        """
    )
    long_text = "word " * 30  # longer than MAX_VALUE_CHARS -> not indexed
    con.execute("INSERT INTO schools VALUES ('Long', ?, 4, NULL)", (long_text,))
    con.commit()
    con.close()
    return path


def test_words():
    assert words("Alameda County's 2012 K-12!") == ["alameda", "county", "s", "2012", "k", "12"]


def test_index_collects_short_distinct_text_values(db):
    idx = ValueIndex.from_db(db)
    vals = {(t, c, v) for t, c, v in idx.entries}
    assert ("schools", "County", "Alameda") in vals
    assert ("lang data", "Language Name", "Chinese Simplified") in vals
    assert sum(v == "Alameda" for _, _, v in idx.entries) == 1  # distinct
    assert not any(c == "Code" for _, c, _ in idx.entries)  # integers not indexed
    assert not any(v.startswith("word word") for _, _, v in idx.entries)  # long text skipped


def test_search_prefers_exact_short_value(db):
    hits = ValueIndex.from_db(db).search("How many schools are in Alameda County?")
    assert hits[0] == ValueHit("schools", "County", "Alameda", 1.0)
    # 'Alameda Unified School District' covers only 1/4 of its words -> dropped
    assert all(h.value != "Alameda Unified School District" for h in hits)


def test_search_multiword_and_per_column_cap(db):
    hits = ValueIndex.from_db(db).search("cards with a Chinese Simplified translation")
    assert hits[0].value == "Chinese Simplified" and hits[0].score == 1.0
    one = ValueIndex.from_db(db).search("Chinese Simplified", per_column=1)
    assert len([h for h in one if h.column == "Language Name"]) == 1


def test_search_ignores_stopwords_and_short_tokens(db):
    assert ValueIndex.from_db(db).search("what is the number of it in a") == []


def test_save_load_roundtrip(db, tmp_path):
    idx = ValueIndex.from_db(db)
    idx.save(tmp_path / "v" / "idx.json")
    again = ValueIndex.load(tmp_path / "v" / "idx.json")
    assert again.entries == idx.entries
    assert again.search("Alameda")[0].value == "Alameda"


def test_skip_columns_with_too_many_values(db):
    idx = ValueIndex.from_db(db, max_distinct=2)
    assert not any(c == "District" for _, c, _ in idx.entries)  # 4 distinct > 2
    assert not any(c == "County" for _, c, _ in idx.entries)  # 3 distinct > 2 as well


def test_format_hits_escapes_quotes():
    text = format_hits([ValueHit("t", "c", "O'Brien", 1.0)])
    assert "- t.`c` = 'O''Brien'" in text
    assert format_hits([]) == ""
