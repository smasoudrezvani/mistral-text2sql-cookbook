import sqlite3
from pathlib import Path

from text2sql_eval.prompting import build_messages, extract_sql
from text2sql_eval.schema import render_schema, table_names


def test_extract_sql_from_fence():
    assert extract_sql("Here:\n```sql\nSELECT 1;\n```") == "SELECT 1"


def test_extract_sql_last_block_wins():
    text = "```sql\nSELECT 1\n```\nfixed:\n```sql\nSELECT 2\n```"
    assert extract_sql(text) == "SELECT 2"


def test_extract_sql_plain_text_and_bare_fence():
    assert extract_sql("SELECT a FROM t;") == "SELECT a FROM t"
    assert extract_sql("```\nSELECT b FROM t\n```") == "SELECT b FROM t"
    assert extract_sql("") == ""


def test_build_messages_empty_hint():
    msgs = build_messages("CREATE TABLE t (a)", "How many?", "  ")
    assert msgs[0]["role"] == "system"
    assert "Hint: (none)" in msgs[1]["content"]
    assert "Question: How many?" in msgs[1]["content"]


def test_render_schema(tmp_path: Path):
    db = tmp_path / "s.sqlite"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE "my table" (id INTEGER, note TEXT)')
    con.executemany(
        'INSERT INTO "my table" VALUES (?, ?)',
        [(1, "x" * 100), (2, None), (3, "a\nb"), (4, "d")],
    )
    con.execute("CREATE TABLE other (k TEXT)")
    con.commit()
    con.close()

    assert table_names(db) == ["my table", "other"]
    text = render_schema(db, n_rows=3)
    assert 'CREATE TABLE "my table"' in text
    assert "3 example rows from my table" in text
    assert "NULL" in text and "a b" in text  # NULL shown, newline flattened
    assert "x" * 100 not in text  # long value cut
    assert "\t4\t" not in text and "\nd" not in text  # only 3 rows
    assert "0 example rows from other" in text
    assert "example rows" not in render_schema(db, n_rows=0)
    only = render_schema(db, tables=["other"])
    assert "my table" not in only
