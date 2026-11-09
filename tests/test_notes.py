import sqlite3
from pathlib import Path

from text2sql_eval.notes import build_notes, format_notes, profile_column, read_descriptions


def test_profile_zeros_and_small_value_set():
    notes = profile_column([0] * 30 + [70, 80, 90] * 10 + [None] * 5)
    assert any("are 0" in n for n in notes)
    assert any(n.startswith("all values:") for n in notes)


def test_profile_placeholders_and_text_numbers():
    vals = ["12", "7.5", "-", "-", "300"] * 10
    notes = profile_column(vals)
    assert any("placeholder '-'" in n for n in notes)
    assert any("numbers stored as text" in n for n in notes)


def test_profile_leading_zero_codes():
    notes = profile_column(["04615150435602", "01100170000000", "12345678901234"] * 10)
    assert any("leading zeros" in n for n in notes)
    assert not any("numbers stored as text" in n for n in notes)


def test_profile_mixed_numbers_and_words():
    notes = profile_column(["negative", "0", "4", "16", "negative", "1"] * 5)
    assert any("mixes numbers and words" in n and "'negative'" in n for n in notes)


def test_profile_comma_lists():
    notes = profile_column(["SLE", "SLE, RA", "RA", "PSS, SLE", "MCTD"] * 6)
    assert any("comma-separated" in n for n in notes)


def test_profile_clean_column_says_nothing():
    assert profile_column([f"name {i}" for i in range(100)]) == []
    assert profile_column(list(range(1, 200))) == []


def test_build_notes_combines_bird_description_and_profile(tmp_path: Path):
    db = tmp_path / "h.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE superhero (name TEXT, weight_kg INTEGER)")
    con.executemany(
        "INSERT INTO superhero VALUES (?, ?)",
        [(f"h{i}", 0 if i % 4 == 0 else 50 + i) for i in range(40)],
    )
    con.commit()
    con.close()
    desc = tmp_path / "database_description"
    desc.mkdir()
    (desc / "superhero.csv").write_text(
        "﻿original_column_name,column_name,column_description,data_format,value_description\n"
        'weight_kg,weight kg,the weight of the superhero,integer,"commonsense evidence:\n'
        'If the weight_kg is NULL or 0, it means the weight is missing."\n',
        encoding="utf-8",
    )
    assert read_descriptions(desc)["superhero"]["weight_kg"].startswith(
        "the weight of the superhero"
    )
    notes = build_notes(db, desc)
    w = notes["superhero"]["weight_kg"]
    assert "NULL or 0, it means the weight is missing" in w and "commonsense" not in w
    assert "[data] 25% of values are 0" in w
    assert "name" not in notes["superhero"]  # nothing to say about a clean column
    block = format_notes("superhero", notes["superhero"])
    assert block.startswith("/* Column notes for superhero:") and block.endswith("*/")
