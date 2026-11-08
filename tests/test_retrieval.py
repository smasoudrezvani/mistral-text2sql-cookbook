import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import text2sql_eval.retrieval as retrieval
from text2sql_eval.retrieval import (
    cosine,
    embed,
    foreign_key_neighbors,
    select_tables,
    table_docs,
    tables_in_sql,
)


@pytest.fixture
def shop(tmp_path: Path) -> tuple[Path, Path]:
    """customers <- orders -> products, plus an unrelated 'weather' table."""
    db = tmp_path / "shop.sqlite"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE products (id INTEGER PRIMARY KEY, title TEXT);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (product_id) REFERENCES Products(id)
        );
        CREATE TABLE weather (day TEXT, temp REAL);
        """
    )
    con.close()
    desc = tmp_path / "database_description"
    desc.mkdir()
    (desc / "customers.csv").write_text(
        "﻿original_column_name,column_name,column_description,data_format,value_description\n"
        "id,,customer id,integer,\n"
        "name,customer name,full name of the customer,text,\n",
        encoding="utf-8",
    )
    (desc / "weather.csv").write_bytes(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "temp,temperature,daily temperature in \xb0C,real,\n".encode("cp1252")
    )
    return db, desc


def test_table_docs(shop):
    db, desc = shop
    docs = table_docs(db, desc)
    assert set(docs) == {"customers", "products", "orders", "weather"}
    assert "name (customer name; full name of the customer)" in docs["customers"]
    assert "°C" in docs["weather"]  # cp1252 file read correctly
    assert docs["products"] == "Table products. Columns: id, title"  # no csv: plain columns
    assert table_docs(db, None)["customers"] == "Table customers. Columns: id, name"


def test_foreign_key_neighbors_both_directions(shop):
    db, _ = shop
    n = foreign_key_neighbors(db)
    assert n["orders"] == {"customers", "products"}  # 'Products' matched case-insensitively
    assert n["customers"] == {"orders"}
    assert n["weather"] == set()


def test_select_tables_adds_neighbours():
    vecs = {
        "customers": [1.0, 0.0],
        "orders": [0.0, 1.0],
        "products": [0.1, 1.0],
        "weather": [-1.0, 0.0],
    }
    nbrs = {
        "customers": {"orders"},
        "orders": {"customers", "products"},
        "products": {"orders"},
        "weather": set(),
    }
    picked = select_tables([1.0, 0.0], vecs, nbrs, k=1)
    assert picked == ["customers", "orders"]  # top-1, then its FK neighbour
    assert "weather" not in select_tables([1.0, 0.0], vecs, nbrs, k=2)


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 0]) == 0.0


def test_tables_in_sql():
    names = ["frpm", "schools", "satscores", "Player Attributes", "order", "status"]
    sql = (
        "SELECT s.School, s.status FROM schools AS s JOIN `Player Attributes` p ON 1 "
        "WHERE s.note = 'frpm' AND x IN (SELECT 1 FROM SATSCORES) ORDER BY s.School"
    )
    # 'order' (keyword), 'status' (column) and 'frpm' (string) are not tables read here
    assert tables_in_sql(sql, names) == {"schools", "Player Attributes", "satscores"}


def test_tables_in_sql_cte_and_real_order_table():
    names = ["order", "loan"]
    sql = "WITH c AS (SELECT * FROM `order`) SELECT * FROM c JOIN loan ON 1 ORDER BY 1"
    assert tables_in_sql(sql, names) == {"order", "loan"}


def test_tables_in_sql_unparseable_falls_back():
    assert tables_in_sql("SELEC x FROM loan WHERE ((", ["loan", "order"]) == {"loan"}


def test_embed_batches_and_retries(monkeypatch):
    monkeypatch.setattr(retrieval.time, "sleep", lambda s: None)
    calls = []

    class Err(Exception):
        status_code = 503

    def create(model, inputs):
        calls.append(len(inputs))
        if len(calls) == 2:
            raise Err("boom")
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(t))]) for t in inputs])

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    vecs = embed(client, ["a", "bb", "ccc"], batch_size=2)
    assert vecs == [[1.0], [2.0], [3.0]]
    assert calls == [2, 1, 1]  # second batch failed once, then succeeded
