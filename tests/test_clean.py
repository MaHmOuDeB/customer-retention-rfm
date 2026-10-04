"""The cleaning SQL on a tiny synthetic raw table: reversed orders vanish on both sides, genuine returns stay."""
from pathlib import Path

import duckdb

SQL = (Path(__file__).resolve().parent.parent / "sql" / "01_clean.sql").read_text().replace("read_parquet('data/online_retail_ii.parquet')", "raw_in")


def run(rows):
    con = duckdb.connect()
    con.execute("CREATE TABLE raw_in (invoice VARCHAR, stock_code VARCHAR, description VARCHAR, quantity INT, invoice_ts TIMESTAMP, unit_price DOUBLE, customer_id DOUBLE, country VARCHAR)")
    for r in rows:
        con.execute("INSERT INTO raw_in VALUES (?,?,?,?,?,?,?,?)", r)
    con.execute(SQL)
    return con


def row(inv, code, qty, ts, price=2.0, cust=1.0, country="United Kingdom"):
    return (inv, code, "ITEM", qty, ts, price, cust, country)


def test_same_day_reversal_is_removed_on_both_sides():
    con = run([row("A1", "10001", 100, "2011-01-18 10:00"), row("C2", "10001", -100, "2011-01-18 10:05"), row("A3", "10002", 1, "2011-01-19 09:00")])
    assert con.execute("SELECT COUNT(*) FROM returns").fetchone()[0] == 0
    assert [r[0] for r in con.execute("SELECT invoice FROM sales ORDER BY invoice").fetchall()] == ["A3"]


def test_genuine_return_after_more_than_a_day_stays_in_both():
    con = run([row("A1", "10001", 5, "2011-01-01 10:00"), row("C2", "10001", -5, "2011-01-10 10:00")])
    assert con.execute("SELECT SUM(revenue) FROM sales").fetchone()[0] == 10.0
    assert con.execute("SELECT SUM(return_value) FROM returns").fetchone()[0] == 10.0


def test_partial_return_of_a_larger_order_is_not_a_reversal():
    con = run([row("A1", "10001", 10, "2011-01-01 10:00"), row("C2", "10001", -3, "2011-01-01 12:00")])
    assert con.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 1
    assert con.execute("SELECT SUM(return_value) FROM returns").fetchone()[0] == 6.0


def test_non_product_codes_and_missing_customers_are_dropped():
    con = run([row("A1", "POST", 1, "2011-01-01 10:00"), row("A2", "10001", 1, "2011-01-01 10:00", cust=None), row("A3", "10001", 1, "2011-01-01 10:00")])
    assert con.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 1
