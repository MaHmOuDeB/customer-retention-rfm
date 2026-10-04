"""Tiny synthetic checks that the feature SQL uses only data up to the cutoff and labels the NEXT 90 days only."""
from pathlib import Path

import duckdb

SQL = (Path(__file__).resolve().parent.parent / "sql" / "02_features.sql").read_text()


def build(orders, returns=()):
    con = duckdb.connect()
    con.execute("CREATE TABLE orders (customer_id BIGINT, invoice VARCHAR, order_date DATE, order_value DOUBLE, n_products INT, country VARCHAR)")
    con.execute("CREATE TABLE returns (customer_id BIGINT, return_date DATE, return_value DOUBLE)")
    for o in orders:
        con.execute("INSERT INTO orders VALUES (?,?,?,?,?,?)", o)
    for r in returns:
        con.execute("INSERT INTO returns VALUES (?,?,?)", r)
    con.execute(SQL.replace("{cutoff}", "2011-06-10"))
    return con.execute("SELECT * FROM features ORDER BY customer_id").df().set_index("customer_id")


def test_future_orders_do_not_change_features_only_the_label():
    f = build([
        (1, "A1", "2011-05-01", 100.0, 3, "United Kingdom"),
        (1, "A2", "2011-06-01", 50.0, 2, "United Kingdom"),
        (1, "A3", "2011-07-15", 999.0, 9, "United Kingdom"),   # after the cutoff: must not enter features
    ])
    assert f.loc[1, "frequency"] == 2 and f.loc[1, "monetary"] == 150.0
    assert f.loc[1, "recency_days"] == 9
    assert f.loc[1, "repurchased_90d"] == 1


def test_label_window_is_90_days_after_cutoff():
    f = build([
        (1, "A1", "2011-06-01", 10.0, 1, "United Kingdom"),
        (1, "A2", "2011-09-08", 10.0, 1, "United Kingdom"),    # day 90: inside the window
        (2, "B1", "2011-06-01", 10.0, 1, "France"),
        (2, "B2", "2011-09-09", 10.0, 1, "France"),            # day 91: outside
    ])
    assert f.loc[1, "repurchased_90d"] == 1
    assert f.loc[2, "repurchased_90d"] == 0


def test_customers_who_first_buy_after_the_cutoff_are_not_scored():
    f = build([(1, "A1", "2011-06-01", 10.0, 1, "United Kingdom"), (3, "C1", "2011-07-01", 10.0, 1, "United Kingdom")])
    assert list(f.index) == [1]


def test_returns_after_the_cutoff_are_ignored_and_ratio_is_computed():
    f = build([(1, "A1", "2011-06-01", 200.0, 1, "United Kingdom")],
              returns=[(1, "2011-06-05", 50.0), (1, "2011-08-01", 500.0)])
    assert abs(f.loc[1, "return_ratio"] - 0.25) < 1e-9


def test_revenue_in_the_label_window_only_counts_the_next_90_days():
    f = build([
        (1, "A1", "2011-06-01", 100.0, 1, "United Kingdom"),
        (1, "A2", "2011-07-01", 40.0, 1, "United Kingdom"),     # inside the window
        (1, "A3", "2011-08-01", 60.0, 1, "United Kingdom"),     # inside the window
        (1, "A4", "2011-12-01", 999.0, 1, "United Kingdom"),    # far outside the window
    ])
    assert f.loc[1, "revenue_next_90d"] == 100.0 and f.loc[1, "monetary"] == 100.0


def test_feature_rows_come_out_sorted_by_customer():
    f = build([(3, "C1", "2011-06-01", 1.0, 1, "UK"), (1, "A1", "2011-06-01", 1.0, 1, "UK"), (2, "B1", "2011-06-01", 1.0, 1, "UK")])
    assert list(f.index) == [1, 2, 3]
