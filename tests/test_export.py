"""Tests for the site export (src/export_site.py).

Most tests read only committed files (reports/results.json and reports/site/retention.json), so they run in CI.
The last one re-runs the export and needs the local parquet file.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from export_site import build_export, cohort_rows, dump, main  # noqa: E402

RESULTS = json.loads((ROOT / "reports" / "results.json").read_text())
EXPORT_PATH = ROOT / "reports" / "site" / "retention.json"
PARQUET = ROOT / "data" / "online_retail_ii.parquet"
XLSX = ROOT / "data" / "online_retail_II.xlsx"


@pytest.fixture(scope="module")
def export() -> dict:
    return json.loads(EXPORT_PATH.read_text())


def test_committed_export_round_trips():
    text = EXPORT_PATH.read_text()
    assert dump(json.loads(text)) == text


def test_dump_is_canonical():
    assert dump({"b": [1, 2], "a": "é"}) == '{\n "a": "é",\n "b": [\n  1,\n  2\n ]\n}\n'
    with pytest.raises(ValueError):
        dump({"x": float("nan")})


def test_meta(export):
    meta = export["_meta"]
    assert meta["schema"] == "retention/1" and meta["source"] == "retention" and meta["provenance"] == "public-data"
    assert meta["dataset"] == {"name": "UCI Online Retail II", "url": "https://archive.ics.uci.edu/dataset/502/online+retail+ii", "licence": "CC BY 4.0"}
    assert meta["inputs"][0]["path"] == "data/online_retail_II.xlsx"
    assert meta["inputs"][0]["sha256"] == "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980"
    assert meta["inputs"][0]["rows"] == RESULTS["data"]["raw_rows"]
    assert "generated" not in json.dumps(meta).lower()


def test_cohort_facts_come_from_results(export):
    facts = export["facts"]
    assert facts["cohort.month1_rate"]["value"] == RESULTS["cohorts"]["retention_month1"] == 0.201
    assert facts["cohort.month12_rate"]["value"] == RESULTS["cohorts"]["retention_month12"]
    assert facts["cohort.month1_rate"]["unit"] == "share" and facts["cohort.month1_rate"]["decimals"] == 0


def test_data_and_model_facts(export):
    f, d = export["facts"], RESULTS["data"]
    assert (f["data.raw_rows"]["value"], f["data.clean_rows"]["value"]) == (d["raw_rows"], d["clean_sales_rows"])
    assert (f["data.customers"]["value"], f["data.orders"]["value"]) == (d["customers"], d["orders"])
    assert f["data.reversed_revenue_gbp"] == {**f["data.reversed_revenue_gbp"], "value": d["revenue_reversed_within_24h_gbp"], "unit": "gbp", "scale": "k"}
    assert (f["data.first_date"]["value"], f["data.last_date"]["value"]) == ("2009-12-01", "2011-12-09")
    assert f["data.first_date"]["unit"] == "date"
    models = RESULTS["model_out_of_time"]
    assert f["model.rule_auc"]["value"] == models["Baseline: most recent buyer first"]["roc_auc"]
    assert f["model.gb_auc"]["value"] == models["Gradient boosting"]["roc_auc"]
    gain = RESULTS["auc_difference_vs_recency_rule"]["all_customers"]
    assert (f["model.auc_gain"]["value"], f["model.auc_gain_low"]["value"], f["model.auc_gain_high"]["value"]) == (gain["auc_difference"], *gain["ci95"])
    assert f["model.auc_gain"]["decimals"] == 2 and f["model.rule_auc"]["decimals"] == 3
    assert f["model.test_base_rate"]["value"] == RESULTS["snapshots"]["test_repurchase_rate"] and f["model.test_base_rate"]["decimals"] == 1


def test_champions_share_is_computed_from_counts_so_it_shows_23_percent(export):
    f = export["facts"]
    champions = RESULTS["rfm_segments"]["Champions"]
    share = f["rfm.champions_customer_share"]
    assert share["value"] == round(champions["customers"] / RESULTS["snapshots"]["test_customers"], 4) == 0.2345
    assert round(share["value"] * 100) == 23
    assert share["from"] == "results.json#/rfm_segments/Champions"
    assert f["rfm.champions_revenue_share"]["value"] == champions["revenue_share"]
    # the table row keeps the rounding of results.json
    row = next(r for r in export["tables"]["rfm_segments"] if r["segment"] == "Champions")
    assert row["customer_share"] == champions["customer_share"] == 0.235


def test_winback_facts(export):
    f = export["facts"]
    n_per_arm = f["winback.n_per_arm"]["value"]
    assert n_per_arm == 1353
    holdout = [r for r in RESULTS["holdout_test_sample_sizes"] if r["customers_per_arm"] == n_per_arm]
    assert len(holdout) == 1 and holdout[0]["baseline_rate"] == f["winback.baseline_rate"]["value"] == 0.29 and holdout[0]["uplift_pp"] == f["winback.uplift_pp"]["value"] == 5
    assert f["winback.lapsed_customers"]["value"] == RESULTS["lapsed_group"]["customers"]
    assert f["winback.top20_customers"]["value"] == RESULTS["lapsed_targeting_table"][2]["contacted"] == 670
    top10 = RESULTS["lapsed_targeting_table"][1]
    assert f["winback.top10_return_rate"]["value"] == round(top10["returners_model"] / top10["contacted"], 4)
    assert f["winback.top10_return_rate"]["from"] == "results.json#/lapsed_targeting_table/1"
    assert f["winback.baseline_rate"]["unit"] == "share" and f["winback.uplift_pp"]["unit"] == "pp"


def test_cohort_matrix_reproduces_weighted_retention(export):
    matrix = export["tables"]["cohort_matrix"]
    assert len(matrix) == 11 and [r["cohort_month"] for r in matrix] == [f"2010-{m:02d}" for m in range(1, 12)]
    total = sum(r["cohort_size"] for r in matrix)
    for month in (1, 3, 6, 12):
        weighted = sum(r["cohort_size"] * r["retention"][month] for r in matrix) / total
        assert round(weighted, 3) == RESULTS["cohorts"][f"retention_month{month}"]
    assert all(len(r["retention"]) == 13 and r["retention"][0] == 1.0 for r in matrix)


def test_rfm_rows_equal_results_and_shares_sum_to_one(export):
    rows = export["tables"]["rfm_segments"]
    assert [r["segment"] for r in rows] == list(RESULTS["rfm_segments"])
    for row in rows:
        src = RESULTS["rfm_segments"][row["segment"]]
        assert row == {"segment": row["segment"], "customers": src["customers"], "customer_share": src["customer_share"], "revenue_share": src["revenue_share"],
                       "repurchase_rate_90d": src["repurchase_rate_90d"], "median_recency_days": src["median_recency"], "median_orders": src["median_orders"]}
    assert sum(r["customer_share"] for r in rows) == pytest.approx(1, abs=1e-3)
    assert sum(r["revenue_share"] for r in rows) == pytest.approx(1, abs=1e-3)


def test_other_tables_match_results(export):
    t = export["tables"]
    assert t["calibration"] == RESULTS["calibration_gradient_boosting"]
    assert t["lapsed_targeting"] == RESULTS["lapsed_targeting_table"]
    assert t["holdout_sizes"] == RESULTS["holdout_test_sample_sizes"]
    assert [m["model"] for m in t["models"]] == list(RESULTS["model_out_of_time"])
    for m in t["models"]:
        assert m["roc_auc"] == RESULTS["model_out_of_time"][m["model"]]["roc_auc"]


def test_no_customer_or_invoice_identifiers(export):
    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    banned = {k for k in keys(export["tables"]) if "customer_id" in k or "invoice" in k}
    assert banned == set()


def test_cohort_rows_fills_missing_months_with_zero():
    table = pd.DataFrame({
        "cohort_month": pd.to_datetime(["2010-01-01", "2010-01-01", "2010-01-01", "2010-02-01", "2010-02-01"]),
        "month_n": [0, 1, 3, 0, 12],
        "cohort_size": [10, 10, 10, 8, 8],
        "active_customers": [10, 5, 1, 8, 2],
        "retention": [1.0, 0.5, 0.1, 1.0, 0.25],
    })
    rows = cohort_rows(table)
    assert [r["cohort_month"] for r in rows] == ["2010-01", "2010-02"]
    assert [r["cohort_size"] for r in rows] == [10, 8]
    for r in rows:
        assert len(r["retention"]) == 13 and None not in r["retention"]
    assert rows[0]["retention"][:5] == [1.0, 0.5, 0.0, 0.1, 0.0]
    assert rows[1]["retention"][1] == 0.0 and rows[1]["retention"][12] == 0.25


def test_build_export_has_no_timestamp_and_is_deterministic():
    table = pd.DataFrame({"cohort_month": pd.to_datetime(["2010-01-01"] * 2), "month_n": [0, 1], "cohort_size": [4, 4], "active_customers": [4, 1], "retention": [1.0, 0.25]})
    a = dump(build_export(RESULTS, table, "0" * 64, 5))
    assert a == dump(build_export(RESULTS, table, "0" * 64, 5))
    assert not re.search(r"\d{4}-\d{2}-\d{2}T|generated|timestamp", a)


@pytest.mark.skipif(not (PARQUET.exists() and XLSX.exists()), reason="needs the local dataset (make data)")
def test_export_twice_is_byte_identical(tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    main(first)
    main(second)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_text() == EXPORT_PATH.read_text()
