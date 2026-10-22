"""Write ``reports/site/retention.json``, the export the portfolio site reads.

Run from anywhere after ``make data`` and ``make analysis``::

    make export

The numbers come from ``reports/results.json`` (copied with its rounding) and from the cohort SQL. The file holds no customer IDs, no
invoice numbers and no generation time, and is written as canonical JSON, so two runs give the same bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "site" / "retention.json"
XLSX = ROOT / "data" / "online_retail_II.xlsx"
EXPECTED_XLSX_SHA256 = "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980"
BASELINE = "Baseline: most recent buyer first"
GRADIENT_BOOSTING = "Gradient boosting"
MONTHS = 13  # month 0 (the first order) to month 12


def dump(obj: dict) -> str:
    """Canonical JSON: sorted keys, one-space indent, no NaN, trailing newline."""
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def cohort_rows(table: pd.DataFrame) -> list[dict]:
    """One dict per acquisition cohort with 13 retention values (months 0 to 12). A month the SQL output does not have is 0.0, never null."""
    rows = []
    for month, group in table.groupby("cohort_month", sort=True):
        by_month = dict(zip(group["month_n"].astype(int), group["retention"].astype(float), strict=True))
        rows.append({"cohort_month": pd.Timestamp(month).strftime("%Y-%m"), "cohort_size": int(group["cohort_size"].iloc[0]),
                     "retention": [round(by_month.get(n, 0.0), 4) for n in range(MONTHS)]})
    return rows


def _fact(value, unit: str, decimals: int, label: str, source: str, **extra) -> dict:
    return {"value": value, "unit": unit, "decimals": decimals, "label": label, "from": source, **extra}


def build_facts(results: dict) -> dict:
    d, c, m = results["data"], results["cohorts"], results["model_out_of_time"]
    gain = results["auc_difference_vs_recency_rule"]["all_customers"]
    champions, lapsed = results["rfm_segments"]["Champions"], results["lapsed_group"]
    top10, top20 = results["lapsed_targeting_table"][1], results["lapsed_targeting_table"][2]
    holdout = next(r for r in results["holdout_test_sample_sizes"] if r["group"] == "all lapsed customers" and r["uplift_pp"] == 5)
    r = "results.json#/"
    return {
        "data.raw_rows": _fact(d["raw_rows"], "count", 0, "Invoice lines in the raw file", r + "data/raw_rows"),
        "data.clean_rows": _fact(d["clean_sales_rows"], "count", 0, "Sales lines after cleaning", r + "data/clean_sales_rows"),
        "data.customers": _fact(d["customers"], "count", 0, "Customers with at least one clean order", r + "data/customers"),
        "data.orders": _fact(d["orders"], "count", 0, "Orders", r + "data/orders"),
        "data.reversed_revenue_gbp": _fact(d["revenue_reversed_within_24h_gbp"], "gbp", 0, "Revenue removed because the order was cancelled within 24 hours",
                                           r + "data/revenue_reversed_within_24h_gbp", scale="k"),
        "data.first_date": _fact(d["first_date"], "date", 0, "First invoice date", r + "data/first_date"),
        "data.last_date": _fact(d["last_date"], "date", 0, "Last invoice date", r + "data/last_date"),
        "cohort.month1_rate": _fact(c["retention_month1"], "share", 0, "Month-1 retention, Jan to Nov 2010 cohorts, weighted by cohort size", r + "cohorts/retention_month1"),
        "cohort.month12_rate": _fact(c["retention_month12"], "share", 0, "Month-12 retention, Jan to Nov 2010 cohorts, weighted by cohort size", r + "cohorts/retention_month12"),
        "model.rule_auc": _fact(m[BASELINE]["roc_auc"], "decimal", 3, "ROC AUC of the most-recent-buyer-first rule", r + f"model_out_of_time/{BASELINE}/roc_auc"),
        "model.gb_auc": _fact(m[GRADIENT_BOOSTING]["roc_auc"], "decimal", 3, "ROC AUC of gradient boosting", r + f"model_out_of_time/{GRADIENT_BOOSTING}/roc_auc"),
        "model.auc_gain": _fact(gain["auc_difference"], "decimal", 2, "AUC gain of gradient boosting over the rule", r + "auc_difference_vs_recency_rule/all_customers/auc_difference"),
        "model.auc_gain_low": _fact(gain["ci95"][0], "decimal", 2, "AUC gain, lower end of the 95% interval", r + "auc_difference_vs_recency_rule/all_customers/ci95/0"),
        "model.auc_gain_high": _fact(gain["ci95"][1], "decimal", 2, "AUC gain, upper end of the 95% interval", r + "auc_difference_vs_recency_rule/all_customers/ci95/1"),
        "model.test_base_rate": _fact(results["snapshots"]["test_repurchase_rate"], "share", 1, "Share of customers who ordered again in the test window", r + "snapshots/test_repurchase_rate"),
        # Computed from counts: results.json rounds the share to 0.235, which a 0-decimal percentage would show as 24%.
        "rfm.champions_customer_share": _fact(round(champions["customers"] / results["snapshots"]["test_customers"], 4), "share", 0,
                                              "Champions as a share of customers at the test cutoff", r + "rfm_segments/Champions"),
        "rfm.champions_revenue_share": _fact(champions["revenue_share"], "share", 0, "Champions' share of spend up to the test cutoff", r + "rfm_segments/Champions/revenue_share"),
        "winback.lapsed_customers": _fact(lapsed["customers"], "count", 0, "Lapsed customers at the test cutoff", r + "lapsed_group/customers"),
        "winback.baseline_rate": _fact(holdout["baseline_rate"], "share", 0, "Share of lapsed customers who order again within 90 days", r + "lapsed_group/reactivation_rate_next_90d"),
        "winback.uplift_pp": _fact(holdout["uplift_pp"], "pp", 0, "Uplift the holdout test is sized to detect", r + "holdout_test_sample_sizes/2/uplift_pp"),
        "winback.n_per_arm": _fact(holdout["customers_per_arm"], "count", 0, "Lapsed customers needed per arm to detect that uplift", r + "holdout_test_sample_sizes/2/customers_per_arm"),
        "winback.top20_customers": _fact(top20["contacted"], "count", 0, "Customers in the top 20% of lapsed by score", r + "lapsed_targeting_table/2/contacted"),
        "winback.top10_return_rate": _fact(round(top10["returners_model"] / top10["contacted"], 4), "share", 0,
                                           "Share of the top 10% of lapsed by score who ordered again within 90 days", r + "lapsed_targeting_table/1"),
    }


def build_tables(results: dict, cohort_table: pd.DataFrame) -> dict:
    models = results["model_out_of_time"]
    return {
        "cohort_matrix": cohort_rows(cohort_table),
        "rfm_segments": [{"segment": name, "customers": s["customers"], "customer_share": s["customer_share"], "revenue_share": s["revenue_share"],
                          "repurchase_rate_90d": s["repurchase_rate_90d"], "median_recency_days": s["median_recency"], "median_orders": s["median_orders"]}
                         for name, s in results["rfm_segments"].items()],
        "models": [{"model": name, **{k: v[k] for k in ("roc_auc", "precision_top10pct", "lift_top10pct", "lapsed_roc_auc")}} for name, v in models.items()],
        "calibration": results["calibration_gradient_boosting"],
        "lapsed_targeting": results["lapsed_targeting_table"],
        "holdout_sizes": results["holdout_test_sample_sizes"],
    }


def build_export(results: dict, cohort_table: pd.DataFrame, xlsx_sha256: str, raw_rows: int) -> dict:
    return {
        "_meta": {
            "schema": "retention/1", "source": "retention", "generator": "src/export_site.py",
            "inputs": [{"path": "data/online_retail_II.xlsx", "sha256": xlsx_sha256, "rows": raw_rows}],
            "engine": {"duckdb": duckdb.__version__, "pandas": pd.__version__, "python": ".".join(platform.python_version_tuple()[:2])},
            "dataset": {"name": "UCI Online Retail II", "url": "https://archive.ics.uci.edu/dataset/502/online+retail+ii", "licence": "CC BY 4.0"},
            "provenance": "public-data",
        },
        "facts": build_facts(results),
        "tables": build_tables(results, cohort_table),
    }


def main(out: Path = OUT) -> None:
    os.chdir(ROOT)  # the SQL reads data/online_retail_ii.parquet relative to the repo root
    digest = hashlib.sha256(XLSX.read_bytes()).hexdigest()
    if digest != EXPECTED_XLSX_SHA256:
        print(f"warning: {XLSX.name} has sha256 {digest}, expected {EXPECTED_XLSX_SHA256}; the export records the hash it saw", file=sys.stderr)
    con = duckdb.connect()
    for name in ("01_clean.sql", "03_cohorts.sql"):
        con.execute((ROOT / "sql" / name).read_text())
    raw_rows = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
    # exactly as analysis.cohort_matrix: Jan to Nov 2010 cohorts, months 0 to 12
    table = con.execute("SELECT * FROM cohort_retention WHERE cohort_month BETWEEN DATE '2010-01-01' AND DATE '2010-11-01' AND month_n <= 12").df()
    results = json.loads((ROOT / "reports" / "results.json").read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump(build_export(results, table, digest, raw_rows)))
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
