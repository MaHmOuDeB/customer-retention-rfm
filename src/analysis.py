"""Customer retention and reactivation analysis on UCI Online Retail II (SQL in DuckDB, models in scikit-learn).

Run from the repo root after ``python src/prepare.py``::

    python src/analysis.py

Outputs ``reports/results.json`` and ``reports/figures/*.png``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from statsmodels.stats.power import NormalIndPower  # noqa: E402
from statsmodels.stats.proportion import proportion_effectsize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
# Label windows end 2011-09-08 and 2011-12-08 (the data ends 2011-12-09).
TRAIN_CUTOFF, TEST_CUTOFF = "2011-06-10", "2011-09-09"
FEATURES = ["recency_days", "frequency", "monetary", "avg_order_value", "tenure_days", "orders_last_90d",
            "revenue_last_90d", "avg_basket_products", "is_uk", "return_ratio"]
LOG_COLS = ["frequency", "monetary", "avg_order_value", "revenue_last_90d", "avg_basket_products", "recency_days", "tenure_days"]
SEED = 42
BLUE, LIGHT = "#2f5fb3", "#9db8e0"


def run_sql(con: duckdb.DuckDBPyConnection, name: str, **params: str) -> None:
    sql = (ROOT / "sql" / name).read_text()
    for key, value in params.items():
        sql = sql.replace("{" + key + "}", str(value))
    con.execute(sql)


# ---------- data and cohorts ----------
def data_summary(con: duckdb.DuckDBPyConnection) -> dict:
    """Row counts from raw to final sales, so every dropped row is accounted for."""
    scalar = lambda q: con.execute(q).fetchone()[0]  # noqa: E731
    raw_n = scalar("SELECT COUNT(*) FROM raw")
    no_id = scalar("SELECT COUNT(*) FROM raw WHERE customer_id IS NULL")
    cancel = scalar("SELECT COUNT(*) FROM raw WHERE customer_id IS NOT NULL AND invoice LIKE 'C%'")
    candidates = scalar("SELECT COUNT(*) FROM raw WHERE customer_id IS NOT NULL AND invoice NOT LIKE 'C%'")
    sales_all = scalar("SELECT COUNT(*) FROM sales_all")
    sales_n, n_cust, d0, d1, revenue = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT customer_id), MIN(invoice_date), MAX(invoice_date), SUM(revenue) FROM sales").fetchone()
    rev_all = scalar("SELECT SUM(revenue) FROM sales_all")
    return {"raw_rows": raw_n, "rows_without_customer_id": no_id, "cancellation_rows_with_customer_id": cancel,
            "non_product_or_invalid_rows": candidates - sales_all, "sales_lines_before_reversal_check": sales_all,
            "order_lines_reversed_within_24h": sales_all - sales_n, "revenue_reversed_within_24h_gbp": round(rev_all - revenue),
            "clean_sales_rows": sales_n, "customers": n_cust, "orders": scalar("SELECT COUNT(*) FROM orders"),
            "first_date": str(d0), "last_date": str(d1), "revenue_gbp": round(revenue)}


def cohort_matrix(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    """Retention by acquisition month for the Jan to Nov 2010 cohorts.

    Dec 2009 is left out (the data starts that month, so it holds customers who were already buying) and so is Dec 2010
    (its month 12 is the nine days of Dec 2011). Averages are weighted by cohort size.
    """
    run_sql(con, "03_cohorts.sql")
    table = con.execute("SELECT * FROM cohort_retention WHERE cohort_month BETWEEN DATE '2010-01-01' AND DATE '2010-11-01' AND month_n <= 12").df()
    pivot = table.pivot(index="cohort_month", columns="month_n", values="retention")
    sizes = table[table.month_n == 0].set_index("cohort_month").cohort_size
    weighted = {m: float(table[table.month_n == m].active_customers.sum() / sizes.sum()) for m in (1, 3, 6, 12)}
    return pivot, weighted


def plot_cohorts(pivot: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    image = ax.imshow(pivot.values * 100, cmap="Blues", aspect="auto", vmin=0, vmax=60)
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.set_yticks(range(pivot.shape[0]), [d.strftime("%b %Y") for d in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(1, pivot.shape[1]):
            value = pivot.values[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value * 100:.0f}", ha="center", va="center", fontsize=7, color="white" if value > .35 else "black")
    ax.set_xlabel("Months since first order")
    ax.set_title("Share of each 2010 acquisition cohort that orders again (%)")
    fig.colorbar(image, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(FIG / "cohort_retention.png")
    plt.close(fig)


# ---------- snapshots and models ----------
def snapshot(con: duckdb.DuckDBPyConnection, cutoff: str) -> pd.DataFrame:
    run_sql(con, "02_features.sql", cutoff=cutoff)
    df = con.execute("SELECT * FROM features").df()
    df["return_ratio"] = df["return_ratio"].fillna(0).clip(0, 1)
    return df


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURES].copy()
    for col in LOG_COLS:
        x[col] = np.log1p(x[col].clip(lower=0))
    return x


def top_fraction(y: np.ndarray, score: np.ndarray, frac: float = 0.10) -> tuple[float, float]:
    """Precision and lift among the top ``frac`` of customers by score."""
    k = max(1, int(len(y) * frac))
    hit = y[np.argsort(-score)[:k]].mean()
    return float(hit), float(hit / y.mean())


def bootstrap_auc_difference(y: np.ndarray, new: np.ndarray, base: np.ndarray, n_boot: int = 500) -> dict:
    """Paired bootstrap of AUC(new) - AUC(base) on the same customers."""
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], new[idx]) - roc_auc_score(y[idx], base[idx]))
    return {"auc_difference": round(roc_auc_score(y, new) - roc_auc_score(y, base), 3),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)]}


def calibration(y: np.ndarray, score: np.ndarray, bins: int = 10) -> list[dict]:
    """Mean predicted probability versus observed rate in score deciles."""
    frame = pd.DataFrame({"y": y, "p": score})
    frame["bin"] = pd.qcut(frame.p.rank(method="first"), bins, labels=False)
    grouped = frame.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), customers=("y", "size")).round(3)
    return grouped.reset_index().to_dict("records")


def fit_models(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, np.ndarray], HistGradientBoostingClassifier, pd.DataFrame]:
    x_train, x_test = design_matrix(train), design_matrix(test)
    y_train = train.repurchased_90d.values
    scaler = StandardScaler().fit(x_train)
    logistic = LogisticRegression(max_iter=2000).fit(scaler.transform(x_train), y_train)
    boosted = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, random_state=SEED).fit(x_train, y_train)
    scores = {"Baseline: most recent buyer first": -test.recency_days.values.astype(float),
              "Logistic regression": logistic.predict_proba(scaler.transform(x_test))[:, 1],
              "Gradient boosting": boosted.predict_proba(x_test)[:, 1]}
    return scores, boosted, x_test


def evaluate(scores: dict[str, np.ndarray], y: np.ndarray, lapsed: np.ndarray) -> dict:
    out = {}
    for name, s in scores.items():
        p10, lift = top_fraction(y, s)
        pl, ll = top_fraction(y[lapsed], s[lapsed])
        out[name] = {"roc_auc": round(roc_auc_score(y, s), 3), "avg_precision": round(average_precision_score(y, s), 3),
                     "precision_top10pct": round(p10, 3), "lift_top10pct": round(lift, 2),
                     "lapsed_roc_auc": round(roc_auc_score(y[lapsed], s[lapsed]), 3),
                     "lapsed_precision_top10pct": round(pl, 3), "lapsed_lift_top10pct": round(ll, 2)}
    return out


def plot_models(scores: dict[str, np.ndarray], y: np.ndarray, importance: pd.Series, calib: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for name, s in scores.items():
        order = np.argsort(-s)
        axes[0].plot(np.arange(1, len(s) + 1) / len(s) * 100, np.cumsum(y[order]) / y.sum() * 100, label=name)
    axes[0].plot([0, 100], [0, 100], "k--", lw=.8, label="Random")
    axes[0].set(xlabel="% of customers contacted (highest score first)", ylabel="% of repurchasers reached", title="Cumulative gains, test period")
    axes[0].legend(fontsize=7)
    top = importance.head(8)
    axes[1].barh(top.index[::-1], top.values[::-1], color=BLUE)
    axes[1].set(xlabel="Drop in AUC when the feature is shuffled", title="What drives the gradient boosting model")
    c = pd.DataFrame(calib)
    axes[2].plot([0, 1], [0, 1], "k--", lw=.8)
    axes[2].plot(c.predicted, c.observed, "o-", color=BLUE)
    axes[2].set(xlabel="Predicted probability (decile mean)", ylabel="Observed repurchase rate", title="Calibration in the test period")
    fig.tight_layout()
    fig.savefig(FIG / "model_lift.png")
    plt.close(fig)


# ---------- segments ----------
def rule_segment(r: pd.Series) -> str:
    if r.R >= 4 and r.F >= 4 and r.M >= 4:
        return "Champions"
    if r.R >= 3 and r.F >= 3:
        return "Loyal"
    if r.R >= 4 and r.F <= 2:
        return "New / recent"
    if r.R <= 2 and r.F >= 4:
        return "At risk (were frequent)"
    if r.R <= 2:
        return "Dormant"
    return "Occasional"


def quintile_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Score 1 to 5 from the average rank, so customers with identical values always get identical scores."""
    ranks = values.rank(method="average", ascending=higher_is_better)
    return np.ceil(5 * ranks / len(values)).clip(1, 5).astype(int)


def rfm_segments(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seg = test.copy()
    seg["R"] = quintile_score(seg.recency_days, higher_is_better=False)
    seg["F"] = quintile_score(seg.frequency)
    seg["M"] = quintile_score(seg.monetary)
    seg["segment"] = seg.apply(rule_segment, axis=1)
    prof = seg.groupby("segment").agg(customers=("customer_id", "size"), revenue_share=("monetary", "sum"), median_recency=("recency_days", "median"),
                                      median_orders=("frequency", "median"), repurchase_rate_90d=("repurchased_90d", "mean"))
    prof["revenue_share"] = prof.revenue_share / prof.revenue_share.sum()
    prof["customer_share"] = prof.customers / prof.customers.sum()
    return seg, prof.sort_values("repurchase_rate_90d", ascending=False)


def kmeans_check(seg: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    z = StandardScaler().fit_transform(np.log1p(seg[["recency_days", "frequency", "monetary"]]))
    sample = np.random.default_rng(SEED).choice(len(z), size=min(3000, len(z)), replace=False)
    sil = {}
    for k in range(2, 8):
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(z)
        sil[k] = round(float(silhouette_score(z[sample], labels[sample])), 3)
    best = max(sil, key=sil.get)
    clustered = seg.assign(cluster=KMeans(n_clusters=best, n_init=10, random_state=SEED).fit_predict(z))
    profile = clustered.groupby("cluster").agg(customers=("customer_id", "size"), median_recency=("recency_days", "median"), median_orders=("frequency", "median"),
                                               median_monetary=("monetary", "median"), repurchase_rate_90d=("repurchased_90d", "mean")).round(3)
    return {"by_k": sil, "best_k": best}, profile


def plot_segments(prof: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    order = prof.index
    axes[0].barh(order[::-1], (prof.repurchase_rate_90d[order] * 100)[::-1], color=BLUE)
    axes[0].set(xlabel="Ordered again within 90 days (%)", title="Repurchase rate by RFM segment, test period")
    x, w = np.arange(len(order)), .4
    axes[1].bar(x - w / 2, prof.customer_share[order] * 100, w, label="% of customers", color=LIGHT)
    axes[1].bar(x + w / 2, prof.revenue_share[order] * 100, w, label="% of lifetime revenue", color=BLUE)
    axes[1].set_xticks(x, order, rotation=35, ha="right", fontsize=7)
    axes[1].legend(fontsize=7)
    axes[1].set_title("Customer vs revenue share")
    fig.tight_layout()
    fig.savefig(FIG / "segments.png")
    plt.close(fig)


# ---------- win-back ----------
def lapsed_targeting(test: pd.DataFrame, score: np.ndarray, lapsed: np.ndarray) -> list[dict]:
    y, revenue, s = test.repurchased_90d.values[lapsed], test.revenue_next_90d.values[lapsed], score[lapsed]
    rows = []
    for frac in (0.05, 0.10, 0.20, 0.30):
        k = int(len(y) * frac)
        idx = np.argsort(-s)[:k]
        rows.append({"contact_share": frac, "contacted": k, "returners_model": int(y[idx].sum()), "returners_random": round(float(y.mean() * k), 1),
                     "revenue_next_90d_model_gbp": round(float(revenue[idx].sum())), "revenue_next_90d_random_gbp": round(float(revenue.mean() * k))})
    return rows


def holdout_sample_sizes(baselines: dict[str, float], uplifts: tuple[float, ...] = (0.02, 0.03, 0.05)) -> list[dict]:
    """Customers needed per arm to detect an uplift in the reactivation rate (two-sided, 5% level, 80% power)."""
    power = NormalIndPower()
    rows = []
    for label, p0 in baselines.items():
        for u in uplifts:
            n = power.solve_power(effect_size=proportion_effectsize(p0 + u, p0), alpha=0.05, power=0.8, ratio=1.0)
            rows.append({"group": label, "baseline_rate": round(p0, 3), "uplift_pp": round(u * 100), "customers_per_arm": int(np.ceil(n))})
    return rows


def main() -> None:
    os.chdir(ROOT)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    con = duckdb.connect()
    run_sql(con, "01_clean.sql")
    results: dict = {"data": data_summary(con)}

    pivot, weighted = cohort_matrix(con)
    plot_cohorts(pivot)
    results["cohorts"] = {"cohorts_shown": "Jan to Nov 2010", "weighted_by_cohort_size": True, **{f"retention_month{m}": round(v, 3) for m, v in weighted.items()}}

    train, test = snapshot(con, TRAIN_CUTOFF), snapshot(con, TEST_CUTOFF)
    results["snapshots"] = {"train_cutoff": TRAIN_CUTOFF, "test_cutoff": TEST_CUTOFF, "train_customers": len(train), "test_customers": len(test),
                            "train_repurchase_rate": round(train.repurchased_90d.mean(), 3), "test_repurchase_rate": round(test.repurchased_90d.mean(), 3)}
    scores, boosted, x_test = fit_models(train, test)
    y = test.repurchased_90d.values
    lapsed = (test.recency_days > 90).values
    results["model_out_of_time"] = evaluate(scores, y, lapsed)
    results["auc_difference_vs_recency_rule"] = {
        "all_customers": bootstrap_auc_difference(y, scores["Gradient boosting"], scores["Baseline: most recent buyer first"]),
        "lapsed_only": bootstrap_auc_difference(y[lapsed], scores["Gradient boosting"][lapsed], scores["Baseline: most recent buyer first"][lapsed])}
    results["calibration_gradient_boosting"] = calibration(y, scores["Gradient boosting"])
    results["lapsed_group"] = {"definition": "no order in the 90 days before the test cutoff", "customers": int(lapsed.sum()),
                               "reactivation_rate_next_90d": round(float(y[lapsed].mean()), 3), "active_group_repurchase_rate": round(float(y[~lapsed].mean()), 3)}
    perm = permutation_importance(boosted, x_test, y, scoring="roc_auc", n_repeats=5, random_state=SEED)
    importance = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    results["permutation_importance_auc_drop"] = {k: round(v, 4) for k, v in importance.head(6).items()}
    plot_models(scores, y, importance, results["calibration_gradient_boosting"])

    seg, prof = rfm_segments(test)
    results["rfm_segments"] = {k: {c: (int(v) if c == "customers" else round(float(v), 3)) for c, v in row.items()} for k, row in prof.iterrows()}
    plot_segments(prof)
    sil, cprof = kmeans_check(seg)
    results["kmeans_silhouette"] = sil
    results["kmeans_profile_k"] = {int(k): {c: float(v) for c, v in row.items()} for k, row in cprof.iterrows()}

    results["lapsed_targeting_table"] = lapsed_targeting(test, scores["Gradient boosting"], lapsed)
    top20 = results["lapsed_targeting_table"][2]
    results["holdout_test_sample_sizes"] = holdout_sample_sizes({
        "all lapsed customers": results["lapsed_group"]["reactivation_rate_next_90d"],
        "top 20% of lapsed by score": top20["returners_model"] / top20["contacted"]})

    (ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
