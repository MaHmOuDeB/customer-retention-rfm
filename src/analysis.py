"""Customer retention and reactivation analysis on UCI Online Retail II (SQL in DuckDB, models in scikit-learn).

Run from the repo root:  python src/prepare.py   (once)   then   python src/analysis.py
Outputs: reports/results.json and reports/figures/*.png
"""
import json
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TRAIN_CUTOFF, TEST_CUTOFF = "2011-06-10", "2011-09-09"      # label windows end 2011-09-08 and 2011-12-08 (data ends 2011-12-09)
FEATURES = ["recency_days", "frequency", "monetary", "avg_order_value", "tenure_days", "orders_last_90d",
            "revenue_last_90d", "avg_basket_products", "is_uk", "return_ratio"]
LOG_COLS = ["frequency", "monetary", "avg_order_value", "revenue_last_90d", "avg_basket_products", "recency_days", "tenure_days"]
SEED = 42
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

con = duckdb.connect()
def run_sql(name, **params):
    sql = (ROOT / "sql" / name).read_text()
    for k, v in params.items():
        sql = sql.replace("{" + k + "}", str(v))
    con.execute(sql)

results = {}

# ---------- 1. Clean + data quality ----------
import os
os.chdir(ROOT)
run_sql("01_clean.sql")
q = lambda s: con.execute(s).fetchone()
raw_n, = q("SELECT COUNT(*) FROM raw")
sales_n, n_cust, d0, d1, total_rev = q("SELECT COUNT(*), COUNT(DISTINCT customer_id), MIN(invoice_date), MAX(invoice_date), SUM(revenue) FROM sales")
n_orders, = q("SELECT COUNT(*) FROM orders")
no_id, = q("SELECT COUNT(*) FROM raw WHERE customer_id IS NULL")
cancel, = q("SELECT COUNT(*) FROM raw WHERE invoice LIKE 'C%'")
results["data"] = {"raw_rows": raw_n, "rows_without_customer_id": no_id, "cancellation_rows": cancel, "clean_sales_rows": sales_n,
                   "customers": n_cust, "orders": n_orders, "first_date": str(d0), "last_date": str(d1), "revenue_gbp": round(total_rev)}

# ---------- 2. Cohort retention ----------
run_sql("03_cohorts.sql")
coh = con.execute("SELECT * FROM cohort_retention").df()
piv = coh.pivot(index="cohort_month", columns="month_n", values="retention")
piv = piv[piv.index <= pd.Timestamp("2010-12-01")]          # cohorts with at least 12 months of follow-up
piv = piv[[c for c in piv.columns if c <= 12]]
fig, ax = plt.subplots(figsize=(8, 4.2))
im = ax.imshow(piv.values * 100, cmap="Blues", aspect="auto", vmin=0, vmax=60)
ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([d.strftime("%b %Y") for d in piv.index])
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if not np.isnan(v) and j > 0:
            ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=7, color="white" if v > .35 else "black")
ax.set_xlabel("Months since first order"); ax.set_title("Share of each acquisition cohort that orders again (%), month 0 = 100")
fig.colorbar(im, ax=ax, shrink=.8); fig.tight_layout(); fig.savefig(FIG / "cohort_retention.png"); plt.close(fig)
m1 = piv[1].mean(); m3 = piv[3].mean(); m6 = piv[6].mean(); m12 = piv[12].mean()
results["cohorts"] = {"cohorts_shown": int(piv.shape[0]), "avg_retention_month1": round(m1, 3), "month3": round(m3, 3),
                      "month6": round(m6, 3), "month12": round(m12, 3)}

# ---------- 3. Snapshots + out-of-time model ----------
def snapshot(cutoff):
    run_sql("02_features.sql", cutoff=cutoff)
    df = con.execute("SELECT * FROM features").df()
    df["return_ratio"] = df["return_ratio"].fillna(0).clip(0, 1)
    return df
train, test = snapshot(TRAIN_CUTOFF), snapshot(TEST_CUTOFF)
results["snapshots"] = {"train_cutoff": TRAIN_CUTOFF, "test_cutoff": TEST_CUTOFF, "train_customers": len(train), "test_customers": len(test),
                        "train_repurchase_rate": round(train.repurchased_90d.mean(), 3), "test_repurchase_rate": round(test.repurchased_90d.mean(), 3)}

def prep(df):
    X = df[FEATURES].copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X
Xtr, Xte, ytr, yte = prep(train), prep(test), train.repurchased_90d.values, test.repurchased_90d.values
scaler = StandardScaler().fit(Xtr)
models = {
    "Baseline: most recent buyer first": None,
    "Logistic regression": LogisticRegression(max_iter=2000, C=1.0),
    "Gradient boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, random_state=SEED),
}
def top_decile(y, s, frac=0.10):
    k = max(1, int(len(y) * frac)); idx = np.argsort(-s)[:k]
    return y[idx].mean(), y[idx].mean() / y.mean()
scores, perf = {}, {}
lapsed_mask = (test.recency_days > 90).values
for name, m in models.items():
    if m is None:
        s = -test.recency_days.values.astype(float)
    elif "Logistic" in name:
        m.fit(scaler.transform(Xtr), ytr); s = m.predict_proba(scaler.transform(Xte))[:, 1]
    else:
        m.fit(Xtr, ytr); s = m.predict_proba(Xte)[:, 1]
    scores[name] = s
    p10, lift = top_decile(yte, s)
    perf[name] = {"roc_auc": round(roc_auc_score(yte, s), 3), "avg_precision": round(average_precision_score(yte, s), 3),
                  "precision_top10pct": round(p10, 3), "lift_top10pct": round(lift, 2)}
    yl, sl = yte[lapsed_mask], s[lapsed_mask]
    pl, ll = top_decile(yl, sl)
    perf[name].update({"lapsed_roc_auc": round(roc_auc_score(yl, sl), 3), "lapsed_precision_top10pct": round(pl, 3), "lapsed_lift_top10pct": round(ll, 2)})
results["model_out_of_time"] = perf
results["lapsed_group"] = {"definition": "no order in the 90 days before the test cutoff", "customers": int(lapsed_mask.sum()),
                           "reactivation_rate_next_90d": round(float(yte[lapsed_mask].mean()), 3),
                           "active_group_repurchase_rate": round(float(yte[~lapsed_mask].mean()), 3)}

# feature importance via permutation on the gradient boosting model
from sklearn.inspection import permutation_importance
gb = models["Gradient boosting"]
pi = permutation_importance(gb, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
results["permutation_importance_auc_drop"] = {k: round(v, 4) for k, v in imp.head(6).items()}

# lift curve figure
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for name, s in scores.items():
    order = np.argsort(-s); cum = np.cumsum(yte[order]) / yte.sum(); x = np.arange(1, len(s) + 1) / len(s)
    axes[0].plot(x * 100, cum * 100, label=name)
axes[0].plot([0, 100], [0, 100], "k--", lw=.8, label="Random"); axes[0].set_xlabel("% of customers contacted (highest score first)")
axes[0].set_ylabel("% of repurchasers reached"); axes[0].set_title("Cumulative gains, test period"); axes[0].legend(fontsize=7)
axes[1].barh(imp.index[::-1][-8:], imp.values[::-1][-8:], color="#3b6fb6"); axes[1].set_xlabel("Drop in AUC when the feature is shuffled")
axes[1].set_title("What drives the gradient boosting model")
fig.tight_layout(); fig.savefig(FIG / "model_lift.png"); plt.close(fig)

# ---------- 4. Segments at the test cutoff ----------
seg = test.copy()
seg["R"] = pd.qcut(seg.recency_days.rank(method="first", ascending=False), 5, labels=[1, 2, 3, 4, 5]).astype(int)
seg["F"] = pd.qcut(seg.frequency.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
seg["M"] = pd.qcut(seg.monetary.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
def rule(r):
    if r.R >= 4 and r.F >= 4 and r.M >= 4: return "Champions"
    if r.R >= 3 and r.F >= 3: return "Loyal"
    if r.R >= 4 and r.F <= 2: return "New / recent"
    if r.R <= 2 and r.F >= 4: return "At risk (were frequent)"
    if r.R <= 2: return "Lapsed"
    return "Occasional"
seg["segment"] = seg.apply(rule, axis=1)
prof = seg.groupby("segment").agg(customers=("customer_id", "size"), revenue_share=("monetary", "sum"), median_recency=("recency_days", "median"),
                                  median_orders=("frequency", "median"), repurchase_rate_90d=("repurchased_90d", "mean"))
prof["revenue_share"] = prof.revenue_share / prof.revenue_share.sum()
prof["customer_share"] = prof.customers / prof.customers.sum()
prof = prof.sort_values("repurchase_rate_90d", ascending=False)
results["rfm_segments"] = {k: {c: round(float(v), 3) for c, v in row.items()} for k, row in prof.iterrows()}

# k-means check on log RFM: how many natural groups?
Z = StandardScaler().fit_transform(np.log1p(seg[["recency_days", "frequency", "monetary"]]))
rng = np.random.default_rng(SEED); sample = rng.choice(len(Z), size=min(3000, len(Z)), replace=False)
sil = {}
for k in range(2, 8):
    labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Z)
    sil[k] = round(float(silhouette_score(Z[sample], labels[sample])), 3)
best_k = max(sil, key=sil.get)
results["kmeans_silhouette"] = {"by_k": sil, "best_k": best_k}
seg["cluster"] = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit_predict(Z)
cprof = seg.groupby("cluster").agg(customers=("customer_id", "size"), median_recency=("recency_days", "median"), median_orders=("frequency", "median"),
                                   median_monetary=("monetary", "median"), repurchase_rate_90d=("repurchased_90d", "mean")).round(3)
results["kmeans_profile_k"] = {int(k): {c: float(v) for c, v in row.items()} for k, row in cprof.iterrows()}

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
order = prof.index
axes[0].barh(order[::-1], (prof.repurchase_rate_90d[order] * 100)[::-1], color="#3b6fb6")
axes[0].set_xlabel("Ordered again within 90 days (%)"); axes[0].set_title("Repurchase rate by RFM segment, test period")
x = np.arange(len(order)); w = .4
axes[1].bar(x - w/2, prof.customer_share[order] * 100, w, label="% of customers", color="#9db8e0")
axes[1].bar(x + w/2, prof.revenue_share[order] * 100, w, label="% of lifetime revenue", color="#3b6fb6")
axes[1].set_xticks(x); axes[1].set_xticklabels(order, rotation=35, ha="right", fontsize=7); axes[1].legend(fontsize=7)
axes[1].set_title("Customer vs revenue share")
fig.tight_layout(); fig.savefig(FIG / "segments.png"); plt.close(fig)

# ---------- 5. Who to contact: lapsed customers, model vs random ----------
sl = scores["Gradient boosting"][lapsed_mask]; yl = yte[lapsed_mask]
rows = []
for frac in (0.05, 0.10, 0.20, 0.30):
    k = int(len(yl) * frac); idx = np.argsort(-sl)[:k]
    rows.append({"contact_share": frac, "contacted": k, "expected_reactivations_model": int(yl[idx].sum()),
                 "expected_reactivations_random": round(float(yl.mean() * k), 1)})
results["lapsed_targeting_table"] = rows

(ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=2, default=str))
print(json.dumps(results, indent=2, default=str))
