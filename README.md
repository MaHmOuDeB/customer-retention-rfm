# Customer retention and reactivation analysis

[![CI](https://github.com/MaHmOuDeB/customer-retention-rfm/actions/workflows/ci.yml/badge.svg)](https://github.com/MaHmOuDeB/customer-retention-rfm/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Who comes back, who has gone quiet, and which lapsed customers are worth a win-back offer? Two years of orders from a UK
online gift retailer, analysed with SQL (DuckDB) and scikit-learn: cohort retention, RFM segments, a repurchase model
tested on a later period, and a sample-size plan for a win-back experiment.

*Rebuilt in 2026 from my Master's project "Customer Segmentation" (2024). That project clustered a customer summary
file; this version uses order-level data, so it can answer retention questions the original could not.*

## Results at a glance

| Question | Answer |
|---|---|
| How many customers come back? | About 20% of a new monthly cohort orders again in month 1, and about 18% still order in month 12 |
| Where does the revenue come from? | 23% of customers ("Champions") bring 70% of lifetime revenue; the lapsed third of customers brings 7% |
| How many real customer groups are there? | Two. Silhouette peaks at k = 2 and falls for every larger k; the finer RFM segments are rules for action, not clusters the data separates |
| Can a model predict who orders in the next 90 days? | Yes, but a recency rule does most of the work: AUC 0.79 versus 0.76, a gain of 0.03 (95% CI 0.02 to 0.04) |
| Which lapsed customers are most likely to return? | The top 10% by model score returned at 64% versus 29% for all lapsed customers |
| What should happen next? | A holdout experiment: the model predicts who returns, not who a campaign would change |

![Model results](reports/figures/model_lift.png)

## Contents

1. [Data](#data) 2. [Method](#method) 3. [Findings](#findings) 4. [Recommendation](#recommendation) 5. [Limits](#limits) 6. [Reproduce](#reproduce) 7. [Repo layout](#repo-layout)

## Data

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii): 1,067,371 invoice lines, 1 Dec 2009 to 9 Dec 2011.

| Step | Rows |
|---|---|
| Raw invoice lines | 1,067,371 |
| Without a customer ID (dropped, they cannot be followed over time) | 243,007 |
| Cancellation lines (kept apart as returns) | 19,494 |
| Clean product sales lines used | 802,632 |
| Customers / orders / revenue | 5,852 / 36,594 / £17.4M |

Postage, bank charges and manual adjustments are removed by keeping only five-digit product codes.

## Method

- **SQL models** (`sql/`): clean sales, returns, one row per order, snapshot features and cohorts.
- **Out-of-time evaluation:** features are built as of a cutoff date and the label is "ordered again in the next 90 days".
  The model trains on the 10 June 2011 snapshot and is scored on the 9 September 2011 snapshot, so it never sees the period it
  is tested on. Unit tests check that features use only data up to the cutoff and that the label window is exactly 90 days.
- **Models:** a "most recent buyer first" rule, logistic regression and gradient boosting. The AUC gain over the rule comes with
  a paired bootstrap interval (500 resamples).
- **Segments:** rule-based RFM segments (quintile scores) for action, plus k-means on log-scaled RFM to test how many real groups exist.
- **Reproducible:** fixed seeds and a fixed row order; two runs give byte-identical results.

## Findings

**Retention is low and flat after month 1.** The Dec 2009 cohort (35 to 50%) is the exception: the data starts that month, so it
contains customers who were already buying.

![Cohort retention](reports/figures/cohort_retention.png)

**A quarter of customers carry the revenue.** Champions order again within 90 days 80% of the time, lapsed customers 19%.

![Segments](reports/figures/segments.png)

**Recency does most of the work.** Scored on the September snapshot (base rate 43.5%):

| Model | ROC AUC | Precision in top 10% | Lift in top 10% |
|---|---|---|---|
| Most recent buyer first | 0.762 | 78% | 1.8x |
| Logistic regression | 0.791 | 91% | 2.1x |
| Gradient boosting | 0.793 | 93% | 2.1x |

Recency, frequency and total spend drive the prediction. Among lapsed customers only, the model's edge over the rule is smaller
(AUC 0.711 versus 0.685, gain 0.026, 95% CI 0.012 to 0.039) but clear of zero.

**The model ranks well but under-predicts.** Observed repurchase is higher than predicted in every decile (right panel above),
because the test window is the pre-Christmas peak (43.5% repurchase versus 32% in the training window). Rankings carry over to
a new period; absolute probabilities do not, and would need recalibrating before use.

## Recommendation

3,353 customers had no order in the 90 days before the cutoff, and 29% came back anyway. Contacting the 335 highest-scored
(10%) reaches 213 returners and about £179k of next-90-day revenue, versus about 97 returners and £59k for 335 random lapsed
customers. **That is not the value of a campaign:** those customers are the likeliest to return on their own, so most of that
revenue would have arrived without any offer.

To measure what an offer is worth, split the target list at random into contact and holdout groups. Sample size needed per arm
(two-sided, 5% level, 80% power):

| Target group | Baseline return rate | +2 pp | +3 pp | +5 pp |
|---|---|---|---|---|
| All lapsed customers | 29% | 8,240 | 3,696 | 1,353 |
| Top 20% of lapsed by score | 54% | 9,722 | 4,312 | 1,545 |

At this retailer's size (3,353 lapsed customers) only a +5 pp or larger uplift is detectable in a single round, and the likeliest
returners are the *hardest* group to test, because their high baseline leaves less room to improve. A practical design is to
test across several months or to target mid-ranked customers, where an offer has more to change.

## Limits

- **Prediction, not causation.** Nobody in this data was contacted.
- **Seasonality.** One test window, the peak season. Ranking metrics transfer better than absolute rates.
- **One retailer, 2009 to 2011.** Many customers are businesses buying wholesale, so patterns may not transfer to a consumer
  subscription business.
- **About 23% of invoice lines have no customer ID** and are excluded.

## Reproduce

```bash
make setup      # pip install -r requirements.txt
make data       # downloads UCI Online Retail II (45 MB) and converts it to Parquet (a few minutes)
make analysis   # writes reports/results.json and reports/figures/
make test       # unit tests on synthetic data, no download needed
make lint
```

## Repo layout

```
sql/        01_clean, 02_features (as of a cutoff + 90-day label), 03_cohorts
src/        download.py, prepare.py, analysis.py (all statistics and figures)
tests/      feature-window and label tests on synthetic data
reports/    results.json and figures
```

Data: Chen, D. (2012). Online Retail II. UCI Machine Learning Repository, CC BY 4.0.
