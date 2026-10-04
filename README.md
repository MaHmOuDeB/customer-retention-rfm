# Customer retention and reactivation analysis

Who comes back, who has gone quiet, and which lapsed customers are worth winning back? An analysis of two years of
orders from a UK online gift retailer, with the SQL in DuckDB and the models in scikit-learn.

Rebuilt in 2026 from my Master's project *Customer Segmentation* (2024, clustering and PCA on a supermarket-style
customer file). This version uses a different public dataset, transaction data instead of a customer summary, so it
can answer retention questions the original could not: cohort retention, repurchase prediction and win-back targeting.

## Data

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii): 1,067,371 invoice lines, 1 Dec 2009 to
9 Dec 2011. Download it with the link above and run `src/prepare.py`; the raw file is not in this repo.

| Step | Rows |
|---|---|
| Raw invoice lines | 1,067,371 |
| Without a customer ID (dropped; they cannot be followed over time) | 243,007 |
| Cancellation lines (kept apart as returns) | 19,494 |
| Clean product sales lines used | 802,632 |
| Customers / orders / revenue | 5,852 / 36,594 / £17.4M |

Non-product codes (postage, bank charges, manual adjustments) are removed by keeping only five-digit stock codes.

## What I did

1. **Cleaning and models in SQL** (`sql/`): sales, returns, one row per order, snapshot features, cohorts.
2. **Cohort retention**: share of each acquisition month that orders again in month N.
3. **Out-of-time prediction**: features as of a cutoff date, label = ordered again in the next 90 days. Train on the
   10 June 2011 snapshot, test on the 9 September 2011 snapshot, so the model never sees the period it is scored on.
   Compared a "most recent buyer first" rule, logistic regression and gradient boosting.
4. **Segments**: rule-based RFM segments for action, and k-means on log-scaled RFM (silhouette) to check whether the
   data really contains more groups than that.
5. **Win-back view**: among customers with no order in the last 90 days, who is most likely to return?
6. **Tests** (`tests/`): the feature SQL uses only data up to the cutoff, the label window is exactly 90 days, and
   customers who first buy after the cutoff are not scored.

## Findings

**Retention is low and flat after the first month.** About 20% of a new monthly cohort orders again in month 1 and about
18% still order in month 12 (cohorts Dec 2009 to Dec 2010). The Dec 2009 cohort is the exception at 35 to 50%, because the
data starts that month, so it contains customers who had already been buying before.

![cohorts](reports/figures/cohort_retention.png)

**A quarter of customers carry the revenue.** Champions are 23% of customers and 70% of lifetime revenue, and 80% of them
order again within 90 days. The lapsed group is 34% of customers and 7% of revenue, and 20% of them return.

![segments](reports/figures/segments.png)

**The data holds two natural groups, not six.** Silhouette is highest at k = 2 (0.46: active repeat buyers versus
one-off and lapsed customers) and drops for every larger k. The finer RFM segments are useful because they tell a team
what to do, but they are rules, not clusters the data separates by itself.

**Recency does most of the work.** Out-of-time on the September snapshot:

| Model | ROC AUC | Precision in top 10% | Lift in top 10% |
|---|---|---|---|
| Most recent buyer first | 0.762 | 78% | 1.8x |
| Logistic regression | 0.791 | 91% | 2.1x |
| Gradient boosting | 0.793 | 93% | 2.1x |

Base rate in the test period: 43.5%. Recency, then frequency, then total spend drive the prediction. The extra models add
about 0.03 AUC over a one-line rule, so a simple recency rule is a strong default.

![model](reports/figures/model_lift.png)

**Win-back.** 3,353 customers had no order in the 90 days before the test cutoff, and 29% of them ordered anyway in the
next 90 days. Ranking them with the gradient boosting model, the top 10% (335 customers) contained 213 who came back (64%),
versus about 97 if the same number were picked at random.

## Limits

- **This is prediction, not a campaign result.** Nobody in this data was contacted. The customers most likely to come back
  are not necessarily the ones a win-back offer would change, since many would return unprompted. The next step would be a
  holdout test: contact half of the high-scoring lapsed customers and compare, so the uplift is measured, not assumed.
- **Seasonality.** The test window is the pre-Christmas peak: 43.5% repurchase versus 32% in the training window. Ranking
  metrics (AUC, lift) carry over better than absolute rates.
- **One retailer, 2009 to 2011.** Many customers are businesses buying wholesale, so the patterns will not transfer to a
  consumer subscription business without checking.
- **Customers without an ID are excluded**, about 23% of lines.

## Run it

```bash
pip install -r requirements.txt
python src/prepare.py      # once, converts the Excel file to Parquet (a few minutes)
python src/analysis.py     # writes reports/results.json and reports/figures/
pytest
```

Data: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository (CC BY 4.0).
