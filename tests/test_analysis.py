"""Unit tests for the analysis helpers (pure functions, no data download needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from analysis import holdout_sample_sizes, quintile_score, rule_segment, top_fraction  # noqa: E402


def test_identical_values_get_identical_scores_in_any_order():
    v = pd.Series([1, 1, 1, 1, 2, 3, 5, 8, 13, 21])
    shuffled = v.sample(frac=1, random_state=0)
    assert quintile_score(v).tolist() == quintile_score(shuffled).reindex(v.index).tolist()
    assert quintile_score(v)[:4].nunique() == 1


def test_recency_scores_are_reversed():
    r = pd.Series([1, 50, 100, 200, 400])
    assert quintile_score(r, higher_is_better=False).tolist() == [5, 4, 3, 2, 1]


def test_segment_rules():
    row = lambda r, f, m: pd.Series({"R": r, "F": f, "M": m})  # noqa: E731
    assert rule_segment(row(5, 5, 5)) == "Champions"
    assert rule_segment(row(1, 1, 2)) == "Dormant"
    assert rule_segment(row(2, 5, 3)) == "At risk (were frequent)"
    assert rule_segment(row(5, 1, 1)) == "New / recent"


def test_top_fraction_precision_and_lift():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1])
    precision, lift = top_fraction(y, score, 0.2)
    assert precision == 1.0 and lift == 5.0


def test_holdout_sample_size_matches_the_textbook_figure():
    rows = holdout_sample_sizes({"x": 0.29}, uplifts=(0.05,))
    assert rows[0]["customers_per_arm"] == 1353
