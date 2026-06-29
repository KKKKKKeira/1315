"""測試 Bayesian 情境產生器邏輯。"""

import pandas as pd
import numpy as np
import pytest

from src.model.scenario_builder import (
    _discretize_features,
    _compute_bayesian_probs,
    SCENARIOS,
    _classify_scenario,
)


def make_feature_row(**kwargs):
    defaults = {
        "rsi12": 50, "taiex_chg_pct": 0.5, "total_inst_net": 100000,
        "margin_change": -5000, "wti_chg_pct": -0.5, "ma_bullish": True,
        "ma_bearish": False, "vol_ratio5": 1.1, "sp500_chg_pct": 0.8,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def make_history_df(n=200):
    np.random.seed(7)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    chg = np.random.randn(n) * 1.0
    df = pd.DataFrame({
        "date": dates,
        "next_chg_pct": chg,
        "next_high_chg": chg + np.abs(np.random.randn(n) * 0.5),
        "next_low_chg": chg - np.abs(np.random.randn(n) * 0.5),
        "rsi12": np.random.uniform(20, 80, n),
        "taiex_chg_pct": np.random.randn(n),
        "total_inst_net": np.random.randint(-1e6, 1e6, n),
        "margin_change": np.random.randint(-1e4, 1e4, n),
        "wti_chg_pct": np.random.randn(n),
        "ma_bullish": np.random.choice([True, False], n),
        "ma_bearish": np.random.choice([True, False], n),
        "vol_ratio5": np.random.uniform(0.3, 2.5, n),
        "sp500_chg_pct": np.random.randn(n),
    })
    # 加入離散特徵
    for i, row in df.iterrows():
        bins = _discretize_features(row)
        for k, v in bins.items():
            df.at[i, k] = v
    return df


def test_discretize_features_keys():
    row = make_feature_row()
    bins = _discretize_features(row)
    expected_keys = ["rsi_zone", "taiex_dir", "inst_dir", "margin", "oil", "ma_align", "vol", "us_dir"]
    for k in expected_keys:
        assert k in bins, f"Missing key: {k}"


def test_discretize_rsi_oversold():
    row = make_feature_row(rsi12=25)
    bins = _discretize_features(row)
    assert bins["rsi_zone"] == "oversold"


def test_discretize_rsi_overbought():
    row = make_feature_row(rsi12=75)
    bins = _discretize_features(row)
    assert bins["rsi_zone"] == "overbought"


def test_bayesian_probs_sum_to_one():
    history = make_history_df()
    row = make_feature_row()
    today_bins = _discretize_features(row)
    probs = _compute_bayesian_probs(history, today_bins)
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_bayesian_probs_all_scenarios():
    history = make_history_df()
    row = make_feature_row()
    today_bins = _discretize_features(row)
    probs = _compute_bayesian_probs(history, today_bins)
    for sc in SCENARIOS:
        assert sc in probs
        assert 0 <= probs[sc] <= 1


def test_classify_scenario():
    assert _classify_scenario(3.0) == "strong_bull"
    assert _classify_scenario(1.0) == "bull"
    assert _classify_scenario(0.0) == "flat"
    assert _classify_scenario(-1.0) == "bear"
    assert _classify_scenario(-3.0) == "strong_bear"
