"""測試技術指標與特徵工程。"""

import pandas as pd
import numpy as np
import pytest

from src.features.technical import build_features, add_rsi, add_macd
from src.features.price_range import compute_levels
from src.features.seasonality import compute_seasonality, get_today_seasonality_score


def make_daily_df(n=100, start_price=25.0):
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    np.random.seed(42)
    closes = start_price + np.cumsum(np.random.randn(n) * 0.3)
    highs  = closes + np.abs(np.random.randn(n) * 0.2)
    lows   = closes - np.abs(np.random.randn(n) * 0.2)
    opens  = closes + np.random.randn(n) * 0.1
    vols   = np.random.randint(50000, 200000, n)
    return pd.DataFrame({
        "date":            dates,
        "open":            np.round(opens, 2),
        "max":             np.round(highs, 2),
        "min":             np.round(lows, 2),
        "close":           np.round(closes, 2),
        "Trading_Volume":  vols,
        "Trading_turnover": vols // 500,
    })


def test_build_features_columns():
    df = make_daily_df(60)
    result = build_features(df)
    expected_cols = ["ma5", "ma20", "rsi12", "macd", "bb_upper", "bb_lower",
                     "k", "d", "vol_ratio5", "chg_pct"]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_rsi_range():
    df = make_daily_df(60)
    df = add_rsi(df)
    valid_rsi = df["rsi12"].dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


def test_macd_signal():
    df = make_daily_df(60)
    df = add_macd(df)
    assert "macd" in df.columns
    assert "macd_signal" in df.columns
    assert "macd_hist" in df.columns


def test_compute_levels():
    df = make_daily_df(252)
    levels = compute_levels(df, current_price=25.0)
    assert "strong_support" in levels
    assert "strong_resistance" in levels
    assert "entry_suggest" in levels
    assert levels["strong_support"] < levels["entry_suggest"]
    assert levels["entry_suggest"] < levels["strong_resistance"]


def test_seasonality_output():
    df = make_daily_df(200)
    result = compute_seasonality(df)
    assert "weekday_effect" in result
    assert "month_effect" in result
    assert "Mon" in result["weekday_effect"]


def test_seasonality_score_range():
    df = make_daily_df(200)
    score = get_today_seasonality_score(df, "2024-03-15")
    assert -1.0 <= score <= 1.0
