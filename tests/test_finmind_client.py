"""測試 FinMind 客戶端的資料解析邏輯（不實際呼叫 API）。"""

from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from src.data.finmind_client import (
    get_daily_price,
    get_institutional,
    get_margin,
    default_start,
)

MOCK_PRICE_DATA = [
    {"date": "2024-01-15", "stock_id": "1815", "open": "25.0", "max": "25.5",
     "min": "24.8", "close": "25.2", "spread": "0.3",
     "Trading_Volume": "100000", "Trading_money": "2500000", "Trading_turnover": "500"},
    {"date": "2024-01-16", "stock_id": "1815", "open": "25.2", "max": "25.8",
     "min": "25.0", "close": "25.6", "spread": "0.4",
     "Trading_Volume": "120000", "Trading_money": "3000000", "Trading_turnover": "600"},
]

MOCK_INST_DATA = [
    {"date": "2024-01-15", "stock_id": "1815", "name": "外資", "buy": "500000", "sell": "300000"},
    {"date": "2024-01-15", "stock_id": "1815", "name": "投信", "buy": "100000", "sell": "50000"},
    {"date": "2024-01-16", "stock_id": "1815", "name": "外資", "buy": "200000", "sell": "400000"},
]


def mock_fetch(dataset, data_id, start_date, end_date=None, retries=4):
    if "Price" in dataset and "Tick" not in dataset:
        return pd.DataFrame(MOCK_PRICE_DATA)
    if "Institutional" in dataset:
        return pd.DataFrame(MOCK_INST_DATA)
    return pd.DataFrame()


@patch("src.data.finmind_client._fetch", side_effect=mock_fetch)
def test_get_daily_price(mock):
    df = get_daily_price("1815", "2024-01-01")
    assert not df.empty
    assert "close" in df.columns
    assert df["close"].dtype == float
    assert df["date"].is_monotonic_increasing


@patch("src.data.finmind_client._fetch", side_effect=mock_fetch)
def test_get_institutional(mock):
    df = get_institutional("1815", "2024-01-01")
    assert not df.empty
    assert "date" in df.columns
    # 應有 net_外資 欄
    net_cols = [c for c in df.columns if c.startswith("net_")]
    assert len(net_cols) >= 1
    # 外資 2024-01-15: buy=500000, sell=300000 → net=200000
    row = df[df["date"] == pd.Timestamp("2024-01-15")]
    assert not row.empty
    assert float(row["net_外資"].iloc[0]) == 200000.0


@patch("src.data.finmind_client._fetch", return_value=pd.DataFrame())
def test_get_daily_price_empty(mock):
    df = get_daily_price("1815", "2024-01-01")
    assert df.empty


def test_default_start():
    start = default_start(365)
    assert len(start) == 10  # YYYY-MM-DD
