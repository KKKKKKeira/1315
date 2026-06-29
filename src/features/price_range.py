"""支撐／壓力區間分析 — Volume Profile + 布林通道 + 歷史分位數。

輸出結構：
{
  "strong_support": float,
  "weak_support":   float,
  "entry_suggest":  float,
  "weak_resistance": float,
  "strong_resistance": float,
  "volume_nodes":   [{"price": float, "volume": int, "is_poc": bool}]
}
"""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
RAW_DIR  = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = os.getenv("TARGET_STOCK", "1815")


def build_volume_profile(tick_df: pd.DataFrame, bins: int = 50) -> pd.DataFrame:
    """從 Tick 資料建立 Volume Profile（價格密集度）。"""
    if tick_df.empty or "deal_price" not in tick_df.columns:
        return pd.DataFrame()

    prices = tick_df["deal_price"].dropna()
    vols   = tick_df["volume"].fillna(1)

    price_min = prices.min()
    price_max = prices.max()
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    vol_per_bin = np.zeros(bins)
    for price, vol in zip(prices, vols):
        idx = min(int((price - price_min) / (price_max - price_min) * bins), bins - 1)
        vol_per_bin[idx] += vol

    poc_idx = int(np.argmax(vol_per_bin))
    result = pd.DataFrame({
        "price":  bin_centers.round(2),
        "volume": vol_per_bin.astype(int),
        "is_poc": [i == poc_idx for i in range(bins)]
    })
    return result


def compute_levels(daily_df: pd.DataFrame, current_price: Optional[float] = None) -> dict:
    """從日K 計算支撐壓力水準。"""
    if daily_df.empty:
        return {}

    df = daily_df.copy().sort_values("date").tail(252)  # 最近一年
    close = df["close"]
    current = current_price or close.iloc[-1]

    # 歷史分位數水準
    q10 = float(close.quantile(0.10))
    q25 = float(close.quantile(0.25))
    q50 = float(close.quantile(0.50))
    q75 = float(close.quantile(0.75))
    q90 = float(close.quantile(0.90))

    # 52週高低點
    high52 = float(df["max"].max())
    low52  = float(df["min"].min())

    # 布林通道（最近20日）
    bb_period = min(20, len(df))
    bb_mid    = float(close.tail(bb_period).mean())
    bb_std    = float(close.tail(bb_period).std())
    bb_upper  = bb_mid + 2 * bb_std
    bb_lower  = bb_mid - 2 * bb_std

    # 近期整數關卡（每0.5元一個）
    def round_half(x: float) -> float:
        return round(x * 2) / 2

    below = sorted([p for p in [q25, q10, bb_lower, low52] if p < current], reverse=True)
    above = sorted([p for p in [q75, q90, bb_upper, high52] if p > current])

    strong_support   = round_half(below[1]) if len(below) > 1 else round_half(below[0]) if below else round_half(current * 0.92)
    weak_support     = round_half(below[0]) if below else round_half(current * 0.96)
    weak_resistance  = round_half(above[0]) if above else round_half(current * 1.04)
    strong_resistance = round_half(above[1]) if len(above) > 1 else round_half(above[0]) if above else round_half(current * 1.08)

    # 建議進場：在弱支撐與強支撐之間、或現價-1個標準差
    entry_suggest = round_half((weak_support + current) / 2)

    return {
        "current_price":    round(current, 2),
        "strong_support":   round(strong_support, 2),
        "weak_support":     round(weak_support, 2),
        "entry_suggest":    round(entry_suggest, 2),
        "weak_resistance":  round(weak_resistance, 2),
        "strong_resistance": round(strong_resistance, 2),
        "bb_upper":         round(bb_upper, 2),
        "bb_mid":           round(bb_mid, 2),
        "bb_lower":         round(bb_lower, 2),
        "high_52w":         round(high52, 2),
        "low_52w":          round(low52, 2),
        "quantiles": {
            "q10": round(q10, 2), "q25": round(q25, 2), "q50": round(q50, 2),
            "q75": round(q75, 2), "q90": round(q90, 2),
        }
    }


def get_price_levels(current_price: Optional[float] = None) -> dict:
    """載入歷史資料並回傳支撐壓力結果。"""
    daily_path = RAW_DIR / f"price_{TARGET}.parquet"
    tick_path  = RAW_DIR / f"tick_{TARGET}.parquet"

    daily_df = pd.read_parquet(daily_path) if daily_path.exists() else pd.DataFrame()
    tick_df  = pd.read_parquet(tick_path)  if tick_path.exists()  else pd.DataFrame()

    levels = compute_levels(daily_df, current_price)
    vp = build_volume_profile(tick_df)
    if not vp.empty:
        levels["volume_nodes"] = vp.to_dict(orient="records")
    else:
        levels["volume_nodes"] = []

    return levels


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    result = get_price_levels()
    print(json.dumps(result, indent=2, ensure_ascii=False))
