"""計算所有技術指標特徵，從日K DataFrame 輸出。"""

import pandas as pd
import numpy as np


def add_moving_averages(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    for n in [5, 10, 20, 60, 120, 240]:
        df[f"ma{n}"] = df[price_col].rolling(n).mean()
        df[f"ema{n}"] = df[price_col].ewm(span=n, adjust=False).mean()
    # 多空排列：MA5 > MA20 > MA60
    df["ma_bullish"] = (df["ma5"] > df["ma20"]) & (df["ma20"] > df["ma60"])
    df["ma_bearish"] = (df["ma5"] < df["ma20"]) & (df["ma20"] < df["ma60"])
    # 乖離率
    df["bias5"]  = (df[price_col] - df["ma5"])  / df["ma5"]  * 100
    df["bias20"] = (df[price_col] - df["ma20"]) / df["ma20"] * 100
    df["bias60"] = (df[price_col] - df["ma60"]) / df["ma60"] * 100
    return df


def add_bollinger(df: pd.DataFrame, price_col: str = "close", period: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = df[price_col].rolling(period).mean()
    sigma = df[price_col].rolling(period).std()
    df["bb_upper"] = mid + std * sigma
    df["bb_mid"]   = mid
    df["bb_lower"] = mid - std * sigma
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"]    = (df[price_col] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def add_rsi(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    for period in [6, 12, 24]:
        delta = df[price_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"rsi{period}"] = 100 - 100 / (1 + rs)
    return df


def add_kd(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    low_min  = df["min"].rolling(period).min()
    high_max = df["max"].rolling(period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    rsv = (df["close"] - low_min) / denom * 100
    df["k"] = rsv.ewm(com=2, adjust=False).mean()
    df["d"] = df["k"].ewm(com=2, adjust=False).mean()
    df["j"] = 3 * df["k"] - 2 * df["d"]
    return df


def add_macd(df: pd.DataFrame, price_col: str = "close",
             fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_volume_features(df: pd.DataFrame, vol_col: str = "Trading_Volume") -> pd.DataFrame:
    for n in [5, 20]:
        df[f"vol_ma{n}"] = df[vol_col].rolling(n).mean()
        df[f"vol_ratio{n}"] = df[vol_col] / df[f"vol_ma{n}"]
    df["turnover"] = df.get("Trading_turnover", pd.Series(dtype=float))
    # 量縮/量增
    df["vol_expand"] = df["vol_ratio5"] > 1.5
    df["vol_shrink"] = df["vol_ratio5"] < 0.5
    return df


def add_price_change(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    df["chg_pct"]   = df[price_col].pct_change() * 100
    df["chg_pct_2d"] = df[price_col].pct_change(2) * 100
    df["chg_pct_5d"] = df[price_col].pct_change(5) * 100
    df["high_low_range"] = (df["max"] - df["min"]) / df["min"] * 100
    # 上影線 / 下影線
    df["upper_shadow"] = (df["max"] - df[["open", "close"]].max(axis=1)) / df["min"] * 100
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["min"])  / df["min"] * 100
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """一次性計算所有技術指標，輸入需有 open/max/min/close/Trading_Volume 欄。"""
    df = df.copy().sort_values("date").reset_index(drop=True)
    df = add_moving_averages(df)
    df = add_bollinger(df)
    df = add_rsi(df)
    df = add_kd(df)
    df = add_macd(df)
    df = add_volume_features(df)
    df = add_price_change(df)
    return df
