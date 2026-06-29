"""星期效應與月份效應分析。

台股已知規律：
- 週一通常較弱（承接週末風險）
- 月初 / 月營收公布後常有反應
- 除息前後行為不同
"""

import os
import logging
from pathlib import Path

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

DAY_NAMES   = ["Mon", "Tue", "Wed", "Thu", "Fri"]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def compute_seasonality(daily_df: pd.DataFrame) -> dict:
    if daily_df.empty or "close" not in daily_df.columns:
        return {}

    df = daily_df.copy().sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    df["chg_pct"] = df["close"].pct_change() * 100
    df["weekday"] = df["date"].dt.dayofweek   # 0=Mon
    df["month"]   = df["date"].dt.month

    # 星期效應
    wd_stats = (
        df.groupby("weekday")["chg_pct"]
        .agg(mean="mean", std="std", win_rate=lambda x: (x > 0).mean())
        .round(4)
    )
    wd_stats.index = DAY_NAMES[:len(wd_stats)]

    # 月份效應
    mo_stats = (
        df.groupby("month")["chg_pct"]
        .agg(mean="mean", std="std", win_rate=lambda x: (x > 0).mean())
        .round(4)
    )
    mo_stats.index = [MONTH_NAMES[i-1] for i in mo_stats.index]

    # 月初（1–5日）vs 月底（26–31日）
    df["day_of_month"] = df["date"].dt.day
    early = df[df["day_of_month"] <= 5]["chg_pct"].mean()
    late  = df[df["day_of_month"] >= 26]["chg_pct"].mean()

    result = {
        "weekday_effect": wd_stats.to_dict(orient="index"),
        "month_effect":   mo_stats.to_dict(orient="index"),
        "month_early_avg_chg": round(float(early), 4),
        "month_late_avg_chg":  round(float(late), 4),
    }
    return result


def get_today_seasonality_score(daily_df: pd.DataFrame, date_str: str) -> float:
    """
    回傳今日的季節性偏分（-1 到 +1），供 scenario_builder 用。
    正值=歷史上今天漲多，負值=跌多。
    """
    stats = compute_seasonality(daily_df)
    if not stats:
        return 0.0

    dt = pd.Timestamp(date_str)
    wd_key = DAY_NAMES[dt.dayofweek] if dt.dayofweek < 5 else "Mon"
    mo_key = MONTH_NAMES[dt.month - 1]

    wd_mean = stats["weekday_effect"].get(wd_key, {}).get("mean", 0)
    mo_mean = stats["month_effect"].get(mo_key, {}).get("mean", 0)

    # 正規化到 -1~+1 (±1.5% 映射到 ±1)
    score = np.clip((wd_mean * 0.5 + mo_mean * 0.5) / 1.5, -1, 1)
    return round(float(score), 4)


def main():
    import json
    logging.basicConfig(level=logging.INFO)
    path = RAW_DIR / f"price_{TARGET}.parquet"
    if not path.exists():
        print("找不到日K資料，請先執行 fetch_historical")
        return
    df = pd.read_parquet(path)
    result = compute_seasonality(df)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
