"""相關性分析：1815 vs 大盤、台積電、油價、匯率等。

用法：
    python -m src.features.correlation
輸出：
    data/features/correlation_report.json
    data/features/correlation_matrix.parquet
"""

import os
import json
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
TAIEX  = os.getenv("TAIEX_ID", "Y9999")
RELATED = os.getenv("RELATED_STOCKS", "2330,1447,1457,1467").split(",")


def _load_close(name: str, col_alias: str) -> pd.Series:
    path = RAW_DIR / f"price_{name}.parquet"
    if not path.exists():
        return pd.Series(dtype=float, name=col_alias)
    df = pd.read_parquet(path, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].rename(col_alias)


def build_master_df() -> pd.DataFrame:
    series = {}

    # 目標股
    s = _load_close(TARGET, f"stock_{TARGET}")
    if not s.empty:
        series[f"stock_{TARGET}"] = s

    # 大盤 (TAIEX close 欄名可能是 Trading_money，用 close)
    s = _load_close(TAIEX, "taiex")
    if not s.empty:
        series["taiex"] = s

    # 相關個股
    for sid in RELATED:
        s = _load_close(sid, f"stock_{sid}")
        if not s.empty:
            series[f"stock_{sid}"] = s

    # 外部數據
    ext_path = RAW_DIR / "external.parquet"
    if ext_path.exists():
        ext = pd.read_parquet(ext_path)
        ext["date"] = pd.to_datetime(ext["date"])
        ext = ext.set_index("date")
        for col in ["wti_close", "usd_twd", "vix_close", "sp500_close",
                    "wti_chg_pct", "usd_twd_chg", "sp500_chg_pct"]:
            if col in ext.columns:
                series[col] = ext[col]

    if not series:
        logger.warning("No data available for correlation analysis")
        return pd.DataFrame()

    master = pd.DataFrame(series).dropna(how="all")
    return master


def compute_correlations(master: pd.DataFrame, target_col: str) -> dict:
    if target_col not in master.columns or master.empty:
        return {}

    result = {}
    # 價格相關性（Pearson）
    price_corr = master.corr(method="pearson")[target_col].drop(target_col)

    # 漲跌幅相關性（更有意義）
    pct_df = master.pct_change().dropna()
    pct_corr = pct_df.corr(method="pearson")[target_col].drop(target_col)

    result["price_correlation"] = price_corr.round(4).to_dict()
    result["return_correlation"] = pct_corr.round(4).to_dict()

    # 滯後相關（外部數據前一天對今天的影響）
    lag_corrs = {}
    target_ret = pct_df[target_col]
    for col in pct_df.columns:
        if col == target_col:
            continue
        lagged = pct_df[col].shift(1)
        valid = pd.concat([target_ret, lagged], axis=1).dropna()
        if len(valid) > 20:
            lag_corrs[col] = round(float(valid.corr().iloc[0, 1]), 4)
    result["lag1_correlation"] = lag_corrs

    return result


def main():
    logger.info("建立相關性分析...")
    master = build_master_df()
    if master.empty:
        logger.error("無可用數據，請先執行 fetch_historical")
        return

    target_col = f"stock_{TARGET}"
    corr_result = compute_correlations(master, target_col)

    # 存相關性矩陣
    pct_df = master.pct_change().dropna()
    corr_matrix = pct_df.corr().round(4)
    corr_matrix.to_parquet(FEAT_DIR / "correlation_matrix.parquet")

    # 輸出 JSON 報告
    report_path = FEAT_DIR / "correlation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(corr_result, f, ensure_ascii=False, indent=2)

    logger.info("相關性報告已存：%s", report_path)

    # 列印重點
    ret_corr = corr_result.get("return_correlation", {})
    lag_corr = corr_result.get("lag1_correlation", {})
    print("\n=== 1815 漲跌幅相關性（同日）===")
    for k, v in sorted(ret_corr.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {k:30s}: {v:+.4f}")
    print("\n=== 前一日對 1815 的滯後影響 ===")
    for k, v in sorted(lag_corr.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {k:30s}: {v:+.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
