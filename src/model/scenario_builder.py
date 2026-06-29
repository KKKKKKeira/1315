"""Bayesian 條件機率走勢情境產生器。

每日開盤前：
  1. 讀取昨日所有特徵
  2. 查歷史條件機率表
  3. 輸出 3 個情境（多/盤整/空）+ 各自的機率與目標價位
  4. 儲存到 data/features/scenarios_YYYY-MM-DD.json

用法：
    python -m src.model.scenario_builder
"""

import os
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from src.features.technical import build_features
from src.features.seasonality import get_today_seasonality_score

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
RAW_DIR  = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = os.getenv("TARGET_STOCK", "1815")
TAIEX  = os.getenv("TAIEX_ID", "Y9999")

# 情境定義（以漲跌幅分桶）
SCENARIOS = {
    "strong_bull":  (2.0, 99),
    "bull":         (0.5, 2.0),
    "flat":         (-0.5, 0.5),
    "bear":         (-2.0, -0.5),
    "strong_bear":  (-99, -2.0),
}

SCENARIO_LABELS = {
    "strong_bull":  "強勢上漲（>+2%）",
    "bull":         "溫和上漲（+0.5%~+2%）",
    "flat":         "盤整（-0.5%~+0.5%）",
    "bear":         "溫和下跌（-0.5%~-2%）",
    "strong_bear":  "強勢下跌（<-2%）",
}


# ── 特徵離散化（Bayesian 分桶）────────────────────────────────────────────────

def _discretize_features(row: pd.Series) -> dict[str, str]:
    """將連續特徵轉為離散標籤，供條件機率計算用。"""
    bins = {}

    # RSI 狀態
    rsi = row.get("rsi12", 50)
    bins["rsi_zone"] = "oversold" if rsi < 35 else ("overbought" if rsi > 65 else "normal")

    # 大盤前日漲跌
    taiex_chg = row.get("taiex_chg_pct", 0)
    bins["taiex_dir"] = "up" if taiex_chg > 0.5 else ("down" if taiex_chg < -0.5 else "flat")

    # 三大法人昨日合計
    inst_net = row.get("total_inst_net", 0)
    bins["inst_dir"] = "buy" if inst_net > 0 else ("sell" if inst_net < 0 else "neutral")

    # 融資變化
    margin_chg = row.get("margin_change", 0)
    bins["margin"] = "up" if margin_chg > 0 else ("down" if margin_chg < 0 else "flat")

    # 油價前日變動
    wti_chg = row.get("wti_chg_pct", 0)
    bins["oil"] = "up" if wti_chg > 1 else ("down" if wti_chg < -1 else "flat")

    # 均線多空排列
    ma_bull = row.get("ma_bullish", False)
    ma_bear = row.get("ma_bearish", False)
    bins["ma_align"] = "bull" if ma_bull else ("bear" if ma_bear else "mixed")

    # 成交量比（5日）
    vol_r5 = row.get("vol_ratio5", 1.0)
    bins["vol"] = "high" if vol_r5 > 1.3 else ("low" if vol_r5 < 0.7 else "normal")

    # 美股前晚
    sp500_chg = row.get("sp500_chg_pct", 0)
    bins["us_dir"] = "up" if sp500_chg > 0.5 else ("down" if sp500_chg < -0.5 else "flat")

    return bins


# ── Bayesian 條件機率計算 ─────────────────────────────────────────────────────

def _compute_bayesian_probs(history_df: pd.DataFrame, today_bins: dict[str, str]) -> dict[str, float]:
    """
    Naive Bayes：P(scenario | features) ∝ P(scenario) × ∏ P(feature_i | scenario)
    """
    if history_df.empty:
        # 無歷史資料時使用均勻分布
        n = len(SCENARIOS)
        return {k: 1/n for k in SCENARIOS}

    # 明日漲跌幅（目標變數，已在 history_df 中 shift）
    priors = {}
    for sc, (lo, hi) in SCENARIOS.items():
        mask = (history_df["next_chg_pct"] > lo) & (history_df["next_chg_pct"] <= hi)
        priors[sc] = mask.mean()

    posteriors = {}
    for sc in SCENARIOS:
        log_prob = np.log(priors[sc] + 1e-9)
        mask_sc = _scenario_mask(history_df, sc)
        sc_subset = history_df[mask_sc]
        for feat, val in today_bins.items():
            if feat not in history_df.columns:
                continue
            if len(sc_subset) == 0:
                cond_p = 0.2
            else:
                cond_p = (sc_subset[feat] == val).mean()
            log_prob += np.log(cond_p + 1e-9)
        posteriors[sc] = np.exp(log_prob)

    total = sum(posteriors.values())
    return {k: round(v / total, 4) for k, v in posteriors.items()}


def _classify_scenario(chg: float) -> str:
    for sc, (lo, hi) in SCENARIOS.items():
        if lo < chg <= hi:
            return sc
    return "flat"


def _scenario_mask(df: pd.DataFrame, scenario: str) -> pd.Series:
    lo, hi = SCENARIOS[scenario]
    return (df["next_chg_pct"] > lo) & (df["next_chg_pct"] <= hi)


# ── 目標價位計算 ──────────────────────────────────────────────────────────────

def _compute_targets(history_df: pd.DataFrame, current_price: float, scenario: str) -> dict:
    """從歷史同情境計算統計高低點目標。"""
    if history_df.empty:
        chg = {"strong_bull": 3.0, "bull": 1.2, "flat": 0.0, "bear": -1.2, "strong_bear": -3.0}
        c = chg.get(scenario, 0)
        return {
            "high_target": round(current_price * (1 + (c + 1) / 100), 2),
            "low_target":  round(current_price * (1 + (c - 1) / 100), 2),
            "close_target": round(current_price * (1 + c / 100), 2),
        }

    mask = _scenario_mask(history_df, scenario)
    subset = history_df[mask]
    if len(subset) < 5:
        subset = history_df

    high_mean  = subset.get("next_high_chg", subset.get("next_chg_pct", pd.Series([0]))).mean()
    low_mean   = subset.get("next_low_chg",  subset.get("next_chg_pct", pd.Series([0]))).mean()
    close_mean = subset["next_chg_pct"].mean()

    return {
        "high_target":  round(current_price * (1 + high_mean / 100), 2),
        "low_target":   round(current_price * (1 + low_mean / 100), 2),
        "close_target": round(current_price * (1 + close_mean / 100), 2),
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def _load_master_features() -> pd.DataFrame:
    """合併所有來源特徵成訓練用 DataFrame。"""
    price_path = RAW_DIR / f"price_{TARGET}.parquet"
    if not price_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(price_path)
    df = build_features(df)

    # 加入大盤漲跌
    taiex_path = RAW_DIR / f"price_{TAIEX}.parquet"
    if taiex_path.exists():
        taiex = pd.read_parquet(taiex_path, columns=["date", "close"])
        taiex["date"] = pd.to_datetime(taiex["date"])
        taiex["taiex_chg_pct"] = taiex["close"].pct_change() * 100
        df = df.merge(taiex[["date", "taiex_chg_pct"]], on="date", how="left")

    # 加入三大法人
    inst_path = RAW_DIR / f"institutional_{TARGET}.parquet"
    if inst_path.exists():
        inst = pd.read_parquet(inst_path)
        inst["date"] = pd.to_datetime(inst["date"])
        net_cols = [c for c in inst.columns if c.startswith("net_")]
        inst["total_inst_net"] = inst[net_cols].sum(axis=1)
        df = df.merge(inst[["date", "total_inst_net"]], on="date", how="left")

    # 加入融資融券
    margin_path = RAW_DIR / f"margin_{TARGET}.parquet"
    if margin_path.exists():
        margin = pd.read_parquet(margin_path, columns=["date", "margin_change"])
        margin["date"] = pd.to_datetime(margin["date"])
        df = df.merge(margin, on="date", how="left")

    # 加入外部數據
    ext_path = RAW_DIR / "external.parquet"
    if ext_path.exists():
        ext = pd.read_parquet(ext_path)
        ext["date"] = pd.to_datetime(ext["date"])
        ext_cols = ["date", "wti_chg_pct", "usd_twd_chg", "sp500_chg_pct", "vix_close"]
        ext_cols = [c for c in ext_cols if c in ext.columns]
        df = df.merge(ext[ext_cols], on="date", how="left")

    # 目標變數：明日漲跌幅
    df["next_chg_pct"]  = df["chg_pct"].shift(-1)
    df["next_high_chg"] = (df["max"].shift(-1) - df["close"]) / df["close"] * 100
    df["next_low_chg"]  = (df["min"].shift(-1) - df["close"]) / df["close"] * 100

    # 離散化特徵列
    for feat in ["rsi_zone", "taiex_dir", "inst_dir", "margin", "oil", "ma_align", "vol", "us_dir"]:
        df[feat] = pd.NA

    for i, row in df.iterrows():
        bins = _discretize_features(row)
        for feat, val in bins.items():
            df.at[i, feat] = val

    return df.dropna(subset=["next_chg_pct"]).reset_index(drop=True)


def build_scenarios(target_date: Optional[str] = None) -> dict:
    target_date = target_date or date.today().isoformat()
    logger.info("產生 %s 走勢情境...", target_date)

    history_df = _load_master_features()
    if history_df.empty:
        logger.error("無歷史特徵，無法計算情境")
        return {}

    # 取最新一筆作為今日輸入
    latest = history_df.iloc[-1]
    current_price = float(latest["close"])
    today_bins = _discretize_features(latest)

    # 加入季節性分數
    season_score = get_today_seasonality_score(
        pd.read_parquet(RAW_DIR / f"price_{TARGET}.parquet"),
        target_date
    )

    probs = _compute_bayesian_probs(history_df, today_bins)

    # 季節性微調
    if season_score > 0.2:
        boost = season_score * 0.05
        probs["bull"] = probs.get("bull", 0) + boost
        probs["strong_bull"] = probs.get("strong_bull", 0) + boost / 2
    elif season_score < -0.2:
        boost = abs(season_score) * 0.05
        probs["bear"] = probs.get("bear", 0) + boost
        probs["strong_bear"] = probs.get("strong_bear", 0) + boost / 2
    total = sum(probs.values())
    probs = {k: round(v / total, 4) for k, v in probs.items()}

    # 排序並取前 3 情境
    top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]

    scenarios_out = []
    for sc_key, prob in top3:
        targets = _compute_targets(history_df, current_price, sc_key)
        scenarios_out.append({
            "key":         sc_key,
            "label":       SCENARIO_LABELS[sc_key],
            "probability": prob,
            "current_price": current_price,
            **targets,
        })

    result = {
        "date":          target_date,
        "current_price": current_price,
        "scenarios":     scenarios_out,
        "all_probs":     probs,
        "input_features": today_bins,
        "seasonality_score": season_score,
    }

    out_path = FEAT_DIR / f"scenarios_{target_date}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("情境已存：%s", out_path)

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = build_scenarios()
    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import json
    main()
