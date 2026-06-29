"""XGBoost 模型：預測當日最高點、最低點、收盤漲跌幅。

用法：
    python -m src.model.price_predictor --train   # 訓練並儲存模型
    python -m src.model.price_predictor --predict # 預測今日
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from dotenv import load_dotenv

from src.model.scenario_builder import _load_master_features

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR   = Path(os.getenv("DATA_DIR", "./data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
FEAT_DIR   = DATA_DIR / "features"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "chg_pct", "chg_pct_2d", "chg_pct_5d",
    "rsi6", "rsi12", "rsi24",
    "k", "d", "j",
    "macd", "macd_hist",
    "bb_pct", "bb_width",
    "bias5", "bias20", "bias60",
    "vol_ratio5", "vol_ratio20",
    "high_low_range", "upper_shadow", "lower_shadow",
    "ma_bullish", "ma_bearish",
    "taiex_chg_pct",
    "total_inst_net",
    "margin_change",
    "wti_chg_pct",
    "usd_twd_chg",
    "sp500_chg_pct",
    "vix_close",
]

TARGETS = {
    "close_chg": "next_chg_pct",
    "high_chg":  "next_high_chg",
    "low_chg":   "next_low_chg",
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


def _prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feats].copy()
    X["weekday"] = pd.to_datetime(df["date"]).dt.dayofweek
    X["month"]   = pd.to_datetime(df["date"]).dt.month
    X = X.fillna(0)
    y = df[[v for v in TARGETS.values() if v in df.columns]].copy()
    return X, y


def train() -> dict:
    logger.info("載入特徵資料...")
    df = _load_master_features()
    if df.empty or len(df) < 60:
        logger.error("資料不足（需至少60天），無法訓練")
        return {}

    X, y = _prepare_data(df)
    metrics = {}
    tscv = TimeSeriesSplit(n_splits=5)

    for target_name, target_col in TARGETS.items():
        if target_col not in y.columns:
            continue

        y_t = y[target_col]
        valid_idx = y_t.notna()
        X_t = X[valid_idx]
        y_t = y_t[valid_idx]

        # 時序交叉驗證評估
        maes = []
        for train_idx, val_idx in tscv.split(X_t):
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X_t.iloc[train_idx], y_t.iloc[train_idx], verbose=False)
            pred = model.predict(X_t.iloc[val_idx])
            maes.append(mean_absolute_error(y_t.iloc[val_idx], pred))

        # 全量訓練
        final_model = XGBRegressor(**XGB_PARAMS)
        final_model.fit(X_t, y_t, verbose=False)

        model_path = MODELS_DIR / f"xgb_{target_name}.joblib"
        joblib.dump(final_model, model_path)

        mae_avg = float(np.mean(maes))
        metrics[target_name] = {"cv_mae_pct": round(mae_avg, 4), "n_samples": len(X_t)}
        logger.info("模型 %s: CV MAE = %.4f%%", target_name, mae_avg)

    # 儲存特徵列名
    feat_path = MODELS_DIR / "feature_columns.json"
    with open(feat_path, "w") as f:
        json.dump(list(X.columns), f)

    logger.info("訓練完成：%s", metrics)
    return metrics


def predict(current_features: dict | None = None) -> dict:
    """
    預測今日高低點與收盤漲跌。
    current_features 可覆蓋最新數據（盤中更新使用）。
    """
    df = _load_master_features()
    if df.empty:
        return {}

    X, _ = _prepare_data(df)

    feat_path = MODELS_DIR / "feature_columns.json"
    if feat_path.exists():
        with open(feat_path) as f:
            saved_cols = json.load(f)
        for col in saved_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[saved_cols]

    # 使用最新一筆
    latest_X = X.iloc[[-1]].copy()
    if current_features:
        for col, val in current_features.items():
            if col in latest_X.columns:
                latest_X[col] = val

    results = {}
    for target_name in TARGETS:
        model_path = MODELS_DIR / f"xgb_{target_name}.joblib"
        if not model_path.exists():
            logger.warning("模型 %s 不存在，請先執行 --train", target_name)
            continue
        model = joblib.load(model_path)
        pred = float(model.predict(latest_X)[0])
        results[target_name] = round(pred, 4)

    current_price = float(df["close"].iloc[-1])
    results["current_price"] = current_price
    if "close_chg" in results:
        results["predicted_close"] = round(current_price * (1 + results["close_chg"] / 100), 2)
    if "high_chg" in results:
        results["predicted_high"] = round(current_price * (1 + results["high_chg"] / 100), 2)
    if "low_chg" in results:
        results["predicted_low"] = round(current_price * (1 + results["low_chg"] / 100), 2)

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    if args.train:
        metrics = train()
        print(json.dumps(metrics, indent=2))
    if args.predict:
        result = predict()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import json
    main()
