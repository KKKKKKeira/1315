"""回測模組：驗證 XGBoost 預測模型與 Bayesian 情境準確率。

用法：
    python -m src.model.backtest
"""

import os
import json
import logging
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from src.model.scenario_builder import (_load_master_features, SCENARIOS, _discretize_features,
                                        _compute_bayesian_probs, _classify_scenario)

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR   = Path(os.getenv("DATA_DIR", "./data"))
FEAT_DIR   = DATA_DIR / "features"
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
FEAT_DIR.mkdir(parents=True, exist_ok=True)



def run_backtest(lookback_days: int = 90) -> dict:
    logger.info("開始回測（最近 %d 天）...", lookback_days)

    df = _load_master_features()
    if df.empty or len(df) < lookback_days + 30:
        logger.error("資料不足，無法回測")
        return {}

    # 用前段訓練，後段測試
    cutoff = len(df) - lookback_days
    train_df = df.iloc[:cutoff].copy()
    test_df  = df.iloc[cutoff:].copy()

    # ── Bayesian 情境準確率 ──────────────────────────────────────────────────
    correct_top1 = 0
    correct_top3 = 0
    scenario_actual = []

    for _, row in test_df.iterrows():
        today_bins = _discretize_features(row)
        probs = _compute_bayesian_probs(train_df, today_bins)
        top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        actual_sc = _classify_scenario(float(row.get("next_chg_pct", 0)))

        if top3[0][0] == actual_sc:
            correct_top1 += 1
        if any(sc == actual_sc for sc, _ in top3):
            correct_top3 += 1
        scenario_actual.append({"predicted": top3[0][0], "actual": actual_sc})

    n = len(test_df)
    bayesian_results = {
        "top1_accuracy": round(correct_top1 / n, 4),
        "top3_accuracy": round(correct_top3 / n, 4),
        "n_test": n,
    }

    # ── XGBoost 預測誤差 ─────────────────────────────────────────────────────
    xgb_results = {}
    try:
        import joblib
        from src.model.price_predictor import _prepare_data, TARGETS

        feat_path = MODELS_DIR / "feature_columns.json"
        if feat_path.exists():
            with open(feat_path) as f:
                saved_cols = json.load(f)

            X_test, y_test = _prepare_data(test_df)
            for col in saved_cols:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[saved_cols]

            for target_name, target_col in TARGETS.items():
                model_path = MODELS_DIR / f"xgb_{target_name}.joblib"
                if not model_path.exists() or target_col not in y_test.columns:
                    continue
                model = joblib.load(model_path)
                preds = model.predict(X_test)
                actual = y_test[target_col].values
                valid = ~np.isnan(actual)
                mae = float(np.mean(np.abs(preds[valid] - actual[valid])))
                xgb_results[target_name] = {"mae_pct": round(mae, 4)}
    except Exception as e:
        logger.warning("XGBoost 回測失敗: %s", e)

    # ── 勝率統計 ──────────────────────────────────────────────────────────────
    actual_chgs = test_df["next_chg_pct"].dropna()
    win_rate = float((actual_chgs > 0).mean())
    avg_gain = float(actual_chgs[actual_chgs > 0].mean()) if (actual_chgs > 0).any() else 0
    avg_loss = float(actual_chgs[actual_chgs < 0].mean()) if (actual_chgs < 0).any() else 0

    result = {
        "period": f"最近 {lookback_days} 個交易日",
        "bayesian_accuracy": bayesian_results,
        "xgboost_mae": xgb_results,
        "base_stats": {
            "win_rate": round(win_rate, 4),
            "avg_gain_pct": round(avg_gain, 4),
            "avg_loss_pct": round(avg_loss, 4),
            "profit_factor": round(avg_gain / abs(avg_loss), 2) if avg_loss else 0,
        },
    }

    out_path = FEAT_DIR / f"backtest_{date.today().isoformat()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("回測完成：%s", result)
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_backtest()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
