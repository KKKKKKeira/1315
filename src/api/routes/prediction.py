"""預測相關 REST endpoints。"""

import json
import os
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from src.features.price_range import get_price_levels
from src.features.correlation import build_master_df, compute_correlations

router = APIRouter()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
RAW_DIR  = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
TARGET   = os.getenv("TARGET_STOCK", "1815")


@router.get("/prediction/today")
async def get_today_prediction(request: Request):
    """今日走勢情境（3個，依機率排序）+ XGBoost 高低點預測。"""
    scenarios = getattr(request.app.state, "scenarios", {})
    xgb_pred  = getattr(request.app.state, "xgb_prediction", {})

    if not scenarios:
        # 嘗試從檔案讀取當日情境
        path = FEAT_DIR / f"scenarios_{date.today().isoformat()}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                scenarios = json.load(f)

    return {
        "scenarios": scenarios,
        "xgb_prediction": xgb_pred,
        "date": date.today().isoformat(),
    }


@router.get("/prediction/price_levels")
async def get_price_levels_api(current_price: Optional[float] = None):
    """支撐壓力區間 + Volume Profile。"""
    try:
        levels = get_price_levels(current_price)
        return levels
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/price_history")
async def get_price_history(days: int = 120):
    """歷史日K，供前端走勢圖使用。"""
    path = RAW_DIR / f"price_{TARGET}.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="歷史資料尚未下載")
    df = pd.read_parquet(path).tail(days)
    df["date"] = df["date"].astype(str)
    return df[["date", "open", "max", "min", "close", "Trading_Volume"]].to_dict(orient="records")


@router.get("/chip")
async def get_chip_data(days: int = 60):
    """籌碼面板：三大法人 + 融資融券 + 集保。"""
    result = {}

    inst_path = RAW_DIR / f"institutional_{TARGET}.parquet"
    if inst_path.exists():
        df = pd.read_parquet(inst_path).tail(days)
        df["date"] = df["date"].astype(str)
        result["institutional"] = df.to_dict(orient="records")

    margin_path = RAW_DIR / f"margin_{TARGET}.parquet"
    if margin_path.exists():
        df = pd.read_parquet(margin_path).tail(days)
        df["date"] = df["date"].astype(str)
        result["margin"] = df.to_dict(orient="records")

    share_path = RAW_DIR / f"shareholding_{TARGET}.parquet"
    if share_path.exists():
        df = pd.read_parquet(share_path).tail(30)
        df["date"] = df["date"].astype(str)
        result["shareholding"] = df.to_dict(orient="records")

    rev_path = RAW_DIR / f"revenue_{TARGET}.parquet"
    if rev_path.exists():
        df = pd.read_parquet(rev_path).tail(12)
        df["date"] = df["date"].astype(str)
        result["monthly_revenue"] = df[["date", "revenue", "revenue_yoy", "revenue_mom"]].to_dict(orient="records")

    return result


@router.get("/correlation")
async def get_correlation():
    """相關性矩陣（漲跌幅相關）。"""
    path = FEAT_DIR / "correlation_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="請先執行相關性分析")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/backtest/latest")
async def get_backtest():
    """最新回測結果。"""
    files = sorted(FEAT_DIR.glob("backtest_*.json"), reverse=True)
    if not files:
        raise HTTPException(status_code=404, detail="尚無回測結果")
    with open(files[0], encoding="utf-8") as f:
        return json.load(f)
