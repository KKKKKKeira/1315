"""FastAPI 主程式。

路由：
  GET  /api/prediction/today          → 今日走勢情境
  GET  /api/prediction/price_levels   → 支撐壓力區間
  GET  /api/price_history             → 歷史日K
  GET  /api/chip                      → 籌碼面板數據
  GET  /api/correlation               → 相關性矩陣
  WS   /ws                            → 盤中即時更新
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api.routes import prediction, realtime
from src.data.ws_client import tick_queue, run as ws_run
from src.model import intraday_updater

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時：載入今日情境，設定基準價
    from src.model.scenario_builder import build_scenarios
    from src.model.price_predictor import predict as xgb_predict

    scenarios = build_scenarios()
    if scenarios:
        current = scenarios.get("current_price", 0)
        intraday_updater.set_base_prices(current, current)
        app.state.scenarios = scenarios
        app.state.xgb_prediction = xgb_predict()
    else:
        app.state.scenarios = {}
        app.state.xgb_prediction = {}

    # 背景任務：盤中更新循環
    updater_task = asyncio.create_task(
        intraday_updater.consume_ticks(
            tick_queue,
            app.state.scenarios,
            broadcast_fn=realtime.broadcast,
        )
    )

    yield

    updater_task.cancel()


app = FastAPI(title="1815 富喬 走勢分析", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router, prefix="/api")
app.include_router(realtime.router)

# 直接回傳 dashboard.html（不需要 npm build）
@app.get("/", response_class=FileResponse)
async def serve_dashboard():
    return FileResponse("dashboard.html")
