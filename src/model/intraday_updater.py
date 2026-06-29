"""盤中滾動更新模組。

從 ws_client.tick_queue 消費即時 Tick，
每分鐘計算滾動指標，並在三個關鍵時間點（10:30 / 12:00 / 13:00）
重算 Bayesian 機率，更新情境並發布給 WebSocket server。
"""

import asyncio
import logging
from datetime import datetime, time as dtime
from collections import deque
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# 更新時間點
UPDATE_TIMES = [dtime(10, 30), dtime(12, 0), dtime(13, 0)]

# 歷史 Tick 滾動窗口（分鐘）
_tick_window: deque = deque(maxlen=5000)

# 開盤基準（由外部在 09:00 設定）
_open_price: Optional[float] = None
_prev_close: Optional[float] = None


def set_base_prices(open_price: float, prev_close: float) -> None:
    global _open_price, _prev_close
    _open_price = open_price
    _prev_close = prev_close


def _compute_intraday_features() -> dict:
    """計算當前盤中滾動特徵。"""
    if not _tick_window:
        return {}

    ticks = list(_tick_window)
    df = pd.DataFrame(ticks)

    if "deal_price" not in df.columns:
        return {}

    prices = df["deal_price"].astype(float)
    vols   = df.get("volume", pd.Series([1] * len(df))).astype(float)

    # VWAP
    vwap = float((prices * vols).sum() / vols.sum()) if vols.sum() > 0 else float(prices.mean())

    current_price = float(prices.iloc[-1])
    cum_volume = int(vols.sum())

    # 買賣力道比（如果有 buy_tick_type 欄）
    if "buy_tick_type" in df.columns:
        buy_vol  = float(vols[df["buy_tick_type"] == "B"].sum())
        sell_vol = float(vols[df["buy_tick_type"] == "S"].sum())
        buy_ratio = buy_vol / (buy_vol + sell_vol + 1e-9)
    else:
        buy_ratio = 0.5

    # 相對前日收盤
    prev_close_chg = ((current_price - _prev_close) / _prev_close * 100) if _prev_close else 0.0

    # 相對開盤
    open_chg = ((current_price - _open_price) / _open_price * 100) if _open_price else 0.0

    # VWAP 偏離
    vwap_deviation = ((current_price - vwap) / vwap * 100) if vwap else 0.0

    # 最近10分鐘趨勢（用最近幾百筆近似）
    recent = prices.tail(min(300, len(prices)))
    trend = float(np.polyfit(range(len(recent)), recent, 1)[0]) if len(recent) > 5 else 0.0

    return {
        "current_price":    round(current_price, 2),
        "vwap":             round(vwap, 2),
        "cum_volume":       cum_volume,
        "buy_ratio":        round(buy_ratio, 4),
        "prev_close_chg":   round(prev_close_chg, 4),
        "open_chg":         round(open_chg, 4),
        "vwap_deviation":   round(vwap_deviation, 4),
        "price_trend":      round(trend, 6),
    }


def _adjust_probabilities(base_probs: dict, intraday_feats: dict) -> dict:
    """
    根據盤中即時特徵微調早盤機率。
    策略：buy_ratio 和 vwap_deviation 是主要訊號。
    """
    if not base_probs or not intraday_feats:
        return base_probs

    probs = dict(base_probs)
    buy_ratio = intraday_feats.get("buy_ratio", 0.5)
    vwap_dev  = intraday_feats.get("vwap_deviation", 0.0)
    open_chg  = intraday_feats.get("open_chg", 0.0)

    # buy_ratio > 0.6 → 偏多
    bull_boost = max(0, (buy_ratio - 0.5) * 0.3)
    bear_boost = max(0, (0.5 - buy_ratio) * 0.3)

    # VWAP 偏離 → 均值回歸訊號（偏高容易往下，偏低容易往上）
    if vwap_dev > 1.0:
        bear_boost += min(vwap_dev * 0.02, 0.05)
    elif vwap_dev < -1.0:
        bull_boost += min(abs(vwap_dev) * 0.02, 0.05)

    probs["bull"]         = probs.get("bull", 0) + bull_boost
    probs["strong_bull"]  = probs.get("strong_bull", 0) + bull_boost * 0.5
    probs["bear"]         = probs.get("bear", 0) + bear_boost
    probs["strong_bear"]  = probs.get("strong_bear", 0) + bear_boost * 0.5

    total = sum(probs.values())
    return {k: round(v / total, 4) for k, v in probs.items()}


async def consume_ticks(tick_queue: asyncio.Queue, base_scenarios: dict,
                        broadcast_fn=None) -> None:
    """
    主循環：持續消費 tick_queue，定時計算並廣播更新。
    broadcast_fn(update_dict) 由 FastAPI WebSocket handler 傳入。
    """
    last_update_time = None

    while True:
        # 非阻塞地清空佇列
        try:
            while True:
                tick = tick_queue.get_nowait()
                _tick_window.append(tick)
        except asyncio.QueueEmpty:
            pass

        now = datetime.now().time()

        # 在三個關鍵時間點廣播更新
        for update_t in UPDATE_TIMES:
            if (now >= update_t and
                    (last_update_time is None or last_update_time < update_t)):
                feats = _compute_intraday_features()
                if feats and base_scenarios:
                    new_probs = _adjust_probabilities(
                        base_scenarios.get("all_probs", {}), feats
                    )
                    update = {
                        "type": "intraday_update",
                        "time": datetime.now().isoformat(),
                        "intraday_features": feats,
                        "updated_probs": new_probs,
                        "base_scenarios": base_scenarios.get("scenarios", []),
                    }
                    logger.info("盤中更新 %s: %s", update_t, new_probs)
                    if broadcast_fn:
                        await broadcast_fn(update)
                last_update_time = update_t

        await asyncio.sleep(30)
