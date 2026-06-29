"""WebSocket endpoint — 盤中即時推播。

連接後立即推送當前情境，之後在 10:30 / 12:00 / 13:00 推送更新。
"""

import asyncio
import json
import logging
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()

_connections: Set[WebSocket] = set()


async def broadcast(data: dict) -> None:
    """廣播給所有已連線的 WebSocket 客戶端。"""
    if not _connections:
        return
    message = json.dumps(data, ensure_ascii=False)
    dead = set()
    for ws in _connections:
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    _connections.difference_update(dead)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _connections.add(websocket)
    logger.info("WebSocket 連線：共 %d 個", len(_connections))

    try:
        # 立即推送當前狀態
        from fastapi import Request
        # 取 app.state 中的情境（透過 app 參考）
        app = websocket.app
        scenarios = getattr(app.state, "scenarios", {})
        xgb_pred  = getattr(app.state, "xgb_prediction", {})

        await websocket.send_text(json.dumps({
            "type": "init",
            "scenarios": scenarios,
            "xgb_prediction": xgb_pred,
        }, ensure_ascii=False))

        # 保持連線，等待 ping 或斷線
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if msg == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "heartbeat"}))

    except WebSocketDisconnect:
        logger.info("WebSocket 斷線")
    finally:
        _connections.discard(websocket)
