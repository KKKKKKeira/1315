import { useEffect, useRef, useState, useCallback } from "react";

export function useWebSocket(url) {
  const ws = useRef(null);
  const [lastMessage, setLastMessage] = useState(null);
  const [status, setStatus] = useState("connecting");
  const reconnectTimer = useRef(null);
  const reconnectDelay = useRef(1000);

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return;

    const socket = new WebSocket(url);
    ws.current = socket;
    setStatus("connecting");

    socket.onopen = () => {
      setStatus("connected");
      reconnectDelay.current = 1000;
      // 定時 ping
      const pingTimer = setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) {
          socket.send("ping");
        }
      }, 25000);
      socket._pingTimer = pingTimer;
    };

    socket.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setLastMessage(data);
      } catch {}
    };

    socket.onclose = () => {
      setStatus("disconnected");
      clearInterval(socket._pingTimer);
      reconnectTimer.current = setTimeout(() => {
        reconnectDelay.current = Math.min(reconnectDelay.current * 2, 30000);
        connect();
      }, reconnectDelay.current);
    };

    socket.onerror = () => {
      socket.close();
    };
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      ws.current?.close();
    };
  }, [connect]);

  return { lastMessage, status };
}
