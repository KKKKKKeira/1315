import { useEffect, useState } from "react";
import ScenarioChart from "./components/ScenarioChart.jsx";
import ChipPanel from "./components/ChipPanel.jsx";
import PriceRangePanel from "./components/PriceRangePanel.jsx";
import IntradayPanel from "./components/IntradayPanel.jsx";
import { useWebSocket } from "./hooks/useWebSocket.js";

const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;

export default function App() {
  const [prediction, setPrediction] = useState(null);
  const [priceHistory, setPriceHistory] = useState([]);
  const [chipData, setChipData] = useState({});
  const [priceLevels, setPriceLevels] = useState(null);
  const [intradayUpdate, setIntradayUpdate] = useState(null);
  const [lastUpdateTime, setLastUpdateTime] = useState(null);

  const { lastMessage, status } = useWebSocket(WS_URL);

  // 初始載入
  useEffect(() => {
    Promise.all([
      fetch("/api/prediction/today").then((r) => r.json()),
      fetch("/api/price_history?days=120").then((r) => r.json()),
      fetch("/api/chip?days=60").then((r) => r.json()),
      fetch("/api/prediction/price_levels").then((r) => r.json()),
    ]).then(([pred, hist, chip, levels]) => {
      setPrediction(pred);
      setPriceHistory(hist);
      setChipData(chip);
      setPriceLevels(levels);
    }).catch(console.error);
  }, []);

  // WebSocket 訊息處理
  useEffect(() => {
    if (!lastMessage) return;
    if (lastMessage.type === "init") {
      setPrediction({
        scenarios: lastMessage.scenarios,
        xgb_prediction: lastMessage.xgb_prediction,
      });
    } else if (lastMessage.type === "intraday_update") {
      setIntradayUpdate(lastMessage);
      setLastUpdateTime(new Date(lastMessage.time).toLocaleTimeString("zh-TW"));
    }
  }, [lastMessage]);

  const today = new Date().toLocaleDateString("zh-TW", {
    year: "numeric", month: "long", day: "numeric", weekday: "short",
  });

  return (
    <div className="min-h-screen p-4 max-w-7xl mx-auto">
      {/* 標題列 */}
      <header className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white">1815 富喬工業</h1>
          <p className="text-gray-400 text-sm">{today}</p>
        </div>
        <div className="flex items-center gap-3">
          {lastUpdateTime && (
            <span className="text-xs text-yellow-400">最後更新 {lastUpdateTime}</span>
          )}
          <span className={`px-2 py-1 rounded text-xs font-semibold ${
            status === "connected" ? "bg-green-900 text-green-300" :
            status === "connecting" ? "bg-yellow-900 text-yellow-300" :
            "bg-red-900 text-red-300"
          }`}>
            {status === "connected" ? "即時連線" : status === "connecting" ? "連線中..." : "已斷線"}
          </span>
        </div>
      </header>

      {/* 主要走勢圖 + 情境 */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 mb-4">
        <div className="xl:col-span-2">
          <ScenarioChart
            priceHistory={priceHistory}
            prediction={prediction}
            priceLevels={priceLevels}
          />
        </div>
        <div className="space-y-4">
          <PriceRangePanel levels={priceLevels} prediction={prediction?.xgb_prediction} />
          {intradayUpdate && <IntradayPanel update={intradayUpdate} />}
        </div>
      </div>

      {/* 籌碼面板 */}
      <ChipPanel data={chipData} />
    </div>
  );
}
