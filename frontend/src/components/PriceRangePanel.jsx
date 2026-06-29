export default function PriceRangePanel({ levels, prediction }) {
  if (!levels) {
    return (
      <div className="bg-gray-900 rounded-xl p-4">
        <h2 className="text-lg font-semibold text-white mb-2">支撐壓力區間</h2>
        <p className="text-gray-500 text-sm">載入中...</p>
      </div>
    );
  }

  const rows = [
    { label: "強壓力", value: levels.strong_resistance, color: "text-red-400" },
    { label: "弱壓力", value: levels.weak_resistance,   color: "text-red-300" },
    { label: "現價",   value: levels.current_price,     color: "text-yellow-400", bold: true },
    { label: "建議進場", value: levels.entry_suggest,   color: "text-yellow-300" },
    { label: "弱支撐", value: levels.weak_support,      color: "text-green-300" },
    { label: "強支撐", value: levels.strong_support,    color: "text-green-400" },
  ];

  return (
    <div className="bg-gray-900 rounded-xl p-4">
      <h2 className="text-lg font-semibold text-white mb-3">支撐壓力區間</h2>

      <div className="space-y-1 mb-4">
        {rows.map((r) => (
          <div key={r.label} className="flex justify-between items-center py-1 border-b border-gray-800">
            <span className="text-sm text-gray-400">{r.label}</span>
            <span className={`font-mono font-semibold ${r.color} ${r.bold ? "text-lg" : ""}`}>
              {r.value?.toFixed(2) ?? "—"}
            </span>
          </div>
        ))}
      </div>

      {/* XGBoost 預測 */}
      {prediction && (
        <div className="bg-gray-800 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-2 font-semibold">XGBoost 今日預測</p>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <p className="text-xs text-gray-500">預測高</p>
              <p className="text-green-400 font-mono font-bold">{prediction.predicted_high?.toFixed(2) ?? "—"}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500">預測收</p>
              <p className="text-yellow-400 font-mono font-bold">{prediction.predicted_close?.toFixed(2) ?? "—"}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500">預測低</p>
              <p className="text-red-400 font-mono font-bold">{prediction.predicted_low?.toFixed(2) ?? "—"}</p>
            </div>
          </div>
          {prediction.close_chg !== undefined && (
            <p className={`text-center text-sm mt-2 font-semibold ${prediction.close_chg >= 0 ? "text-green-400" : "text-red-400"}`}>
              {prediction.close_chg >= 0 ? "▲" : "▼"} {Math.abs(prediction.close_chg).toFixed(2)}%
            </p>
          )}
        </div>
      )}
    </div>
  );
}
