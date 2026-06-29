const LABELS = {
  strong_bull: "強勢上漲", bull: "溫和上漲", flat: "盤整",
  bear: "溫和下跌", strong_bear: "強勢下跌",
};
const COLORS = {
  strong_bull: "text-green-400", bull: "text-green-300", flat: "text-gray-400",
  bear: "text-red-300", strong_bear: "text-red-400",
};

export default function IntradayPanel({ update }) {
  if (!update) return null;
  const { intraday_features: feats, updated_probs: probs } = update;
  const sorted = Object.entries(probs || {})
    .sort(([, a], [, b]) => b - a)
    .slice(0, 3);

  return (
    <div className="bg-gray-900 rounded-xl p-4 border border-yellow-900">
      <div className="flex items-center gap-2 mb-3">
        <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
        <h2 className="text-sm font-semibold text-yellow-300">盤中滾動更新</h2>
      </div>

      {feats && (
        <div className="grid grid-cols-2 gap-2 mb-3 text-xs">
          <div className="bg-gray-800 rounded p-2">
            <p className="text-gray-500">現價</p>
            <p className="text-white font-mono font-bold">{feats.current_price}</p>
          </div>
          <div className="bg-gray-800 rounded p-2">
            <p className="text-gray-500">VWAP</p>
            <p className="text-blue-300 font-mono">{feats.vwap}</p>
          </div>
          <div className="bg-gray-800 rounded p-2">
            <p className="text-gray-500">買賣比</p>
            <p className={feats.buy_ratio > 0.5 ? "text-green-400" : "text-red-400"}>
              {(feats.buy_ratio * 100).toFixed(1)}% 買
            </p>
          </div>
          <div className="bg-gray-800 rounded p-2">
            <p className="text-gray-500">累積量</p>
            <p className="text-white">{feats.cum_volume?.toLocaleString()}</p>
          </div>
        </div>
      )}

      <div className="space-y-1">
        {sorted.map(([sc, prob]) => (
          <div key={sc} className="flex items-center gap-2">
            <div className="flex-1 bg-gray-800 rounded-full h-2">
              <div className="h-2 rounded-full bg-yellow-500" style={{ width: `${prob * 100}%` }} />
            </div>
            <span className={`text-xs w-24 ${COLORS[sc]}`}>{LABELS[sc]}</span>
            <span className="text-xs text-gray-400 w-10 text-right">{(prob * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
