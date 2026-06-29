function StatCard({ title, value, sub, color = "text-white" }) {
  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <p className="text-xs text-gray-500 mb-1">{title}</p>
      <p className={`font-bold text-lg font-mono ${color}`}>{value ?? "—"}</p>
      {sub && <p className="text-xs text-gray-500 mt-1">{sub}</p>}
    </div>
  );
}

function fmt(n, decimals = 0) {
  if (n == null) return "—";
  const abs = Math.abs(n);
  const sign = n >= 0 ? "+" : "-";
  if (abs >= 1e8) return `${sign}${(abs / 1e8).toFixed(1)}億`;
  if (abs >= 1e4) return `${sign}${(abs / 1e4).toFixed(0)}萬`;
  return `${sign}${abs.toFixed(decimals)}`;
}

function color(v) {
  if (!v && v !== 0) return "text-white";
  return v > 0 ? "text-green-400" : v < 0 ? "text-red-400" : "text-gray-400";
}

export default function ChipPanel({ data }) {
  const inst = data?.institutional;
  const margin = data?.margin;
  const revenue = data?.monthly_revenue;

  const latestInst = inst?.at(-1) || {};
  const latestMargin = margin?.at(-1) || {};

  // 三大法人合計
  const netCols = Object.keys(latestInst).filter((k) => k.startsWith("net_"));
  const totalNet = netCols.reduce((s, k) => s + (latestInst[k] || 0), 0);

  // 月營收
  const latestRev = revenue?.at(-1) || {};

  return (
    <div className="bg-gray-900 rounded-xl p-4">
      <h2 className="text-lg font-semibold text-white mb-4">籌碼面板</h2>

      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-3 mb-4">
        <StatCard
          title="三大法人合計"
          value={fmt(totalNet)}
          sub={latestInst.date?.slice(0, 10)}
          color={color(totalNet)}
        />
        {netCols.map((k) => (
          <StatCard
            key={k}
            title={k.replace("net_", "")}
            value={fmt(latestInst[k])}
            color={color(latestInst[k])}
          />
        ))}
        <StatCard
          title="融資餘額"
          value={fmt(latestMargin.MarginPurchaseTodayBalance)}
          sub={`變化 ${fmt(latestMargin.margin_change)}`}
          color={color(latestMargin.margin_change)}
        />
        <StatCard
          title="融券餘額"
          value={fmt(latestMargin.ShortSaleTodayBalance)}
          sub={`變化 ${fmt(latestMargin.short_change)}`}
          color={color(-latestMargin.short_change)}
        />
        <StatCard
          title="月營收"
          value={latestRev.revenue ? `${(latestRev.revenue / 1e6).toFixed(0)}百萬` : "—"}
          sub={`YoY ${latestRev.revenue_yoy?.toFixed(1) ?? "—"}%`}
          color={color(latestRev.revenue_yoy)}
        />
      </div>

      {/* 三大法人歷史趨勢（文字版） */}
      {inst && inst.length > 5 && (
        <div className="mt-2">
          <p className="text-xs text-gray-500 mb-2">三大法人近 10 日合計</p>
          <div className="flex gap-1">
            {inst.slice(-10).map((d, i) => {
              const net = netCols.reduce((s, k) => s + (d[k] || 0), 0);
              const h = Math.min(Math.abs(net) / 1e6 * 10, 40);
              return (
                <div key={i} className="flex flex-col items-center flex-1">
                  <div
                    className={`w-full rounded-sm ${net >= 0 ? "bg-green-600" : "bg-red-600"}`}
                    style={{ height: `${h + 4}px` }}
                    title={`${d.date?.slice(0, 10)}: ${fmt(net)}`}
                  />
                  <span className="text-gray-600 text-xs mt-1">{d.date?.slice(5, 10)}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
