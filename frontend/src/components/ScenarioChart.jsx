import { useEffect, useRef, useState } from "react";
import { createChart, ColorType, LineStyle } from "lightweight-charts";

const SCENARIO_COLORS = {
  strong_bull: "#16a34a",
  bull:        "#4ade80",
  flat:        "#6b7280",
  bear:        "#f87171",
  strong_bear: "#dc2626",
};

function ScenarioCard({ scenario, rank }) {
  const color = SCENARIO_COLORS[scenario.key] || "#6b7280";
  const prob = Math.round(scenario.probability * 100);
  return (
    <div className="bg-gray-800 rounded-lg p-3 border border-gray-700"
         style={{ borderLeftColor: color, borderLeftWidth: 3 }}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-gray-400">#{rank}</span>
        <span className="text-lg font-bold" style={{ color }}>{prob}%</span>
      </div>
      <p className="text-sm font-medium text-white mb-2">{scenario.label}</p>
      <div className="grid grid-cols-3 gap-1 text-xs text-gray-400">
        <div>高: <span className="text-white">{scenario.high_target}</span></div>
        <div>目標: <span style={{ color }}>{scenario.close_target}</span></div>
        <div>低: <span className="text-white">{scenario.low_target}</span></div>
      </div>
    </div>
  );
}

export default function ScenarioChart({ priceHistory, prediction, priceLevels }) {
  const chartRef = useRef(null);
  const chart = useRef(null);
  const candleSeries = useRef(null);

  useEffect(() => {
    if (!chartRef.current || chart.current) return;

    chart.current = createChart(chartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#111827" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
      crosshair: { mode: 1 },
      timeScale: { borderColor: "#374151", timeVisible: true },
      rightPriceScale: { borderColor: "#374151" },
      width: chartRef.current.clientWidth,
      height: 320,
    });

    candleSeries.current = chart.current.addCandlestickSeries({
      upColor: "#16a34a",
      downColor: "#dc2626",
      borderVisible: false,
      wickUpColor: "#16a34a",
      wickDownColor: "#dc2626",
    });

    const ro = new ResizeObserver(() => {
      chart.current?.applyOptions({ width: chartRef.current.clientWidth });
    });
    ro.observe(chartRef.current);
    return () => { ro.disconnect(); chart.current?.remove(); chart.current = null; };
  }, []);

  // 更新 K線數據
  useEffect(() => {
    if (!candleSeries.current || !priceHistory?.length) return;
    const data = priceHistory.map((d) => ({
      time: d.date,
      open:  Number(d.open),
      high:  Number(d.max),
      low:   Number(d.min),
      close: Number(d.close),
    })).filter((d) => d.open && d.high && d.low && d.close);
    candleSeries.current.setData(data);
    chart.current?.timeScale().fitContent();
  }, [priceHistory]);

  // 支撐壓力線
  useEffect(() => {
    if (!chart.current || !priceLevels) return;
    const lines = [
      { price: priceLevels.strong_resistance, color: "#dc2626", style: LineStyle.Dashed, title: "強壓力" },
      { price: priceLevels.weak_resistance,   color: "#f87171", style: LineStyle.Dotted,  title: "弱壓力" },
      { price: priceLevels.entry_suggest,     color: "#facc15", style: LineStyle.Dashed,  title: "建議進場" },
      { price: priceLevels.weak_support,      color: "#4ade80", style: LineStyle.Dotted,  title: "弱支撐" },
      { price: priceLevels.strong_support,    color: "#16a34a", style: LineStyle.Dashed,  title: "強支撐" },
    ];
    lines.forEach(({ price, color, style, title }) => {
      if (!price) return;
      candleSeries.current.createPriceLine({ price, color, lineWidth: 1, lineStyle: style, axisLabelVisible: true, title });
    });
  }, [priceLevels]);

  const scenarios = prediction?.scenarios?.scenarios || [];
  const currentPrice = prediction?.scenarios?.current_price;

  return (
    <div className="bg-gray-900 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-white">日K 走勢圖</h2>
        {currentPrice && (
          <span className="text-xl font-bold text-yellow-400">$ {currentPrice}</span>
        )}
      </div>

      <div ref={chartRef} className="w-full mb-4" />

      {/* 情境卡片 */}
      {scenarios.length > 0 && (
        <div>
          <h3 className="text-sm text-gray-400 mb-2 font-medium">今日走勢情境（依機率排序）</h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            {scenarios.map((sc, i) => (
              <ScenarioCard key={sc.key} scenario={sc} rank={i + 1} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
