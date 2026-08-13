"use client";

import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  LineSeries,
  LineStyle,
  createChart,
  createSeriesMarkers,
} from "lightweight-charts";
import type { IChartApi, Time } from "lightweight-charts";

export type HoldingChartData = {
  ticker: string;
  name: string;
  candles: { time: string; open: number; high: number; low: number; close: number }[];
  ma: { time: string; value: number }[];
  prior_high: { time: string; value: number }[];
};

type Props = {
  chart: HoldingChartData;
  /** 진입일·진입가 — 진입 예정 종목은 아직 없으므로 마커를 찍지 않는다. */
  entryDate?: string | null;
  entryPrice?: number | null;
  maDays: number;
  height?: number;
};

// 한국 관례 — 상승 빨강, 하락 파랑. 다른 화면(티커 상세)과 같은 색을 쓴다.
const UP = "#e03131";
const DOWN = "#206bc4";
const MA_COLOR = "#f08c00";
const HIGH_COLOR = "#7048e8";

export function HoldingChart({ chart, entryDate, entryPrice, maDays, height = 240 }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || chart.candles.length === 0) return;

    const api = createChart(container, {
      width: container.clientWidth,
      height,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#5b6778", fontSize: 11 },
      grid: { vertLines: { color: "#f0f2f5" }, horzLines: { color: "#f0f2f5" } },
      rightPriceScale: { borderColor: "#e6e8ec", scaleMargins: { top: 0.12, bottom: 0.08 } },
      timeScale: { borderColor: "#e6e8ec", timeVisible: false },
      handleScroll: false,
      handleScale: false,
    });
    chartRef.current = api;

    const candles = api.addSeries(CandlestickSeries, {
      upColor: UP, downColor: DOWN,
      borderUpColor: UP, borderDownColor: DOWN,
      wickUpColor: UP, wickDownColor: DOWN,
      priceFormat: { type: "price", precision: 0, minMove: 1 },
    });
    candles.setData(chart.candles.map((row) => ({ ...row, time: row.time as Time })));

    // 직전 최고 종가선 — 이 선을 종가가 넘은 날이 돌파다.
    api.addSeries(LineSeries, {
      color: HIGH_COLOR, lineWidth: 1, lineStyle: LineStyle.Dashed,
      priceLineVisible: false, lastValueVisible: false,
    }).setData(chart.prior_high.map((row) => ({ ...row, time: row.time as Time })));

    // 이탈 이평선 — 종가가 이 선을 하회하면 청산한다.
    api.addSeries(LineSeries, {
      color: MA_COLOR, lineWidth: 2,
      priceLineVisible: false, lastValueVisible: false,
    }).setData(chart.ma.map((row) => ({ ...row, time: row.time as Time })));

    if (entryPrice != null && Number.isFinite(entryPrice) && entryPrice > 0) {
      candles.createPriceLine({
        price: entryPrice,
        color: "var(--text-muted)",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "진입",
      });
    }
    // 차트 구간보다 진입일이 이르면 마커를 못 찍는다 — 없는 자리에 찍지 않는다.
    if (entryDate && chart.candles.some((row) => row.time === entryDate)) {
      createSeriesMarkers(candles, [
        { time: entryDate as Time, position: "belowBar", color: UP, shape: "arrowUp", text: "진입" },
      ]);
    }
    api.timeScale().fitContent();

    const observer = new ResizeObserver(() => {
      api.applyOptions({ width: container.clientWidth });
      api.timeScale().fitContent();
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
      api.remove();
      chartRef.current = null;
    };
  }, [chart, entryDate, entryPrice, height]);

  return (
    <div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginBottom: 4 }}>
        <strong style={{ fontSize: "var(--fs-sm)" }}>{chart.name}</strong>
        <span style={{ color: "var(--text-muted)", fontSize: "var(--fs-xs)" }}>{chart.ticker}</span>
        <span style={{ marginLeft: "auto", fontSize: "var(--fs-xs)", color: MA_COLOR }}>{maDays}일선</span>
        <span style={{ fontSize: "var(--fs-xs)", color: HIGH_COLOR }}>직전 최고 종가</span>
      </div>
      <div ref={containerRef} />
    </div>
  );
}
