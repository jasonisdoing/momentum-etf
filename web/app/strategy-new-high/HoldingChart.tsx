"use client";

import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  LineSeries,
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
  /** 보유 수익률(%) — 진입 예정 종목은 없다. */
  returnPct?: number | null;
  /** 보유일 — 진입 예정 종목은 없다. */
  days?: number | null;
  maDays: number;
  height?: number;
};

// 한국 관례 — 상승 빨강, 하락 파랑. 다른 화면(티커 상세)과 같은 색을 쓴다.
const UP = "#e03131";
const DOWN = "#206bc4";
// 이탈 이평선 — 참고 스타일과 같은 청록 계열 녹색.
const MA_COLOR = "#12b886";
const BUY_MARKER_COLOR = "#111827";

function formatPrice(value: number): string {
  return value.toLocaleString("ko-KR", { maximumFractionDigits: value >= 1000 ? 0 : 2 });
}

export function HoldingChart({ chart, entryDate, entryPrice, returnPct, days, maDays, height = 320 }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || chart.candles.length === 0) return;

    const api = createChart(container, {
      width: container.clientWidth,
      height,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#5b6778", fontSize: 12 },
      grid: { vertLines: { color: "#f0f2f5" }, horzLines: { color: "#f0f2f5" } },
      rightPriceScale: { borderColor: "#e6e8ec", scaleMargins: { top: 0.1, bottom: 0.08 } },
      timeScale: { borderColor: "#e6e8ec", timeVisible: false },
      handleScroll: false,
      handleScale: false,
    });
    chartRef.current = api;

    // 현재가 점선·축 라벨은 캔들 시리즈 기본값(마지막 봉 방향 색)을 그대로 쓴다.
    const candles = api.addSeries(CandlestickSeries, {
      upColor: UP, downColor: DOWN,
      borderUpColor: UP, borderDownColor: DOWN,
      wickUpColor: UP, wickDownColor: DOWN,
      priceFormat: { type: "price", precision: 0, minMove: 1 },
    });
    candles.setData(chart.candles.map((row) => ({ ...row, time: row.time as Time })));

    // 이탈 이평선 — 종가가 이 선을 하회하면 청산한다.
    api.addSeries(LineSeries, {
      color: MA_COLOR, lineWidth: 2,
      priceLineVisible: false, lastValueVisible: false,
    }).setData(chart.ma.map((row) => ({ ...row, time: row.time as Time })));

    // 진입 마커 — 매수가를 함께 적는다 (별도 매수가 점선은 두지 않는다).
    // 차트 구간보다 진입일이 이르면 마커를 못 찍는다 — 없는 자리에 찍지 않는다.
    if (entryDate && chart.candles.some((row) => row.time === entryDate)) {
      const label = entryPrice != null && Number.isFinite(entryPrice) ? `Buy ${formatPrice(entryPrice)}` : "Buy";
      createSeriesMarkers(candles, [
        { time: entryDate as Time, position: "belowBar", color: BUY_MARKER_COLOR, shape: "arrowUp", text: label },
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

  const returnColor = returnPct == null || returnPct === 0 ? "inherit" : returnPct > 0 ? UP : DOWN;

  return (
    <div className="card appCard" style={{ padding: "12px 14px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <strong style={{ fontSize: "var(--fs-base)" }}>{chart.name}</strong>
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}>
          {returnPct != null ? (
            <span
              style={{
                border: "1px solid rgba(148,163,184,0.45)",
                borderRadius: 8,
                padding: "3px 10px",
                fontSize: "var(--fs-sm)",
                fontWeight: 700,
              }}
            >
              수익률 <span style={{ color: returnColor }}>{`${returnPct >= 0 ? "+" : ""}${returnPct.toFixed(2)}%`}</span>
            </span>
          ) : (
            <span
              style={{
                borderRadius: 8,
                padding: "3px 10px",
                fontSize: "var(--fs-sm)",
                fontWeight: 700,
                background: "#fff0f0",
                color: UP,
              }}
            >
              진입 예정
            </span>
          )}
          {days != null ? (
            <span
              style={{
                borderRadius: 8,
                padding: "3px 10px",
                fontSize: "var(--fs-sm)",
                fontWeight: 700,
                background: "#e6fcf1",
                color: "#0ca678",
              }}
            >
              {days}일
            </span>
          ) : null}
        </span>
      </div>
      <div style={{ position: "relative" }}>
        {/* 이평선 범례 — 참고 스타일처럼 차트 안 좌상단에 겹쳐 둔다. */}
        <span
          style={{
            position: "absolute",
            top: 6,
            left: 8,
            zIndex: 2,
            fontSize: "var(--fs-sm)",
            fontWeight: 700,
            color: MA_COLOR,
            pointerEvents: "none",
          }}
        >
          MA{maDays}
        </span>
        <div ref={containerRef} />
      </div>
    </div>
  );
}
