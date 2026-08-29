"use client";

/** 보유 종목 차트 카드 — 전략 화면(신고가·모멘텀·합성) 공용.
 *
 *  캔들 + 이동평균선(들) + 진입(Buy) 마커 + 수익률·보유기간 배지.
 *  어떤 선을 그릴지는 백엔드(`utils/holding_chart_service.py`)가 `ma_lines` 로 내려준다 —
 *  신고가는 이탈 이평선 1줄, 모멘텀은 단기·장기 2줄. 색은 여기 팔레트 순서로 정한다.
 */

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
  ma_lines: { ma_days: number; points: { time: string; value: number }[] }[];
};

type Props = {
  chart: HoldingChartData;
  /** 진입일·진입가 — 진입 예정 종목은 아직 없으므로 마커를 찍지 않는다. */
  entryDate?: string | null;
  entryPrice?: number | null;
  /** 보유 수익률(%) — 진입 예정 종목은 없다. */
  returnPct?: number | null;
  /** 보유 기간 — 신고가는 일, 모멘텀은 주. 단위 문구는 `daysUnit`. */
  days?: number | null;
  daysUnit?: string;
  /** 완성된 보유 기간 문구("3주"·"12일") — 합성처럼 백엔드가 문자열로 주는 화면용. days 보다 우선. */
  daysLabel?: string | null;
  height?: number;
};

// 한국 관례 — 상승 빨강, 하락 파랑. 다른 화면(티커 상세)과 같은 색을 쓴다.
const UP = "#e03131";
const DOWN = "#206bc4";
// 이평선 팔레트 — 첫 선(단기/이탈선)은 청록, 둘째 선(장기)은 주황.
const MA_COLORS = ["#12b886", "#f76707", "#7048e8"];
const BUY_MARKER_COLOR = "#111827";

function formatPrice(value: number): string {
  return value.toLocaleString("ko-KR", { maximumFractionDigits: value >= 1000 ? 0 : 2 });
}

export function HoldingChart({ chart, entryDate, entryPrice, returnPct, days, daysUnit = "일", daysLabel, height = 320 }: Props) {
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

    chart.ma_lines.forEach((line, index) => {
      api.addSeries(LineSeries, {
        color: MA_COLORS[index % MA_COLORS.length], lineWidth: 2,
        priceLineVisible: false, lastValueVisible: false,
      }).setData(line.points.map((row) => ({ ...row, time: row.time as Time })));
    });

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
          {daysLabel != null || days != null ? (
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
              {daysLabel ?? `${days}${daysUnit}`}
            </span>
          ) : null}
        </span>
      </div>
      <div style={{ position: "relative" }}>
        {/* 이평선 범례 — 차트 안 좌상단에 겹쳐 둔다. 선 순서 = 색 순서. */}
        <span
          style={{
            position: "absolute",
            top: 6,
            left: 8,
            zIndex: 2,
            display: "flex",
            gap: 10,
            fontSize: "var(--fs-sm)",
            fontWeight: 700,
            pointerEvents: "none",
          }}
        >
          {chart.ma_lines.map((line, index) => (
            <span key={line.ma_days} style={{ color: MA_COLORS[index % MA_COLORS.length] }}>
              MA{line.ma_days}
            </span>
          ))}
        </span>
        <div ref={containerRef} />
      </div>
    </div>
  );
}
