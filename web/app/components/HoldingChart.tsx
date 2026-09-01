"use client";

/** 보유 종목 차트 카드 — 전략 화면(신고가·모멘텀·합성) 공용.
 *
 *  캔들 + 이동평균선(들) + 진입(Buy) 마커 + 수익률·보유기간 배지.
 *  어떤 선을 그릴지는 백엔드(`utils/holding_chart_service.py`)가 `ma_lines` 로 내려준다 —
 *  신고가는 이탈 이평선 1줄, 모멘텀은 단기·장기 2줄. 색은 여기 팔레트 순서로 정한다.
 */

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  CandlestickSeries,
  ColorType,
  LineSeries,
  LineStyle,
  createChart,
  createSeriesMarkers,
} from "lightweight-charts";
import type { IChartApi, Time } from "lightweight-charts";

import { formatMonthDayWithWeekday } from "@/lib/datetime";
import { getSignedNullableClass } from "../assets/assets-helpers";
import { formatCurrencyPrice, priceDecimals } from "@/lib/price-format";

export type HoldingChartData = {
  ticker: string;
  name: string;
  /** 전략명(모멘텀·신고가·포트폴리오) — 합성처럼 카드마다 다른 화면은 백엔드가 내려준다. */
  strategy_label?: string;
  candles: { time: string; open: number; high: number; low: number; close: number }[];
  ma_lines: { ma_days: number; points: { time: string; value: number }[] }[];
  /** 내 평균 매입가 — 실제로 들고 있는 종목에만 온다(`/ticker` 상세와 같은 값). */
  avg_buy_price?: number | null;
  /** 통화(KRW·USD·AUD) — 가격에 기호를 붙이는 데 쓴다. 풀마다 다르다. */
  currency?: string | null;
  /** 이 요청의 차트들이 공유하는 날짜 축 — 전체 거래일 수와, 창 시작부터 첫 캔들까지의 빈 칸 수.
   *  상장한 지 얼마 안 된 종목이 캔들 몇 개로 가로 폭을 다 채우지 않게 한다. */
  window_bars?: number | null;
  leading_bars?: number | null;
};

/** 카드 오른쪽 배지 하나. 색은 화면이 정한다(전략 배지와 같은 모양을 쓴다). */
export type ChartBadge = {
  key: string;
  text: ReactNode;
  background?: string;
  color?: string;
  /** 테두리만 있는 흰 배지 — 수익률처럼 값이 주인공인 배지에 쓴다. */
  outlined?: boolean;
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
  /** 전략명 — 화면 전체가 한 전략이면 여기로 넘긴다. 없으면 데이터의 strategy_label. */
  strategyLabel?: string;
  /** 오른쪽 배지를 직접 정한다 — 주면 기본(진입일·보유기간·수익률) 대신 이걸 그린다.
   *  순위 화면처럼 보유 개념이 없고 순위·고점 같은 다른 값을 보여줄 때 쓴다. */
  badges?: ChartBadge[];
  height?: number;
};

// 한국 관례 — 상승 빨강, 하락 파랑. 다른 화면(티커 상세)과 같은 색을 쓴다.
const UP = "#e03131";
const DOWN = "#206bc4";
// 이평선 팔레트 — 첫 선(단기/이탈선)은 청록, 둘째 선(장기)은 주황.
const MA_COLORS = ["#12b886", "#f76707", "#7048e8"];
const BUY_MARKER_COLOR = "#111827";
// 내 평균 매입가 점선 — 이평선 팔레트와 겹치지 않는 회색.
const AVG_BUY_LINE_COLOR = "#868e96";

// 카드 오른쪽 배지 공통 모양 — 색만 배지마다 다르다.
const badgeStyle: React.CSSProperties = {
  borderRadius: 8,
  padding: "3px 10px",
  fontSize: "var(--fs-sm)",
  fontWeight: 700,
};

export function HoldingChart({ chart, entryDate, entryPrice, returnPct, days, daysUnit = "일", daysLabel, strategyLabel, badges, height = 320 }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  // 내 평균 배지 — 평단 선 높이에 맞춰 왼쪽에 띄운다(`/ticker` 상세와 같은 표기).
  // 선의 y 좌표는 차트가 그려진 뒤에야 알 수 있어 상태로 들고 있는다.
  const [averageBadge, setAverageBadge] = useState<{ top: number; returnPct: number | null } | null>(null);

  const currency = chart.currency ?? "";

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
      // 축·툴팁 가격에 통화 기호를 붙인다 — `/ticker` 상세와 같은 표기(20,795원 · $23.45 · A$23.32).
      priceFormat: {
        type: "custom",
        minMove: priceDecimals(currency) === 0 ? 1 : 0.01,
        formatter: (price: number) => formatCurrencyPrice(price, currency),
      },
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
      const label =
        entryPrice != null && Number.isFinite(entryPrice) ? `Buy ${formatCurrencyPrice(entryPrice, currency)}` : "Buy";
      createSeriesMarkers(candles, [
        { time: entryDate as Time, position: "belowBar", color: BUY_MARKER_COLOR, shape: "arrowUp", text: label },
      ]);
    }
    // 내 평균 매입가 — 실제로 들고 있으면 점선으로 긋는다(`/ticker` 상세와 같은 표기).
    // 전략의 진입가(Buy 마커)와 다르다: 이건 여러 계좌를 합친 내 실제 평단이다.
    const avgBuyPrice = chart.avg_buy_price;
    if (avgBuyPrice != null && Number.isFinite(avgBuyPrice) && avgBuyPrice > 0) {
      candles.createPriceLine({
        price: avgBuyPrice,
        color: AVG_BUY_LINE_COLOR,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "",
      });
    }
    /** 보이는 구간을 공용 창에 맞춘다 — 캔들이 적으면 왼쪽을 비우고 오른쪽 일부만 채운다.
     *  창 정보가 없으면(옛 응답) 예전처럼 데이터에 꽉 맞춘다. */
    function fitToWindow() {
      const windowBars = chart.window_bars ?? 0;
      const leadingBars = chart.leading_bars ?? 0;
      if (windowBars > chart.candles.length) {
        api.timeScale().setVisibleLogicalRange({ from: -leadingBars - 0.5, to: windowBars - leadingBars - 0.5 });
        return;
      }
      api.timeScale().fitContent();
    }
    fitToWindow();

    /** 평단 선의 화면 높이와 그 대비 수익률 — 차트 크기가 바뀌면 다시 잰다. */
    function updateAverageBadge(el: HTMLDivElement) {
      if (avgBuyPrice == null || !Number.isFinite(avgBuyPrice) || avgBuyPrice <= 0) {
        setAverageBadge(null);
        return;
      }
      const y = candles.priceToCoordinate(avgBuyPrice);
      if (y === null) {
        setAverageBadge(null);
        return;
      }
      const lastClose = chart.candles[chart.candles.length - 1]?.close ?? null;
      setAverageBadge({
        top: Math.max(4, Math.min(y - 14, el.clientHeight - 30)),
        returnPct: lastClose != null && lastClose > 0 ? (lastClose / avgBuyPrice - 1) * 100 : null,
      });
    }
    updateAverageBadge(container);

    const observer = new ResizeObserver(() => {
      api.applyOptions({ width: container.clientWidth });
      fitToWindow();
      updateAverageBadge(container);
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
        {/* [전략명] 티커 종목명 — 세 화면 공통 표기. */}
        {/* 이름이 길어도 카드 높이가 늘지 않게 한 줄로 자른다 — 카드가 2열이라 한 장만 높아지면
            옆 카드까지 같이 늘어난다. minWidth:0 이 없으면 flex 자식이 줄지 않아 말줄임이 안 걸린다. */}
        <strong
          style={{
            fontSize: "var(--fs-base)",
            minWidth: 0,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
          title={`${chart.ticker} ${chart.name}`}
        >
          {(() => {
            const label = strategyLabel ?? chart.strategy_label;
            return `${label ? `[${label}] ` : ""}${chart.ticker} ${chart.name}`;
          })()}
        </strong>
        {/* 오른쪽 배지 — 산 날 → 들고 있는 기간 → 수익률 순. 왼쪽부터 시간 순으로 읽힌다.
            `badges` 를 주면 화면이 정한 배지로 통째로 갈아 끼운다(순위 화면). */}
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
          {badges ? (
            badges.map((badge) => (
              <span
                key={badge.key}
                style={{
                  ...badgeStyle,
                  ...(badge.outlined ? { border: "1px solid rgba(148,163,184,0.45)" } : {}),
                  ...(badge.background ? { background: badge.background } : {}),
                  ...(badge.color ? { color: badge.color } : {}),
                }}
              >
                {badge.text}
              </span>
            ))
          ) : (
            <>
          {entryDate ? (
            <span style={{ ...badgeStyle, background: "#f1f3f5", color: "var(--text-muted)" }}>
              {formatMonthDayWithWeekday(entryDate)}
            </span>
          ) : null}
          {daysLabel != null || days != null ? (
            <span style={{ ...badgeStyle, background: "#e6fcf1", color: "#0ca678" }}>
              {daysLabel ?? `${days}${daysUnit}`}
            </span>
          ) : null}
          {returnPct != null ? (
            <span style={{ ...badgeStyle, border: "1px solid rgba(148,163,184,0.45)" }}>
              수익률 <span style={{ color: returnColor }}>{`${returnPct >= 0 ? "+" : ""}${returnPct.toFixed(2)}%`}</span>
            </span>
          ) : (
            <span style={{ ...badgeStyle, background: "#fff0f0", color: UP }}>진입 예정</span>
          )}
            </>
          )}
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
        {averageBadge ? (
          <div className="appChartAverageBadge" style={{ top: averageBadge.top }}>
            <span>내 평균 </span>
            <span className={getSignedNullableClass(averageBadge.returnPct)}>
              {averageBadge.returnPct == null
                ? "-"
                : `${averageBadge.returnPct > 0 ? "+" : ""}${averageBadge.returnPct.toFixed(2)}%`}
            </span>
          </div>
        ) : null}
      </div>
    </div>
  );
}
