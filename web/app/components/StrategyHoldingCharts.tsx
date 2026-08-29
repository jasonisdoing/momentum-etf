"use client";

/** 전략 화면의 「차트」 탭 — 로딩·에러·빈 상태와 2열 격자를 담당한다.
 *
 *  네 화면(모멘텀·신고가·합성·포트폴리오)이 같은 블록을 각자 들고 있었다. 문구와
 *  카드에 넘길 값만 다르고 나머지는 글자 하나까지 같아서, 한 곳을 고치면 나머지 셋이
 *  어긋났다. 차트 카드 자체는 `HoldingChart`, 데이터는 `utils/holding_chart_service` 다.
 *
 *  카드에 넘길 값은 화면마다 다르다(모멘텀은 연속 주수, 신고가는 진입가, 합성은 슬롯) —
 *  그래서 `chartProps` 로 화면이 직접 만든다. 여기서 전략을 분기하지 않는다.
 */

import type { ReactNode } from "react";

import { HoldingChart, type ChartBadge, type HoldingChartData } from "./HoldingChart";

const hintStyle: React.CSSProperties = { color: "var(--text-muted)", fontSize: "var(--fs-sm)" };
const centeredHint: React.CSSProperties = { ...hintStyle, padding: "24px 0", textAlign: "center" };

/** `HoldingChart` 에 넘길 값 중 화면이 정하는 것들(차트 데이터 자체는 여기서 넣는다). */
export type HoldingChartExtras = {
  strategyLabel?: string;
  entryDate?: string | null;
  entryPrice?: number | null;
  returnPct?: number | null;
  days?: number | null;
  daysUnit?: string;
  daysLabel?: string | null;
  /** 배지를 직접 정할 때 — 주면 위 값들 대신 이 배지가 그려진다(순위 화면). */
  badges?: ChartBadge[];
};

export function StrategyHoldingCharts({
  charts,
  loading,
  error,
  hint,
  emptyMessage,
  chartProps,
}: {
  /** 아직 안 받았으면 null — 로딩과 구분한다. */
  charts: HoldingChartData[] | null;
  loading: boolean;
  error: string | null;
  /** 이 전략에서 선이 뜻하는 바. 전략마다 판정이 달라 화면이 쓴다. */
  hint: ReactNode;
  emptyMessage: string;
  chartProps: (chart: HoldingChartData) => HoldingChartExtras;
}) {
  if (loading || (!charts && !error)) return <div style={centeredHint}>차트를 불러오는 중…</div>;
  if (error) return <div className="alert alert-danger">{error}</div>;
  if (!charts || charts.length === 0) return <div style={centeredHint}>{emptyMessage}</div>;

  return (
    <>
      <div style={{ ...hintStyle, margin: "4px 0 10px" }}>{hint}</div>
      {/* 최대 2열, 좁으면 1열 — 규칙은 globals.css 의 `.appChartGrid` 한 곳에 있다. */}
      <div className="appChartGrid">
        {charts.map((item) => (
          <HoldingChart key={item.ticker} chart={item} {...chartProps(item)} />
        ))}
      </div>
    </>
  );
}
