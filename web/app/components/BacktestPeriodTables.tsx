"use client";

/** 백테스트 기간별 표(연간·월간·주간·일간) — 모든 전략 화면(모멘텀·신고가·포트폴리오·합성·사고팔기)이 쓴다.
 *
 *  입력은 백엔드가 내려주는 **일별 누적 수익률(%)** 이고, 기간별 자르기는 `@/lib/backtest-periods` 가
 *  한 곳에서 한다. 탭 한 줄에 연간·월간·주간·일간과 화면 고유 탭(`extraTab`, 예: 「체결」)을 함께 둔다.
 *  주간 라벨은 그 주의 마지막 거래일이다.
 */

import type { ColDef } from "ag-grid-community";
import { type ReactNode, useMemo, useState } from "react";

import { type BacktestCumulativeRow, type PeriodRow, toPeriodRows, weekKeyOf } from "@/lib/backtest-periods";
import { formatDateWithWeekday } from "@/lib/datetime";
import { adrColumn, formatSignedPct, signColor } from "@/lib/grid-cells";

import { AppAgGrid } from "./AppAgGrid";
import { createAppGridTheme } from "./app-grid-theme";
import { NavTabs, type NavTabItem } from "./NavTabs";

const gridTheme = createAppGridTheme();

const PERIOD_VIEWS = [
  { key: "yearly", label: "연간" },
  { key: "monthly", label: "월간" },
  { key: "weekly", label: "주간" },
  { key: "daily", label: "일간" },
] as const;
export type BacktestPeriodView = (typeof PERIOD_VIEWS)[number]["key"];

const EXTRA_TAB_KEY = "extra";
const PERIOD_LABEL: Record<BacktestPeriodView, string> = { yearly: "연도", monthly: "월", weekly: "주", daily: "일자" };
const KEY_LENGTH = { yearly: 4, monthly: 7, daily: 10 } as const;

/** 초과(%p) 컬럼 — 전략 − 벤치마크. 모든 백테스트 화면이 같은 정의를 쓴다. */
export function excessColumn<T extends { strategy_pct: number | null; benchmark_pct: number | null }>(): ColDef<T> {
  return {
    headerName: "초과",
    colId: "excess_pp",
    flex: 1,
    minWidth: 100,
    type: "numericColumn",
    valueGetter: (p) =>
      p.data && p.data.strategy_pct != null && p.data.benchmark_pct != null
        ? p.data.strategy_pct - p.data.benchmark_pct
        : null,
    valueFormatter: (p) => (p.value == null ? "-" : `${formatSignedPct(p.value as number, 2)}p`),
    cellStyle: (p) => ({ color: signColor(p.value as number) }),
  };
}

function periodRowsOf(daily: BacktestCumulativeRow[], view: BacktestPeriodView): PeriodRow[] {
  return view === "weekly"
    ? toPeriodRows(daily, weekKeyOf, true)
    : toPeriodRows(daily, (date) => date.slice(0, KEY_LENGTH[view]));
}

/** 시장 ADR 컬럼 — 일간=당일, 주간=판정일(주 마지막 거래일), 월간·연간=기간 최저. */
function adrColumnFor(view: BacktestPeriodView): ColDef<PeriodRow> {
  if (view === "daily") {
    return adrColumn<PeriodRow>({
      headerName: "ADR",
      headerTooltip: "그날의 시장 ADR(20일 등락비율)",
      hide: false,
      getter: (row) => row.adr,
    });
  }
  if (view === "weekly") {
    return adrColumn<PeriodRow>({
      headerName: "판정일 ADR",
      headerTooltip: "그 주 마지막 거래일의 시장 ADR — 다음 거래일 진입 게이트를 결정한 값",
      hide: false,
      getter: (row) => row.adr,
    });
  }
  return adrColumn<PeriodRow>({
    headerName: "최저 ADR",
    headerTooltip: `${view === "yearly" ? "그 해" : "그 달"}의 시장 ADR 최저값`,
    hide: false,
    getter: (row) => row.adr_min,
  });
}

export function BacktestPeriodTables({
  daily,
  benchmarkLabel,
  benchmarkTooltip,
  strategyLabel,
  strategyTooltip,
  defaultView,
  extraTab,
  middleColumns,
}: {
  daily: BacktestCumulativeRow[];
  benchmarkLabel: string;
  benchmarkTooltip?: string;
  /** 전략 열 이름 — 합성은 「합성」, 나머지는 「전략」. */
  strategyLabel: string;
  strategyTooltip?: string;
  defaultView: BacktestPeriodView;
  /** 기간 탭 뒤에 이어 붙는 탭 하나(예: 「체결」) — 다른 화면과 한 줄 탭으로 보이게 한다. */
  extraTab?: { label: string; content: ReactNode };
  /** 전략 열과 벤치마크 열 사이에 끼우는 열(합성의 인출·슬리브). 렌더마다 새로 만들지 말고 useCallback 으로 넘긴다. */
  middleColumns?: (view: BacktestPeriodView) => ColDef<PeriodRow>[];
}) {
  const [view, setView] = useState<BacktestPeriodView | typeof EXTRA_TAB_KEY>(defaultView);
  const tabs = useMemo<NavTabItem<BacktestPeriodView | typeof EXTRA_TAB_KEY>[]>(
    () => (extraTab ? [...PERIOD_VIEWS, { key: EXTRA_TAB_KEY, label: extraTab.label }] : [...PERIOD_VIEWS]),
    [extraTab],
  );
  // ADR 열 — 값이 하나도 없으면(레짐 시장이 없는 풀·합성) 열을 만들지 않는다.
  const hasAdr = useMemo(() => daily.some((row) => row.adr != null), [daily]);
  const periodView = view === EXTRA_TAB_KEY ? null : view;
  const rows = useMemo(() => (periodView ? periodRowsOf(daily, periodView) : []), [daily, periodView]);

  const columns = useMemo<ColDef<PeriodRow>[]>(() => {
    if (!periodView) return [];
    return [
      {
        field: "period",
        headerName: PERIOD_LABEL[periodView],
        width: 148,
        valueFormatter: (p) =>
          periodView === "weekly" || periodView === "daily"
            ? formatDateWithWeekday(String(p.value ?? ""))
            : String(p.value ?? ""),
        cellStyle: { fontWeight: 700 },
      },
      {
        field: "strategy_pct",
        headerName: strategyLabel,
        headerTooltip: strategyTooltip,
        flex: 1,
        minWidth: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatSignedPct(p.value as number, 2),
        cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 700 }),
      },
      ...(middleColumns ? middleColumns(periodView) : []),
      {
        field: "benchmark_pct",
        headerName: benchmarkLabel,
        headerTooltip: benchmarkTooltip,
        flex: 1,
        minWidth: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatSignedPct(p.value as number, 2),
        cellStyle: (p) => ({ color: signColor(p.value as number) }),
      },
      excessColumn<PeriodRow>(),
      ...(hasAdr ? [adrColumnFor(periodView)] : []),
    ];
  }, [periodView, strategyLabel, strategyTooltip, benchmarkLabel, benchmarkTooltip, middleColumns, hasAdr]);

  return (
    <div>
      <NavTabs
        items={tabs}
        value={view}
        onChange={setView}
        label="백테스트 보기 단위"
        style={{ marginBottom: 10 }}
      />
      {view === EXTRA_TAB_KEY && extraTab ? (
        extraTab.content
      ) : (
        // 표는 모두 autoHeight — 카드 안에서 따로 스크롤하지 않고 브라우저 스크롤로 본다.
        <AppAgGrid<PeriodRow>
          rowData={rows}
          columnDefs={columns}
          getRowId={(p) => p.data.period}
          theme={gridTheme}
          minHeight={0}
          height="auto"
          gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
        />
      )}
    </div>
  );
}
