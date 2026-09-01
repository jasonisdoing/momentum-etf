"use client";

/** 백테스트 기간별 표(연간·월간·주간·일간) — 전략 화면과 자산 헬퍼가 함께 쓴다.
 *
 *  입력은 **일간 변동률(%)** 시계열 하나다. 나머지 단위는 `@/lib/backtest-periods` 가
 *  복리로 합성하므로 화면마다 기준이 갈리지 않는다.
 *
 *  주간은 **달력 주** 기준이다. 모멘텀 화면의 주간 탭은 교체 구간 기준이고 매매 내역
 *  (편입·편출·교체율)이 함께 붙어서 다른 표다 — 그쪽은 그 화면이 직접 그린다.
 */

import type { ColDef } from "ag-grid-community";
import { useMemo, useState } from "react";

import {
  type BacktestCalendarWeekRow,
  type BacktestDayRow,
  type BacktestMonthRow,
  type BacktestYearRow,
  toCalendarMonthRows,
  toCalendarWeekRows,
  toYearRows,
} from "@/lib/backtest-periods";
import { formatSignedPct, signColor } from "@/lib/grid-cells";

import { AppAgGrid } from "./AppAgGrid";
import { createAppGridTheme } from "./app-grid-theme";
import { NavTabs } from "./NavTabs";

const gridTheme = createAppGridTheme();

const VIEW_MODES = [
  { key: "yearly", label: "연간" },
  { key: "monthly", label: "월간" },
  { key: "weekly", label: "주간" },
  { key: "daily", label: "일간" },
] as const;
export type BacktestPeriodView = (typeof VIEW_MODES)[number]["key"];

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

/** 전략·벤치마크 수익률 컬럼 한 쌍 + 초과 — 기간 단위만 다르고 나머지는 같다. */
function pctColumns<T extends { strategy_pct: number | null; benchmark_pct: number | null }>(
  benchmarkLabel: string,
  benchmarkTooltip?: string,
): ColDef<T>[] {
  const column = (headerName: string, field: "strategy_pct" | "benchmark_pct", headerTooltip?: string): ColDef<T> => ({
    headerName,
    field: field as ColDef<T>["field"],
    headerTooltip,
    flex: 1,
    minWidth: 110,
    type: "numericColumn",
    valueFormatter: (p) => (p.value == null ? "-" : formatSignedPct(p.value as number, 2)),
    cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: field === "strategy_pct" ? 700 : 400 }),
  });
  return [column("전략", "strategy_pct"), column(benchmarkLabel, "benchmark_pct", benchmarkTooltip), excessColumn<T>()];
}

const labelColumn = <T,>(headerName: string, field: string, width = 148): ColDef<T> => ({
  headerName,
  field: field as ColDef<T>["field"],
  width,
  cellStyle: () => ({ fontWeight: 700 }),
});

export function BacktestPeriodTables({
  daily,
  benchmarkLabel,
  benchmarkTooltip,
  defaultView = "monthly",
}: {
  daily: BacktestDayRow[];
  benchmarkLabel: string;
  benchmarkTooltip?: string;
  defaultView?: BacktestPeriodView;
}) {
  const [view, setView] = useState<BacktestPeriodView>(defaultView);

  const monthRows = useMemo(() => toCalendarMonthRows(daily), [daily]);
  const weekRows = useMemo(() => toCalendarWeekRows(daily), [daily]);
  const yearRows = useMemo(() => toYearRows(monthRows), [monthRows]);
  // 최신이 위 — 기간별 표와 순서를 맞춘다(백엔드가 주는 순서를 그대로 믿지 않는다).
  const dayRows = useMemo(() => [...daily].sort((a, b) => b.date.localeCompare(a.date)), [daily]);

  const yearColumns = useMemo<ColDef<BacktestYearRow>[]>(() => {
    // 부분 기간은 /compare 와 같은 규칙으로 값 뒤에 `*` 를 붙인다.
    const column = (
      headerName: string,
      field: "strategy_pct" | "benchmark_pct",
      partialField: "strategy_partial" | "benchmark_partial",
      headerTooltip?: string,
    ): ColDef<BacktestYearRow> => ({
      headerName,
      field,
      headerTooltip,
      flex: 1,
      minWidth: 110,
      type: "numericColumn",
      valueFormatter: (p) =>
        p.value == null ? "-" : `${formatSignedPct(p.value as number, 2)}${p.data?.[partialField] ? "*" : ""}`,
      cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: field === "strategy_pct" ? 700 : 400 }),
      tooltipValueGetter: (p) =>
        p.data?.[partialField] ? "12개월이 다 차지 않은 해 — 있는 달만 합성한 부분 기간" : undefined,
    });
    return [
      labelColumn<BacktestYearRow>("연도", "year"),
      column("전략", "strategy_pct", "strategy_partial"),
      column(benchmarkLabel, "benchmark_pct", "benchmark_partial", benchmarkTooltip),
      excessColumn<BacktestYearRow>(),
    ];
  }, [benchmarkLabel, benchmarkTooltip]);

  const monthColumns = useMemo<ColDef<BacktestMonthRow>[]>(
    () => [labelColumn<BacktestMonthRow>("월", "month"), ...pctColumns<BacktestMonthRow>(benchmarkLabel, benchmarkTooltip)],
    [benchmarkLabel, benchmarkTooltip],
  );
  const weekColumns = useMemo<ColDef<BacktestCalendarWeekRow>[]>(
    () => [
      labelColumn<BacktestCalendarWeekRow>("주 종료일", "week_end"),
      ...pctColumns<BacktestCalendarWeekRow>(benchmarkLabel, benchmarkTooltip),
    ],
    [benchmarkLabel, benchmarkTooltip],
  );
  const dayColumns = useMemo<ColDef<BacktestDayRow>[]>(
    () => [labelColumn<BacktestDayRow>("날짜", "date"), ...pctColumns<BacktestDayRow>(benchmarkLabel, benchmarkTooltip)],
    [benchmarkLabel, benchmarkTooltip],
  );

  // 표는 모두 autoHeight — 카드 안에서 따로 스크롤하지 않고 브라우저 스크롤로 본다.
  const gridProps = {
    theme: gridTheme,
    minHeight: 0,
    height: "auto" as const,
    gridOptions: { domLayout: "autoHeight" as const },
  };

  return (
    <div>
      <NavTabs
        items={VIEW_MODES}
        value={view}
        onChange={setView}
        label="백테스트 보기 단위"
        style={{ marginBottom: 10 }}
      />
      {view === "yearly" ? (
        <>
          <AppAgGrid<BacktestYearRow>
            rowData={yearRows}
            columnDefs={yearColumns}
            getRowId={(p) => p.data.year}
            {...gridProps}
          />
          {yearRows.some((row) => row.strategy_partial || row.benchmark_partial) ? (
            <span style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
              * 12개월이 다 차지 않은 해 — 있는 달만 합성한 부분 기간
            </span>
          ) : null}
        </>
      ) : view === "monthly" ? (
        <AppAgGrid<BacktestMonthRow>
          rowData={monthRows}
          columnDefs={monthColumns}
          getRowId={(p) => p.data.month}
          {...gridProps}
        />
      ) : view === "weekly" ? (
        <AppAgGrid<BacktestCalendarWeekRow>
          rowData={weekRows}
          columnDefs={weekColumns}
          getRowId={(p) => p.data.week_end}
          {...gridProps}
        />
      ) : (
        <AppAgGrid<BacktestDayRow>
          rowData={dayRows}
          columnDefs={dayColumns}
          getRowId={(p) => p.data.date}
          {...gridProps}
        />
      )}
    </div>
  );
}
