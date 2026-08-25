"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { CellStyle, ColDef } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";
import { createAppGridTheme } from "../components/app-grid-theme";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { TrendSparkline, type TrendPoint } from "../components/TrendSparkline";
import { useToast } from "../components/ToastProvider";
import {
  EMPTY_FILTERS,
  countActiveFilters,
  passesFilters,
  readRememberedFilters,
  writeRememberedFilters,
  type DividendFilters,
} from "./dividend-filters";

/** 배치가 저장한 연도별 원시값. */
type YearEntry = {
  dps: number | null;
  operating_income: number | null;
  net_income: number | null;
  is_consensus: boolean;
};

/** 백엔드가 계산한 한 종목의 행. 가격이 필요한 값은 조회 시점에 계산돼 온다. */
type DividendRow = {
  ticker: string;
  name: string | null;
  current_price: number | null;
  market_cap: number | null;
  /** 개월 → 수익률(%). 키는 문자열("1","3","6","12"). */
  returns: Record<string, number | null>;
  /** 연도 → 배당률(%). 지난 해는 연말 종가, 올해는 현재가 기준. */
  dividend_yield_by_year: Record<string, number | null>;
  dividend_yield: number | null;
  dividend_yield_is_forward: boolean;
  buyback_yield: number | null;
  shareholder_yield: number | null;
  payout_ratio_gross: number | null;
  payout_ratio_net: number | null;
  payout_base_year: string | null;
  trend_operating_ratio: number | null;
  trend_net_ratio: number | null;
  trend_dividend_ratio: number | null;
  per: number | null;
  pbr: number | null;
  /** 연도 → 원시값. 추세 막대그래프가 그린다. */
  by_year: Record<string, YearEntry>;
  notes: string[];
};

type ApiResponse = {
  rows?: DividendRow[];
  years?: string[];
  consensus_year?: string | null;
  return_months?: string[];
  updated_at?: string | null;
  error?: string;
};

const gridTheme = createAppGridTheme({ rowHeight: 34 });

/** 숫자 입력 필터 — 라벨과 판정 기준을 한 줄로 묶어 화면과 로직이 어긋나지 않게 한다. */
const NUMERIC_FILTERS: readonly {
  key: "minDividendYield" | "minShareholderYield" | "minPayoutRatio" | "maxPer" | "maxPbr";
  label: string;
  placeholder: string;
  step: number;
  title: string;
}[] = [
  { key: "minDividendYield", label: "배당률 ≥", placeholder: "4", step: 0.1, title: "지금 사면 받을 배당수익률(%) 하한" },
  { key: "minShareholderYield", label: "총환원율 ≥", placeholder: "0", step: 0.1, title: "배당률 + 자사주률(%) 하한" },
  { key: "minPayoutRatio", label: "환원율 ≥", placeholder: "30", step: 1, title: "(배당지급 + 자사주취득) ÷ 순이익(%) 하한" },
  { key: "maxPer", label: "PER ≤", placeholder: "20", step: 0.5, title: "PER 상한 (컨센서스 연도 기준)" },
  { key: "maxPbr", label: "PBR ≤", placeholder: "1", step: 0.1, title: "PBR 상한 (컨센서스 연도 기준)" },
];

/** 추세 필터 — 확정 회계연도 내내 우상향한 종목만 남긴다. */
const TREND_FILTERS: readonly {
  key: "requireOperatingTrend" | "requireNetTrend" | "requireDividendTrend";
  label: string;
  title: string;
}[] = [
  { key: "requireOperatingTrend", label: "영업익 우상향", title: "확정 회계연도 내내 전년 대비 증가한 종목만" },
  { key: "requireNetTrend", label: "순이익 우상향", title: "확정 회계연도 내내 전년 대비 증가한 종목만" },
  { key: "requireDividendTrend", label: "배당 우상향", title: "주당배당금이 확정 회계연도 내내 증가한 종목만" },
];

const percent = (value: number | null | undefined, digits = 2): string =>
  value === null || value === undefined ? "-" : `${value.toFixed(digits)}%`;

const signedPercent = (value: number | null | undefined): string =>
  value === null || value === undefined ? "-" : `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`;

/** 툴팁용 — 재무 금액은 억 단위가 읽기 쉽다. */
const formatEok = (value: number): string => `${Math.round(value / 1e8).toLocaleString("ko-KR")}억`;

/** 툴팁용 — 주당배당금은 원 단위 그대로. */
const formatWon = (value: number): string => `${Math.round(value).toLocaleString("ko-KR")}원`;

export function KorDividendManager({ onSummaryChange }: { onSummaryChange?: (count: number) => void }) {
  const toast = useToast();
  const [data, setData] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // 서버 렌더에는 localStorage 가 없어 초기값으로 못 쓴다 — 마운트 후 복원한다.
  const [filters, setFilters] = useState<DividendFilters>(EMPTY_FILTERS);

  useEffect(() => {
    setFilters(readRememberedFilters());
  }, []);

  const updateFilter = useCallback(<K extends keyof DividendFilters>(key: K, value: DividendFilters[K]) => {
    setFilters((current) => {
      const next = { ...current, [key]: value };
      writeRememberedFilters(next);
      return next;
    });
  }, []);

  const resetFilters = useCallback(() => {
    setFilters(EMPTY_FILTERS);
    writeRememberedFilters(EMPTY_FILTERS);
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch("/api/kor-dividend", { cache: "no-store" });
      const payload = (await resp.json()) as ApiResponse;
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "배당주 데이터를 불러오지 못했습니다.");
      setData(payload);
    } catch (err) {
      const message = err instanceof Error ? err.message : "배당주 데이터를 불러오지 못했습니다.";
      setError(message);
      toast.error(message);
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    void load();
  }, [load]);

  const allRows = useMemo(() => data?.rows ?? [], [data]);
  const rows = useMemo(() => allRows.filter((row) => passesFilters(row, filters)), [allRows, filters]);
  const activeFilterCount = countActiveFilters(filters);

  useEffect(() => {
    onSummaryChange?.(rows.length);
  }, [rows.length, onSummaryChange]);

  const columnDefs = useMemo<ColDef<DividendRow>[]>(() => {
    const consensusYear = data?.consensus_year ?? null;
    // 배당률 연도 컬럼 — 최신(컨센서스) → 과거. 표가 넓어지지 않게 3개만.
    const years = (data?.years ?? []).slice(0, 3);
    const returnMonths = data?.return_months ?? ["1", "3", "6", "12"];

    /** 추세 컬럼 — 연도별 원시값을 막대그래프로. 정렬은 최근 확정연도 값 기준. */
    const trendColumn = (
      field: "operating_income" | "net_income" | "dps",
      headerName: string,
      headerTooltip: string,
    ): ColDef<DividendRow> => ({
      colId: `trend_${field}`,
      headerName,
      width: 86,
      headerTooltip,
      // 정렬용 값 — 그래프는 모양만 주므로 최근 확정값으로 줄을 세운다.
      valueGetter: (params) => {
        const byYear = params.data?.by_year ?? {};
        const latest = Object.keys(byYear)
          .filter((year) => !byYear[year].is_consensus && byYear[year][field] !== null)
          .sort()
          .pop();
        return latest ? byYear[latest][field] : null;
      },
      cellRenderer: (params: { data?: DividendRow }) => {
        const byYear = params.data?.by_year ?? {};
        const points: TrendPoint[] = Object.keys(byYear)
          .sort()
          .map((year) => ({ label: year, value: byYear[year][field], estimated: byYear[year].is_consensus }));
        return <TrendSparkline points={points} format={field === "dps" ? formatWon : formatEok} />;
      },
    });


    return [
      {
        headerName: "순위",
        colId: "rank",
        width: 68,
        pinned: "left",
        valueGetter: (params) => (params.node?.rowIndex ?? 0) + 1,
        type: "numericColumn",
      },
      {
        // 티커·종목명 표기는 다른 시장 화면(/kor-market-stock 등)과 같은 방식을 쓴다.
        field: "ticker",
        headerName: "티커",
        width: 100,
        minWidth: 84,
        pinned: "left",
        cellStyle: {
          fontFamily: "var(--font-mono, monospace)",
          fontSize: "var(--fs-sm)",
        } as CellStyle,
        cellRenderer: (params: { value?: string }) => <TickerDetailLink ticker={String(params.value ?? "")} />,
      },
      {
        // 남는 가로는 종목명이 가져간다 — 이름이 길어 잘리는 게 숫자 컬럼이 넓어지는 것보다 아깝다.
        field: "name",
        headerName: "종목명",
        flex: 1,
        minWidth: 180,
        pinned: "left",
      },
      {
        field: "current_price",
        headerName: "현재가",
        width: 100,
        type: "numericColumn",
        valueFormatter: (params) => (params.value ? Number(params.value).toLocaleString("ko-KR") : "-"),
      },
      ...returnMonths.map<ColDef<DividendRow>>((months) => ({
        colId: `return_${months}`,
        headerName: Number(months) % 12 === 0 ? `${Number(months) / 12}년` : `${months}개월`,
        width: 84,
        type: "numericColumn",
        valueGetter: (params) => params.data?.returns?.[months] ?? null,
        valueFormatter: (params) => signedPercent(params.value),
        // 다른 화면과 같은 등락 색 클래스를 쓴다.
        cellClass: (params) =>
          params.value === null || params.value === undefined
            ? ""
            : params.value > 0
              ? "metricPositive"
              : params.value < 0
                ? "metricNegative"
                : "",
      })),
      ...years.map<ColDef<DividendRow>>((year) => ({
        colId: `dy_${year}`,
        headerName: year === consensusYear ? `${year}(예상)` : year,
        width: year === consensusYear ? 108 : 92,
        type: "numericColumn",
        valueGetter: (params) => params.data?.dividend_yield_by_year?.[year] ?? null,
        valueFormatter: (params) => percent(params.value),
        // 연도별 배당률은 이 화면의 본론이라 값을 굵게 둔다.
        cellStyle: { fontWeight: 700 },
        headerTooltip:
          year === consensusYear
            ? "컨센서스 예상 주당배당금 ÷ 현재가"
            : `${year}년 확정 주당배당금 ÷ ${year}년 연말 종가 (그 해에 실제로 받았을 수익률)`,
      })),
      {
        field: "shareholder_yield",
        headerName: "총환원율",
        width: 96,
        type: "numericColumn",
        valueFormatter: (params) => percent(params.value),
        headerTooltip: "배당률 + 자사주률 (DIVB 지수 정의)",
      },
      {
        field: "dividend_yield",
        headerName: "배당률",
        width: 90,
        type: "numericColumn",
        valueFormatter: (params) => percent(params.value),
        headerTooltip: "지금 사면 받을 배당수익률 — 컨센서스 예상 DPS 우선, 없으면 최근 확정 DPS ÷ 현재가",
      },
      {
        field: "buyback_yield",
        headerName: "자사주률",
        width: 96,
        type: "numericColumn",
        valueFormatter: (params) => percent(params.value),
        headerTooltip: "최근 2개 회계연도 자사주 순매입(취득−처분−발행) 연평균 ÷ 시가총액. 증자가 크면 음수",
      },
      {
        field: "payout_ratio_gross",
        headerName: "환원율",
        width: 92,
        type: "numericColumn",
        valueGetter: (params) =>
          params.data?.payout_ratio_gross === null || params.data?.payout_ratio_gross === undefined
            ? null
            : params.data.payout_ratio_gross * 100,
        valueFormatter: (params) => percent(params.value, 1),
        headerTooltip: "(배당지급 + 자사주취득) ÷ 순이익. 배당은 이듬해 현금흐름표에서 온 귀속연도 기준",
      },
      {
        field: "per",
        headerName: "PER",
        width: 78,
        type: "numericColumn",
        valueFormatter: (params) => (params.value === null || params.value === undefined ? "-" : Number(params.value).toFixed(1)),
      },
      {
        field: "pbr",
        headerName: "PBR",
        width: 78,
        type: "numericColumn",
        valueFormatter: (params) => (params.value === null || params.value === undefined ? "-" : Number(params.value).toFixed(2)),
      },
      trendColumn("operating_income", "영업익", "영업이익 연도별 추이 — 막대 높이가 값, 색은 전년 대비(증가 빨강·감소 파랑·유지 회색). 옅은 막대는 컨센서스 예상."),
      trendColumn("net_income", "순이익", "당기순이익 연도별 추이 — 막대 높이가 값, 색은 전년 대비(증가 빨강·감소 파랑·유지 회색). 옅은 막대는 컨센서스 예상."),
      trendColumn("dps", "배당", "주당배당금 연도별 추이 — 막대 높이가 값, 색은 전년 대비(증가 빨강·감소 파랑·유지 회색). 옅은 막대는 컨센서스 예상."),
    ];
  }, [data]);

  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                {/* 화면 이름은 PageFrame 제목이 이미 보여준다 — 여기서는 기준만 밝힌다. */}
                <div className="tableFooterMeta" style={{ margin: 0, color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
                  코스피 200 기준 · 재무는 매일 배치가 적재하고 배당률·자사주률은 현재가로 계산합니다.
                  지난 해 배당률의 분모는 그 해 연말 종가, 올해는 현재가입니다.
                </div>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: "var(--fs-sm)", color: "var(--text-muted)" }}>
                  {allRows.length}개 중 <strong style={{ color: "var(--ink)" }}>{rows.length}개</strong>
                  {activeFilterCount > 0 ? ` · 조건 ${activeFilterCount}개` : ""}
                </span>
                <button
                  type="button"
                  className="btn btn-sm btn-outline-secondary"
                  disabled={activeFilterCount === 0}
                  onClick={resetFilters}
                >
                  조건 초기화
                </button>
              </div>
            </div>
          </div>

          {/* 필터 — 빈칸은 조건 없음. 값이 없는(판정 불가) 종목은 조건이 걸리면 빠진다. */}
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              alignItems: "center",
              gap: 12,
              padding: "8px 16px",
              borderBottom: "1px solid var(--line)",
            }}
          >
            {NUMERIC_FILTERS.map((filter) => (
              <label key={filter.key} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                <span style={{ fontSize: "var(--fs-sm)", fontWeight: 600, color: "var(--text-muted)", whiteSpace: "nowrap" }}>
                  {filter.label}
                </span>
                <input
                  type="number"
                  step={filter.step}
                  placeholder={filter.placeholder}
                  title={filter.title}
                  style={{
                    width: 78,
                    border: "1px solid rgba(148,163,184,0.45)",
                    borderRadius: 6,
                    padding: "3px 6px",
                    fontSize: "var(--fs-sm)",
                  }}
                  value={filters[filter.key]}
                  onChange={(event) => updateFilter(filter.key, event.target.value)}
                />
              </label>
            ))}
            <span style={{ width: 1, alignSelf: "stretch", background: "var(--line)" }} />
            {TREND_FILTERS.map((filter) => (
              <label
                key={filter.key}
                style={{ display: "inline-flex", alignItems: "center", gap: 5, cursor: "pointer", fontSize: "var(--fs-sm)" }}
                title={filter.title}
              >
                <input
                  type="checkbox"
                  checked={filters[filter.key]}
                  onChange={(event) => updateFilter(filter.key, event.target.checked)}
                />
                <span style={{ fontWeight: 600 }}>{filter.label}</span>
              </label>
            ))}
          </div>

          <div className="card-body appCardBodyTight appTableCardBodyFill">
            {error ? (
              <div className="alert alert-warning mb-0" style={{ margin: 12 }}>
                {error}
              </div>
            ) : (
              <div className="appGridFillWrap">
                <AppAgGrid<DividendRow>
                  className="settingsAgGrid"
                  rowData={rows}
                  columnDefs={columnDefs}
                  loading={loading}
                  theme={gridTheme}
                  minHeight="100%"
                  getRowId={(params) => params.data.ticker}
                  gridOptions={{
                    suppressMovableColumns: true,
                    defaultColDef: {
                      sortable: true,
                      resizable: true,
                      tooltipValueGetter: (params) => params.valueFormatted ?? params.value,
                    },
                  }}
                />
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
