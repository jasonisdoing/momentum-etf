"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { CellStyle, ColDef, ColGroupDef } from "ag-grid-community";

import { marketCapRankColumn, renderHighDrawdownCell } from "@/lib/grid-cells";
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
  /** 시장 전체 시가총액 순위 (배치 기준). */
  market_cap_rank: number | null;
  /** 최근 12개월 최고가 대비(%). 0 이면 신고점. */
  high_drawdown: number | null;
  current_price: number | null;
  market_cap: number | null;
  /** 개월 → 수익률(%). 키는 문자열("1","3","6","12"). */
  returns: Record<string, number | null>;
  /** 연도 → 배당률(%). 지난 해는 연말 종가, 올해는 현재가 기준. */
  dividend_yield_by_year: Record<string, number | null>;
  /** 연도 → 자사주률(%). 그 해 자사주 순매입 ÷ 그 해 시가총액. */
  buyback_yield_by_year: Record<string, number | null>;
  /** 연도 → 총환원율(%) = 배당률 + 자사주률. 같은 해, 같은 분모라 기준이 섞이지 않는다. */
  shareholder_yield_by_year: Record<string, number | null>;
  dividend_yield: number | null;
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
  {
    key: "minDividendYield",
    label: "2026(예상) 배당률 ≥",
    placeholder: "4",
    step: 0.1,
    title: "컨센서스 연도 배당률(%) 하한 — 표의 2026(예상) 칸과 같은 값. 그 칸이 빈 종목은 조건을 걸면 빠집니다.",
  },
  { key: "minShareholderYield", label: "총환원율 ≥", placeholder: "0", step: 0.1, title: "배당률 + 자사주률(%) 하한" },
  { key: "minPayoutRatio", label: "이익대비 환원 ≥", placeholder: "30", step: 1, title: "(배당지급 + 자사주취득) ÷ 당기순이익(%) 하한" },
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

/**
 * 표를 읽을 때 알아야 하는 것 — 숫자만 봐서는 **오해하는** 항목만 남긴다.
 * (컬럼별 정의는 각 헤더의 툴팁에 있으므로 여기서 되풀이하지 않는다.)
 */
const TABLE_NOTES: readonly { title: string; body: string }[] = [
  {
    title: "배당률 분모는 해마다 다름",
    body: "지난 해는 그 해 연말 종가, 올해(예상)는 현재가. 배당이 늘어도 주가가 더 오르면 배당률은 떨어집니다.",
  },
  {
    title: "환원율 = 그 해 배당률 + 자사주률",
    body: "같은 해, 같은 분모(그 해 시가총액)로 냅니다. 유상증자가 자사주 취득보다 크면 음수가 됩니다. 2026 은 자사주 컨센서스가 없어 배당률만 있습니다.",
  },
  {
    title: "이익대비 환원은 기준연도가 한 해 전",
    body: "(배당지급 + 자사주취득) ÷ 순이익입니다. 현금흐름표의 배당 지급은 전년도 결산배당이라 귀속연도로 되돌리며, 최근 확정연도는 다음 사업보고서가 나와야 채워집니다. 값 옆 괄호가 기준연도입니다.",
  },
  {
    title: "추세 막대",
    body: "높이가 값, 색은 전년 대비(증가 빨강·감소 파랑·유지 회색). 옅은 막대는 예상치. 마우스를 올리면 연도별 값이 보입니다.",
  },
  {
    title: "우선주 없음 · 빈 값은 제외",
    body: "코스피 200 이 보통주로만 구성되고, 자사주률이 우선주에서 부풀려지는 문제도 함께 피합니다. 필터를 걸면 값이 없어 판정 못 하는 종목은 빠집니다.",
  },
];

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

  const columnDefs = useMemo<(ColDef<DividendRow> | ColGroupDef<DividendRow>)[]>(() => {
    const consensusYear = data?.consensus_year ?? null;
    // 연도 그룹 — 최신(컨센서스) → 과거 4개.
    const years = (data?.years ?? []).slice(0, 4);
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
        valueGetter: (params) => (params.node?.rowIndex ?? 0) + 1,
        type: "numericColumn",
      },
      // 시총 순위·고점 대비 — 종목풀 순위(/pools-rank) 화면과 같은 공용 컬럼·같은 정의.
      marketCapRankColumn<DividendRow>("market_cap_rank", false),
      {
        field: "high_drawdown",
        headerName: "고점",
        width: 80,
        minWidth: 80,
        type: "rightAligned",
        headerTooltip: "최근 12개월 최고가 대비(%) — 0 이면 신고점",
        cellRenderer: (params: { value: number | null | undefined }) => renderHighDrawdownCell(params.value, 2),
      },
      {
        // 티커·종목명 표기는 다른 시장 화면(/kor-market-stock 등)과 같은 방식을 쓴다.
        field: "ticker",
        headerName: "티커",
        width: 100,
        minWidth: 84,
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
      // 연도별 배당률·총환원율 — 여러 해를 평균 내면 배당은 예상, 자사주는 지난 실적이 되어
      // 기준이 섞인다. 같은 해, 같은 분모(그 해 시가총액)로 나란히 둔다.
      ...years.map<ColGroupDef<DividendRow>>((year) => {
        const isConsensus = year === consensusYear;
        return {
          groupId: `year_${year}`,
          headerName: isConsensus ? `${year}(예상)` : year,
          marryChildren: true,
          children: [
            {
              colId: `dy_${year}`,
              headerName: "배당률",
              // 컨센서스 연도는 그룹 헤더('2026(예상)')가 자식 하나뿐이라, 자식 폭이 좁으면
              // 그룹 헤더 글자가 잘린다. 그 해만 넓게 둔다.
              width: isConsensus ? 104 : 92,
              type: "numericColumn",
              valueGetter: (params) => params.data?.dividend_yield_by_year?.[year] ?? null,
              valueFormatter: (params) => percent(params.value),
              // 배당률은 이 화면의 본론이라 값을 굵게 둔다.
              cellStyle: { fontWeight: 700 },
              headerTooltip: isConsensus
                ? "컨센서스 예상 주당배당금 ÷ 현재가"
                : `${year}년 확정 주당배당금 ÷ ${year}년 연말 종가 (그 해에 실제로 받았을 수익률)`,
            },
            // 컨센서스 연도는 자사주 예상치가 없어(DART·네이버 모두) 환원율 컬럼을 두지 않는다.
            ...(isConsensus
              ? []
              : [
                  {
                    colId: `sy_${year}`,
                    headerName: "환원율",
                    width: 92,
                    type: "numericColumn",
                    valueGetter: (params) => params.data?.shareholder_yield_by_year?.[year] ?? null,
                    valueFormatter: (params) => percent(params.value),
                    headerTooltip: `${year}년 배당률 + 자사주률 (자사주 순매입 ÷ ${year}년 시가총액). 유상증자가 자사주 취득보다 크면 음수`,
                    cellClass: (params) =>
                      params.value !== null && params.value !== undefined && params.value < 0 ? "metricNegative" : "",
                  } satisfies ColDef<DividendRow>,
                ]),
          ],
        };
      }),
      {
        // 연도 그룹의 '환원율'(시가총액 대비)과 이름이 겹치지 않게 '이익대비 환원'으로 둔다.
        // 기준연도가 종목마다 다를 수 있어(배당지급이 이듬해 공시라) 값 옆에 연도를 붙인다.
        field: "payout_ratio_gross",
        headerName: "이익대비 환원",
        width: 132,
        type: "numericColumn",
        valueGetter: (params) =>
          params.data?.payout_ratio_gross === null || params.data?.payout_ratio_gross === undefined
            ? null
            : params.data.payout_ratio_gross * 100,
        valueFormatter: (params) => {
          if (params.value === null || params.value === undefined) return "-";
          const year = params.data?.payout_base_year;
          return `${params.value.toFixed(1)}%${year ? ` (${year})` : ""}`;
        },
        headerTooltip:
          "(배당지급 + 자사주취득) ÷ 당기순이익 — 번 돈의 몇 %를 주주에게 돌려줬나. 괄호는 기준 회계연도. " +
          "배당지급은 이듬해 현금흐름표에서 와 귀속연도로 되돌린 값이라, 가장 최근 확정연도는 아직 채워지지 않는다.",
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
                {/* tableFooterMeta 는 표 하단용이라 우측 정렬이 걸려 있다 — 헤더 설명에는 쓰지 않는다. */}
                <div style={{ margin: 0, color: "var(--text-muted)", fontSize: "var(--fs-sm)", textAlign: "left" }}>
                  코스피 200 기준 · 재무는 매일 배치가 적재하고 배당률·자사주률은 현재가로 계산합니다.
                  <ul style={{ margin: "4px 0 0", paddingLeft: 16, lineHeight: 1.6 }}>
                    {TABLE_NOTES.map((note) => (
                      <li key={note.title}>
                        <strong>{note.title}</strong> — {note.body}
                      </li>
                    ))}
                  </ul>
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
