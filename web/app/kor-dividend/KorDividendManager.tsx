"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { ColDef } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";
import { createAppGridTheme } from "../components/app-grid-theme";
import { useToast } from "../components/ToastProvider";

/** 백엔드가 계산한 한 종목의 행. 가격이 필요한 값은 조회 시점에 계산돼 온다. */
type DividendRow = {
  ticker: string;
  name: string | null;
  index_weight: number | null;
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
  trend_operating: string | null;
  trend_net: string | null;
  trend_dividend: string | null;
  per: number | null;
  pbr: number | null;
  score: number;
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

const percent = (value: number | null | undefined, digits = 2): string =>
  value === null || value === undefined ? "-" : `${value.toFixed(digits)}%`;

const signedPercent = (value: number | null | undefined): string =>
  value === null || value === undefined ? "-" : `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`;

export function KorDividendManager({ onSummaryChange }: { onSummaryChange?: (count: number) => void }) {
  const toast = useToast();
  const [data, setData] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  const rows = useMemo(() => data?.rows ?? [], [data]);

  useEffect(() => {
    onSummaryChange?.(rows.length);
  }, [rows.length, onSummaryChange]);

  const columnDefs = useMemo<ColDef<DividendRow>[]>(() => {
    const consensusYear = data?.consensus_year ?? null;
    // 배당률 연도 컬럼 — 최신(컨센서스) → 과거. 표가 넓어지지 않게 3개만.
    const years = (data?.years ?? []).slice(0, 3);
    const returnMonths = data?.return_months ?? ["1", "3", "6", "12"];

    return [
      {
        headerName: "순위",
        colId: "rank",
        width: 68,
        pinned: "left",
        valueGetter: (params) => (params.node?.rowIndex ?? 0) + 1,
        type: "numericColumn",
      },
      { field: "ticker", headerName: "티커", width: 92, pinned: "left" },
      { field: "name", headerName: "종목명", width: 160, pinned: "left" },
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
        headerName: year === consensusYear ? `${year}(예)` : year,
        width: 92,
        type: "numericColumn",
        valueGetter: (params) => params.data?.dividend_yield_by_year?.[year] ?? null,
        valueFormatter: (params) => percent(params.value),
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
        field: "trend_operating",
        headerName: "영업익",
        width: 82,
        valueFormatter: (params) => params.value ?? "-",
        headerTooltip: "확정 회계연도 중 전년 대비 증가한 해의 비율",
      },
      { field: "trend_net", headerName: "순이익", width: 82, valueFormatter: (params) => params.value ?? "-" },
      { field: "trend_dividend", headerName: "배당", width: 76, valueFormatter: (params) => params.value ?? "-" },
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
      {
        field: "score",
        headerName: "점수",
        width: 84,
        type: "numericColumn",
        valueFormatter: (params) => (params.value === null || params.value === undefined ? "-" : Number(params.value).toFixed(1)),
        headerTooltip: "품질(실적·배당·환원 60) + 가격(PER·배당률 40) = 100점",
      },
      {
        field: "index_weight",
        headerName: "지수비중",
        flex: 1,
        minWidth: 96,
        type: "numericColumn",
        valueFormatter: (params) => percent(params.value),
        headerTooltip: "KOSPI200(KODEX 200) 내 비중",
      },
    ];
  }, [data]);

  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <div>
                  <h2 style={{ fontSize: "var(--fs-lg)", fontWeight: 800, margin: 0 }}>한국 배당주</h2>
                  <div className="tableFooterMeta" style={{ margin: 0, color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
                    KOSPI200(KODEX 200 보유종목) · 재무는 매일 배치가 적재하고 배당률·자사주률은 현재가로 계산합니다.
                    지난 해 배당률의 분모는 그 해 연말 종가, 올해는 현재가입니다.
                  </div>
                </div>
              </div>
            </div>
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
