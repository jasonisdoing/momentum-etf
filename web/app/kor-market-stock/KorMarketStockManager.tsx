"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { iconSetQuartzBold, themeQuartz } from "ag-grid-community";
import type { ColDef } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";

type KorMarketStockRow = {
  rank: number;
  ticker: string;
  name: string;
  current_price: number | null;
  change_pct: number | null;
  volume: number | null;
  market_cap: number | null;
};

type KorMarketStocksResponse = {
  market: string;
  total_count: number;
  count: number;
  rows: KorMarketStockRow[];
  error?: string;
};

const korMarketStockGridTheme = themeQuartz
  .withPart(iconSetQuartzBold)
  .withParams({
    accentColor: "#206bc4",
    backgroundColor: "#ffffff",
    foregroundColor: "#182433",
    headerBackgroundColor: "#f8fafc",
    headerTextColor: "#5b6778",
    spacing: 8,
    fontSize: 14,
    wrapperBorderRadius: 10,
    rowHeight: 38,
    headerHeight: 38,
    cellHorizontalPadding: 12,
    headerColumnBorder: true,
    headerColumnBorderHeight: "70%",
    columnBorder: true,
    oddRowBackgroundColor: "#fbfdff",
    headerCellHoverBackgroundColor: "#eef4fb",
    headerCellMovingBackgroundColor: "#e8f0fb",
    iconButtonHoverBackgroundColor: "#eef4fb",
    iconButtonHoverColor: "#206bc4",
    iconSize: 18,
  });

function formatKrw(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR").format(value);
}

function formatPercent(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function formatVolume(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR").format(value);
}

function formatMarketCap(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (value >= 10000) {
    const jo = Math.floor(value / 10000);
    const eok = value % 10000;
    return eok > 0
      ? `${new Intl.NumberFormat("ko-KR").format(jo)}조 ${new Intl.NumberFormat("ko-KR").format(eok)}억`
      : `${new Intl.NumberFormat("ko-KR").format(jo)}조`;
  }
  return `${new Intl.NumberFormat("ko-KR").format(value)}억`;
}

const MARKET_OPTIONS = ["KOSPI", "KOSDAQ"] as const;
const LIMIT_OPTIONS = [30, 50, 100] as const;

const columnDefs: ColDef<KorMarketStockRow>[] = [
  {
    headerName: "#",
    field: "rank",
    width: 64,
    minWidth: 56,
    maxWidth: 76,
    sortable: false,
    resizable: false,
    cellStyle: { textAlign: "center", color: "#8896a6" },
  },
  {
    headerName: "티커",
    field: "ticker",
    width: 100,
    minWidth: 84,
    cellStyle: { fontFamily: "var(--font-mono, monospace)", fontSize: "13px" },
  },
  {
    headerName: "종목명",
    field: "name",
    flex: 1,
    minWidth: 180,
  },
  {
    headerName: "현재가",
    field: "current_price",
    width: 130,
    minWidth: 108,
    type: "rightAligned",
    valueFormatter: (p) => formatKrw(p.value),
  },
  {
    headerName: "등락률",
    field: "change_pct",
    width: 110,
    minWidth: 96,
    type: "rightAligned",
    valueFormatter: (p) => formatPercent(p.value),
    cellClassRules: {
      metricPositive: (p) => p.value != null && p.value > 0,
      metricNegative: (p) => p.value != null && p.value < 0,
    },
  },
  {
    headerName: "거래량",
    field: "volume",
    width: 140,
    minWidth: 120,
    type: "rightAligned",
    valueFormatter: (p) => formatVolume(p.value),
  },
  {
    headerName: "시가총액",
    field: "market_cap",
    width: 160,
    minWidth: 140,
    type: "rightAligned",
    valueFormatter: (p) => formatMarketCap(p.value),
  },
];

export function KorMarketStockManager({
  onSummaryChange,
}: {
  onSummaryChange?: (summary: { market: string; count: number; totalCount: number }) => void;
}) {
  const [market, setMarket] = useState<(typeof MARKET_OPTIONS)[number]>("KOSPI");
  const [limit, setLimit] = useState<(typeof LIMIT_OPTIONS)[number]>(50);
  const [rows, setRows] = useState<KorMarketStockRow[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (m: string, l: number) => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`/api/kor-market-stocks?market=${m}&limit=${l}`, { cache: "no-store" });
      const data = (await resp.json()) as KorMarketStocksResponse;
      if (!resp.ok) {
        throw new Error(data.error ?? "데이터를 불러오지 못했습니다.");
      }
      setRows(data.rows ?? []);
      setTotalCount(data.total_count ?? 0);
    } catch (e) {
      setError(e instanceof Error ? e.message : "데이터를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(market, limit);
  }, [market, limit, load]);

  useEffect(() => {
    onSummaryChange?.({ market, count: rows.length, totalCount });
  }, [market, rows.length, totalCount, onSummaryChange]);

  return (
    <section className="appSection appSectionFill">
      <div className="card appCard appTableCardFill">
        {/* 메인 헤더 */}
        <div className="card-header">
          <div className="appMainHeader">
            <div className="appMainHeaderLeft korMarketStockMainHeaderLeft">
              <label className="appLabeledField">
                <span className="appLabeledFieldLabel">마켓</span>
                <div className="appSegmentedToggle appSegmentedToggleCompact" role="group" aria-label="마켓 선택">
                  {MARKET_OPTIONS.map((opt) => (
                    <button
                      key={opt}
                      type="button"
                      className={market === opt ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                      onClick={() => setMarket(opt)}
                    >
                      {opt === "KOSPI" ? "코스피" : "코스닥"}
                    </button>
                  ))}
                </div>
              </label>

              <label className="appLabeledField">
                <span className="appLabeledFieldLabel">시가총액 상위</span>
                <select
                  className="form-select"
                  value={limit}
                  onChange={(e) => setLimit(Number(e.target.value) as (typeof LIMIT_OPTIONS)[number])}
                >
                  {LIMIT_OPTIONS.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}개
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>
        </div>

        <div className="card-body appCardBodyTight appTableCardBodyFill">
          {error && (
            <div style={{ padding: "0.5rem 0.75rem", marginBottom: "0.5rem", background: "#fef2f2", color: "#dc2626", borderRadius: "6px", fontSize: "0.85rem" }}>
              {error}
            </div>
          )}

          <div className="appGridFillWrap">
            <AppAgGrid<KorMarketStockRow>
              rowData={rows}
              columnDefs={columnDefs}
              loading={loading}
              theme={korMarketStockGridTheme}
              minHeight="32rem"
              gridOptions={{
                overlayNoRowsTemplate: '<span style="color:#667382;">데이터 없음</span>',
              }}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
