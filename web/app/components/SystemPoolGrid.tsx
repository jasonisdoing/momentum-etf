"use client";

import { useEffect, useState } from "react";
import type { ColDef } from "ag-grid-community";

import { AppAgGrid } from "./AppAgGrid";
import { createAppGridTheme } from "./app-grid-theme";

type SystemPoolRow = {
  id: string;
  order: number;
  pool: string;
  ticker_type: string;
  country_code: string;
  stock_count: number;
  rising_count: number;
  rising_ratio: number;
  score_up_count: number;
  score_total_count: number;
  score_up_ratio: number;
  etf_count: number;
};

function formatCount(value: number): string {
  return new Intl.NumberFormat("ko-KR").format(value);
}

function formatPercent(value: number): string {
  return `${new Intl.NumberFormat("ko-KR", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)}%`;
}

const appGridTheme = createAppGridTheme();

const poolColumns: ColDef<SystemPoolRow>[] = [
  {
    field: "order",
    headerName: "순서",
    minWidth: 72,
    flex: 0.45,
    type: "rightAligned",
    cellRenderer: (params: { value: number }) => formatCount(params.value),
  },
  { field: "pool", headerName: "종목풀", minWidth: 180, flex: 1.8 },
  { field: "ticker_type", headerName: "ID", minWidth: 100, flex: 0.7 },
  { field: "country_code", headerName: "국가", minWidth: 82, flex: 0.55 },
  {
    field: "stock_count",
    headerName: "종목수",
    minWidth: 100,
    flex: 0.65,
    type: "rightAligned",
    cellRenderer: (params: { value: number }) => formatCount(params.value),
  },
  {
    field: "rising_count",
    headerName: "상승수(일간)",
    minWidth: 100,
    flex: 0.75,
    type: "rightAligned",
    cellRenderer: (params: { value: number; data?: SystemPoolRow }) => {
      const total = params.data?.stock_count ?? 0;
      return `${formatCount(params.value)}/${formatCount(total)}`;
    },
  },
  {
    field: "rising_ratio",
    headerName: "상승비율(일간)",
    minWidth: 100,
    flex: 0.75,
    type: "rightAligned",
    cellStyle: { color: "#dc2626" },
    cellRenderer: (params: { value: number }) => formatPercent(params.value),
  },
  {
    field: "score_up_count",
    headerName: "상승수",
    minWidth: 100,
    flex: 0.75,
    type: "rightAligned",
    cellRenderer: (params: { value: number; data?: SystemPoolRow }) => {
      const total = params.data?.score_total_count ?? 0;
      return `${formatCount(params.value)}/${formatCount(total)}`;
    },
  },
  {
    field: "score_up_ratio",
    headerName: "상승비율",
    minWidth: 100,
    flex: 0.75,
    type: "rightAligned",
    cellStyle: { color: "#dc2626" },
    cellRenderer: (params: { value: number }) => formatPercent(params.value),
  },
  {
    field: "etf_count",
    headerName: "ETF",
    minWidth: 82,
    flex: 0.55,
    type: "rightAligned",
    cellRenderer: (params: { value: number }) => formatCount(params.value),
  },
];

/** 종목풀 요약 그리드 — `/api/system` 의 pool_rows 를 자체 조회해 렌더링한다. */
export function SystemPoolGrid() {
  const [rows, setRows] = useState<SystemPoolRow[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    async function load() {
      try {
        const resp = await fetch("/api/system", { cache: "no-store" });
        const payload = (await resp.json()) as { pool_rows?: SystemPoolRow[] };
        if (alive && resp.ok) setRows(payload.pool_rows ?? []);
      } catch {
        /* 무시 */
      } finally {
        if (alive) setLoading(false);
      }
    }
    void load();
    return () => {
      alive = false;
    };
  }, []);

  return (
    <section className="appSection">
      <div className="card appCard">
        <div className="card-header">
          <div className="appMainHeader">
            <div className="appMainHeaderLeft">
              <span className="appHeaderMetricValue">종목풀</span>
            </div>
          </div>
        </div>
        <div className="card-body appCardBodyTight">
          <AppAgGrid
            rowData={rows}
            columnDefs={poolColumns}
            loading={loading}
            minHeight="18rem"
            theme={appGridTheme}
            getRowId={(params) => params.data.id}
            gridOptions={{ suppressMovableColumns: true, domLayout: "autoHeight" }}
          />
        </div>
      </div>
    </section>
  );
}
