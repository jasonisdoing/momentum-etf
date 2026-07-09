"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { ColDef, GridOptions } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";
import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";

type TopPickRow = {
  ticker: string;
  name: string;
  ticker_type?: string;
  country_code?: string;
  trend_pct: number | null;
  trend_score: number | null;
  sortino_score: number | null;
  sortino?: number | null;
  score: number | null;
  target_weight_pct: number | null;
  current_price?: number | null;
  target_amount_krw?: number | null;
  target_quantity?: number | null;
  current_quantity?: number | null;
  current_amount_krw?: number | null;
  change_quantity?: number | null;
  change_weight_pct?: number | null;
  unallocated_amount_krw?: number | null;
};

type TopPickTradeSummary = {
  account_id?: string;
  account_name?: string;
  account_amount_krw?: number;
  target_asset_amount_krw?: number;
  remaining_cash_krw?: number;
};

type TopPickPayload = {
  as_of_date?: string;
  rows?: TopPickRow[];
  missing_tickers?: string[];
  settings?: {
    MA_TYPE: string;
    MA_MONTHS: number;
    MIN_WEIGHT: number;
    MAX_WEIGHT: number;
    CASH_MAX_WEIGHT: number;
    ACCOUNT_ID: string;
  };
  trade_summary?: TopPickTradeSummary;
  error?: string;
};

function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return value.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatWeightPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${Number(value.toFixed(1))}%`;
}

function formatKrw(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${Math.round(value).toLocaleString("ko-KR")}원`;
}

function formatQuantity(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${Math.floor(value).toLocaleString("ko-KR")}주`;
}

function formatChangeQuantity(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  const quantity = Math.trunc(value);
  if (quantity > 0) return `+${quantity.toLocaleString("ko-KR")}주`;
  return `${quantity.toLocaleString("ko-KR")}주`;
}

function signedQuantityColor(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) {
    return "#475569";
  }
  return value > 0 ? "#d63939" : "#206bc4";
}

function formatTicker(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return value === "__CASH__" ? "CASH" : value;
}

const gridTheme = createAppGridTheme();

export function TopPickClient() {
  const toast = useToast();
  const [payload, setPayload] = useState<TopPickPayload | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const resp = await fetch("/api/top-pick", { cache: "no-store" });
      const data = (await resp.json()) as TopPickPayload;
      if (!resp.ok || data.error) {
        throw new Error(data.error ?? "탑픽 비중을 불러오지 못했습니다.");
      }
      setPayload(data);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "탑픽 비중을 불러오지 못했습니다.");
      setPayload(null);
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    void load();
  }, [load]);

  const rows = payload?.rows ?? [];
  const settings = payload?.settings;
  const missing = payload?.missing_tickers ?? [];
  const tradeSummary = payload?.trade_summary ?? {};
  const cashWeight = rows.find((row) => row.ticker === "__CASH__")?.target_weight_pct ?? null;
  const etfCount = rows.filter((row) => row.ticker !== "__CASH__").length;

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>종목:</span>
          <span className="appHeaderMetricValue">{etfCount}개</span>
        </div>
        <div className="appHeaderMetric">
          <span>현금:</span>
          <span className="appHeaderMetricValue">{formatWeightPct(cashWeight)} / {formatKrw(tradeSummary.remaining_cash_krw)}</span>
        </div>
        <div className="appHeaderMetric">
          <span>계좌:</span>
          <span className="appHeaderMetricValue">{tradeSummary.account_name ?? settings?.ACCOUNT_ID ?? "-"}</span>
        </div>
        <div className="appHeaderMetric">
          <span>총자산:</span>
          <span className="appHeaderMetricValue">{formatKrw(tradeSummary.account_amount_krw)}</span>
        </div>
        <div className="appHeaderMetric">
          <span>기준일:</span>
          <span className="appHeaderMetricValue">{payload?.as_of_date ?? "-"}</span>
        </div>
      </div>
    ),
    [cashWeight, etfCount, payload?.as_of_date, settings?.ACCOUNT_ID, tradeSummary.account_amount_krw, tradeSummary.account_name, tradeSummary.remaining_cash_krw],
  );

  const columns = useMemo<ColDef<TopPickRow>[]>(
    () => [
      {
        field: "ticker",
        headerName: "티커",
        width: 110,
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: string | null | undefined }) => formatTicker(params.value),
      },
      {
        field: "name",
        headerName: "종목명",
        minWidth: 240,
        flex: 1,
      },
      {
        field: "trend_pct",
        headerName: "추세(%)",
        width: 100,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatNumber(params.value),
      },
      {
        field: "trend_score",
        headerName: "추세점수",
        width: 100,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatNumber(params.value),
      },
      {
        field: "sortino",
        headerName: "Sortino",
        width: 100,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatNumber(params.value),
      },      {
        field: "target_weight_pct",
        headerName: "목표비중",
        width: 110,
        type: "rightAligned",
        cellStyle: { fontWeight: 800 },
        cellRenderer: (params: { value: number | null | undefined }) => formatWeightPct(params.value),
      },
      {
        field: "current_price",
        headerName: "현재가",
        width: 112,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatKrw(params.value),
      },
      {
        field: "target_amount_krw",
        headerName: "목표금액",
        width: 128,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatKrw(params.value),
      },
      {
        field: "target_quantity",
        headerName: "목표수량",
        width: 110,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) => formatQuantity(params.value),
      },
      {
        field: "current_quantity",
        headerName: "현재수량",
        width: 110,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatQuantity(params.value),
      },
      {
        field: "current_amount_krw",
        headerName: "현재금액",
        width: 128,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatKrw(params.value),
      },
      {
        field: "change_weight_pct",
        headerName: "변동비중",
        width: 100,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) => {
          if (params.value == null || Number.isNaN(params.value)) return "-";
          const val = params.value;
          const formatted = `${val > 0 ? "+" : ""}${val.toFixed(1)}%`;
          const color = val === 0 ? "#475569" : val > 0 ? "#d63939" : "#206bc4";
          return <span style={{ color }}>{formatted}</span>;
        },
      },
      {
        field: "change_quantity",
        headerName: "변동수량",
        width: 110,
        type: "rightAligned",
        cellStyle: { fontWeight: 800 },
        cellRenderer: (params: { value: number | null | undefined }) => (
          <span style={{ color: signedQuantityColor(params.value) }}>{formatChangeQuantity(params.value)}</span>
        ),
      },
    ],
    [],
  );

  const gridOptions = useMemo<GridOptions<TopPickRow>>(
    () => ({
      domLayout: "autoHeight",
      suppressMovableColumns: true,
      overlayNoRowsTemplate: '<span style="color:#667382;">표시할 탑픽 비중이 없습니다.</span>',
    }),
    [],
  );

  return (
    <PageFrame title="탑픽 비중" fullWidth titleRight={titleRight}>
      <div className="appPageStack">
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>목표 비중</h2>
                <div style={{ color: "#64748b", fontSize: "0.86rem" }}>
                  기준일 {payload?.as_of_date ?? "-"} · {settings ? `${settings.MA_TYPE} ${settings.MA_MONTHS}개월 (추세 100%)` : "설정 없음"}
                </div>
              </div>
              <div className="appMainHeaderRight">
                <button type="button" className="btn btn-sm btn-outline-secondary" disabled={loading} onClick={() => void load()}>
                  새로고침
                </button>
              </div>
            </div>
            {settings ? (
              <div style={{ color: "#64748b", fontSize: "0.82rem", marginTop: 8 }}>
                적용계좌 {tradeSummary.account_name ?? settings.ACCOUNT_ID} · 최소 {settings.MIN_WEIGHT}% · 최대 {settings.MAX_WEIGHT}% · 현금 최대 {settings.CASH_MAX_WEIGHT}%
              </div>
            ) : null}
            {missing.length > 0 ? (
              <div className="alert alert-warning" style={{ marginTop: 12, marginBottom: 0 }}>
                가격 캐시 누락: {missing.join(", ")}
              </div>
            ) : null}
          </div>
        </div>

        <div className="card appCard">
          <div className="card-body appCardBodyTight">
            <AppAgGrid<TopPickRow>
              rowData={rows}
              columnDefs={columns}
              loading={loading}
              minHeight="auto"
              className="topPickWeightGrid"
              theme={gridTheme}
              getRowId={(params) => params.data.ticker}
              gridOptions={gridOptions}
            />
          </div>
        </div>
      </div>
      <style jsx global>{`
        .topPickWeightGrid {
          height: auto !important;
        }

        .topPickWeightGrid .appAgGridTheme {
          height: auto;
        }
      `}</style>
    </PageFrame>
  );
}
