"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import type { ColDef, RowClassParams, GridOptions } from "ag-grid-community";
import { themeQuartz, iconSetQuartzBold } from "ag-grid-community";
import { PageFrame } from "../components/PageFrame";
import { AppAgGrid } from "../components/AppAgGrid";

// ─── 타입 ────────────────────────────────────────────────────────────────────
type AccountOption = {
  account_id: string;
  name: string;
};

type ComponentSource = {
  etf_ticker: string;
  etf_name: string;
  weight: number;
};

type ComponentRow = {
  ticker: string;
  name: string;
  total_weight: number;
  sources: ComponentSource[];
  current_price?: number | null;
  change_pct?: number | null;
};

type EtfDetail = {
  ticker: string;
  name: string;
  quantity: number;
  component_count: number;
  has_components: boolean;
};

type HoldingsComponentsData = {
  account_id: string;
  account_name: string;
  held_etf_count: number;
  components: ComponentRow[];
  etf_details: EtfDetail[];
};

type MainGridRow = ComponentRow & { rowType: "main" };
type DetailGridRow = { rowType: "detail"; parentTicker: string; sources: ComponentSource[] };
type GridRow = MainGridRow | DetailGridRow;

// ─── 유틸 ────────────────────────────────────────────────────────────────────
function formatWeight(w: number): string {
  return `${w.toFixed(2)}%`;
}

function formatPrice(val: number | null | undefined): string {
  if (val == null) return "-";
  return `${Math.floor(val).toLocaleString()}원`;
}

function formatSignedPercent(val: number | null | undefined): string {
  if (val == null) return "-";
  const sign = val > 0 ? "+" : "";
  return `${sign}${val.toFixed(2)}%`;
}

function getSignedClass(val: number | null | undefined): string {
  if (val == null) return "";
  if (val > 0) return "text-success";
  if (val < 0) return "text-danger";
  return "";
}

const gridTheme = themeQuartz
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

function isDetailRow(row: GridRow | undefined): row is DetailGridRow {
  return row?.rowType === "detail";
}

// ─── 컴포넌트 ─────────────────────────────────────────────────────────────────
export function HoldingsDetailsPageClient() {
  const [accounts, setAccounts] = useState<AccountOption[]>([]);
  const [selectedAccount, setSelectedAccount] = useState<string>("");
  const [data, setData] = useState<HoldingsComponentsData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);

  // 계좌 목록 로드
  useEffect(() => {
    async function fetchAccounts() {
      try {
        const res = await fetch("/api/holdings-components/accounts", { cache: "no-store" });
        if (!res.ok) throw new Error("계좌 목록을 불러오지 못했습니다.");
        const list = (await res.json()) as AccountOption[];
        setAccounts(list);
        if (list.length > 0) setSelectedAccount(list[0].account_id);
      } catch (e) {
        setError(e instanceof Error ? e.message : "오류가 발생했습니다.");
      }
    }
    void fetchAccounts();
  }, []);

  // 선택된 계좌 데이터 로드
  const loadData = useCallback(async (accountId: string) => {
    if (!accountId) return;
    setIsLoading(true);
    setError(null);
    setData(null);
    setExpandedTicker(null);
    try {
      const res = await fetch(`/api/holdings-components?account_id=${encodeURIComponent(accountId)}`, {
        cache: "no-store",
      });
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? "데이터를 불러오지 못했습니다.");
      }
      const result = (await res.json()) as HoldingsComponentsData;
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selectedAccount) void loadData(selectedAccount);
  }, [selectedAccount, loadData]);

  // 그리드 데이터 가공
  const gridRows = useMemo<GridRow[]>(() => {
    if (!data) return [];
    const result: GridRow[] = [];
    for (const comp of data.components) {
      result.push({ ...comp, rowType: "main" });
      if (expandedTicker === comp.ticker && comp.sources.length > 0) {
        result.push({
          rowType: "detail",
          parentTicker: comp.ticker,
          sources: comp.sources,
        });
      }
    }
    return result;
  }, [data, expandedTicker]);

  const columnDefs = useMemo<ColDef<GridRow>[]>(() => [
    {
      headerName: "티커",
      field: "ticker",
      width: 100,
      cellRenderer: (params: { data?: GridRow; value?: string }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        return <span className="text-muted fw-semibold" style={{ fontFamily: "monospace" }}>{params.value}</span>;
      },
    },
    {
      headerName: "종목명",
      field: "name",
      flex: 1,
      minWidth: 180,
      cellRenderer: (params: { data?: GridRow; value?: string }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const mainRow = params.data as MainGridRow;
        const isExpanded = expandedTicker === mainRow.ticker;
        const hasSources = mainRow.sources.length > 0;
        return (
          <div className={`d-flex align-items-center gap-2 ${hasSources ? "cursor-pointer" : ""}`} style={{ userSelect: "none" }}>
            {hasSources && (
              <span className="text-primary d-flex align-items-center" style={{ fontSize: "10px", transition: "transform 0.15s", transform: isExpanded ? "rotate(90deg)" : "none" }}>
                ▶
              </span>
            )}
            <span className="fw-bold text-dark">{params.value}</span>
          </div>
        );
      },
    },
    {
      headerName: "비중",
      field: "total_weight",
      width: 90,
      type: "rightAligned",
      cellRenderer: (params: { value?: number }) => (
        <span className="fw-bold" style={{ color: "#206bc4" }}>
          {params.value != null ? formatWeight(params.value) : "-"}
        </span>
      ),
    },
    {
      headerName: "현재가",
      field: "current_price",
      width: 110,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: number | null }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const row = params.data as MainGridRow;
        if (row.ticker === "-") return "-";
        return formatPrice(params.value);
      },
    },
    {
      headerName: "일간(%)",
      field: "change_pct",
      width: 100,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: number | null }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const row = params.data as MainGridRow;
        if (row.ticker === "-") return "-";
        return <span className={getSignedClass(params.value)}>{formatSignedPercent(params.value)}</span>;
      },
    },
  ], [expandedTicker]);

  // 상세 패널 렌더러
  const DetailRenderer = useCallback((params: { data?: GridRow }) => {
    if (!params.data || !isDetailRow(params.data)) return null;
    return (
      <div className="bg-light px-3 py-2 border-bottom" style={{ marginLeft: "40px" }}>
        <div className="d-flex flex-column gap-1">
          {params.data.sources.map((src, idx) => (
            <div key={idx} className="d-flex align-items-center justify-content-between py-1 px-2 bg-white rounded border shadow-sm" style={{ fontSize: "12.5px" }}>
              <div className="d-flex align-items-center gap-2 text-muted">
                <span>└ {src.etf_name}</span>
                <span className="badge bg-secondary-subtle text-secondary py-1 px-2" style={{ fontSize: "10px" }}>{src.etf_ticker}</span>
              </div>
              <span className="fw-bold text-primary">{formatWeight(src.weight)}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }, []);

  const gridOptions = useMemo<GridOptions<GridRow>>(() => ({
    getRowId: (params) => (isDetailRow(params.data) ? `detail:${params.data.parentTicker}` : params.data.ticker),
    isFullWidthRow: (params) => isDetailRow(params.rowNode.data),
    fullWidthCellRenderer: DetailRenderer,
    getRowHeight: (params) => (isDetailRow(params.data) ? 20 + params.data.sources.length * 36 : 38),
    onCellClicked: (params) => {
      if (params.data && !isDetailRow(params.data) && params.colDef.field === "name") {
        const ticker = (params.data as MainGridRow).ticker;
        setExpandedTicker((prev) => (prev === ticker ? null : ticker));
      }
    },
    overlayNoRowsTemplate: '<span class="ag-overlay-no-rows-center">데이터가 없습니다.</span>',
  }), [DetailRenderer]);

  // 헤더 우측 정보
  const titleRight = data ? (
    <div className="appHeaderMetrics rankToolbarMeta">
      <div className="appHeaderMetric">
        <span>보유 ETF:</span>
        <span className="appHeaderMetricValue">{data.held_etf_count}개</span>
      </div>
      <div className="appHeaderMetric">
        <span>구성종목:</span>
        <span className="appHeaderMetricValue">{data.components.length}개</span>
      </div>
    </div>
  ) : null;

  return (
    <PageFrame title="보유종목 상세" fullHeight fullWidth titleRight={titleRight}>
      {error && (
        <div className="appBannerStack">
          <div className="bannerError alert alert-danger mb-0">{error}</div>
        </div>
      )}

      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft rankMainHeaderLeft">
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">계좌</span>
                  <select
                    className="form-select"
                    value={selectedAccount}
                    onChange={(e) => setSelectedAccount(e.target.value)}
                    disabled={accounts.length === 0}
                  >
                    {accounts.length === 0 ? (
                      <option value="">계좌 불러오는 중...</option>
                    ) : (
                      accounts.map((acc) => (
                        <option key={acc.account_id} value={acc.account_id}>
                          {acc.name}
                        </option>
                      ))
                    )}
                  </select>
                </label>
              </div>
            </div>
          </div>
          
          <div className="card-body appCardBodyTight appTableCardBodyFill">
            <div className="appGridFillWrap">
              <AppAgGrid
                rowData={isLoading ? [] : gridRows}
                columnDefs={columnDefs}
                loading={isLoading}
                theme={gridTheme}
                minHeight="100%"
                gridOptions={gridOptions}
                getRowClass={(params: RowClassParams<GridRow>) => {
                  if (isDetailRow(params.data)) return "holdingsDetailFullRow";
                  return "";
                }}
              />
            </div>
          </div>
        </div>
      </section>

      <style jsx global>{`
        .holdingsDetailFullRow {
          background-color: transparent !important;
          border-bottom: 1px solid #e2e8f0;
        }
        .bg-light { background-color: #f8fafc !important; }
        .cursor-pointer { cursor: pointer; }
      `}</style>
    </PageFrame>
  );
}
