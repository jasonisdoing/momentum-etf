"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ColDef, RowClassParams, GridOptions } from "ag-grid-community";
import { themeQuartz, iconSetQuartzBold } from "ag-grid-community";
import { PageFrame } from "../components/PageFrame";
import { AppAgGrid } from "../components/AppAgGrid";
import { ResponsiveFiltersSection } from "../components/ResponsiveFiltersSection";
import {
  readRememberedMomentumEtfAccountId,
  writeRememberedMomentumEtfAccountId,
} from "../components/account-selection";
import { readSessionTtlCache, writeSessionTtlCache } from "../../lib/session-ttl-cache";

// ─── 타입 ────────────────────────────────────────────────────────────────────
type AccountOption = {
  account_id: string;
  name: string;
};

type ComponentSource = {
  etf_ticker: string;
  etf_name: string;
  weight: number;
  current_price?: number | null;
  change_pct?: number | null;
  currency?: string;
  return_pct?: number | null;
  daily_profit_krw?: number | null;
  cumulative_profit_krw?: number | null;
};

type ComponentRow = {
  ticker: string;
  name: string;
  has_components?: boolean;
  total_weight: number;
  sources: ComponentSource[];
  current_price?: number | null;
  change_pct?: number | null;
  currency?: string;
  return_pct?: number | null;
  daily_profit_krw?: number | null;
  cumulative_profit_krw?: number | null;
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

const HOLDINGS_COMPONENT_ACCOUNTS_CACHE_KEY = "momentum-etf:holdings-details:accounts";
const HOLDINGS_COMPONENTS_CACHE_KEY_PREFIX = "momentum-etf:holdings-details:data:";
const HOLDINGS_COMPONENT_ACCOUNTS_CACHE_TTL_MS = 300_000;
const HOLDINGS_COMPONENTS_CACHE_TTL_MS = 30_000;

// ─── 유틸 ────────────────────────────────────────────────────────────────────
function formatWeight(w: number): string {
  return `${w.toFixed(2)}%`;
}

function formatPrice(val: number | null | undefined, currency?: string): string {
  if (val == null) return "-";
  if (currency === "USD") {
    return `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  if (currency === "AUD") {
    return `A$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  return `${Math.floor(val).toLocaleString()}원`;
}

function formatSignedPercent(val: number | null | undefined): string {
  if (val == null) return "-";
  return `${val.toFixed(2)}%`;
}

function formatKrw(val: number | null | undefined): string {
  if (val == null) return "-";
  const rounded = Math.round(val);
  return `${rounded.toLocaleString()}원`;
}

function getSignedClass(val: number | null | undefined): string {
  if (val == null) return "";
  if (val > 0) return "metricPositive";
  if (val < 0) return "metricNegative";
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
  const [selectedAccount, setSelectedAccount] = useState<string>(
    readRememberedMomentumEtfAccountId() || "",
  );
  const [data, setData] = useState<HoldingsComponentsData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);
  const requestSequenceRef = useRef(0);

  // 계좌 목록 로드
  useEffect(() => {
    async function fetchAccounts() {
      try {
        const cached = readSessionTtlCache<AccountOption[]>(
          HOLDINGS_COMPONENT_ACCOUNTS_CACHE_KEY,
          HOLDINGS_COMPONENT_ACCOUNTS_CACHE_TTL_MS,
        );
        if (cached && cached.length > 0) {
          setAccounts(cached);
          if (!selectedAccount) {
            setSelectedAccount("TOTAL");
            writeRememberedMomentumEtfAccountId("TOTAL");
          }
          return;
        }

        const res = await fetch("/api/holdings-components/accounts", { cache: "no-store" });
        if (!res.ok) throw new Error("계좌 목록을 불러오지 못했습니다.");
        const list = (await res.json()) as AccountOption[];
        writeSessionTtlCache(HOLDINGS_COMPONENT_ACCOUNTS_CACHE_KEY, list);
        setAccounts(list);
        if (list.length > 0 && !selectedAccount) {
          setSelectedAccount("TOTAL");
          writeRememberedMomentumEtfAccountId("TOTAL");
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "오류가 발생했습니다.");
      }
    }
    void fetchAccounts();
  }, []);

  // 선택된 계좌 데이터 로드
  const loadData = useCallback(async (accountId: string) => {
    if (!accountId) return;
    const requestSequence = requestSequenceRef.current + 1;
    requestSequenceRef.current = requestSequence;
    const cacheKey = `${HOLDINGS_COMPONENTS_CACHE_KEY_PREFIX}${accountId}`;
    const cached = readSessionTtlCache<HoldingsComponentsData>(cacheKey, HOLDINGS_COMPONENTS_CACHE_TTL_MS);
    if (cached) {
      setError(null);
      setData(cached);
      setExpandedTicker(null);
      setIsLoading(false);
      return;
    }
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
      if (requestSequenceRef.current !== requestSequence) {
        return;
      }
      writeSessionTtlCache(cacheKey, result);
      setData(result);
    } catch (e) {
      if (requestSequenceRef.current !== requestSequence) {
        return;
      }
      setError(e instanceof Error ? e.message : "오류가 발생했습니다.");
    } finally {
      if (requestSequenceRef.current === requestSequence) {
        setIsLoading(false);
      }
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
      if (expandedTicker === comp.ticker && comp.has_components === true && comp.sources.length > 0) {
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
        return <span className="text-muted fw-semibold" style={{ fontFamily: "monospace", fontSize: "13px" }}>{params.value}</span>;
      },
    },
    {
      headerName: "종목명",
      field: "name",
      flex: 1,
      minWidth: 140,
      cellRenderer: (params: { data?: GridRow; value?: string }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const mainRow = params.data as MainGridRow;
        const isExpanded = expandedTicker === mainRow.ticker;
        const hasSources = mainRow.has_components === true && mainRow.sources.length > 0;
        return (
          <div
            className={`d-flex align-items-center gap-2 holdingsDetailsNameCell ${hasSources ? "cursor-pointer" : ""}`}
            style={{ userSelect: "none" }}
          >
            {hasSources && (
              <span className="text-primary d-flex align-items-center" style={{ fontSize: "10px", transition: "transform 0.15s", transform: isExpanded ? "rotate(90deg)" : "none" }}>
                ▶
              </span>
            )}
            <span className="fw-bold text-dark holdingsDetailsNameText">{params.value}</span>
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
        return formatPrice(params.value, row.currency);
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
    {
      headerName: "수익률",
      field: "return_pct",
      width: 100,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: number | null }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        return <span className={getSignedClass(params.value)}>{formatSignedPercent(params.value)}</span>;
      },
    },
    {
      headerName: "금일 손익",
      field: "daily_profit_krw",
      width: 140,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: number | null }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        return <span className={getSignedClass(params.value)}>{formatKrw(params.value)}</span>;
      },
    },
    {
      headerName: "누적 손익",
      field: "cumulative_profit_krw",
      width: 140,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: number | null }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        return <span className={getSignedClass(params.value)}>{formatKrw(params.value)}</span>;
      },
    },
  ], [expandedTicker]);

  // 상세 패널 렌더러 (부모 행과 컬럼/정렬 완벽 정밀 타격)
  const DetailRenderer = useCallback((params: { data?: GridRow }) => {
    if (!params.data || !isDetailRow(params.data)) return null;
    
    return (
      <div className="holdingsDetailNestedRows">
        {params.data.sources.map((src, idx) => (
          <div key={idx} className="holdingsDetailRow">
            {/* 1. 티커 (100px) - 좌측 패딩 12px, 폰트 13px 고정 */}
            <div className="hdColTicker fw-semibold text-muted" style={{ fontFamily: "monospace", fontSize: "13px" }}>
              {src.etf_ticker}
            </div>
            
            {/* 2. 종목명 (flex:1) - 좌측 패딩 12px */}
            <div className="hdColName fw-bold text-dark">
              {src.etf_name}
            </div>
            
            {/* 3. 비중 (90px) - 우측 패딩 12px */}
            <div className="hdColWeight fw-bold text-primary">
              {formatWeight(src.weight)}
            </div>
            
            {/* 4. 현재가 (110px) - 우측 패딩 12px */}
            <div className="hdColPrice">
              {formatPrice(src.current_price, src.currency)}
            </div>
            
            {/* 5. 일간(%) (100px) - 우측 패딩 12px */}
            <div className="hdColChange">
              <span className={getSignedClass(src.change_pct)}>
                {formatSignedPercent(src.change_pct)}
              </span>
            </div>

            <div className="hdColReturn">
              <span className={getSignedClass(src.return_pct)}>
                {formatSignedPercent(src.return_pct)}
              </span>
            </div>

            <div className="hdColDailyProfit">
              <span className={getSignedClass(src.daily_profit_krw)}>
                {formatKrw(src.daily_profit_krw)}
              </span>
            </div>

            <div className="hdColCumulativeProfit">
              <span className={getSignedClass(src.cumulative_profit_krw)}>
                {formatKrw(src.cumulative_profit_krw)}
              </span>
            </div>
          </div>
        ))}
      </div>
    );
  }, []);

  const gridOptions = useMemo<GridOptions<GridRow>>(() => ({
    getRowId: (params) => (isDetailRow(params.data) ? `detail:${params.data.parentTicker}` : params.data.ticker),
    isFullWidthRow: (params) => isDetailRow(params.rowNode.data),
    fullWidthCellRenderer: DetailRenderer,
    // 정밀 계산: Padding-top(8) + RowHeight(38) * N + Padding-bottom(8)
    getRowHeight: (params) => (isDetailRow(params.data) ? 16 + params.data.sources.length * 38 : 38),
    onCellClicked: (params) => {
      if (
        params.data &&
        !isDetailRow(params.data) &&
        params.colDef.field === "name" &&
        params.data.has_components === true
      ) {
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
            <ResponsiveFiltersSection>
              <div className="appMainHeader">
                <div className="appMainHeaderLeft rankMainHeaderLeft">
                  <label className="appLabeledField">
                    <span className="appLabeledFieldLabel">계좌</span>
                    <select
                      className="form-select"
                      value={selectedAccount}
                      onChange={(e) => {
                        const nextId = e.target.value;
                        setSelectedAccount(nextId);
                        writeRememberedMomentumEtfAccountId(nextId);
                      }}
                      disabled={accounts.length === 0}
                    >
                      {accounts.length === 0 ? (
                        <option value="">계좌 불러오는 중...</option>
                      ) : (
                        <>
                          <option value="TOTAL">전체</option>
                          {accounts.map((acc) => (
                            <option key={acc.account_id} value={acc.account_id}>
                              {acc.name}
                            </option>
                          ))}
                        </>
                      )}
                    </select>
                  </label>
                </div>
              </div>
            </ResponsiveFiltersSection>
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
          background-color: #fbfcfe !important;
          border-bottom: 1px solid #e2e8f0;
        }
        .holdingsDetailNestedRows {
          background-color: #f1f5f9;
          padding: 8px 0;
          box-sizing: border-box;
        }
        .holdingsDetailRow {
          display: flex;
          align-items: center;
          height: 38px;
          border-bottom: 1px solid #e2e8f0;
          transition: background-color 0.15s;
          box-sizing: border-box;
        }
        .holdingsDetailRow:hover {
          background-color: #ffffff;
        }
        .holdingsDetailRow > div {
          height: 100%;
          display: flex;
          align-items: center;
          border-right: 1px solid #e2e8f0;
          box-sizing: border-box;
          font-size: 14px;
        }
        .holdingsDetailRow > div:last-child {
          border-right: none;
        }
        /* 각 컬럼 너비 및 패딩 정밀 동기화 (Ag-Grid 12px 기준) */
        .hdColTicker { width: 100px; padding-left: 12px; }
        .hdColName   { flex: 1; min-width: 0; padding-left: 12px; overflow: hidden; }
        .hdColWeight { width: 90px; padding-right: 12px; justify-content: flex-end; }
        .hdColPrice  { width: 110px; padding-right: 12px; justify-content: flex-end; }
        .hdColChange { width: 100px; padding-right: 12px; justify-content: flex-end; }
        .hdColReturn { width: 100px; padding-right: 12px; justify-content: flex-end; }
        .hdColDailyProfit { width: 140px; padding-right: 12px; justify-content: flex-end; }
        .hdColCumulativeProfit { width: 140px; padding-right: 12px; justify-content: flex-end; }
        .holdingsDetailsNameCell {
          width: 100%;
          min-width: 0;
          overflow: hidden;
        }
        .holdingsDetailsNameText {
          min-width: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          display: inline-block;
        }
        .hdColName {
          white-space: nowrap;
          text-overflow: ellipsis;
        }

        .cursor-pointer { cursor: pointer; }
      `}</style>
    </PageFrame>
  );
}
