"use client";

import { iconSetQuartzBold, themeQuartz } from "ag-grid-community";
import type { ColDef, RowClassParams } from "ag-grid-community";
import type { GridOptions } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { AppAgGrid } from "../components/AppAgGrid";

type HoldingsRow = {
  account_name: string;
  currency: string;
  ticker: string;
  name: string;
  quantity: number;
  current_price: string;
  current_price_num: number;
  days_held: string;
  pnl_krw: number;
  pnl_krw_num: number;
  return_pct: number;
  daily_change_pct: number | null;
  buy_amount_krw: number;
  valuation_krw: number;
  bucket_id: number;
  bucket: string;
  memo: string;
};

type AccountSummary = {
  account_id: string;
  cash_balance_krw: number;
};

type AggregatedHoldingRow = HoldingsRow & { portfolio_weight_pct: number };

type HoldingsHeaderSummary = {
  accountCount: number;
  holdingCount: number;
  totalValuation: number;
};

type ConstituentRow = {
  ticker: string;
  name: string;
  weight: number | null;
  current_price: number | null;
  change_pct: number | null;
};

// 부모 그리드에 올라갈 row 타입: main 행 또는 detail(자식) 행
type ParentRow =
  | (AggregatedHoldingRow & { rowType: "main" })
  | { rowType: "detail"; parentTicker: string; constituents: ConstituentRow[]; loading: boolean };

const holdingsGridTheme = themeQuartz
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

// 구성종목이 있을 수 있는 종목인지 판별 (한국 6자리 코드 + 현금 아님)
// 0113D0, 0091P0 같은 알파뉴메릭 ETF 코드도 포함
function canHaveConstituents(row: AggregatedHoldingRow): boolean {
  if (row.ticker === "__CASH__") return false;
  return row.currency === "KRW" && row.ticker.length === 6;
}

// ticker 페이지의 tickerDetailHoldingsPanel 높이와 동일
const DETAIL_PANEL_HEIGHT = 420;
const DETAIL_PANEL_PADDING = 12; // fullWidth row 상하 여백

function getDetailRowHeight(_count: number): number {
  return DETAIL_PANEL_HEIGHT + DETAIL_PANEL_PADDING * 2;
}

// ticker 페이지 gridTheme과 동일한 파라미터
const constituentGridTheme = holdingsGridTheme.withParams({
  rowHeight: 34,
  headerHeight: 36,
  wrapperBorderRadius: 10,
  fontSize: 14,
});

export function HoldingsManager({
  onHeaderSummaryChange,
}: {
  onHeaderSummaryChange?: (summary: HoldingsHeaderSummary) => void;
}) {
  const router = useRouter();
  const [holdings, setHoldings] = useState<HoldingsRow[]>([]);
  const [totalCashKrw, setTotalCashKrw] = useState(0);
  const [loading, setLoading] = useState(true);
  const [showAmounts, setShowAmounts] = useState(true);

  // 펼쳐진 ticker + 구성종목 캐시
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);
  const constituentsCacheRef = useRef<Map<string, ConstituentRow[] | null>>(new Map());
  const [detailLoading, setDetailLoading] = useState(false);

  const constituentColDefs = useMemo<ColDef<ConstituentRow>[]>(() => [
    {
      field: "ticker",
      headerName: "종목코드",
      minWidth: 120,
      width: 120,
      cellClass: "tickerDetailCodeCell",
      cellStyle: { fontWeight: 700 },
    },
    {
      field: "name",
      headerName: "종목명",
      minWidth: 148,
      flex: 1.2,
      cellClass: "tickerDetailNameCell",
    },
    {
      field: "weight",
      headerName: "비중",
      minWidth: 76,
      width: 76,
      type: "rightAligned",
      cellRenderer: (params: { value: number | null }) =>
        params.value != null ? `${Number(params.value).toFixed(2)}%` : "-",
    },
    {
      field: "current_price",
      headerName: "현재가",
      minWidth: 108,
      width: 108,
      type: "rightAligned",
      cellRenderer: (params: { value: number | null }) =>
        params.value != null ? `${Math.floor(params.value).toLocaleString()}원` : "-",
    },
    {
      field: "change_pct",
      headerName: "일간(%)",
      minWidth: 88,
      width: 88,
      type: "rightAligned",
      cellRenderer: (params: { value: number | null }) => (
        <span className={getSignedClass(params.value)}>{formatSignedPercent(params.value)}</span>
      ),
    },
  ], []);

  const loadHoldings = useCallback(async () => {
    try {
      setLoading(true);
      const res = await fetch("/api/assets", { cache: "no-store" });
      if (!res.ok) {
        setHoldings([]);
        setTotalCashKrw(0);
        return;
      }
      const data = await res.json();
      const rows = (data.rows || []).filter((r: HoldingsRow) => r.ticker && r.quantity > 0);
      const cashSum = ((data.account_summaries || []) as AccountSummary[]).reduce(
        (sum: number, acc: AccountSummary) => sum + (acc.cash_balance_krw || 0),
        0,
      );
      setHoldings(rows);
      setTotalCashKrw(cashSum);
    } catch (err) {
      console.error(err);
      setHoldings([]);
      setTotalCashKrw(0);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadHoldings();
  }, [loadHoldings]);

  const accountNames = [...new Set(holdings.map((h) => h.account_name))];
  const aggregatedBaseHoldings = Array.from(
    holdings.reduce((acc, row) => {
      const key = `${row.currency}:${row.ticker}`;
      const existing = acc.get(key);
      if (!existing) {
        acc.set(key, { ...row });
        return acc;
      }
      const nextBuyAmount = existing.buy_amount_krw + row.buy_amount_krw;
      const nextValuation = existing.valuation_krw + row.valuation_krw;
      const nextPnl = existing.pnl_krw + row.pnl_krw;
      const existingDaysHeldInt = parseHoldingDaysToInt(existing.days_held);
      const rowDaysHeldInt = parseHoldingDaysToInt(row.days_held);
      const weightedDailyChange =
        existing.daily_change_pct !== null && row.daily_change_pct !== null
          ? ((existing.daily_change_pct * existing.valuation_krw) + (row.daily_change_pct * row.valuation_krw)) /
            Math.max(nextValuation, 1)
          : existing.daily_change_pct ?? row.daily_change_pct;
      acc.set(key, {
        ...existing,
        quantity: existing.quantity + row.quantity,
        buy_amount_krw: nextBuyAmount,
        valuation_krw: nextValuation,
        pnl_krw: nextPnl,
        pnl_krw_num: nextPnl,
        return_pct: nextBuyAmount > 0 ? Number(((nextPnl / nextBuyAmount) * 100).toFixed(2)) : 0,
        daily_change_pct: weightedDailyChange === null ? null : Number(weightedDailyChange.toFixed(2)),
        bucket_id: row.valuation_krw > existing.valuation_krw ? row.bucket_id : existing.bucket_id,
        bucket: row.valuation_krw > existing.valuation_krw ? row.bucket : existing.bucket,
        memo: existing.memo || row.memo,
        account_name: accountNames.join(", "),
        days_held: rowDaysHeldInt > existingDaysHeldInt ? row.days_held : existing.days_held,
      });
      return acc;
    }, new Map<string, HoldingsRow>()).values(),
  );

  const holdingsValuation = aggregatedBaseHoldings.reduce((sum, row) => sum + row.valuation_krw, 0);
  const totalValuation = holdingsValuation + totalCashKrw;
  const aggregatedHoldings: AggregatedHoldingRow[] = aggregatedBaseHoldings.map((row) => ({
    ...row,
    portfolio_weight_pct: totalValuation > 0 ? Number(((row.valuation_krw / totalValuation) * 100).toFixed(1)) : 0,
  }));

  if (totalCashKrw > 0) {
    aggregatedHoldings.push({
      account_name: "",
      currency: "KRW",
      ticker: "__CASH__",
      name: "현금",
      quantity: 0,
      current_price: "-",
      current_price_num: 0,
      days_held: "-",
      pnl_krw: 0,
      pnl_krw_num: 0,
      return_pct: 0,
      daily_change_pct: null,
      buy_amount_krw: totalCashKrw,
      valuation_krw: totalCashKrw,
      bucket_id: 0,
      bucket: "",
      memo: "",
      portfolio_weight_pct: totalValuation > 0 ? Number(((totalCashKrw / totalValuation) * 100).toFixed(1)) : 0,
    });
  }

  aggregatedHoldings.sort((a, b) => b.valuation_krw - a.valuation_krw);

  useEffect(() => {
    onHeaderSummaryChange?.({
      accountCount: accountNames.length,
      holdingCount: aggregatedHoldings.length,
      totalValuation,
    });
  }, [accountNames.length, aggregatedHoldings.length, onHeaderSummaryChange, totalValuation]);

  // 구성종목 fetch
  const fetchConstituents = useCallback(async (ticker: string): Promise<ConstituentRow[] | null> => {
    const cached = constituentsCacheRef.current.get(ticker);
    if (cached !== undefined) return cached;
    try {
      const params = new URLSearchParams({ ticker });
      const res = await fetch(`/api/ticker-detail?${params.toString()}`);
      if (!res.ok) {
        constituentsCacheRef.current.set(ticker, null);
        return null;
      }
      const data = await res.json();
      const items: ConstituentRow[] = (data.holdings || [])
        .slice(0, 50)
        .map((h: { ticker: string; name: string; weight?: number | null; current_price?: number | null; change_pct?: number | null }) => ({
          ticker: h.ticker,
          name: h.name,
          weight: h.weight ?? null,
          current_price: h.current_price ?? null,
          change_pct: h.change_pct ?? null,
        }));
      constituentsCacheRef.current.set(ticker, items.length > 0 ? items : null);
      return items.length > 0 ? items : null;
    } catch {
      constituentsCacheRef.current.set(ticker, null);
      return null;
    }
  }, []);

  // 종목명 클릭 → 펼치기/닫기
  const handleNameClick = useCallback(
    async (row: AggregatedHoldingRow) => {
      if (!canHaveConstituents(row)) return;
      const ticker = row.ticker;
      if (expandedTicker === ticker) {
        setExpandedTicker(null);
        return;
      }
      if (constituentsCacheRef.current.has(ticker)) {
        const cached = constituentsCacheRef.current.get(ticker);
        if (cached === null) return;
        setExpandedTicker(ticker);
        return;
      }
      setDetailLoading(true);
      const result = await fetchConstituents(ticker);
      setDetailLoading(false);
      if (result && result.length > 0) {
        setExpandedTicker(ticker);
      }
    },
    [expandedTicker, fetchConstituents],
  );

  const moveToTickerDetail = useCallback(
    (ticker: string | null | undefined) => {
      const normalizedTicker = normalizeDisplayTicker(String(ticker ?? "-"));
      if (!normalizedTicker || normalizedTicker === "-" || normalizedTicker === "IS") return;
      router.push(`/ticker?ticker=${encodeURIComponent(normalizedTicker)}`);
    },
    [router],
  );

  const isCashRow = useCallback((row: AggregatedHoldingRow | undefined) => row?.ticker === "__CASH__", []);

  // 부모 그리드 rowData: main 행 + 펼쳐진 경우 detail 행 삽입
  const parentRows = useMemo<ParentRow[]>(() => {
    const result: ParentRow[] = [];
    for (const row of aggregatedHoldings) {
      result.push({ ...row, rowType: "main" });
      if (expandedTicker === row.ticker) {
        const constituents = constituentsCacheRef.current.get(row.ticker) ?? [];
        result.push({
          rowType: "detail",
          parentTicker: row.ticker,
          constituents,
          loading: detailLoading,
        });
      }
    }
    return result;
  }, [aggregatedHoldings, expandedTicker, detailLoading]);

  const isDetailRow = useCallback(
    (row: ParentRow | undefined): row is Extract<ParentRow, { rowType: "detail" }> =>
      row?.rowType === "detail",
    [],
  );

  const columnDefs = useMemo<ColDef<ParentRow>[]>(() => [
    {
      headerName: "버킷",
      field: "bucket",
      width: 108,
      sortable: true,
      comparator: (_a, _b, nodeA, nodeB) => {
        const aId = Number((nodeA.data as AggregatedHoldingRow)?.bucket_id ?? 0);
        const bId = Number((nodeB.data as AggregatedHoldingRow)?.bucket_id ?? 0);
        return aId - bId;
      },
      cellClass: (params) => {
        if (isDetailRow(params.data)) return "";
        return `${getBucketCellClass(String((params.data as AggregatedHoldingRow)?.bucket ?? ""))} tableAlignCenter`;
      },
      cellRenderer: (params: { data?: ParentRow }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const row = params.data as AggregatedHoldingRow;
        if (isCashRow(row)) return <span style={{ color: "#8b949e" }}>-</span>;
        return <span>{row.bucket || "-"}</span>;
      },
    },
    {
      headerName: "티커",
      field: "ticker",
      width: 110,
      sortable: true,
      cellClass: "tableAlignCenter holdingsTickerCell",
      cellRenderer: (params: { value?: string | null; data?: ParentRow }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const rawTicker = String(params.value ?? "-");
        if (rawTicker === "__CASH__") return <span className="appCodeText" style={{ color: "#8b949e" }}>-</span>;
        const normalizedTicker = normalizeDisplayTicker(rawTicker);
        if (normalizedTicker === "IS") return <span className="appCodeText">{normalizedTicker}</span>;
        return (
          <button
            type="button"
            className="appCodeText"
            style={{ color: "inherit", textDecoration: "none", background: "none", border: "none", padding: 0 }}
            onClick={() => moveToTickerDetail(normalizedTicker)}
          >
            {normalizedTicker}
          </button>
        );
      },
    },
    {
      headerName: "종목명",
      field: "name",
      flex: 1.4,
      minWidth: 210,
      sortable: true,
      cellClass: "holdingsNameCell",
      cellRenderer: (params: { value?: string | null; data?: ParentRow }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const row = params.data as AggregatedHoldingRow;
        if (!params.value) return "-";
        const isCash = row.ticker === "__CASH__";
        const expandable = canHaveConstituents(row);
        const isExpanded = expandedTicker === row.ticker;
        return (
          <span
            className={`holdingsNameMain${expandable ? " holdingsNameExpandable" : ""}`}
            title={params.value}
            style={isCash ? { color: "#8b949e", fontWeight: 500 } : undefined}
          >
            {expandable && (
              <span className={`holdingsExpandIcon${isExpanded ? " is-open" : ""}`}>▶</span>
            )}
            {params.value}
          </span>
        );
      },
    },
    {
      headerName: "비중",
      field: "portfolio_weight_pct",
      width: 92,
      type: "rightAligned",
      valueFormatter: (params) => {
        if (isDetailRow(params.data)) return "";
        return `${Number((params.data as AggregatedHoldingRow)?.portfolio_weight_pct ?? 0).toFixed(1)}%`;
      },
    },
    {
      headerName: "보유일",
      field: "days_held",
      width: 92,
      cellClass: "tableAlignCenter",
      cellRenderer: (params: { data?: ParentRow }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        return (params.data as AggregatedHoldingRow).days_held;
      },
    },
    {
      headerName: "평가금액",
      field: "valuation_krw",
      width: 160,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentRow }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const row = params.data as AggregatedHoldingRow;
        return showAmounts ? formatHoldingsKrw(row.valuation_krw) : "••••";
      },
    },
    {
      headerName: "현재가",
      field: "current_price_num",
      width: 140,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentRow }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const row = params.data as AggregatedHoldingRow;
        if (isCashRow(row)) return "-";
        const value = showAmounts ? formatHoldingPrice(row.current_price_num, row.currency) : "••••";
        return <span className={getSignedClass(row.daily_change_pct)}>{value}</span>;
      },
    },
    {
      headerName: "일간(%)",
      field: "daily_change_pct",
      width: 120,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentRow }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const row = params.data as AggregatedHoldingRow;
        if (row.ticker === "__CASH__") return "-";
        return <span className={getSignedClass(row.daily_change_pct)}>{formatSignedPercent(row.daily_change_pct)}</span>;
      },
    },
    {
      headerName: "평가손익",
      field: "pnl_krw",
      width: 150,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentRow }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const row = params.data as AggregatedHoldingRow;
        if (isCashRow(row)) return "-";
        const value = showAmounts ? formatHoldingsKrw(row.pnl_krw) : "••••";
        return <span className={getSignedClass(row.return_pct)}>{value}</span>;
      },
    },
    {
      headerName: "수익률(%)",
      field: "return_pct",
      width: 120,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentRow }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const row = params.data as AggregatedHoldingRow;
        if (row.ticker === "__CASH__") return "-";
        return <span className={getSignedClass(row.return_pct)}>{formatSignedPercent(row.return_pct)}</span>;
      },
    },
  ], [isCashRow, isDetailRow, moveToTickerDetail, showAmounts, expandedTicker, handleNameClick]);

  // detail(자식) fullWidth renderer — ticker 페이지 tickerDetailHoldingsPanel 구조 그대로 사용
  const DetailRenderer = useCallback(
    (params: { data?: ParentRow }) => {
      if (!params.data || !isDetailRow(params.data)) return null;
      const { constituents } = params.data;
      return (
        <div style={{ height: "100%", padding: `${DETAIL_PANEL_PADDING}px 16px`, display: "flex", alignItems: "flex-start" }}>
          <div className="tickerDetailHoldingsPanel" style={{ width: "50%", minWidth: 440 }}>
            <div className="tickerDetailTableHeader">
              <span className="tickerDetailTableTitle">구성종목</span>
              <span className="tickerDetailTableMeta">상위 {constituents.length}개</span>
            </div>
            <div className="appGridFillWrap">
              <AppAgGrid
                className="tickerDetailHoldingsGrid"
                rowData={constituents}
                columnDefs={constituentColDefs}
                loading={false}
                theme={constituentGridTheme}
                gridOptions={{
                  suppressMovableColumns: true,
                  getRowId: (p) => String(p.data.ticker),
                }}
              />
            </div>
          </div>
        </div>
      );
    },
    [isDetailRow, constituentColDefs],
  );

  const holdingsGridOptions = useMemo<GridOptions<ParentRow>>(
    () => ({
      suppressMovableColumns: true,
      getRowId: (params) => {
        const d = params.data as ParentRow;
        if (isDetailRow(d)) return `detail:${d.parentTicker}`;
        const row = d as AggregatedHoldingRow;
        return `${row.currency}:${row.ticker}`;
      },
      isFullWidthRow: (params) => isDetailRow(params.rowNode.data as ParentRow),
      fullWidthCellRenderer: DetailRenderer,
      getRowHeight: (params) => {
        const d = params.data as ParentRow;
        if (isDetailRow(d)) return getDetailRowHeight(d.constituents.length);
        return 38;
      },
      onCellClicked: (params) => {
        if (!params.data || isDetailRow(params.data as ParentRow)) return;
        if (params.colDef.field !== "name") return;
        const row = params.data as AggregatedHoldingRow;
        void handleNameClick(row);
      },
      overlayNoRowsTemplate: '<span class="ag-overlay-no-rows-center">보유 종목이 없습니다.</span>',
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [DetailRenderer, isDetailRow, handleNameClick],
  );

  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft" />
              <div className="appMainHeaderRight">
                <button
                  type="button"
                  className={`btn btn-sm shadow-sm ${showAmounts ? "btn-outline-secondary" : "btn-dark"}`}
                  onClick={() => setShowAmounts((prev) => !prev)}
                >
                  {showAmounts ? "금액 가리기" : "금액 보기"}
                </button>
              </div>
            </div>
          </div>

          <div className="card-body appCardBodyTight appTableCardBodyFill">
            <div className="appGridFillWrap">
              <AppAgGrid
                rowData={loading ? [] : parentRows}
                columnDefs={columnDefs}
                loading={loading || detailLoading}
                minHeight="100%"
                className="holdingsGrid"
                theme={holdingsGridTheme}
                getRowClass={(params: RowClassParams<ParentRow>) => {
                  if (isDetailRow(params.data)) return "holdingsRow holdingsDetailFullRow";
                  return (params.data as AggregatedHoldingRow)?.ticker === "__CASH__"
                    ? "holdingsRow holdingsRowCash"
                    : "holdingsRow";
                }}
                gridOptions={holdingsGridOptions}
              />
            </div>
          </div>
        </div>
      </section>

      <style jsx global>{`
        .holdingsGrid .ag-cell {
          display: flex;
          align-items: center;
        }
        .holdingsGrid .tableAlignCenter {
          justify-content: center;
          text-align: center;
        }
        .holdingsGrid .holdingsTickerCell {
          font-weight: 600;
          color: #5f6b82;
        }
        .holdingsGrid .holdingsNameCell {
          min-width: 0;
          overflow: hidden;
        }
        .holdingsGrid .ag-row.holdingsRowCash .ag-cell {
          background-color: #f4f5f7;
        }
        .holdingsGrid .ag-row.holdingsDetailFullRow {
          background-color: #ffffff !important;
          border-bottom: 1px solid #d0daea;
        }
        .holdingsGrid .rankBucketCell {
          justify-content: center;
          font-weight: 600;
          white-space: nowrap;
        }
        .holdingsGrid .rankBucketCell1 { background: var(--bucket-1); color: #fff; }
        .holdingsGrid .rankBucketCell2 { background: var(--bucket-2); color: #fff; }
        .holdingsGrid .rankBucketCell3 { background: var(--bucket-3); color: #fff; }
        .holdingsGrid .rankBucketCell4 { background: var(--bucket-4); color: #fff; }

      `}</style>

      <style jsx>{`
        .holdingsNameMain {
          display: flex;
          align-items: center;
          gap: 6px;
          width: 100%;
          min-width: 0;
          font-size: 0.95rem;
          font-weight: 700;
          color: #1d273b;
          word-break: keep-all;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .holdingsNameExpandable {
          cursor: pointer;
        }
        .holdingsNameExpandable:hover {
          color: #206bc4;
        }
        .holdingsExpandIcon {
          font-size: 9px;
          color: #8b949e;
          flex-shrink: 0;
          transition: transform 0.15s;
          display: inline-block;
        }
        .holdingsExpandIcon.is-open {
          transform: rotate(90deg);
          color: #206bc4;
        }
      `}</style>
    </div>
  );
}

function formatHoldingPrice(val: number, currency: string) {
  if (currency === "AUD") return `A$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  if (currency === "USD") return `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  return `${Math.floor(val).toLocaleString()}원`;
}

function formatHoldingsKrw(val: number) {
  const abs = Math.abs(val);
  const sign = val < 0 ? "-" : "";
  if (abs >= 100_000_000) return `${sign}${(abs / 100_000_000).toFixed(1)}억`;
  if (abs >= 10_000) return `${sign}${Math.floor(abs / 10_000).toLocaleString()}만`;
  return `${sign}${Math.floor(abs).toLocaleString()}원`;
}

function formatSignedPercent(val: number | null): string {
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

function getBucketCellClass(bucket: string): string {
  const lower = bucket.toLowerCase();
  if (lower.includes("1") || lower.startsWith("a")) return "rankBucketCell rankBucketCell1";
  if (lower.includes("2") || lower.startsWith("b")) return "rankBucketCell rankBucketCell2";
  if (lower.includes("3") || lower.startsWith("c")) return "rankBucketCell rankBucketCell3";
  if (lower.includes("4") || lower.startsWith("d")) return "rankBucketCell rankBucketCell4";
  return "rankBucketCell";
}

function normalizeDisplayTicker(ticker: string): string {
  if (!ticker || ticker === "-") return "-";
  const upper = ticker.toUpperCase();
  if (/^\d{6}$/.test(upper)) return upper;
  if (upper.endsWith(".KS") || upper.endsWith(".KQ") || upper.endsWith(".AX")) return upper.split(".")[0];
  return upper;
}

function parseHoldingDaysToInt(daysHeld: string): number {
  if (!daysHeld || daysHeld === "-") return 0;
  const match = daysHeld.match(/\d+/);
  return match ? parseInt(match[0], 10) : 0;
}
