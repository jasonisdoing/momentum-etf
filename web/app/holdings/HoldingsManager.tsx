"use client";

import { iconSetQuartzBold, themeQuartz } from "ag-grid-community";
import type { ColDef, RowClassParams } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useState } from "react";

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

type AggregatedHoldingRow = HoldingsRow;

type HoldingsHeaderSummary = {
  accountCount: number;
  holdingCount: number;
  totalValuation: number;
};

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

export function HoldingsManager({
  onHeaderSummaryChange,
}: {
  onHeaderSummaryChange?: (summary: HoldingsHeaderSummary) => void;
}) {
  const [holdings, setHoldings] = useState<HoldingsRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAmounts, setShowAmounts] = useState(true);

  const loadHoldings = useCallback(async () => {
    try {
      setLoading(true);
      const res = await fetch("/api/assets", { cache: "no-store" });
      if (!res.ok) {
        setHoldings([]);
        return;
      }
      const data = await res.json();
      const rows = (data.rows || []).filter((r: HoldingsRow) => r.ticker && r.quantity > 0);
      setHoldings(rows);
    } catch (err) {
      console.error(err);
      setHoldings([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadHoldings();

    function handlePageShow() {
      void loadHoldings();
    }

    window.addEventListener("pageshow", handlePageShow);
    return () => {
      window.removeEventListener("pageshow", handlePageShow);
    };
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
          ? (
              ((existing.daily_change_pct * existing.valuation_krw) + (row.daily_change_pct * row.valuation_krw)) /
              Math.max(nextValuation, 1)
            )
          : (existing.daily_change_pct ?? row.daily_change_pct);

      acc.set(key, {
        ...existing,
        quantity: existing.quantity + row.quantity,
        buy_amount_krw: nextBuyAmount,
        valuation_krw: nextValuation,
        pnl_krw: nextPnl,
        pnl_krw_num: nextPnl,
        return_pct: nextBuyAmount > 0 ? Number(((nextPnl / nextBuyAmount) * 100).toFixed(2)) : 0,
        daily_change_pct:
          weightedDailyChange === null ? null : Number(weightedDailyChange.toFixed(2)),
        bucket_id: row.valuation_krw > existing.valuation_krw ? row.bucket_id : existing.bucket_id,
        bucket: row.valuation_krw > existing.valuation_krw ? row.bucket : existing.bucket,
        memo: existing.memo || row.memo,
        account_name: accountNames.join(", "),
        days_held:
          rowDaysHeldInt > existingDaysHeldInt
            ? row.days_held
            : existing.days_held,
      });
      return acc;
    }, new Map<string, AggregatedHoldingRow>()).values(),
  );

  const totalValuation = aggregatedBaseHoldings.reduce((sum, row) => sum + row.valuation_krw, 0);
  const aggregatedHoldings = aggregatedBaseHoldings
    .map((row) => ({
      ...row,
      portfolio_weight_pct: totalValuation > 0 ? Number(((row.valuation_krw / totalValuation) * 100).toFixed(1)) : 0,
    }))
    .sort((a, b) => b.valuation_krw - a.valuation_krw);

  useEffect(() => {
    onHeaderSummaryChange?.({
      accountCount: accountNames.length,
      holdingCount: aggregatedHoldings.length,
      totalValuation,
    });
  }, [accountNames.length, aggregatedHoldings.length, onHeaderSummaryChange, totalValuation]);

  const columnDefs = useMemo<ColDef<(AggregatedHoldingRow & { portfolio_weight_pct: number })>[]>(() => [
    {
      headerName: "티커",
      field: "ticker",
      width: 110,
      sortable: true,
      cellClass: "tableAlignCenter holdingsTickerCell",
      cellRenderer: (params: { value?: string | null }) => {
        const rawTicker = String(params.value ?? "-");
        const normalizedTicker = normalizeDisplayTicker(rawTicker);
        if (normalizedTicker === "IS") {
          return <span className="appCodeText">{normalizedTicker}</span>;
        }
        const href = `/ticker?ticker=${encodeURIComponent(normalizedTicker)}`;
        return (
          <a href={href} className="appCodeText" style={{ color: "inherit", textDecoration: "none" }}>
            {normalizedTicker}
          </a>
        );
      },
    },
    {
      headerName: "종목명",
      field: "name",
      flex: 1.4,
      minWidth: 210,
      sortable: true,
      cellRenderer: (params: { value?: string | null }) => {
        if (!params.value) {
          return "-";
        }
        return <span className="holdingsNameMain" title={params.value}>{params.value}</span>;
      },
    },
    {
      headerName: "비중",
      field: "portfolio_weight_pct",
      width: 92,
      type: "rightAligned",
      valueFormatter: (params) => `${Number(params.value ?? 0).toFixed(1)}%`,
    },
    {
      headerName: "보유일",
      field: "days_held",
      width: 92,
      cellClass: "tableAlignCenter",
    },
    {
      headerName: "평가금액",
      field: "valuation_krw",
      width: 160,
      type: "rightAligned",
      valueFormatter: (params) => showAmounts ? formatHoldingsKrw(Number(params.value ?? 0)) : "••••",
    },
    {
      headerName: "현재가",
      field: "current_price_num",
      width: 140,
      type: "rightAligned",
      cellRenderer: (params: { data?: AggregatedHoldingRow & { portfolio_weight_pct: number } }) => {
        const row = params.data;
        if (!row) {
          return "-";
        }
        const value = showAmounts ? formatHoldingPrice(row.current_price_num, row.currency) : "••••";
        return <span className={getSignedClass(row.daily_change_pct)}>{value}</span>;
      },
    },
    {
      headerName: "일간(%)",
      field: "daily_change_pct",
      width: 120,
      type: "rightAligned",
      cellRenderer: (params: { value?: number | null }) => {
        const value = params.value ?? null;
        return <span className={getSignedClass(value)}>{formatSignedPercent(value)}</span>;
      },
    },
    {
      headerName: "평가손익",
      field: "pnl_krw",
      width: 150,
      type: "rightAligned",
      cellRenderer: (params: { data?: AggregatedHoldingRow & { portfolio_weight_pct: number } }) => {
        const row = params.data;
        if (!row) {
          return "-";
        }
        const value = showAmounts ? formatHoldingsKrw(row.pnl_krw) : "••••";
        return <span className={getSignedClass(row.return_pct)}>{value}</span>;
      },
    },
    {
      headerName: "수익률(%)",
      field: "return_pct",
      width: 120,
      type: "rightAligned",
      cellRenderer: (params: { value?: number | null }) => {
        const value = params.value ?? null;
        return <span className={getSignedClass(value)}>{formatSignedPercent(value)}</span>;
      },
    },
  ], [showAmounts]);

  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <div className="holdingsToolbar">
                  {[
                    { id: 1, name: "모멘텀", color: "#1e6bb8", sub: "#e7f1ff" },
                    { id: 2, name: "시장지수", color: "#2fb344", sub: "#eaf8ed" },
                    { id: 3, name: "미국배당", color: "#d63384", sub: "#fbebf3" },
                    { id: 4, name: "대체헷지", color: "#f76707", sub: "#fef0e7" },
                  ].map((b) => (
                    <div
                      key={b.id}
                      className="bucket-legend-badge"
                      style={{ backgroundColor: b.sub, color: b.color }}
                    >
                      {b.id}. {b.name}
                    </div>
                  ))}
                </div>
              </div>
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
                rowData={loading ? [] : aggregatedHoldings}
                columnDefs={columnDefs}
                loading={loading}
                minHeight="100%"
                className="holdingsGrid"
                theme={holdingsGridTheme}
                getRowClass={(params: RowClassParams<AggregatedHoldingRow & { portfolio_weight_pct: number }>) => `holdingsRow holdingsRowBucket${params.data?.bucket_id ?? 0}`}
                gridOptions={{
                  rowHeight: 38,
                  suppressMovableColumns: true,
                  getRowId: (params) => `${params.data.currency}:${params.data.ticker}`,
                  overlayNoRowsTemplate: '<span class="ag-overlay-no-rows-center">보유 종목이 없습니다.</span>',
                }}
              />
            </div>
          </div>
        </div>
      </section>

      <style jsx global>{`
        .holdingsGrid .ag-row.holdingsRowBucket1 .ag-cell {
          background-color: #e7f1ff;
        }
        .holdingsGrid .ag-row.holdingsRowBucket2 .ag-cell {
          background-color: #eaf8ed;
        }
        .holdingsGrid .ag-row.holdingsRowBucket3 .ag-cell {
          background-color: #fbebf3;
        }
        .holdingsGrid .ag-row.holdingsRowBucket4 .ag-cell {
          background-color: #fef0e7;
        }
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
      `}</style>

      <style jsx>{`
        .holdingsToolbar {
          display: flex;
          align-items: center;
          gap: 8px;
          flex-wrap: wrap;
        }
        .bucket-legend-badge {
          display: inline-flex;
          align-items: center;
          padding: 2px 8px;
          border-radius: 999px;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: -0.01em;
        }
        .holdingsNameMain {
          min-width: 0;
          font-size: 0.95rem;
          font-weight: 700;
          color: #1d273b;
          word-break: keep-all;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
      `}</style>
    </div>
  );
}

function formatHoldingPrice(val: number, currency: string) {
  if (currency === "AUD") {
    return `A$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  if (currency === "USD") {
    return `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  return `${Math.floor(val).toLocaleString()}원`;
}

function formatHoldingsKrw(val: number) {
  return `${Math.round(val).toLocaleString()}원`;
}

function formatSignedPercent(value: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function normalizeDisplayTicker(value: string | null | undefined) {
  const raw = String(value ?? "").trim().toUpperCase();
  if (!raw) {
    return "-";
  }
  return raw.replace(/^ASX:/, "");
}

function getSignedClass(value: number | null) {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) {
    return "";
  }
  return value > 0 ? "metricPositive" : "metricNegative";
}

function parseHoldingDaysToInt(value: string | null | undefined): number {
  const raw = String(value ?? "").trim().toUpperCase();
  if (!raw || raw === "-") {
    return 0;
  }

  let totalDays = 0;
  const weekMatch = raw.match(/(\d+)\s*W/);
  const dayMatch = raw.match(/(\d+)\s*D/);

  if (weekMatch) {
    totalDays += Number(weekMatch[1]) * 7;
  }
  if (dayMatch) {
    totalDays += Number(dayMatch[1]);
  }

  return totalDays;
}
