"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { IconPlus } from "@tabler/icons-react";
import type { CellStyle, ColDef } from "ag-grid-community";

import { BUCKET_OPTIONS } from "@/lib/bucket-theme";
import { addStockCandidate, loadStocksTable } from "@/lib/stocks-store";
import type { StocksAccountItem } from "@/lib/stocks-store";
import { AppAgGrid } from "../components/AppAgGrid";
import { AppModal } from "../components/AppModal";
import { ResponsiveFiltersSection } from "../components/ResponsiveFiltersSection";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";
import {
  readRememberedTickerType,
  writeRememberedTickerType,
} from "../components/account-selection";

type UsMarketStockRow = {
  rank: number;
  ticker: string;
  name: string;
  english_name: string;
  industry: string;
  market: string;
  ticker_pools: string;
  is_held: boolean;
  current_price: number | null;
  change_pct: number | null;
  volume: number | null;
  market_cap: number | null;
};

type UsMarketStockGridRow = UsMarketStockRow & {
  __selected__?: boolean;
};

type UsMarketStocksResponse = {
  market: string;
  total_count: number;
  count: number;
  rows: UsMarketStockRow[];
  error?: string;
};

const usMarketStockGridTheme = createAppGridTheme();
const MARKET_OPTIONS = ["NYS", "NSQ"] as const;
const LIMIT_OPTIONS = [50, 100, 150, 200] as const;

function formatUsd(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `$${new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 2 }).format(value)}`;
}

function formatPercent(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value.toFixed(2)}%`;
}

function formatVolume(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR").format(value);
}

function formatUsdMarketCap(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (value >= 1_000_000_000_000) {
    return `${new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 2 }).format(value / 1_000_000_000_000)}조 달러`;
  }
  if (value >= 100_000_000) {
    return `${new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 1 }).format(value / 100_000_000)}억 달러`;
  }
  return `${new Intl.NumberFormat("ko-KR").format(value)}달러`;
}

function formatMarketLabel(market: string): string {
  if (market === "NYS") return "뉴욕";
  if (market === "NSQ") return "나스닥";
  return market;
}

export function UsMarketStockManager({
  onSummaryChange,
}: {
  onSummaryChange?: (summary: { market: string; count: number; totalCount: number }) => void;
}) {
  const [market, setMarket] = useState<(typeof MARKET_OPTIONS)[number]>("NSQ");
  const [limit, setLimit] = useState<number>(100);
  const [rows, setRows] = useState<UsMarketStockRow[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [tickerPools, setTickerPools] = useState<StocksAccountItem[]>([]);
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [addModalOpen, setAddModalOpen] = useState(false);
  const [selectedTickerPool, setSelectedTickerPool] = useState("");
  const [selectedBucketId, setSelectedBucketId] = useState<number | "">("");
  const [adding, setAdding] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const toast = useToast();

  const load = useCallback(async (m: string, l: number) => {
    setLoading(true);
    setError(null);
    try {
      const [resp, stocksPayload] = await Promise.all([
        fetch(`/api/us-market-stocks?market=${m}&limit=${l}`, { cache: "no-store" }),
        loadStocksTable().catch(() => ({ ticker_types: [], rows: [], ticker_type: "" })),
      ]);
      const data = (await resp.json()) as UsMarketStocksResponse;
      if (!resp.ok) {
        throw new Error(data.error ?? "데이터를 불러오지 못했습니다.");
      }
      setRows(data.rows ?? []);
      setTotalCount(data.total_count ?? 0);
      setTickerPools(stocksPayload.ticker_types ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "데이터를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(market, limit);
  }, [market, limit, load]);

  useEffect(() => {
    onSummaryChange?.({ market, count: rows.length, totalCount });
  }, [market, rows.length, totalCount, onSummaryChange]);

  const gridRows = useMemo(
    () =>
      [...rows].sort((left, right) => {
        const leftMarketCap = left.market_cap ?? Number.NEGATIVE_INFINITY;
        const rightMarketCap = right.market_cap ?? Number.NEGATIVE_INFINITY;
        if (leftMarketCap !== rightMarketCap) {
          return rightMarketCap - leftMarketCap;
        }
        return left.ticker.localeCompare(right.ticker);
      }),
    [rows],
  );

  const allVisibleSelected = useMemo(
    () => gridRows.length > 0 && gridRows.every((row) => selectedTickers.includes(row.ticker)),
    [gridRows, selectedTickers],
  );

  const toggleTickerSelection = useCallback((ticker: string) => {
    setSelectedTickers((current) =>
      current.includes(ticker) ? current.filter((item) => item !== ticker) : [...current, ticker],
    );
  }, []);

  const toggleSelectAllVisible = useCallback(() => {
    const visibleTickers = gridRows.map((row) => row.ticker);
    setSelectedTickers((current) => {
      if (visibleTickers.length === 0) return current;
      const allSelected = visibleTickers.every((ticker) => current.includes(ticker));
      if (allSelected) {
        return current.filter((ticker) => !visibleTickers.includes(ticker));
      }
      return [...new Set([...current, ...visibleTickers])];
    });
  }, [gridRows]);

  const handleOpenAddModal = useCallback(() => {
    if (selectedTickers.length === 0) return;

    const stockPools = tickerPools.filter((p) => p.name.includes("미국 개별주"));
    const remembered = readRememberedTickerType();

    if (remembered && stockPools.some((p) => p.ticker_type === remembered)) {
      setSelectedTickerPool(remembered);
    } else if (stockPools.length === 1) {
      setSelectedTickerPool(stockPools[0].ticker_type);
    } else {
      setSelectedTickerPool("");
    }

    setSelectedBucketId("");
    setAddModalOpen(true);
  }, [selectedTickers.length, tickerPools]);

  const handleCloseAddModal = useCallback(() => {
    if (adding) return;
    setAddModalOpen(false);
  }, [adding]);

  const handleAddSelected = useCallback(async () => {
    const tickerPool = String(selectedTickerPool || "").trim().toLowerCase();
    const bucketId = Number(selectedBucketId || 0);
    if (!tickerPool || !bucketId) {
      toast.error("종목풀과 버킷을 모두 선택하세요.");
      return;
    }

    setAdding(true);
    let addedCount = 0;
    let duplicateCount = 0;
    const failedTickers: string[] = [];

    for (const ticker of selectedTickers) {
      try {
        await addStockCandidate(tickerPool, ticker, bucketId);
        addedCount += 1;
      } catch (addError) {
        const message = addError instanceof Error ? addError.message : "종목 추가 처리에 실패했습니다.";
        if (message.includes("이미 등록된 종목입니다.")) {
          duplicateCount += 1;
          continue;
        }
        failedTickers.push(ticker);
      }
    }

    setAdding(false);
    setAddModalOpen(false);

    if (addedCount > 0) {
      toast.success(`종목 ${addedCount}개를 추가했습니다.`);
    }
    if (duplicateCount > 0) {
      toast.error(`이미 등록된 종목 ${duplicateCount}개는 건너뛰었습니다.`);
    }
    if (failedTickers.length > 0) {
      toast.error(`추가 실패: ${failedTickers.join(", ")}`);
    }

    if (addedCount > 0) {
      setSelectedTickers([]);
      await load(market, limit);
    }
  }, [load, market, limit, selectedBucketId, selectedTickerPool, selectedTickers, toast]);

  const columnDefs = useMemo<ColDef<UsMarketStockGridRow>[]>(
    () => [
      {
        headerName: "#",
        field: "rank",
        width: 64,
        minWidth: 56,
        maxWidth: 76,
        sortable: true,
        resizable: false,
        cellStyle: { textAlign: "center", color: "#8896a6" } as CellStyle,
      },
      {
        headerName: "종목풀",
        field: "ticker_pools",
        width: 108,
        maxWidth: 160,
        cellRenderer: (params: { value: string }) => String(params.value ?? "").trim() || "-",
      },
      {
        headerName: "티커",
        field: "ticker",
        width: 104,
        minWidth: 88,
        cellStyle: {
          fontFamily: "var(--font-mono, monospace)",
          fontSize: "13px",
        } as CellStyle,
        cellRenderer: (params: { value?: string }) => {
          // 미국 티커는 접두사 없이 그대로 사용. 백엔드 resolve에서 미국 우선 처리.
          const raw = String(params.value ?? "").trim();
          return <TickerDetailLink ticker={raw} displayTicker={raw} />;
        },
      },
      {
        headerName: "종목명",
        field: "name",
        flex: 1,
        minWidth: 180,
      },
      {
        headerName: "영문명",
        field: "english_name",
        flex: 1,
        minWidth: 220,
      },
      {
        headerName: "거래소",
        field: "market",
        width: 92,
        minWidth: 84,
        valueFormatter: (p) => formatMarketLabel(String(p.value ?? "")),
      },
      {
        headerName: "업종",
        field: "industry",
        width: 150,
        minWidth: 120,
        cellRenderer: (params: { value?: string }) => {
          const value = String(params.value ?? "").trim();
          return <span title={value}>{value || "-"}</span>;
        },
      },
      {
        headerName: "현재가",
        field: "current_price",
        width: 130,
        minWidth: 108,
        type: "rightAligned",
        valueFormatter: (p) => formatUsd(p.value),
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
        sort: "desc",
        valueFormatter: (p) => formatUsdMarketCap(p.value),
      },
      {
        field: "__selected__",
        headerName: "",
        width: 52,
        maxWidth: 52,
        sortable: false,
        filter: false,
        suppressHeaderMenuButton: true,
        suppressColumnsToolPanel: true,
        headerComponent: () => (
          <input
            type="checkbox"
            aria-label="전체 선택"
            checked={allVisibleSelected}
            onChange={() => toggleSelectAllVisible()}
          />
        ),
        cellRenderer: (params: { data?: UsMarketStockGridRow }) => {
          const ticker = String(params.data?.ticker ?? "").trim();
          if (!ticker) return null;
          return (
            <input
              type="checkbox"
              aria-label={`${ticker} 선택`}
              checked={selectedTickers.includes(ticker)}
              onChange={() => toggleTickerSelection(ticker)}
              onClick={(event) => event.stopPropagation()}
            />
          );
        },
      },
    ],
    [allVisibleSelected, selectedTickers, toggleSelectAllVisible, toggleTickerSelection],
  );

  return (
    <section className="appSection appSectionFill">
      <div className="card appCard appTableCardFill">
        <div className="card-header">
          <ResponsiveFiltersSection>
            <div className="appMainHeader">
              <div className="appMainHeaderLeft usMarketStockMainHeaderLeft">
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
                        {formatMarketLabel(opt)}
                      </button>
                    ))}
                  </div>
                </label>

                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">시가총액 상위</span>
                  <select
                    className="form-select"
                    value={limit}
                    onChange={(e) => setLimit(Number(e.target.value))}
                  >
                    {LIMIT_OPTIONS.map((opt) => (
                      <option key={opt} value={opt}>
                        {opt}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="appMainHeaderRight">
                <button
                  type="button"
                  className="btn btn-success btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                  onClick={handleOpenAddModal}
                  disabled={selectedTickers.length === 0}
                >
                  <IconPlus size={16} stroke={2} />
                  추가
                </button>
              </div>
            </div>
          </ResponsiveFiltersSection>
        </div>

        <div className="card-body appCardBodyTight appTableCardBodyFill">
          {error && (
            <div style={{ padding: "0.5rem 0.75rem", marginBottom: "0.5rem", background: "#fef2f2", color: "#dc2626", borderRadius: "6px", fontSize: "0.85rem" }}>
              {error}
            </div>
          )}

          <div className="appGridFillWrap">
            <AppAgGrid<UsMarketStockGridRow>
              rowData={gridRows}
              columnDefs={columnDefs}
              loading={loading}
              theme={usMarketStockGridTheme}
              minHeight="32rem"
              getRowClass={(params) => (params.data?.is_held ? "appHeldRow" : "")}
              gridOptions={{
                overlayNoRowsTemplate: '<span style="color:#667382;">데이터 없음</span>',
                suppressMovableColumns: true,
              }}
            />
          </div>
        </div>
      </div>

      <AppModal
        open={addModalOpen}
        title="종목풀 추가"
        subtitle={`선택한 종목 ${selectedTickers.length}개를 추가합니다.`}
        onClose={handleCloseAddModal}
        footer={
          <>
            <button type="button" className="btn btn-ghost-secondary" onClick={handleCloseAddModal} disabled={adding}>
              취소
            </button>
            <button
              type="button"
              className="btn btn-success"
              onClick={() => void handleAddSelected()}
              disabled={!selectedTickerPool || !selectedBucketId || adding}
            >
              {adding ? "추가 중..." : "추가"}
            </button>
          </>
        }
      >
        <div className="appModalFormStack" style={{ display: "grid", gap: "0.875rem" }}>
          <label className="appLabeledField">
            <span className="appLabeledFieldLabel">종목풀</span>
            <select
              className="field compactField"
              value={selectedTickerPool}
              onChange={(event) => {
                const nextType = event.target.value;
                setSelectedTickerPool(nextType);
                if (nextType) writeRememberedTickerType(nextType);
              }}
            >
              <option value="">종목풀 선택</option>
              {tickerPools
                .filter((p) => p.name.includes("미국 개별주"))
                .map((pool) => (
                  <option key={pool.ticker_type} value={pool.ticker_type}>
                    {pool.name}
                  </option>
                ))}
            </select>
          </label>
          <label className="appLabeledField">
            <span className="appLabeledFieldLabel">버킷</span>
            <select
              className="field compactField"
              value={selectedBucketId}
              onChange={(event) => setSelectedBucketId(event.target.value ? Number(event.target.value) : "")}
            >
              <option value="">버킷 선택</option>
              {BUCKET_OPTIONS.map((bucket) => (
                <option key={bucket.id} value={bucket.id}>
                  {bucket.name}
                </option>
              ))}
            </select>
          </label>
        </div>
      </AppModal>
    </section>
  );
}
