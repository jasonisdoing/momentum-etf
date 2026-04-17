"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { IconPlus } from "@tabler/icons-react";
import { iconSetQuartzBold, themeQuartz } from "ag-grid-community";
import type { CellStyle, ColDef } from "ag-grid-community";

import { BUCKET_OPTIONS } from "@/lib/bucket-theme";
import { addStockCandidate, loadStocksTable } from "@/lib/stocks-store";
import type { StocksAccountItem } from "@/lib/stocks-store";
import { AppAgGrid } from "../components/AppAgGrid";
import { AppModal } from "../components/AppModal";
import { useToast } from "../components/ToastProvider";
import {
  readRememberedTickerType,
  writeRememberedTickerType,
} from "../components/account-selection";

type KorMarketStockRow = {
  rank: number;
  ticker: string;
  name: string;
  ticker_pools: string;
  is_held: boolean;
  current_price: number | null;
  change_pct: number | null;
  volume: number | null;
  market_cap: number | null;
};

type KorMarketStockGridRow = KorMarketStockRow & {
  __selected__?: boolean;
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
        fetch(`/api/kor-market-stocks?market=${m}&limit=${l}`, { cache: "no-store" }),
        loadStocksTable().catch(() => ({ ticker_types: [], rows: [], ticker_type: "" })),
      ]);
      const data = (await resp.json()) as KorMarketStocksResponse;
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
    load(market, limit);
  }, [market, limit, load]);

  useEffect(() => {
    onSummaryChange?.({ market, count: rows.length, totalCount });
  }, [market, rows.length, totalCount, onSummaryChange]);

  const allVisibleSelected = useMemo(
    () => rows.length > 0 && rows.every((row) => selectedTickers.includes(row.ticker)),
    [rows, selectedTickers],
  );

  const toggleTickerSelection = useCallback((ticker: string) => {
    setSelectedTickers((current) =>
      current.includes(ticker) ? current.filter((item) => item !== ticker) : [...current, ticker],
    );
  }, []);

  const toggleSelectAllVisible = useCallback(() => {
    const visibleTickers = rows.map((row) => row.ticker);
    setSelectedTickers((current) => {
      if (visibleTickers.length === 0) return current;
      const allSelected = visibleTickers.every((ticker) => current.includes(ticker));
      if (allSelected) {
        return current.filter((ticker) => !visibleTickers.includes(ticker));
      }
      return [...new Set([...current, ...visibleTickers])];
    });
  }, [rows]);

  const handleOpenAddModal = useCallback(() => {
    if (selectedTickers.length === 0) return;

    const stockPools = tickerPools.filter((p) => p.name.includes("한국 개별주"));
    const remembered = readRememberedTickerType();
    
    if (remembered && stockPools.some(p => p.ticker_type === remembered)) {
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

  const columnDefs = useMemo<ColDef<KorMarketStockGridRow>[]>(
    () => [
      {
        headerName: "#",
        field: "rank",
        width: 64,
        minWidth: 56,
        maxWidth: 76,
        sortable: false,
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
        width: 100,
        minWidth: 84,
        cellStyle: { fontFamily: "var(--font-mono, monospace)", fontSize: "13px" } as CellStyle,
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
        cellRenderer: (params: { data?: KorMarketStockGridRow }) => {
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
        </div>

        <div className="card-body appCardBodyTight appTableCardBodyFill">
          {error && (
            <div style={{ padding: "0.5rem 0.75rem", marginBottom: "0.5rem", background: "#fef2f2", color: "#dc2626", borderRadius: "6px", fontSize: "0.85rem" }}>
              {error}
            </div>
          )}

          <div className="appGridFillWrap">
            <AppAgGrid<KorMarketStockGridRow>
              rowData={rows}
              columnDefs={columnDefs}
              loading={loading}
              theme={korMarketStockGridTheme}
              minHeight="32rem"
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
                .filter((p) => p.name.includes("한국 개별주"))
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
