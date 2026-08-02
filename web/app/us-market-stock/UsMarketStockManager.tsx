"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { IconPlus } from "@tabler/icons-react";
import type { CellStyle, ColDef } from "ag-grid-community";

import { BUCKET_OPTIONS } from "@/lib/bucket-theme";
import { formatPoolLabel } from "@/lib/pool-label";
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
  sector: string;
  market: string;
  ticker_pools: string;
  is_held: boolean;
  current_price: number | null;
  change_pct: number | null;
  volume: number | null;
  market_cap: number | null;
  return_1m_base_date: string | null;
  return_1m_base_price: number | null;
  return_1m_pct: number | null;
  return_3m_base_date: string | null;
  return_3m_base_price: number | null;
  return_3m_pct: number | null;
  return_12m_base_date: string | null;
  return_12m_base_price: number | null;
  return_12m_pct: number | null;
  mdd_12m_pct: number | null;
};

type UsMarketStockGridRow = UsMarketStockRow & {
  __selected__?: boolean;
};

type UsMarketStocksResponse = {
  index: string;
  updated_at: string;
  total_count: number;
  count: number;
  rows: UsMarketStockRow[];
  error?: string;
};

const usMarketStockGridTheme = createAppGridTheme();
// 화면에서 고르는 보기 — `통합` 은 두 지수를 합쳐 중복 종목을 한 번만 센다.
const VIEW_OPTIONS = [
  { key: "COMBINED", label: "통합", indices: ["SP500", "NDX100"] },
  { key: "SP500", label: "S&P", indices: ["SP500"] },
  { key: "NDX100", label: "나스닥", indices: ["NDX100"] },
] as const;
type ViewOption = (typeof VIEW_OPTIONS)[number]["key"];

function viewIndices(view: ViewOption): readonly string[] {
  return VIEW_OPTIONS.find((option) => option.key === view)?.indices ?? [];
}

// 시총 상위 몇 개까지 볼지 — 응답이 시총 순 정렬이라 상위 N 절단으로 처리한다.
// null 이면 전체(절단 없음).
const TOP_OPTIONS = [null, 100, 200, 300, 400, 500] as const;

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

function renderTruncatedText(value: string | null | undefined) {
  const text = String(value ?? "").trim();
  return (
    <span className="usMarketStockTruncate" title={text}>
      {text || "-"}
    </span>
  );
}

function isUsTickerPool(pool: StocksAccountItem): boolean {
  return String(pool.country_code ?? "").trim().toLowerCase() === "us";
}

export function UsMarketStockManager({
  onSummaryChange,
}: {
  onSummaryChange?: (summary: { index: string; count: number; totalCount: number }) => void;
}) {
  const [view, setView] = useState<ViewOption>("COMBINED");
  const [topCount, setTopCount] = useState<number | null>(null);
  const [minMarketCapUkm, setMinMarketCapUkm] = useState<string>("");
  const [rows, setRows] = useState<UsMarketStockRow[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [tickerPools, setTickerPools] = useState<StocksAccountItem[]>([]);
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [registeredTickers, setRegisteredTickers] = useState<Set<string>>(new Set());
  const [addModalOpen, setAddModalOpen] = useState(false);
  const [selectedTickerPool, setSelectedTickerPool] = useState("");
  const [selectedBucketId, setSelectedBucketId] = useState<number | "">("");
  const [adding, setAdding] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const toast = useToast();

  const load = useCallback(async (currentView: ViewOption, minCapUkmText: string) => {
    setLoading(true);
    setError(null);
    try {
      const minCap = String(minCapUkmText || "").trim() || "0";
      const indices = viewIndices(currentView);
      const [responses, allStocksPayload, usStocksPayload] = await Promise.all([
        Promise.all(
          indices.map((idx) =>
            fetch(
              `/api/us-market-stocks?index=${encodeURIComponent(idx)}&min_market_cap_ukm=${encodeURIComponent(minCap)}`,
              { cache: "no-store" },
            ),
          ),
        ),
        loadStocksTable().catch(() => ({ ticker_types: [], rows: [], ticker_type: "" })),
        loadStocksTable("us").catch(() => ({ ticker_types: [], rows: [], ticker_type: "" })),
      ]);
      const payloads: UsMarketStocksResponse[] = [];
      for (const resp of responses) {
        const data = (await resp.json()) as UsMarketStocksResponse;
        if (!resp.ok) throw new Error(data.error ?? "데이터를 불러오지 못했습니다.");
        payloads.push(data);
      }
      // 합집합 — 같은 종목이 두 지수에 있으면 한 번만 담고, 시총 내림차순으로 다시 정렬한다.
      const merged = new Map<string, UsMarketStockRow>();
      for (const data of payloads) {
        for (const row of data.rows ?? []) {
          if (!merged.has(row.ticker)) merged.set(row.ticker, row);
        }
      }
      const mergedRows = [...merged.values()].sort(
        (a, b) => (b.market_cap ?? 0) - (a.market_cap ?? 0),
      );
      setRows(mergedRows);
      setTotalCount(mergedRows.length);
      setTickerPools(allStocksPayload.ticker_types ?? []);

      const registered = new Set<string>();
      (usStocksPayload.rows ?? []).forEach((row: any) => {
        if (row.ticker) {
          registered.add(String(row.ticker).trim().toUpperCase());
        }
      });
      setRegisteredTickers(registered);
    } catch (e) {
      setError(e instanceof Error ? e.message : "데이터를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(view, minMarketCapUkm);
  }, [view, minMarketCapUkm, load]);

  // 시총 상위 N 만 표시 (전체면 절단 없음). rows 는 이미 시총 내림차순이다.
  const visibleRows = useMemo(
    () => (topCount === null ? rows : rows.slice(0, topCount)),
    [rows, topCount],
  );

  useEffect(() => {
    onSummaryChange?.({ index: view, count: visibleRows.length, totalCount });
  }, [view, visibleRows.length, totalCount, onSummaryChange]);

  const gridRows = useMemo(() => [...visibleRows], [visibleRows]);

  const allVisibleSelected = useMemo(() => {
    return gridRows.length > 0 && gridRows.every((row) => selectedTickers.includes(row.ticker));
  }, [gridRows, selectedTickers]);

  const toggleTickerSelection = useCallback((ticker: string) => {
    setSelectedTickers((current) =>
      current.includes(ticker) ? current.filter((item) => item !== ticker) : [...current, ticker],
    );
  }, []);

  const toggleSelectAllVisible = useCallback(() => {
    const selectableTickers = gridRows
      .map((row) => row.ticker);
    setSelectedTickers((current) => {
      if (selectableTickers.length === 0) return current;
      const allSelected = selectableTickers.every((ticker) => current.includes(ticker));
      if (allSelected) {
        return current.filter((ticker) => !selectableTickers.includes(ticker));
      }
      return [...new Set([...current, ...selectableTickers])];
    });
  }, [gridRows, registeredTickers]);

  const handleOpenAddModal = useCallback(() => {
    if (selectedTickers.length === 0) return;

    const stockPools = tickerPools.filter(isUsTickerPool);
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
      await load(view, minMarketCapUkm);
    }
  }, [load, view, minMarketCapUkm, selectedBucketId, selectedTickerPool, selectedTickers, toast]);

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
        cellClass: "usMarketStockTextCell",
        cellRenderer: (params: { value: string }) => renderTruncatedText(params.value),
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
          const raw = String(params.value ?? "").trim();
          return <TickerDetailLink ticker={raw} displayTicker={raw} />;
        },
      },
      {
        headerName: "종목명",
        field: "name",
        flex: 1,
        minWidth: 180,
        cellClass: "usMarketStockTextCell",
        cellRenderer: (params: { value?: string }) => renderTruncatedText(params.value),
      },
      {
        headerName: "섹터",
        field: "sector",
        width: 160,
        minWidth: 120,
        cellClass: "usMarketStockTextCell",
        cellRenderer: (params: { value?: string }) => renderTruncatedText(params.value),
      },
      {
        headerName: "업종",
        field: "industry",
        width: 180,
        minWidth: 120,
        cellClass: "usMarketStockTextCell",
        cellRenderer: (params: { value?: string }) => renderTruncatedText(params.value),
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
        headerName: "현재가",
        field: "current_price",
        width: 130,
        minWidth: 108,
        type: "rightAligned",
        valueFormatter: (p) => formatUsd(p.value),
      },
      {
        headerName: "1개월",
        field: "return_1m_pct",
        width: 104,
        minWidth: 96,
        type: "rightAligned",
        valueFormatter: (p) => formatPercent(p.value),
        cellClassRules: {
          metricPositive: (p) => p.value != null && p.value > 0,
          metricNegative: (p) => p.value != null && p.value < 0,
        },
      },
      {
        headerName: "3개월",
        field: "return_3m_pct",
        width: 104,
        minWidth: 96,
        type: "rightAligned",
        valueFormatter: (p) => formatPercent(p.value),
        cellClassRules: {
          metricPositive: (p) => p.value != null && p.value > 0,
          metricNegative: (p) => p.value != null && p.value < 0,
        },
      },
      {
        headerName: "12개월",
        field: "return_12m_pct",
        width: 108,
        minWidth: 100,
        type: "rightAligned",
        valueFormatter: (p) => formatPercent(p.value),
        cellClassRules: {
          metricPositive: (p) => p.value != null && p.value > 0,
          metricNegative: (p) => p.value != null && p.value < 0,
        },
      },
      {
        headerName: "MDD(12개월)",
        field: "mdd_12m_pct",
        width: 132,
        minWidth: 122,
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
        width: 120,
        minWidth: 110,
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
    [allVisibleSelected, selectedTickers, toggleSelectAllVisible, toggleTickerSelection, registeredTickers],
  );

  return (
    <section className="appSection appSectionFill">
      <div className="card appCard appTableCardFill">
        <div className="card-header">
          <ResponsiveFiltersSection>
            <div className="appMainHeader">
              <div className="appMainHeaderLeft usMarketStockMainHeaderLeft">
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">인덱스</span>
                  <div className="appSegmentedToggle appSegmentedToggleCompact" role="group" aria-label="인덱스 선택">
                    {VIEW_OPTIONS.map((option) => (
                      <button
                        key={option.key}
                        type="button"
                        className={
                          view === option.key
                            ? "btn appSegmentedToggleButton is-active"
                            : "btn appSegmentedToggleButton"
                        }
                        title={option.key === "COMBINED" ? "S&P 500 과 나스닥 100 의 합집합" : undefined}
                        onClick={() => setView(option.key)}
                      >
                        {option.label}
                      </button>
                    ))}
                    <select
                      value={topCount === null ? "all" : String(topCount)}
                      onChange={(event) =>
                        setTopCount(event.target.value === "all" ? null : Number(event.target.value))
                      }
                      style={{
                        border: "1px solid rgba(148,163,184,0.4)",
                        borderRadius: 6,
                        padding: "3px 6px",
                        fontSize: "0.84rem",
                        marginLeft: 6,
                      }}
                    >
                      {TOP_OPTIONS.map((count) => (
                        <option key={count ?? "all"} value={count === null ? "all" : String(count)}>
                          {count === null ? "전체" : `상위 ${count}`}
                        </option>
                      ))}
                    </select>
                  </div>
                </label>

                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">최소 시가총액(억 달러)</span>
                  <input
                    className="form-control"
                    inputMode="numeric"
                    value={minMarketCapUkm}
                    onChange={(e) => setMinMarketCapUkm(e.target.value.replace(/[^\d]/g, ""))}
                    placeholder="최소 시가총액(억 달러)"
                  />
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
              className="usMarketStockGrid"
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
                .filter(isUsTickerPool)
                .map((pool) => (
                  <option key={pool.ticker_type} value={pool.ticker_type}>
                    {formatPoolLabel(pool)}
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
