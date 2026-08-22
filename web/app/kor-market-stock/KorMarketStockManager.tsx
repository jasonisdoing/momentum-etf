"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { IconPlus } from "@tabler/icons-react";
import type { CellStyle, ColDef } from "ag-grid-community";

import { BUCKET_OPTIONS } from "@/lib/bucket-theme";
import {
  INDUSTRY_COLUMN_MIN_WIDTH,
  INDUSTRY_COLUMN_WIDTH,
  renderIndustryCell,
} from "@/lib/grid-cells";
import { formatKorMarketCap } from "@/lib/market-cap-format";
import { useLatestRequest } from "@/lib/use-latest-request";
import { formatPoolLabel } from "@/lib/pool-label";
import { loadStocksTable } from "@/lib/stocks-store";
import { addTickersToPool, describePoolAddPlan } from "@/lib/pool-add";
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

type KorMarketStockRow = {
  rank: number;
  ticker: string;
  /** 이 종목이 이미 들어 있는 종목풀 id 목록 — 추가 시 중복을 미리 거른다. */
  ticker_pool_types?: string[];
  name: string;
  industry: string;
  ticker_pools: string;
  is_held: boolean;
  current_price: number | null;
  change_pct: number | null;
  volume: number | null;
  market_cap: number | null;
  return_1m_pct: number | null;
  return_3m_pct: number | null;
  return_12m_pct: number | null;
  mdd_12m_pct: number | null;
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

const korMarketStockGridTheme = createAppGridTheme();

function formatKrw(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR").format(value);
}

function formatPercent(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value.toFixed(2)}%`;
}

function formatVolume(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR").format(value);
}


const MARKET_OPTIONS = ["KOSPI", "KOSDAQ"] as const;
const KOSPI_LIMIT_OPTIONS = [200, 150, 100, 50] as const;
const KOSDAQ_LIMIT_OPTIONS = [150, 100, 50] as const;

export function KorMarketStockManager({
  onSummaryChange,
}: {
  onSummaryChange?: (summary: { market: string; count: number; totalCount: number }) => void;
}) {
  const [market, setMarket] = useState<(typeof MARKET_OPTIONS)[number]>("KOSPI");
  const [limit, setLimit] = useState<number>(200);
  const [minMarketCapJo, setMinMarketCapJo] = useState("");
  const [rows, setRows] = useState<KorMarketStockRow[]>([]);
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
  const { begin, isLatest } = useLatestRequest();

  const load = useCallback(async (m: string, l: number, minCapJoText: string) => {
    // 코스피(200종목)가 코스닥(150종목)보다 느려서, 시장을 바꾸면 늦게 온 코스피 응답이
    // 먼저 그려진 코스닥 화면을 덮었다. 마지막 요청의 응답만 반영한다.
    const token = begin();
    setLoading(true);
    setError(null);
    try {
      const minCapJo = String(minCapJoText || "").trim() || "0";
      const [resp, allStocksPayload, korStocksPayload] = await Promise.all([
        fetch(`/api/kor-market-stocks?market=${m}&limit=${l}&min_market_cap_jo=${encodeURIComponent(minCapJo)}`, { cache: "no-store" }),
        loadStocksTable().catch(() => ({ ticker_types: [], rows: [], ticker_type: "" })),
        loadStocksTable("kor").catch(() => ({ ticker_types: [], rows: [], ticker_type: "" })),
      ]);
      const data = (await resp.json()) as KorMarketStocksResponse;
      if (!isLatest(token)) return;
      if (!resp.ok) {
        throw new Error(data.error ?? "데이터를 불러오지 못했습니다.");
      }
      setRows(data.rows ?? []);
      setTotalCount(data.total_count ?? 0);
      setTickerPools(allStocksPayload.ticker_types ?? []);

      const registered = new Set<string>();
      (korStocksPayload.rows ?? []).forEach((row: any) => {
        if (row.ticker) {
          registered.add(String(row.ticker).trim().toUpperCase());
        }
      });
      setRegisteredTickers(registered);
    } catch (e) {
      if (!isLatest(token)) return;
      setError(e instanceof Error ? e.message : "데이터를 불러오지 못했습니다.");
    } finally {
      if (isLatest(token)) setLoading(false);
    }
  }, [begin, isLatest]);

  useEffect(() => {
    load(market, limit, minMarketCapJo);
  }, [market, limit, minMarketCapJo, load]);

  const limitOptions = useMemo<number[]>(
    () => (market === "KOSPI" ? [...KOSPI_LIMIT_OPTIONS] : [...KOSDAQ_LIMIT_OPTIONS]),
    [market],
  );

  useEffect(() => {
    if (!limitOptions.includes(limit)) {
      setLimit(limitOptions[0]);
    }
  }, [limit, limitOptions]);

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

    const stockPools = tickerPools.filter((p) => p.country_code === "kor");
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
    // 이미 그 풀에 있는 종목은 보내지 않는다 — 표가 풀 목록을 들고 있어 조회가 필요 없다.
    const { added, skipped, failed } = await addTickersToPool(selectedTickers, rows, tickerPool, bucketId);

    setAdding(false);
    setAddModalOpen(false);

    if (added > 0) {
      toast.success(`종목 ${added}개를 추가했습니다.`);
    }
    if (skipped > 0) {
      toast.warning(`선택한 ${selectedTickers.length}개 중 이미 있는 ${skipped}개는 제외했습니다.`);
    }
    if (failed.length > 0) {
      toast.error(`추가 실패: ${failed.join(", ")}`);
    }

    if (added > 0) {
      setSelectedTickers([]);
      await load(market, limit, minMarketCapJo);
    }
  }, [load, market, limit, minMarketCapJo, selectedBucketId, selectedTickerPool, selectedTickers, toast]);

  const columnDefs = useMemo<ColDef<KorMarketStockGridRow>[]>(
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
        width: 180,
        maxWidth: 320,
        cellRenderer: (params: { value: string }) => String(params.value ?? "").trim() || "-",
      },
      {
        headerName: "티커",
        field: "ticker",
        width: 100,
        minWidth: 84,
        cellStyle: {
          fontFamily: "var(--font-mono, monospace)",
          fontSize: "var(--fs-sm)",
        } as CellStyle,
        cellClass: "korMarketStockTickerCell",
        cellRenderer: (params: { value?: string }) => <TickerDetailLink ticker={String(params.value ?? "")} />,
      },
      {
        headerName: "종목명",
        field: "name",
        flex: 1,
        minWidth: 180,
      },
      {
        headerName: "업종",
        field: "industry",
        width: INDUSTRY_COLUMN_WIDTH,
        minWidth: INDUSTRY_COLUMN_MIN_WIDTH,
        headerTooltip: "네이버 업종 분류 — 신고가 돌파·종목풀 순위 화면과 같은 값.",
        cellRenderer: (params: { value?: string }) => renderIndustryCell(params.value),
      },
      {
        headerName: "일간(%)",
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
        valueFormatter: (p) => formatKrw(p.value),
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
        valueFormatter: (p) => formatKorMarketCap(p.value),
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
    [allVisibleSelected, selectedTickers, toggleSelectAllVisible, toggleTickerSelection, registeredTickers],
  );

  return (
    <section className="appSection appSectionFill">
      <div className="card appCard appTableCardFill">
        {/* 메인 헤더 */}
        <div className="card-header">
          <ResponsiveFiltersSection>
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
                        onClick={() => {
                          setMarket(opt);
                          setLimit(opt === "KOSPI" ? 200 : 150);
                        }}
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
                    onChange={(e) => setLimit(Number(e.target.value))}
                  >
                    {limitOptions.map((opt) => (
                      <option key={opt} value={opt}>
                        {market} {opt}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">최소 시가총액(조)</span>
                  <input
                    className="form-control"
                    inputMode="numeric"
                    value={minMarketCapJo}
                    onChange={(e) => setMinMarketCapJo(e.target.value.replace(/[^\d]/g, ""))}
                    placeholder="최소 시가총액(조)"
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
            <div style={{ padding: "0.5rem 0.75rem", marginBottom: "0.5rem", background: "#fef2f2", color: "#dc2626", borderRadius: "6px", fontSize: "var(--fs-sm)" }}>
              {error}
            </div>
          )}

          <div className="appGridFillWrap">
            <AppAgGrid<KorMarketStockGridRow>
              rowData={gridRows}
              columnDefs={columnDefs}
              loading={loading}
              theme={korMarketStockGridTheme}
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
        subtitle={describePoolAddPlan(selectedTickers, rows, selectedTickerPool)}
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
                .filter((p) => p.country_code === "kor")
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
