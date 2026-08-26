"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { IconPlus } from "@tabler/icons-react";
import type { CellStyle, ColDef } from "ag-grid-community";

import { BUCKET_OPTIONS } from "@/lib/bucket-theme";
import { INDUSTRY_COLUMN_MIN_WIDTH, INDUSTRY_COLUMN_WIDTH, renderIndustryCell } from "@/lib/grid-cells";
import { formatPoolLabel } from "@/lib/pool-label";
import { useLatestRequest } from "@/lib/use-latest-request";
import { loadStocksTable } from "@/lib/stocks-store";
import { addTickersToPool, buildPoolAddSkipNotice, splitByPoolMembership } from "@/lib/pool-add";
import type { PoolAddProgress } from "@/lib/pool-add";
import type { StocksAccountItem } from "@/lib/stocks-store";
import { AppAgGrid } from "../components/AppAgGrid";
import { AppModal } from "../components/AppModal";
import { PoolAddProgressBar } from "../components/PoolAddProgressBar";
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
  /** 이 종목이 이미 들어 있는 종목풀 id 목록 — 추가 시 중복을 미리 거른다. */
  ticker_pool_types?: string[];
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
// 통합은 두 지수를 합쳐 600 종목에 가까워 100 단위로는 구간이 너무 성기다 → 50 단위.
// 단일 지수는 종목 수가 적어 100 단위 그대로 둔다.
// 마지막으로 고른 상위 N — 다음 방문에도 같은 범위로 열리게 기억한다.
// 키 형식은 시스템 공통(`momentum-etf:<화면>:<항목>`)을 따른다.
const US_MARKET_TOP_COUNT_KEY = "momentum-etf:us-market-stock:top-count";

/** 저장된 상위 N. `"all"`(전체)이거나 값이 없으면 null. */
function readRememberedTopCount(): number | null {
  if (typeof window === "undefined") {
    return null;
  }
  const raw = window.localStorage.getItem(US_MARKET_TOP_COUNT_KEY);
  if (!raw || raw === "all") {
    return null;
  }
  const parsed = Number(raw);
  // 못 읽는 값은 전체로 둔다 — 임의의 숫자로 잘라 보여주면 무엇이 적용됐는지 알 수 없다.
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function writeRememberedTopCount(value: number | null): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(US_MARKET_TOP_COUNT_KEY, value === null ? "all" : String(value));
}

const COMBINED_TOP_STEP = 50;
const SINGLE_TOP_OPTIONS: readonly (number | null)[] = [null, 100, 200, 300, 400, 500];

/**
 * 보기별 상위 N 선택지. 통합은 실제 종목 수까지만 50 단위로 채운다
 * (전체 개수 이상은 `전체` 와 같은 결과라 만들지 않는다).
 *
 * 지금 고른 값이 목록에 없으면 함께 노출한다 — 빼면 셀렉트가 빈칸이 되어
 * 무엇이 적용 중인지 알 수 없다.
 */
function topOptions(view: ViewOption, rowCount: number, current: number | null): (number | null)[] {
  const options: (number | null)[] =
    view === "COMBINED"
      ? [null, ...Array.from(
          { length: Math.max(0, Math.ceil(rowCount / COMBINED_TOP_STEP) - 1) },
          (_, i) => (i + 1) * COMBINED_TOP_STEP,
        )]
      : [...SINGLE_TOP_OPTIONS];
  if (current !== null && !options.includes(current)) {
    options.push(current);
    options.sort((a, b) => (a ?? -1) - (b ?? -1));
  }
  return options;
}

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
  // 종목당 수 초씩 걸려서 진행도를 보여주지 않으면 멈춘 것처럼 보인다.
  const [addProgress, setAddProgress] = useState<PoolAddProgress | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // 마지막으로 고른 상위 N 복원 — 서버 렌더에는 localStorage 가 없어 초기값으로 못 쓴다.
  // 상위 N 은 이미 받아둔 행을 자르기만 해서(재조회 없음) 늦게 반영돼도 값싸다.
  useEffect(() => {
    setTopCount(readRememberedTopCount());
  }, []);

  const toast = useToast();
  const { begin, isLatest } = useLatestRequest();

  const load = useCallback(async (currentView: ViewOption, minCapUkmText: string) => {
    // 늦게 도착한 옛 응답이 새 결과를 덮지 않게 한다(공용 훅).
    const token = begin();
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
      // 시총 내림차순으로 세운 뒤 1번부터 다시 번호를 붙인다. 서버는 지수별로 번호를
      // 매기므로(SP500 7위 · NDX100 7위), 합쳐 놓고 그대로 두면 같은 번호가 두 번 나오고
      // 아래로 갈수록 순서와 어긋난다.
      const mergedRows = [...merged.values()]
        .sort((a, b) => (b.market_cap ?? 0) - (a.market_cap ?? 0))
        .map((row, index) => ({ ...row, rank: index + 1 }));
      if (!isLatest(token)) return;
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
      if (!isLatest(token)) return;
      setError(e instanceof Error ? e.message : "데이터를 불러오지 못했습니다.");
    } finally {
      if (isLatest(token)) setLoading(false);
    }
  }, [begin, isLatest]);

  useEffect(() => {
    void load(view, minMarketCapUkm);
  }, [view, minMarketCapUkm, load]);

  // 시총 상위 N 만 표시 (전체면 절단 없음). rows 는 이미 시총 내림차순이다.
  const visibleRows = useMemo(
    () => (topCount === null ? rows : rows.slice(0, topCount)),
    [rows, topCount],
  );

  const topChoices = useMemo(() => topOptions(view, rows.length, topCount), [view, rows.length, topCount]);

  useEffect(() => {
    onSummaryChange?.({ index: view, count: visibleRows.length, totalCount });
  }, [view, visibleRows.length, totalCount, onSummaryChange]);

  // `#` 은 지금 보고 있는 목록에서의 위치다. 백엔드가 준 rank 는 지수 안에서의 순위라
  // 통합 보기에서는 두 지수의 번호가 섞여 중복·누락이 생긴다(상위 150 인데 144 까지만 보이는 등).
  // 시총 내림차순으로 합쳐 자른 뒤 여기서 1 번부터 다시 매긴다.
  const gridRows = useMemo(
    () => visibleRows.map((row, index) => ({ ...row, rank: index + 1 })),
    [visibleRows],
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

    // 이미 어딘가의 풀에 있는 종목은 보내지 않는다 — 표가 풀 목록을 들고 있어 조회가 필요 없다.
    // 건너뛰는 건 정상 동작이라 시작 전에 노란색으로 알린다.
    const split = splitByPoolMembership(selectedTickers, rows, tickerPool);
    const skipNotice = buildPoolAddSkipNotice(selectedTickers.length, split);
    if (skipNotice) {
      toast.warning(skipNotice);
    }
    if (split.fresh.length === 0) {
      setAddModalOpen(false);
      return;
    }

    setAdding(true);
    setAddProgress(null);
    const { added, skipped, blocked, failed } = await addTickersToPool(split.fresh, tickerPool, bucketId, setAddProgress);

    setAdding(false);
    setAddProgress(null);
    setAddModalOpen(false);

    if (added > 0) {
      toast.success(`종목 ${added}개를 추가했습니다.`);
    }
    if (skipped > 0) {
      toast.error(`이미 등록된 종목 ${skipped}개는 건너뛰었습니다.`);
    }
    if (blocked > 0) {
      toast.error(`다른 종목풀에 있는 ${blocked}개는 건너뛰었습니다.`);
    }
    if (failed.length > 0) {
      toast.error(`추가 실패: ${failed.join(", ")}`);
    }

    if (added > 0) {
      setSelectedTickers([]);
      await load(view, minMarketCapUkm);
    }
  }, [load, view, minMarketCapUkm, rows, selectedBucketId, selectedTickerPool, selectedTickers, toast]);

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
          fontSize: "var(--fs-sm)",
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
        width: INDUSTRY_COLUMN_WIDTH,
        minWidth: INDUSTRY_COLUMN_MIN_WIDTH,
        cellClass: "usMarketStockTextCell",
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
                      onChange={(event) => {
                        const next = event.target.value === "all" ? null : Number(event.target.value);
                        setTopCount(next);
                        writeRememberedTopCount(next);
                      }}
                      style={{
                        border: "1px solid rgba(148,163,184,0.4)",
                        borderRadius: 6,
                        padding: "3px 6px",
                        fontSize: "var(--fs-sm)",
                        marginLeft: 6,
                      }}
                    >
                      {topChoices.map((count) => (
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
            <div style={{ padding: "0.5rem 0.75rem", marginBottom: "0.5rem", background: "#fef2f2", color: "#dc2626", borderRadius: "6px", fontSize: "var(--fs-sm)" }}>
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
          <PoolAddProgressBar progress={addProgress} />
        </div>
      </AppModal>
    </section>
  );
}
