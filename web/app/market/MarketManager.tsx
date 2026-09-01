"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { IconPlus } from "@tabler/icons-react";
import type { ColDef, RowClassParams } from "ag-grid-community";

import { BUCKET_OPTIONS } from "@/lib/bucket-theme";
import { formatPoolLabel } from "@/lib/pool-label";
import { addTickersToPool, buildPoolAddSkipNotice, splitByPoolMembership } from "@/lib/pool-add";
import type { PoolAddProgress } from "@/lib/pool-add";
import { loadStocksTable } from "@/lib/stocks-store";
import { AppAgGrid } from "../components/AppAgGrid";
import { PoolAddProgressBar } from "../components/PoolAddProgressBar";
import { ResponsiveFiltersSection } from "../components/ResponsiveFiltersSection";
import { AppModal } from "../components/AppModal";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";
import {
  readRememberedTickerType,
  writeRememberedTickerType,
} from "../components/account-selection";

type MarketRowItem = {
  ticker: string;
  ticker_pools: string;
  ticker_pool_types?: string[];
  name: string;
  listed_at: string;
  daily_change_pct: number | null;
  current_price: number | null;
  nav: number | null;
  deviation: number | null;
  return_1m_pct: number | null;
  return_2m_pct: number | null;
  return_3m_pct: number | null;
  prev_volume: number;
  market_cap: number;
  is_held: boolean;
  /** PTP(Publicly Traded Partnership) — 국내 매도 시 총액 10% 원천징수라 사실상 거래 대상이 아니다.
   *  이름으로는 못 가르는 값이라 배치가 판정해 내려준다(`utils/us_etf_market_service`). */
  is_ptp?: boolean;
  /** 매매차익 비과세 여부(국내 주식형 ETF만). 분류를 못 받은 종목은 없음 = 모름. */
  is_tax_free?: boolean | null;
};

type MarketResponse = {
  updated_at?: string | null;
  rows?: MarketRowItem[];
  error?: string;
};

type MarketTickerPool = {
  ticker_type: string;
  order: number;
  name: string;
  icon: string;
};

type MarketGridRow = MarketRowItem & {
  row_number: number;
  __selected__?: boolean;
};

// 시장별 차이(데이터 원천·컬럼 구성·제외 키워드)는 전부 여기서만 정한다 — 화면은 하나다.
type MarketCode = "kor" | "us";

type MarketVariantConfig = {
  apiPath: string;
  errorLabel: string;
  // 이름 키워드 제외 필터 — 대소문자 무시 비교이므로 키워드는 대문자로 적는다.
  exclusionGroups: Record<string, string[]>;
  /** 이름으로 못 가르는 제외 그룹 — 배치가 판정해 둔 불리언 필드를 본다.
   *  키워드 그룹과 같은 칩으로 보이고 같은 방식으로 켜고 끈다. */
  flagExclusions?: Record<string, keyof MarketRowItem>;
  /** 과세 구분 토글(모두/과세/비과세)을 보일지 — 국내 ETF 만 이 구분이 있다. */
  showTaxFilter?: boolean;
  defaultExcluded: string[];
  showNavColumns: boolean; // Nav·괴리율 — 한국 실시간 스냅샷에만 있다
  showListing: boolean; // 상장일 컬럼 + 신규 필터 — 미국 마스터에는 상장일이 없다
  capHeader: string; // 규모 컬럼: 한국은 시가총액, 미국은 20일 평균 거래대금
  capFilterDefault: string; // 규모 최소값 필터 초기값 ("" = 없음)
  volumeFilterDefault: string; // 거래량 최소값 필터 초기값 ("" = 없음)
};

// 과세 구분 선택지 — 분류를 못 받은 종목은 '모두'에서만 보인다(어느 쪽으로도 넘겨짚지 않는다).
const TAX_FILTER_OPTIONS = [
  { key: "all", label: "모두", title: "과세 구분과 무관하게 전부" },
  { key: "taxed", label: "과세", title: "해외·파생·원자재·채권 — 매매차익에 배당소득세" },
  { key: "free", label: "비과세", title: "국내 주식형 — 매매차익 비과세" },
] as const;

const MARKET_VARIANTS: Record<MarketCode, MarketVariantConfig> = {
  kor: {
    apiPath: "/api/market",
    errorLabel: "ETF 마켓 데이터를 불러오지 못했습니다.",
    exclusionGroups: {
      "채권(모든종류)": ["채권", "미국채", "국채", "회사채", "단기채", "장기채"],
      혼합: ["혼합"],
      리츠: ["리츠"],
      인버스: ["인버스"],
      "2X": ["2X"],
      레버리지: ["레버리지"],
      합성: ["합성"],
      커버드콜: ["커버드콜"],
      선물: ["선물"],
    },
    defaultExcluded: ["채권(모든종류)", "혼합", "리츠", "인버스", "2X", "레버리지"],
    showNavColumns: true,
    showListing: true,
    capHeader: "시가총액(억)",
    capFilterDefault: "",
    volumeFilterDefault: "",
    // 국내 주식형 ETF 는 매매차익이 비과세고 그 밖(해외·파생·원자재·채권)은 과세다.
    showTaxFilter: true,
  },
  us: {
    apiPath: "/api/market/us-etf",
    errorLabel: "미국 ETF 마켓 데이터를 불러오지 못했습니다.",
    exclusionGroups: {
      // 채권·현금성 — 이름 표기가 다양해 키워드를 넓게 건다. 앞뒤 공백이 있는 키워드는
      // 단어 단위 매칭이다(비교 시 이름 양끝에 공백을 붙인다) — "CLO" 를 그냥 걸면 CLOUD 가 걸린다.
      채권: [
        "BOND", "TREASURY", "T-BILL", "TBILL", " CLO ", "MUNICIPAL", " MUNI ", "CORPORATE",
        "HIGH YIELD", "FLOATING RATE", " GOVT ", "GOVERNMENT", "AGGREGATE", "FIXED INCOME",
        "CREDIT", "1-3M BOX", "MORTGAGE", " MBS ", "BANK LOAN", "SENIOR LOAN", "PREFERRED",
      ],
      "인버스/숏": ["INVERSE", "SHORT", "BEAR", "-0.5X", "-1X"],
      // 배수 표기가 제각각이라(2X·4X ETN·1.5X) 배수 키워드를 넓게 건다. "-2X" 류는 2X 에 걸린다.
      레버리지: ["1.5X", "2X", "3X", "4X", "5X", "LEVERAGED", "ULTRA", "레버리지"],
      // INCOME 은 옵션 인컴·채권형·배당형에 섞여 쓰이지만 전부 모멘텀 관점의 제외 대상이라 같이 건다.
      "커버드콜/인컴": ["COVERED CALL", "BUYWRITE", "INCOME", "커버드콜"],
      리츠: ["REAL ESTATE", " REIT ", "리츠"],
      // 가상자산 — 국내 증권사가 현물 ETF 중개를 못 해 거래 불가. 이름 키워드라 선물형(BITO)도
      // 같이 빠진다(현물/선물을 이름만으로 구분할 수 없음). 코인명은 신상품을 못 쫓아가므로
      // 가상자산 전문 운용사명(BITWISE 등)을 함께 건다 — BHYP("BITWISE HYPERLIQUID") 같은 케이스.
      가상자산: [
        "BITCOIN", "ETHEREUM", "ETHER", "SOLANA", "XRP", "CRYPTO", "STAKING",
        "BITWISE", "GRAYSCALE", "21SHARES", "HASHDEX", "CANARY", "OSPREY",
        "HYPERLIQUID", "DOGECOIN", "LITECOIN", "AVALANCHE", "CHAINLINK", "CARDANO", "DIGITAL ASSET",
        "비트코인", "이더리움",
      ],
    },
    // PTP — 원자재·선물형 상품(BWET·DBA·USO 등)이 여기 속한다. 짧은 이름에는 법적 구조가
    // 안 드러나 키워드로는 못 거른다(배치가 긴 이름으로 판정해 `is_ptp` 로 내려준다).
    flagExclusions: { PTP: "is_ptp" },
    defaultExcluded: ["채권", "인버스/숏", "레버리지", "커버드콜/인컴", "리츠", "가상자산", "PTP"],
    showNavColumns: false,
    showListing: false,
    capHeader: "거래대금($M)",
    capFilterDefault: "50",
    volumeFilterDefault: "500000",
  },
};

const marketGridTheme = createAppGridTheme();

function formatKrwEok(value: number): string {
  return new Intl.NumberFormat("ko-KR").format(value);
}

function formatCount(value: number): string {
  return new Intl.NumberFormat("ko-KR").format(value);
}

function formatNullableNumber(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 2 }).format(value);
}

function formatPercent(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${value.toFixed(2)}%`;
}

function getSignedMetricClass(value: number | null): string | undefined {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return undefined;
  }
  if (value > 0) {
    return "metricPositive";
  }
  if (value < 0) {
    return "metricNegative";
  }
  return undefined;
}

function getDeviationClass(value: number | null): string | undefined {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return undefined;
  }
  if (value >= 2) {
    return "metricPositive metricStrong";
  }
  if (value <= -2) {
    return "metricNegative metricStrong";
  }
  return undefined;
}

export function MarketManager({
  market = "kor",
  onHeaderSummaryChange,
}: {
  market?: MarketCode;
  onHeaderSummaryChange?: (summary: { filteredCount: number; totalCount: number; updatedAt: string | null }) => void;
}) {
  const variant = MARKET_VARIANTS[market];
  const toast = useToast();
  const [rows, setRows] = useState<MarketRowItem[]>([]);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [tickerPools, setTickerPools] = useState<MarketTickerPool[]>([]);
  const [query, setQuery] = useState("");
  const [minMarketCap, setMinMarketCap] = useState(variant.capFilterDefault); // 규모(시총/거래대금) 최소값
  const [minPrevVolume, setMinPrevVolume] = useState(variant.volumeFilterDefault); // 거래량(주)
  const [excludedGroups, setExcludedGroups] = useState<string[]>(variant.defaultExcluded);
  // 과세 구분 — all(모두) · taxed(과세) · free(비과세). 분류를 못 받은 종목은 '모두'에서만 보인다.
  const [taxFilter, setTaxFilter] = useState<"all" | "taxed" | "free">("all");
  const [newOnly, setNewOnly] = useState(false);
  const [newListingDays, setNewListingDays] = useState("14");
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [addModalOpen, setAddModalOpen] = useState(false);
  const [selectedTickerPool, setSelectedTickerPool] = useState("");
  const [selectedBucketId, setSelectedBucketId] = useState<number | "">("");
  const [adding, setAdding] = useState(false);
  const [addProgress, setAddProgress] = useState<PoolAddProgress | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const previousMarketFiltersRef = useRef<{
    minMarketCap: string;
    minPrevVolume: string;
    excludedGroups: string[];
  } | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [marketResponse, stocksPayload] = await Promise.all([
        fetch(variant.apiPath, { cache: "no-store" }),
        loadStocksTable().catch(
          () =>
            ({
              ticker_types: [],
              rows: [],
              ticker_type: "",
            }) as { ticker_types: MarketTickerPool[]; rows: unknown[]; ticker_type: string },
        ),
      ]);
      const payload = (await marketResponse.json()) as MarketResponse;
      if (!marketResponse.ok) {
        throw new Error(payload.error ?? variant.errorLabel);
      }
      setRows(payload.rows ?? []);
      setUpdatedAt(payload.updated_at ?? null);
      setTickerPools(stocksPayload.ticker_types ?? []);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : variant.errorLabel);
    } finally {
      setLoading(false);
    }
  }, [variant]);

  useEffect(() => {
    void load();
  }, [load]);

  const filteredRows = useMemo(() => {
    const normalizedQuery = query.trim().toUpperCase();
    const expandedKeywords = excludedGroups.flatMap((group) => variant.exclusionGroups[group] ?? []);
    // 이름으로 못 가르는 그룹(PTP) — 배치가 판정해 둔 불리언 필드를 본다.
    const excludedFlagFields = excludedGroups
      .map((group) => variant.flagExclusions?.[group])
      .filter((field): field is keyof MarketRowItem => Boolean(field));
    const marketCapFilter = Number(minMarketCap || 0);
    const volumeFilter = Number(minPrevVolume || 0);
    const today = new Date();
    const todayKst = new Date(today.toLocaleString("en-US", { timeZone: "Asia/Seoul" }));
    const cutoff = new Date(todayKst);
    const normalizedNewListingDays = Math.max(1, Number.parseInt(newListingDays || "14", 10) || 14);
    cutoff.setHours(0, 0, 0, 0);
    cutoff.setDate(cutoff.getDate() - (normalizedNewListingDays - 1));

    return rows
      .filter((row) => {
        if (
          normalizedQuery &&
          !row.ticker.toUpperCase().includes(normalizedQuery) &&
          !row.name.toUpperCase().includes(normalizedQuery)
        ) {
          return false;
        }

        if (!newOnly && excludedFlagFields.some((field) => row[field] === true)) {
          return false;
        }

        // 과세 구분 — 분류가 없는 종목(null)은 어느 쪽으로도 넘겨짚지 않고 '모두'에서만 보인다.
        if (taxFilter === "free" && row.is_tax_free !== true) return false;
        if (taxFilter === "taxed" && row.is_tax_free !== false) return false;

        // 양끝 공백 패딩 — 공백 포함 키워드(" CLO " 등)가 이름 첫/끝 단어에도 걸리게 한다.
        const nameUpper = ` ${row.name.toUpperCase()} `;
        if (!newOnly && expandedKeywords.some((keyword) => nameUpper.includes(keyword.toUpperCase()))) {
          return false;
        }

        if (!newOnly && marketCapFilter > 0 && row.market_cap < marketCapFilter) {
          return false;
        }

        if (!newOnly && volumeFilter > 0 && row.prev_volume < volumeFilter) {
          return false;
        }

        if (newOnly) {
          const listedAt = String(row.listed_at || "").trim();
          if (!listedAt) {
            return false;
          }
          const listedDate = new Date(`${listedAt}T00:00:00+09:00`);
          if (Number.isNaN(listedDate.getTime())) {
            return false;
          }
          if (listedDate < cutoff) {
            return false;
          }
        }

        return true;
      })
      .sort((left, right) => {
        const leftValue = left.daily_change_pct ?? Number.NEGATIVE_INFINITY;
        const rightValue = right.daily_change_pct ?? Number.NEGATIVE_INFINITY;
        if (leftValue !== rightValue) {
          return rightValue - leftValue;
        }
        return left.ticker.localeCompare(right.ticker);
      });
  }, [excludedGroups, minMarketCap, minPrevVolume, newListingDays, newOnly, query, rows, taxFilter]);

  const gridRows = useMemo<MarketGridRow[]>(
    () => filteredRows.map((row, index) => ({ ...row, row_number: index + 1 })),
    [filteredRows],
  );

  useEffect(() => {
    const visibleTickers = new Set(gridRows.map((row) => row.ticker));
    setSelectedTickers((current) => current.filter((ticker) => visibleTickers.has(ticker)));
  }, [gridRows]);

  useEffect(() => {
    onHeaderSummaryChange?.({
      filteredCount: filteredRows.length,
      totalCount: rows.length,
      updatedAt,
    });
  }, [filteredRows.length, onHeaderSummaryChange, rows.length, updatedAt]);

  const hasSelectedRows = selectedTickers.length > 0;
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
      if (visibleTickers.length === 0) {
        return current;
      }
      const allSelected = visibleTickers.every((ticker) => current.includes(ticker));
      if (allSelected) {
        return current.filter((ticker) => !visibleTickers.includes(ticker));
      }
      return [...new Set([...current, ...visibleTickers])];
    });
  }, [gridRows]);

  const handleOpenAddModal = useCallback(() => {
    if (!hasSelectedRows) {
      return;
    }
    setSelectedTickerPool(readRememberedTickerType() || "");
    setSelectedBucketId("");
    setAddModalOpen(true);
  }, [hasSelectedRows]);

  const handleCloseAddModal = useCallback(() => {
    if (adding) {
      return;
    }
    setAddModalOpen(false);
  }, [adding]);

  const handleAddSelected = useCallback(async () => {
    const tickerPool = String(selectedTickerPool || "").trim().toLowerCase();
    const bucketId = Number(selectedBucketId || 0);
    if (!tickerPool || !bucketId) {
      toast.error("종목풀과 버킷을 모두 선택하세요.");
      return;
    }

    // 이미 어딘가의 풀에 있는 종목은 보내지 않는다 — 개별주 화면들과 같은 공용 흐름(pool-add).
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
    const { added, skipped, blocked, failed } = await addTickersToPool(
      split.fresh,
      tickerPool,
      bucketId,
      setAddProgress,
      // 진행도에 종목명을 보여주기 위한 표의 이름 — 추가 조회 없이 넘긴다.
      new Map(rows.map((row) => [String(row.ticker).trim().toUpperCase(), row.name])),
    );

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
      await load();
    }
  }, [load, rows, selectedBucketId, selectedTickerPool, selectedTickers, toast]);

  const columns = useMemo<ColDef<MarketGridRow>[]>(
    () => [
      { field: "row_number", headerName: "#", width: 72, maxWidth: 80 },
      {
        field: "ticker_pools",
        headerName: "종목풀",
        width: 108,
        maxWidth: 116,
        cellRenderer: (params: { value: string }) => String(params.value ?? "").trim() || "-",
      },
      {
        field: "ticker",
        headerName: "티커",
        width: 104,
        cellRenderer: (params: { value: string }) => {
          const value = String(params.value ?? "-");
          return <TickerDetailLink ticker={value} />;
        },
      },
      {
        field: "name",
        headerName: "종목명",
        minWidth: 220,
        flex: 1,
        cellClass: "marketNameCell",
        cellRenderer: (params: { value: string | null | undefined }) => {
          const value = String(params.value ?? "-");
          return (
            <span className="marketNameMain" title={value}>
              {value}
            </span>
          );
        },
      },
      {
        field: "daily_change_pct",
        headerName: "일간(%)",
        width: 112,
        type: "rightAligned",
        sort: "desc",
        comparator: (a, b) => (a ?? Number.NEGATIVE_INFINITY) - (b ?? Number.NEGATIVE_INFINITY),
        cellRenderer: (params: { value: number | null }) => (
          <span className={getSignedMetricClass(params.value)}>{formatPercent(params.value)}</span>
        ),
      },
      {
        field: "current_price",
        headerName: "현재가",
        width: 110,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatNullableNumber(params.value),
      },
      ...(variant.showNavColumns
        ? ([
            {
              field: "nav",
              headerName: "Nav",
              width: 110,
              type: "rightAligned",
              cellRenderer: (params: { value: number | null }) => formatNullableNumber(params.value),
            },
            {
              field: "deviation",
              headerName: "괴리율",
              width: 96,
              type: "rightAligned",
              cellRenderer: (params: { value: number | null }) => (
                <span className={getDeviationClass(params.value)}>{formatPercent(params.value)}</span>
              ),
            },
          ] as ColDef<MarketGridRow>[])
        : []),
      {
        field: "return_1m_pct",
        headerName: "1달(%)",
        width: 96,
        type: "rightAligned",
        comparator: (a, b) => (a ?? Number.NEGATIVE_INFINITY) - (b ?? Number.NEGATIVE_INFINITY),
        cellRenderer: (params: { value: number | null }) => (
          <span className={getSignedMetricClass(params.value)}>{formatPercent(params.value)}</span>
        ),
      },
      {
        field: "return_2m_pct",
        headerName: "2달(%)",
        width: 96,
        type: "rightAligned",
        comparator: (a, b) => (a ?? Number.NEGATIVE_INFINITY) - (b ?? Number.NEGATIVE_INFINITY),
        cellRenderer: (params: { value: number | null }) => (
          <span className={getSignedMetricClass(params.value)}>{formatPercent(params.value)}</span>
        ),
      },
      {
        field: "return_3m_pct",
        headerName: "3달(%)",
        width: 96,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => (
          <span className={getSignedMetricClass(params.value)}>{formatPercent(params.value)}</span>
        ),
      },
      ...(variant.showListing ? ([{ field: "listed_at", headerName: "상장일", width: 112 }] as ColDef<MarketGridRow>[]) : []),
      {
        field: "prev_volume",
        headerName: "전일거래량(주)",
        width: 128,
        type: "rightAligned",
        cellRenderer: (params: { value: number }) => formatCount(params.value),
      },
      {
        field: "market_cap",
        headerName: variant.capHeader,
        width: 128,
        type: "rightAligned",
        cellRenderer: (params: { value: number }) => formatKrwEok(params.value),
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
        cellRenderer: (params: { data?: MarketGridRow }) => {
          const ticker = String(params.data?.ticker ?? "").trim();
          if (!ticker) {
            return null;
          }
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
    [allVisibleSelected, selectedTickers, toggleSelectAllVisible, toggleTickerSelection, variant],
  );

  function toggleGroup(group: string) {
    setExcludedGroups((current) =>
      current.includes(group) ? current.filter((item) => item !== group) : [...current, group],
    );
  }

  function toggleNewOnly() {
    setNewOnly((current) => {
      if (!current) {
        previousMarketFiltersRef.current = {
          minMarketCap,
          minPrevVolume,
          excludedGroups,
        };
        setMinMarketCap("");
        setMinPrevVolume("");
        setExcludedGroups([]);
        return true;
      }

      const previous = previousMarketFiltersRef.current;
      setMinMarketCap(previous?.minMarketCap ?? "500");
      setMinPrevVolume(previous?.minPrevVolume ?? "100000");
      setExcludedGroups(previous?.excludedGroups ?? variant.defaultExcluded);
      previousMarketFiltersRef.current = null;
      return false;
    });
  }

  function handleNewListingDaysChange(value: string) {
    if (value === "") {
      setNewListingDays("");
      return;
    }
    const parsedValue = Number.parseInt(value, 10);
    if (Number.isNaN(parsedValue)) {
      return;
    }
    setNewListingDays(String(Math.max(1, parsedValue)));
  }

  return (
    <div className="appPageStack appPageStackFill">
      {error ? (
        <div className="appBannerStack">
          <div className="bannerError">{error}</div>
        </div>
      ) : null}
      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <ResponsiveFiltersSection>
              <div className="appMainHeader marketMainHeader">
                <div className="appMainHeaderLeft marketMainHeaderLeft">
                  <label className="appLabeledField">
                    <span className="appLabeledFieldLabel">티커/종목명</span>
                    <input
                      className="field compactField"
                      type="text"
                      placeholder="티커 또는 종목명을 입력"
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                    />
                  </label>
                  <label className="appLabeledField">
                    <span className="appLabeledFieldLabel">{variant.capHeader}</span>
                    <input
                      className="field compactField"
                      type="number"
                      placeholder="최소값"
                      value={minMarketCap}
                      onChange={(event) => setMinMarketCap(event.target.value)}
                    />
                  </label>
                  <label className="appLabeledField">
                    <span className="appLabeledFieldLabel">거래량(주)</span>
                    <div className="marketVolumeInlineRow">
                      <input
                        className="field compactField marketVolumeInput"
                        type="number"
                        placeholder="최소 전일 거래량"
                        value={minPrevVolume}
                        onChange={(event) => setMinPrevVolume(event.target.value)}
                      />
                      {variant.showListing ? (
                      <div className="marketNewOnlyRow">
                        <button
                          type="button"
                          className={newOnly ? "filterPill filterPillActive" : "filterPill"}
                          onClick={toggleNewOnly}
                        >
                          신규
                        </button>
                        {newOnly ? (
                          <div className="marketNewOnlyDaysRow">
                            <input
                              className="field compactField marketNewOnlyDaysInput"
                              type="number"
                              min={1}
                              step={1}
                              value={newListingDays}
                              onChange={(event) => handleNewListingDaysChange(event.target.value)}
                              aria-label="신규 ETF 최근 일수"
                            />
                            <span className="marketNewOnlyDaysLabel">일</span>
                          </div>) : null}
                        {newOnly ? (
                          <span className="marketNewOnlyHint">최근 {Math.max(1, Number.parseInt(newListingDays || "14", 10) || 14)}일 상장 ETF</span>
                        ) : null}
                      </div>
                      ) : null}
                    </div>
                  </label>
                  {/* 과세 구분 — 국내 주식형만 매매차익 비과세다. 이름으로는 못 가르므로
                      (KODEX 레버리지는 국내 주식이지만 파생이라 과세) 배치가 받아 둔 분류를 쓴다. */}
                  {variant.showTaxFilter ? (
                    <label className="appLabeledField">
                      <span className="appLabeledFieldLabel">과세</span>
                      <div className="appSegmentedToggle appSegmentedToggleCompact" role="group" aria-label="과세 구분">
                        {TAX_FILTER_OPTIONS.map(({ key, label, title }) => (
                          <button
                            key={key}
                            type="button"
                            className={taxFilter === key ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                            onClick={() => setTaxFilter(key)}
                            title={title}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    </label>
                  ) : null}
                </div>
                <div className="appMainHeaderRight">
                  <button
                    type="button"
                    className="btn btn-success btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                    onClick={handleOpenAddModal}
                    disabled={!hasSelectedRows}
                  >
                    <IconPlus size={16} stroke={2} />
                    추가
                  </button>
                </div>
              </div>
            </ResponsiveFiltersSection>
          </div>
          <div className="card-body appCardBodyTight appTableCardBodyFill">
            <div className="pillRow">
              {[...Object.keys(variant.exclusionGroups), ...Object.keys(variant.flagExclusions ?? {})].map((group) => {
                const isActive = excludedGroups.includes(group);
                return (
                  <button
                    key={group}
                    type="button"
                    className={isActive ? "filterPill filterPillActive" : "filterPill"}
                    onClick={() => toggleGroup(group)}
                  >
                    {group}
                  </button>
                );
              })}
            </div>

            <div className="appGridFillWrap" style={{ minHeight: 0 }}>
              <AppAgGrid
                rowData={gridRows}
                columnDefs={columns}
                loading={loading}
                minHeight="100%"
                theme={marketGridTheme}
                getRowClass={(params: RowClassParams<MarketGridRow>) => (params.data?.is_held ? "appHeldRow" : "")}
                gridOptions={{
                  suppressMovableColumns: true,
                }}
              />
            </div>
          </div>
        </div>
      </section>

      <style jsx global>{`
        .marketNameCell {
          min-width: 0;
          overflow: hidden;
        }
      `}</style>

      <style jsx>{`
        .marketNameMain {
          display: block;
          width: 100%;
          min-width: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .marketVolumeInlineRow {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          flex-wrap: nowrap;
        }
        .marketVolumeInput {
          width: 140px;
          min-width: 140px;
          flex: 0 0 140px;
        }
        .marketNewOnlyRow {
          display: flex;
          align-items: center;
          gap: 0.625rem;
          flex-wrap: nowrap;
          white-space: nowrap;
        }
        .marketNewOnlyDaysRow {
          display: flex;
          align-items: center;
          gap: 0.375rem;
        }
        .marketNewOnlyDaysInput {
          width: 76px;
          min-width: 76px;
        }
        .marketNewOnlyDaysLabel {
          font-size: var(--fs-base);
          font-weight: 600;
          color: #5b6778;
          white-space: nowrap;
        }
        .marketNewOnlyHint {
          font-size: var(--fs-base);
          font-weight: 700;
          color: #206bc4;
          white-space: nowrap;
        }
        .appModalFormStack {
          display: grid;
          gap: 0.875rem;
        }
      `}</style>

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
        <div className="appModalFormStack">
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
              {tickerPools.map((pool) => (
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
    </div>
  );
}
