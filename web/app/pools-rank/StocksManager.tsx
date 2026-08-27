"use client";

import { IconArrowsExchange, IconDeviceFloppy, IconPlus, IconTrash } from "@tabler/icons-react";
import type { ColDef, RowClassParams } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useRef, useState, useTransition } from "react";

import { BUCKET_OPTIONS } from "@/lib/bucket-theme";
import { MaDaysSelect, type MaOptionsPayload } from "../components/MaDaysSelect";
import { formatPoolLabel } from "@/lib/pool-label";
import { poolHasIndustry, poolHasMarketCap } from "@/lib/pool-industry";
import {
  INDUSTRY_COLUMN_MIN_WIDTH,
  INDUSTRY_COLUMN_WIDTH,
  marketBadgeCellStyle,
  renderHighDrawdownCell,
  renderIndustryCell,
  tradeValueMultStyle,
  marketCapRankColumn,
  stockMemoColumn,
} from "@/lib/grid-cells";
import { isTrendBroken, renderStockNameCell } from "@/lib/name-highlight";
import { readSessionTtlCache, writeSessionTtlCache } from "@/lib/session-ttl-cache";
import type { PoolAddProgress } from "@/lib/pool-add";
import { PoolAddProgressBar } from "../components/PoolAddProgressBar";
import { addStockCandidate, deleteStock, loadMovablePools, moveStockToPool, updateStockBucket, updateStockMemo, validateStockCandidate, updateStockExclude, type StocksAccountItem } from "@/lib/stocks-store";
import {
  readRememberedTickerType,
  writeRememberedTickerType,
} from "../components/account-selection";
import { formatPrice } from "../../lib/price-format";
import { AppAgGrid } from "../components/AppAgGrid";
import { ResponsiveFiltersSection } from "../components/ResponsiveFiltersSection";
import { AppModal } from "../components/AppModal";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";

type RankTickerType = {
  ticker_type: string;
  order: number;
  name: string;
  icon: string;
  country_code: string;
  // 풀 성격(stock/etf) — 종목풀 설정의 '구분' 토글. 미설정이면 빈 값.
  pool_kind?: string;
  top_n_hold?: number;
  /** 업종 상한 — 종목풀 저장값. null/미설정 = 제한 없음. */
  max_per_industry?: number | null;
  currency?: string;
  include?: string[];
};

type RankMaRule = {
  short_ma_days: number;
  long_ma_days: number;
  score_column: string;
  ma_type: string;
};

type RankRow = {
  [key: string]: any;
  source_ticker_type?: string;
  종목풀?: string;
  currency?: string;
  backtest_stats?: {
    cagr: number;
    mdd: number;
    sortino: number;
    is_partial?: boolean;
  } | null;
  순번: string;
  순위: number | null;
  이전순위: number | null;
  "1주순위": number | null;
  버킷: string;
  bucket: number;
  티커: string;
  시총순위?: number | null;
  마켓?: string;
  종목명: string;
  /** 종목 메모 — 자산 관리 화면과 같은 값(종목에 붙는다). */
  메모?: string;
  상장일: string;
  추세: number | null;
  이격?: number | null;
  단기이격?: number | null;
  보유대상?: boolean;
  보유: string;
  현재가: number | null;
  /** 20일 평균 거래대금 대비 배수. 신고가 돌파 화면과 같은 값이다. */
  거래대금: number | null;
  "괴리율": number | null;
  "일간(%)": number | null;
  "1주(%)": number | null;
  "2주(%)": number | null;
  "3주(%)": number | null;
  "4주(%)": number | null;
  "1달(%)": number | null;
  "2달(%)": number | null;
  "3달(%)": number | null;
  "4달(%)": number | null;
  "5달(%)": number | null;
  "6달(%)": number | null;
  "7달(%)": number | null;
  "8달(%)": number | null;
  "9달(%)": number | null;
  "10달(%)": number | null;
  "11달(%)": number | null;
  "12달(%)": number | null;
  "24달(%)": number | null;
  "36달(%)": number | null;
  고점: number | null;
  RSI: number | null;
  배당률: number | null;
  보수: number | null;
  순자산총액: number | null;
  "전일 거래량(주)": number | null;
  exclude_from_ranking?: boolean;
  is_benchmark?: boolean;
  is_below_benchmark?: boolean;
};

type RankResponse = {
  ticker_types?: RankTickerType[];
  ticker_type?: string;
  ma_rules?: RankMaRule[];
  /** 이평선 일수 선택지 — 백엔드 상수(utils/ma_options)가 단일 소스. */
  short_ma_options?: number[];
  long_ma_options?: number[];
  as_of_date?: string | null;
  monthly_return_labels?: string[];
  rows?: RankRow[];
  cache_blocked?: boolean;
  latest_trading_day?: string | null;
  cache_updated_at?: string | null;
  ranking_computed_at?: string | null;
  realtime_fetched_at?: string | null;
  previous_trading_day?: string | null;
  missing_tickers?: string[];
  missing_ticker_labels?: string[];
  stale_tickers?: string[];
  error?: string;
};

type RankGridRow = RankRow & {
  id: string;
  __isAddingRow?: boolean;
};

type RankAddingRowState = {
  ticker: string;
  name: string;
  listing_date: string;
  bucket: number;
  status: "active" | "deleted" | "new" | null;
  is_validating: boolean;
  is_validated: boolean;
};

/** 업종 상한 선택지 — 백엔드 `config.MAX_PER_INDUSTRY_OPTIONS` 와 같은 목록.
 *  `-1` 이 '없음'(제한 없음)이다 — 쿼리로 넘길 때 '미지정' 과 구분하려고 숫자로 둔다. */
const MAX_PER_INDUSTRY_CHOICES: number[] = [1, 2, 3, -1];

const rankGridTheme = createAppGridTheme();
const RANK_SESSION_CACHE_TTL_MS = 60_000;
const RANK_SESSION_CACHE_PREFIX = "stocks:rank";
const DEFAULT_TICKER_TYPE = "";

/** 그리드에 어떤 컬럼 묶음을 보여줄지. 화면 전환용 `pageMode` 와는 다른 축이다. */
type MetricMode = "basic" | "ranking" | "monthly" | "info";

const METRIC_MODE_OPTIONS: { value: MetricMode; label: string }[] = [
  { value: "basic", label: "기본" },
  { value: "ranking", label: "랭킹" },
  { value: "monthly", label: "월별" },
  { value: "info", label: "정보" },
];

type RankToolbarCache = {
  ticker_types: RankTickerType[];
  ticker_type: string;
  ma_rule: RankMaRule | null;
  ma_options: Partial<MaOptionsPayload>;
};

type RankHeaderSummary = {
  upCount: number;
  upPct: number;
  totalCount: number;
  ruleSummary: string;
};

let rankToolbarCache: RankToolbarCache | null = null;



function getTodayDateInputValue(): string {
  return new Date().toLocaleDateString("en-CA", { timeZone: "Asia/Seoul" });
}

function toDateInputValue(value: string | null | undefined): string {
  if (!value) {
    return getTodayDateInputValue();
  }
  return String(value).slice(0, 10);
}

function formatNumber(value: number | null, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function formatPercent(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${value.toFixed(2)}%`;
}

/** 티커 목록을 짧게 — 수백 개를 그대로 나열하면 경고가 화면을 덮는다. */
function summarizeTickers(tickers: string[], limit = 5): string {
  if (tickers.length <= limit) return tickers.join(", ");
  return `${tickers.slice(0, limit).join(", ")} 외 ${tickers.length - limit}개`;
}

function getSignedClass(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) {
    return "";
  }
  return value > 0 ? "metricPositive" : "metricNegative";
}

function renderRankDelta(value: number | null | undefined) {
  const rankDelta = value ?? null;
  if (rankDelta === null || rankDelta === undefined || rankDelta === 0) {
    return <span style={{ fontWeight: 600, color: "#6c757d" }}>-</span>;
  }

  const isRise = rankDelta > 0;
  const delta = Math.abs(rankDelta);
  return (
    <span style={{ color: isRise ? "#d63939" : "#206bc4", fontWeight: 700 }}>
      {isRise ? `+${delta}` : `-${delta}`} {isRise ? "▲" : "▼"}
    </span>
  );
}

/** 종목명 강조 키워드 → 색상/이모지 (대소문자 무관, 볼드 공통).
 *  키워드를 추가하려면 여기에 한 줄만 더하면 된다. 겹치는 키워드는 긴 것을 앞에 둘 것 (예: UltraPro > Ultra).
 *  emoji 는 종목명 맨 뒤에 붙는다 (여러 키워드 매칭 시 중복 없이 순서대로). */
function getBucketCellClass(bucketLabel: string): string {
  const match = /^(\d+)/.exec(String(bucketLabel || "").trim());
  if (!match) {
    return "rankBucketCell";
  }
  return `rankBucketCell rankBucketCell${match[1]}`;
}

function normalizeTicker(value: string): string {
  return String(value || "").trim().toUpperCase();
}

function getBucketName(bucketId: number): string {
  return BUCKET_OPTIONS.find((option) => option.id === bucketId)?.name ?? BUCKET_OPTIONS[0]?.name ?? "-";
}

function getBucketIdByName(bucketName: string): number {
  return BUCKET_OPTIONS.find((option) => option.name === bucketName)?.id ?? BUCKET_OPTIONS[0]?.id ?? 1;
}

function buildDirtyCellKey(rowId: string, field: string): string {
  return `${rowId}::${field}`;
}

function formatMetaTime(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("ko-KR", {
    dateStyle: "short",
    timeStyle: "short",
  }).format(date);
}

function renderSignedPercentCell(value: number | null) {
  return <span className={getSignedClass(value ?? null)}>{formatPercent(value ?? null)}</span>;
}

function renderRsiCell(value: number | null) {
  return formatNumber(value, 1);
}

function formatCurrencyValue(value: number | null, countryCode?: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  const digits = countryCode === "au" ? 2 : 0;
  return formatNumber(value, digits);
}

function formatAssetInEok(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${formatNumber(value / 100_000_000, 0)}억`;
}

function formatUsdMarketCap(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  if (value >= 1_000_000_000_000) {
    return `${formatNumber(value / 1_000_000_000_000, 2)}조 달러`;
  }
  if (value >= 100_000_000) {
    return `${formatNumber(value / 100_000_000, 1)}억 달러`;
  }
  return `${formatNumber(value, 0)}달러`;
}

function formatAudMarketCap(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  if (value >= 100_000_000) {
    return `${formatNumber(value / 100_000_000, 1)}억 AUD`;
  }
  return `${formatNumber(value, 0)} AUD`;
}

function buildRankSessionCacheKey(query: string): string {
  return `${RANK_SESSION_CACHE_PREFIX}:${query || "default"}`;
}


export function StocksManager({ onHeaderSummaryChange }: { onHeaderSummaryChange?: (summary: RankHeaderSummary) => void }) {
  const toast = useToast();
  const lastBlockedToastRef = useRef<string | null>(null);
  const addingTickerDraftRef = useRef("");
  const loadSequenceRef = useRef(0);
  const toolbarFetchAbortRef = useRef<AbortController | null>(null);
  const rankFetchAbortRef = useRef<AbortController | null>(null);
  const [isPending, startTransition] = useTransition();
  const [pageMode, setPageMode] = useState<"rank" | "manage">("rank");
  const [ticker_types, setAccounts] = useState<RankTickerType[]>(rankToolbarCache?.ticker_types ?? []);
  const [selectedTickerType, setSelectedAccountId] = useState(
    rankToolbarCache?.ticker_type ?? DEFAULT_TICKER_TYPE,
  );
  const [maRule, setMaRule] = useState<RankMaRule | null>(rankToolbarCache?.ma_rule ?? null);
  // 업종 상한 — 화면에서 바꿔 보는 값(저장하지 않는다). null = 아직 종목풀 저장값 그대로.
  // `-1` 이 '제한 없음' 이다 — null(미지정)과 구분해야 저장값으로 되돌아가지 않는다.
  const [maxPerIndustry, setMaxPerIndustry] = useState<number | null>(null);

  // 백엔드가 내려주는 선택지를 쓴다 — 화면이 복사본을 들고 있으면 값이 추가될 때 여기만 옛 목록이 남는다.
  const [maOptions, setMaOptions] = useState<Partial<MaOptionsPayload>>(rankToolbarCache?.ma_options ?? {});
  const [metricMode, setMetricMode] = useState<MetricMode>("basic");
  const [monthlyReturnLabels, setMonthlyReturnLabels] = useState<string[]>([]);
  const [selectedAsOfDate, setSelectedAsOfDate] = useState<string>(getTodayDateInputValue());
  const [rows, setRows] = useState<RankRow[]>([]);
  const [cacheBlocked, setCacheBlocked] = useState(false);
  const [rankingComputedAt, setRankingComputedAt] = useState<string | null>(null);
  const [realtimeFetchedAt, setRealtimeFetchedAt] = useState<string | null>(null);
  const [missingTickers, setMissingTickers] = useState<string[]>([]);
  const [missingTickerLabels, setMissingTickerLabels] = useState<string[]>([]);
  const [staleTickers, setStaleTickers] = useState<string[]>([]);
  const [addingRow, setAddingRow] = useState<RankAddingRowState | null>(null);
  const [dirtyRowIds, setDirtyRowIds] = useState<string[]>([]);
  const [dirtyCellKeys, setDirtyCellKeys] = useState<string[]>([]);
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  // 종목 이동 — 지금 보고 있는 풀에서 같은 국가·구분의 다른 풀로 옮긴다.
  const [moveOpen, setMoveOpen] = useState(false);
  const [movablePools, setMovablePools] = useState<StocksAccountItem[] | null>(null);
  const [moveTarget, setMoveTarget] = useState("");
  const [moveProgress, setMoveProgress] = useState<PoolAddProgress | null>(null);

  // gridRows의 id 계산과 동일한 방식으로 row id를 생성한다.
  const getRowId = useCallback(
    (row: { 티커?: unknown; source_ticker_type?: unknown }) =>
      `${String(row.source_ticker_type ?? selectedTickerType).trim().toLowerCase()}:${normalizeTicker(String(row.티커 ?? ""))}`,
    [selectedTickerType],
  );
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const todayDateInputValue = useMemo(() => getTodayDateInputValue(), []);
  function clearCacheWarningState() {
    setCacheBlocked(false);
    setMissingTickers([]);
    setMissingTickerLabels([]);
    setStaleTickers([]);
  }

  function applyRankPayload(payload: RankResponse) {
    setAccounts(payload.ticker_types ?? []);
    const nextAccountId = payload.ticker_type ?? "";
    setSelectedAccountId(nextAccountId);
    writeRememberedTickerType(nextAccountId);
    setMaRule(payload.ma_rules?.[0] ?? null);
    const nextMaOptions = { short_ma_options: payload.short_ma_options, long_ma_options: payload.long_ma_options };
    setMaOptions(nextMaOptions);
    setSelectedAsOfDate(toDateInputValue(payload.as_of_date));
    setMonthlyReturnLabels(payload.monthly_return_labels ?? []);
    rankToolbarCache = {
      ticker_types: payload.ticker_types ?? [],
      ticker_type: nextAccountId,
      ma_rule: payload.ma_rules?.[0] ?? null,
      ma_options: nextMaOptions,
    };
    setAddingRow(null);
    addingTickerDraftRef.current = "";
    setDirtyRowIds([]);
    setDirtyCellKeys([]);
    setSelectedTickers([]);
    setDeleteConfirmOpen(false);
    setRows(payload.rows ?? []);
    setCacheBlocked(Boolean(payload.cache_blocked));

    setRankingComputedAt(payload.ranking_computed_at ?? null);
    setRealtimeFetchedAt(payload.realtime_fetched_at ?? null);
    setMissingTickers(payload.missing_tickers ?? []);
    setMissingTickerLabels(payload.missing_ticker_labels ?? []);
    setStaleTickers(payload.stale_tickers ?? []);
  }

  function applyRankToolbarPayload(payload: RankResponse) {
    const nextTickerTypes = payload.ticker_types ?? [];
    const nextAccountId = payload.ticker_type ?? readRememberedTickerType() ?? "";
    setAccounts(nextTickerTypes);
    setSelectedAccountId(nextAccountId);
    if (nextAccountId) {
      writeRememberedTickerType(nextAccountId);
    }
    setMaRule(payload.ma_rules?.[0] ?? null);
    const nextMaOptions = { short_ma_options: payload.short_ma_options, long_ma_options: payload.long_ma_options };
    setMaOptions(nextMaOptions);

    rankToolbarCache = {
      ticker_types: nextTickerTypes,
      ticker_type: nextAccountId,
      ma_rule: payload.ma_rules?.[0] ?? null,
      ma_options: nextMaOptions,
    };
  }

  async function loadToolbar() {
    if (rankToolbarCache) {
      return;
    }

    toolbarFetchAbortRef.current?.abort();
    const abortController = new AbortController();
    toolbarFetchAbortRef.current = abortController;

    try {
      const search = new URLSearchParams();
      const rememberedTickerType = readRememberedTickerType();
      if (rememberedTickerType) {
        search.set("ticker_type", rememberedTickerType);
      }
      const query = search.size > 0 ? `?${search.toString()}` : "";
      const response = await fetch(`/api/rank-toolbar${query}`, {
        cache: "no-store",
        signal: abortController.signal,
      });
      const payload = (await response.json()) as RankResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "종목풀 정보를 불러오지 못했습니다.");
      }
      applyRankToolbarPayload(payload);
    } catch (loadError) {
      if (loadError instanceof DOMException && loadError.name === "AbortError") {
        return;
      }
    } finally {
      if (toolbarFetchAbortRef.current === abortController) {
        toolbarFetchAbortRef.current = null;
      }
    }
  }

  async function load(next?: {
    ticker_type?: string;
    ma_rule_override?: RankMaRule;
    as_of_date?: string;
    /** 업종 상한 — 화면에서 바꿔 본 값. -1 = 제한 없음. */
    max_per_industry?: number;
    bootstrap?: boolean;
    skip_session_cache?: boolean;
  }) {
    const requestSequence = ++loadSequenceRef.current;
    setLoading(true);
    setError(null);
    clearCacheWarningState();
    let showedCachedPayload = false;

    try {
      const search = new URLSearchParams();
      if (next?.ticker_type) {
        search.set("ticker_type", next.ticker_type);
      }
      if (next?.as_of_date) {
        search.set("as_of_date", next.as_of_date);
      }
      if (next?.ma_rule_override) {
        search.set("short_ma_days", String(next.ma_rule_override.short_ma_days));
        search.set("long_ma_days", String(next.ma_rule_override.long_ma_days));
      }
      if (next?.max_per_industry != null) {
        search.set("max_per_industry", String(next.max_per_industry));
      }

      const query = search.size > 0 ? `?${search.toString()}` : "";
      const sessionCacheKey = buildRankSessionCacheKey(query);
      if (!next?.skip_session_cache) {
        const cachedPayload = readSessionTtlCache<RankResponse>(sessionCacheKey, RANK_SESSION_CACHE_TTL_MS);
        if (cachedPayload && requestSequence === loadSequenceRef.current) {
          applyRankPayload(cachedPayload);
          setLoading(false);
          showedCachedPayload = true;
        }
      }

      rankFetchAbortRef.current?.abort();
      const abortController = new AbortController();
      rankFetchAbortRef.current = abortController;

      const response = await fetch(`/api/rank${query}`, {
        cache: "no-store",
        signal: abortController.signal,
      });
      const payload = (await response.json()) as RankResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "순위 데이터를 불러오지 못했습니다.");
      }
      if (requestSequence !== loadSequenceRef.current) {
        return;
      }

      writeSessionTtlCache(sessionCacheKey, payload);
      applyRankPayload(payload);
    } catch (loadError) {
      if (loadError instanceof DOMException && loadError.name === "AbortError") {
        return;
      }
      if (requestSequence !== loadSequenceRef.current) {
        return;
      }
      if (showedCachedPayload) {
        return;
      }
      let msg = loadError instanceof Error ? loadError.message : "순위 데이터를 불러오지 못했습니다.";
      if (msg.includes("Unexpected token") || msg.includes("fetch failed") || msg === "순위 데이터를 불러오지 못했습니다.") {
        msg = "몽고디비 데이터베이스 응답 지연(타임아웃)으로 인해 순위 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.";
      }
      setError(msg);
      if (msg.includes("몽고디비 데이터베이스")) {
        if (typeof window !== "undefined") {
          window.dispatchEvent(new Event("db_error_occurred"));
        }
      }
    } finally {
      if (requestSequence === loadSequenceRef.current) {
        rankFetchAbortRef.current = null;
      }
      if (requestSequence === loadSequenceRef.current) {
        setLoading(false);
      }
    }
  }

  useEffect(() => {
    void loadToolbar();
    void load({
      ticker_type: readRememberedTickerType() ?? undefined,
      as_of_date: getTodayDateInputValue(),
      bootstrap: true,
    });
  }, []);

  const selectedTickerTypeItem = useMemo(
    () => ticker_types.find((account) => account.ticker_type === selectedTickerType) ?? null,
    [ticker_types, selectedTickerType],
  );
  const isAllTickerType = selectedTickerType === "all";

  useEffect(() => {
    if (isAllTickerType && pageMode === "manage") {
      setPageMode("rank");
    }
  }, [isAllTickerType, pageMode]);

  const gridRows = useMemo<RankGridRow[]>(
    () =>
      rows.map((row, index) => ({
        ...row,
        id: `${String(row.source_ticker_type ?? selectedTickerType).trim().toLowerCase()}:${normalizeTicker(String(row.티커 ?? `${index}`))}`,
      })),
    [rows, selectedTickerType],
  );

  const showDeviationColumn = useMemo(() => {
    const tickerType = String(selectedTickerTypeItem?.ticker_type || "").trim().toLowerCase();
    return tickerType === "kor_kr" || tickerType === "kor_us";
  }, [selectedTickerTypeItem?.ticker_type]);

  const displayGridRows = useMemo<RankGridRow[]>(() => {
    const rows = gridRows;
    if (pageMode !== "manage" || !addingRow) {
      return rows;
    }
    return [
      {
        id: "adding-row",
        __isAddingRow: true,
        순번: "-",
        순위: null,
        이전순위: null,
        "1주순위": null,
        버킷: getBucketName(addingRow.bucket),
        bucket: addingRow.bucket,
        티커: addingRow.ticker,
        종목명: addingRow.name,
        상장일: addingRow.listing_date || "-",
        추세: null,
        이격: null,
        단기이격: null,
        보유: "",
        보유대상: false,
        현재가: null,
        거래대금: null,
        괴리율: null,
        "일간(%)": null,
        "1주(%)": null,
        "2주(%)": null,
        "3주(%)": null,
        "4주(%)": null,
        "1달(%)": null,
        "2달(%)": null,
        "3달(%)": null,
        "4달(%)": null,
        "5달(%)": null,
        "6달(%)": null,
        "7달(%)": null,
        "8달(%)": null,
        "9달(%)": null,
        "10달(%)": null,
        "11달(%)": null,
        "12달(%)": null,
        "24달(%)": null,
        "36달(%)": null,
        고점: null,
        RSI: null,
        배당률: null,
        보수: null,
        순자산총액: null,
        "전일 거래량(주)": null,
        exclude_from_ranking: false,
      },
      ...rows,
    ];
  }, [addingRow, gridRows, pageMode]);

  const maRuleSummary = useMemo(
    () => (maRule ? [`${maRule.ma_type} 단기 ${maRule.short_ma_days}일 · 장기 ${maRule.long_ma_days}일`] : []),
    [maRule],
  );


  // 업종 컬럼 노출 여부 — 종목풀 설정의 풀 성격(pool_kind) 토글이 1순위
  // (개별주=표시, ETF=숨김), 미설정 풀은 행 값 유무로 추정 (strategy-momentum 과 같은 기준).
  // 업종 컬럼·업종 상한 노출 — 판정은 전 화면 공용(`@/lib/pool-industry`).
  const hasIndustryData = poolHasIndustry(selectedTickerTypeItem);
  // 화면에서 바꾼 값이 있으면 그것, 없으면 종목풀 저장값. 저장값이 없으면 '없음'(-1).
  const effectiveMaxPerIndustry =
    maxPerIndustry ?? (selectedTickerTypeItem?.max_per_industry == null ? -1 : selectedTickerTypeItem.max_per_industry);
  const hasMarketCap = poolHasMarketCap(selectedTickerTypeItem);

  const columns = useMemo<ColDef<RankGridRow>[]>(() => {
    const leadingColumns: ColDef<RankGridRow>[] = [
      {
        field: "순위",
        headerName: "순위",
        minWidth: 86,
        width: 86,
        cellStyle: { justifyContent: "center", textAlign: "center", overflow: "hidden", paddingLeft: 2, paddingRight: 2 },
        cellRenderer: (params: { data?: RankGridRow; value: number | null | undefined }) => {
          if (params.data?.is_benchmark) {
            return (
              <span style={{ fontSize: "var(--fs-base)", display: "inline-flex", alignItems: "center", gap: "4px", fontWeight: 700 }} title="벤치마크 종목">
                ⭐ {params.value == null ? "-" : formatNumber(params.value, 0)}
              </span>
            );
          }
          if (pageMode === "rank" && params.data?.exclude_from_ranking) {
            return (
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "4px",
                  borderRadius: "999px",
                  border: "1px solid #c7d2fe",
                  background: "#eef2ff",
                  color: "#4338ca",
                  fontSize: "var(--fs-sm)",
                  fontWeight: 700,
                  lineHeight: 1,
                  padding: "0.15rem 0.4rem",
                  whiteSpace: "nowrap",
                  maxWidth: "100%",
                }}
              >
                📌제외 {params.value == null ? "-" : formatNumber(params.value, 0)}
              </span>
            );
          }
          return (
            <span style={{ fontWeight: 700 }}>{params.value == null ? "-" : formatNumber(params.value, 0)}</span>
          );
        },
      },
      {
        field: "이전순위",
        headerName: "이전",
        minWidth: 58,
        width: 58,
        cellStyle: { justifyContent: "center", textAlign: "center" },
        sortable: true,
        cellRenderer: (params: { data?: RankGridRow; value: number | null | undefined }) => {
          const currentRank = params.data?.순위 ?? null;
          const previousRank = params.value ?? null;
          if (currentRank === null || currentRank === undefined || previousRank === null || previousRank === undefined) {
            return <span style={{ fontWeight: 600 }}>-</span>;
          }

          if (currentRank === previousRank) {
            return <span style={{ fontWeight: 600 }}>{previousRank}</span>;
          }

          const isRise = currentRank < previousRank;
          return (
            <span style={{ color: isRise ? "#d63939" : "#206bc4", fontWeight: 700 }}>
              {previousRank}
            </span>
          );
        },
      },
      {
        colId: "1주순위변동",
        headerName: "1주",
        minWidth: 66,
        width: 66,
        cellStyle: { textAlign: "center" },
        sortable: true,
        valueGetter: (params) => {
          const currentRank = params.data?.순위 ?? null;
          const weeklyRank = params.data?.["1주순위"] ?? null;
          if (currentRank === null || currentRank === undefined || weeklyRank === null || weeklyRank === undefined) {
            return null;
          }
          return weeklyRank - currentRank;
        },
        cellRenderer: (params: { value: number | null | undefined }) => {
          return renderRankDelta(params.value);
        },
      },
      {
        colId: "추천",
        headerName: "✓",
        headerTooltip: "추천 — 제외 종목·벤치마크가 아니고, 장기가 양수이며, 단기가 음수가 아닌 종목 중 장기 상위 N개(보유 종목수)",
        minWidth: 44,
        width: 44,
        sortable: true,
        filter: false,
        cellStyle: { textAlign: "center" },
        valueGetter: (params) => (params.data?.보유대상 ? 1 : 0),
        cellRenderer: (params: { data?: RankGridRow; value: number | null | undefined }) => {
          if (!params.value) return <span style={{ color: "var(--text-muted)" }}>-</span>;
          return <span style={{ fontSize: "var(--fs-base)" }}>✅</span>;
        },
      },
      {
        field: "고점",
        headerName: "고점",
        minWidth: 80,
        width: 80,
        type: "rightAligned",
        // 공용 고점 렌더러 (strategy-momentum 과 동일) — 0 이면 ⭐신고점.
        cellRenderer: (params: { data?: RankGridRow; value: number | null | undefined }) =>
          renderHighDrawdownCell(params.value, 2),
      },
      {
        field: "버킷",
        headerName: "버킷",
        minWidth: 108,
        width: 108,
        sortable: true,
        comparator: (_a, _b, nodeA, nodeB) => {
          const aId = Number(nodeA.data?.bucket ?? 0);
          const bId = Number(nodeB.data?.bucket ?? 0);
          return aId - bId;
        },
        cellClass: (params) => {
          const dirtyClass =
            params.data && dirtyCellKeys.includes(buildDirtyCellKey(params.data.id, "버킷")) ? " rankDirtyCell" : "";
          return `${getBucketCellClass(String(params.value ?? ""))}${dirtyClass}`;
        },
        editable: (params) => pageMode === "manage" && !params.data?.__isAddingRow,
        cellEditor: "agSelectCellEditor",
        cellEditorParams: {
          values: BUCKET_OPTIONS.map((option) => option.name),
        },
        valueGetter: (params) => getBucketName(Number(params.data?.bucket ?? 1)),
        valueSetter: (params) => {
          if (!params.data || params.data.__isAddingRow) {
            return false;
          }
          const nextBucketId = getBucketIdByName(String(params.newValue ?? ""));
          params.data.bucket = nextBucketId;
          params.data.버킷 = getBucketName(nextBucketId);
          return true;
        },
        cellRenderer: (params: { data?: RankGridRow }) => {
          if (params.data?.__isAddingRow) {
            return (
              <select
                className="form-select form-select-sm"
                value={addingRow?.bucket ?? 1}
                onChange={(event) =>
                  setAddingRow((prev) =>
                    prev
                      ? {
                        ...prev,
                        bucket: Number(event.target.value),
                      }
                      : null,
                  )
                }
              >
                {BUCKET_OPTIONS.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.name}
                  </option>
                ))}
              </select>
            );
          }
          return <span>{getBucketName(Number(params.data?.bucket ?? 1))}</span>;
        },
      },
      ...(pageMode === "manage"
        ? [
          {
            field: "exclude_from_ranking",
            headerName: "제외 종목",
            minWidth: 84,
            width: 84,
            cellStyle: { textAlign: "center" },
            cellRenderer: (params: { data?: RankGridRow; value: boolean | null | undefined }) => {
              if (params.data?.__isAddingRow) return null;
              const ticker = params.data?.티커;
              if (!ticker) return null;
              return (
                <div className="form-check form-switch d-flex justify-content-center align-items-center h-100 mb-0">
                  <input
                    type="checkbox"
                    className="form-check-input"
                    checked={!!params.value}
                    style={{ cursor: "pointer", marginTop: 0 }}
                    onChange={(e) => {
                      const checked = e.target.checked;
                      startTransition(async () => {
                        try {
                          await updateStockExclude(selectedTickerType, ticker, checked);
                          toast.success(`[${ticker}] 제외 종목 ${checked ? "설정" : "해제"} 완료`);
                          void load({
                            ticker_type: selectedTickerType,
                            ma_rule_override: maRule ?? undefined,
                            as_of_date: selectedAsOfDate,
                            skip_session_cache: true,
                          });
                        } catch (error) {
                          showErrorToast(error instanceof Error ? error.message : "제외 종목 설정에 실패했습니다.");
                        }
                      });
                    }}
                  />
                </div>
              );
            },
          } as ColDef<RankGridRow>,
        ]
        : []),
      ...(isAllTickerType
        ? [
          {
            field: "종목풀",
            headerName: "종목풀",
            minWidth: 125,
            width: 125,
            cellClass: "appTextEllipsisCell",
            cellRenderer: (params: { value: string | null | undefined; data?: RankGridRow }) => {
              const value = String(params.value ?? "").trim();
              if (!value) {
                return <span>-</span>;
              }
              // 1순위: source_ticker_type 식별자 매칭, 2순위: name 매칭
              const sourceTickerType = String(params.data?.source_ticker_type ?? "").trim();
              const matched =
                (sourceTickerType && ticker_types.find((tt) => tt.ticker_type === sourceTickerType)) ||
                ticker_types.find((tt) => tt.name === value);
              const icon = matched?.icon ?? "";
              const display = icon ? `${icon} ${value}` : value;
              return <span title={display}>{display}</span>;
            },
          } as ColDef<RankGridRow>,
        ]
        : []),
      ...(selectedTickerType === "kor"
        ? [
          {
            field: "마켓",
            headerName: "마켓",
            minWidth: 80,
            width: 80,
            // 공용 마켓 배지 스타일 (strategy-momentum 과 동일).
            cellStyle: (params) => marketBadgeCellStyle(params.value),
          } as ColDef<RankGridRow>,
        ]
        : []),
      // 시총은 개별주에만 있는 값이라 업종과 판정이 다르다(`@/lib/pool-industry`).
      marketCapRankColumn<RankGridRow>("시총순위", !hasMarketCap),
      {
        field: "티커",
        headerName: "티커",
        minWidth: 95,
        width: 95,
        cellRenderer: (params: { value: string | null | undefined; data?: RankGridRow }) => {
          if (params.data?.__isAddingRow) {
            return (
              <div className="stocksTickerLookup">
                <input
                  type="text"
                  className="form-control form-control-sm"
                  defaultValue={addingTickerDraftRef.current}
                  autoFocus
                  onChange={(event) => {
                    addingTickerDraftRef.current = event.target.value;
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      void handleValidateAddingTicker(event.currentTarget.value);
                    }
                  }}
                />
              </div>
            );
          }
          const value = String(params.value ?? "-");
          // 호주 계좌는 ASX: 접두사로 표시하여 미국 동일 심볼과 구분
          const rowCountryCode = String(params.data?.country_code || selectedTickerTypeItem?.country_code || "").toLowerCase();
          const isAusPool = rowCountryCode === "au";
          const displayValue = isAusPool && value !== "-" && !value.startsWith("ASX:")
            ? `ASX:${value}`
            : value;
          return <TickerDetailLink ticker={displayValue} displayTicker={displayValue} />;
        },
      },
      {
        field: "종목명",
        headerName: "종목명",
        minWidth: 249,
        flex: 1.05,
        cellRenderer: (params: { value: string | null | undefined; data?: RankGridRow }) => {
          if (params.data?.__isAddingRow) {
            const draftTicker = normalizeTicker(addingTickerDraftRef.current);
            const validatedTicker = normalizeTicker(addingRow?.ticker ?? "");
            const isDraftDirty = Boolean(draftTicker) && draftTicker !== validatedTicker;
            if (addingRow?.is_validating) {
              return <span className="text-muted">티커 확인 중...</span>;
            }
            if (!isDraftDirty && addingRow?.status === "active") {
              return <span className="text-danger fw-bold">이미 등록된 종목입니다.</span>;
            }
            if (!isDraftDirty && addingRow?.is_validated) {
              return (
                <span className="appNameCellText fw-semibold" title={addingRow.name}>
                  {addingRow.name}
                </span>
              );
            }
            return (
              <div className="rankAddingNameCell">
                <span className="text-muted">티커 확인 후 종목명이 표시됩니다.</span>
                <button
                  className="btn btn-outline-primary btn-sm"
                  type="button"
                  onClick={() => void handleValidateAddingTicker(addingTickerDraftRef.current)}
                  disabled={addingRow?.is_validating}
                >
                  확인
                </button>
              </div>
            );
          }
          return renderStockNameCell(params.value, {
            // MDD·소르티노 노란색과 같은 기준 — 상장 기간이 백테스트 기준 창(METRIC_WINDOW_MONTHS)보다 짧은 종목.
            isNew: params.data?.backtest_stats?.is_partial === true,
            trendBroken: isTrendBroken(params.data?.단기이격, params.data?.이격),
          });
        },
      },
      // 종목 메모 — 전 화면 공용 컬럼(`@/lib/grid-cells`). 순위와 무관한 수기 칸이라
      // 모드와 상관없이 바로 고칠 수 있다(셀을 벗어나면 저장).
      stockMemoColumn<RankGridRow>({
        field: "메모",
        onSave: (row, memo) => void handleMemoChange(String(row.티커 ?? ""), memo),
        editable: (row) => !row?.__isAddingRow,
      }),
      {
        field: "업종",
        headerName: "업종",
        hide: !hasIndustryData,
        minWidth: INDUSTRY_COLUMN_MIN_WIDTH,
        width: INDUSTRY_COLUMN_WIDTH,
        headerTooltip: "한국은 네이버 분류, 미국은 지수 구성종목의 yfinance 분류",
        cellRenderer: (params: { value?: string }) => renderIndustryCell(params.value),
      },
      {
        field: "일간(%)",
        headerName: "일간(%)",
        hide: metricMode !== "basic",
        minWidth: 86,
        width: 86,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => renderSignedPercentCell(params.value ?? null),
      },
      {
        field: "현재가",
        headerName: "현재가",
        hide: metricMode !== "basic",
        minWidth: 88,
        width: 88,
        type: "rightAligned",
        cellRenderer: (params: { data?: RankGridRow; value: number | null | undefined }) => {
          const rowCurrency = params.data?.currency || selectedTickerTypeItem?.currency;
          return formatPrice(params.value ?? null, rowCurrency);
        },
      },
      {
        field: "거래대금",
        headerName: "거래대금",
        hide: metricMode !== "basic",
        minWidth: 88,
        width: 88,
        type: "rightAligned",
        headerTooltip:
          "20일 평균 거래대금 대비 배수 — 신고가 돌파 화면과 같은 값. 국내 상장은 장중 실시간, 그 외는 직전 마감 기준.",
        valueFormatter: (params) => (params.value == null ? "-" : `${(params.value as number).toFixed(1)}배`),
        cellStyle: (params) => tradeValueMultStyle(params.value as number | null | undefined),
      },
    ];

    // 순위 산정에 직접 쓰이는 지표들. 종목을 고르는 눈으로 볼 때만 필요하다.
    const rankingColumns: ColDef<RankGridRow>[] = [
      {
        // 순위 점수 — 표를 줄 세우는 값 자체(core/strategy/scoring.rank_score).
        // 단기·장기 왼쪽에 둬서 "이 둘로 이 점수가 나왔다" 가 읽히게 한다.
        field: "점수",
        headerName: "점수",
        minWidth: 86,
        width: 86,
        type: "rightAligned",
        headerTooltip: "순위 점수 — 단기·장기 이격률의 평균. 이 값 내림차순이 표의 순위다.",
        cellRenderer: (params: { value: number | null | undefined }) => renderSignedPercentCell(params.value ?? null),
      },
      {
        field: "단기이격",
        headerName: "단기",
        minWidth: 86,
        width: 86,
        type: "rightAligned",
        headerTooltip: "종가와 단기 이평선의 이격률. 음수면 단기 추세가 꺾인 것으로 보고 보유하지 않는다.",
        cellRenderer: (params: { value: number | null | undefined }) => renderSignedPercentCell(params.value ?? null),
      },
      {
        field: "이격",
        headerName: "장기",
        minWidth: 86,
        width: 86,
        type: "rightAligned",
        headerTooltip: "종가와 장기 이평선의 이격률. 단기와 함께 순위 점수를 이룬다.",
        cellRenderer: (params: { value: number | null | undefined }) => renderSignedPercentCell(params.value ?? null),
      },
      {
        field: "RSI",
        headerName: "RSI",
        minWidth: 68,
        width: 68,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) =>
          renderRsiCell(params.value ?? null),
      },
      {
        headerName: "MDD",
        minWidth: 80,
        width: 80,
        type: "rightAligned",
        valueGetter: (params) => {
          const stats = params.data?.backtest_stats;
          const val = stats?.mdd;
          return val != null ? val : null;
        },
        cellRenderer: (params: { data?: RankGridRow; value: number | null | undefined }) => {
          if (params.value == null) return "-";
          return `${params.value.toFixed(2)}%`;
        },
        cellStyle: (params: { data?: RankGridRow }) => {
          const stats = params.data?.backtest_stats;
          if (stats?.is_partial) {
            return { color: "#ca8a04", fontWeight: 700 };
          }
          return null;
        },
      },
      {
        headerName: "소르티노",
        minWidth: 80,
        width: 80,
        type: "rightAligned",
        valueGetter: (params) => {
          const stats = params.data?.backtest_stats;
          const val = stats?.sortino;
          return val != null ? val : null;
        },
        cellRenderer: (params: { data?: RankGridRow; value: number | null | undefined }) => {
          if (params.value == null) return "-";
          return params.value.toFixed(2);
        },
        cellStyle: (params: { data?: RankGridRow }) => {
          const stats = params.data?.backtest_stats;
          if (stats?.is_partial) {
            return { color: "#ca8a04", fontWeight: 700 };
          }
          return null;
        },
      },
    ];

    // 가격과 기간별 수익률. 종목의 성적을 훑어볼 때 보는 기본 화면이다.
    const basicColumns: ColDef<RankGridRow>[] = [
      ...(showDeviationColumn
        ? [
          {
            field: "괴리율",
            headerName: "괴리율",
            minWidth: 78,
            width: 78,
            type: "rightAligned",
            cellRenderer: (params: { value: number | null | undefined }) => {
              const val = params.value ?? 0;
              const isExtreme = val > 2.0 || val < -2.0;
              return (
                <span style={{ color: isExtreme ? "#d63939" : "inherit", fontWeight: isExtreme ? 700 : 400 }}>
                  {formatPercent(params.value ?? null)}
                </span>
              );
            },
          } as ColDef<RankGridRow>,
        ]
        : []),
      {
        field: "1주(%)",
        headerName: "1주",
        minWidth: 88,
        width: 88,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => renderSignedPercentCell(params.value ?? null),
      },
      {
        field: "2주(%)",
        headerName: "2주",
        minWidth: 88,
        width: 88,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => renderSignedPercentCell(params.value ?? null),
      },
      ...[
        { field: "1달(%)", headerName: "1달" },
        { field: "3달(%)", headerName: "3달" },
        { field: "6달(%)", headerName: "6달" },
        { field: "12달(%)", headerName: "1년" },
        { field: "24달(%)", headerName: "2년" },
        { field: "36달(%)", headerName: "3년" },
      ].map(
        ({ field, headerName }) =>
          ({
            field,
            headerName,
            minWidth: headerName.length > 4 ? 94 : 78,
            width: headerName.length > 4 ? 94 : 78,
            type: "rightAligned",
            cellRenderer: (params: { value: number | null | undefined }) => renderSignedPercentCell(params.value ?? null),
          }) as ColDef<RankGridRow>,
      ),
    ];

    const monthlyColumns: ColDef<RankGridRow>[] = monthlyReturnLabels.map(
      (label) =>
        ({
          field: label,
          // 데이터 키는 "YYYY-MM(%)" 그대로 두고 헤더만 (%) 없이 표시한다.
          headerName: label.replace("(%)", ""),
          // 헤더가 "YYYY-MM" 7자로 짧아져 폭도 그에 맞춰 줄인다 (84는 헤더가 잘렸다).
          minWidth: 92,
          width: 92,
          type: "rightAligned",
          cellRenderer: (params: { value: number | null | undefined }) => renderSignedPercentCell(params.value ?? null),
        }) as ColDef<RankGridRow>,
    );

    const isUsTickerType = String(selectedTickerTypeItem?.country_code || "").toLowerCase() === "us";
    const isAuTickerType = String(selectedTickerTypeItem?.country_code || "").toLowerCase() === "au";
    const infoColumns: ColDef<RankGridRow>[] = [
      {
        field: "배당률",
        headerName: "배당률",
        minWidth: 92,
        width: 92,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatPercent(params.value ?? null),
      },
      ...(!isUsTickerType
        ? [
          {
            field: "보수",
            headerName: "보수",
            minWidth: 92,
            width: 92,
            type: "rightAligned",
            cellRenderer: (params: { value: number | null | undefined }) => formatPercent(params.value ?? null),
          } as ColDef<RankGridRow>,
        ]
        : []),
      {
        field: "순자산총액",
        headerName: isUsTickerType ? "시가총액" : "순자산총액",
        minWidth: 132,
        width: 132,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => {
          const val = params.value ?? null;
          if (isUsTickerType) return formatUsdMarketCap(val);
          if (isAuTickerType) return formatAudMarketCap(val);
          return formatAssetInEok(val);
        },
      },
      {
        field: "거래량",
        headerName: "거래량",
        minWidth: 100,
        width: 100,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatNumber(params.value ?? null, 0),
      },
      {
        field: "상장일",
        headerName: "상장일",
        minWidth: 110,
        width: 110,
        cellRenderer: (params: { value: string | null | undefined }) => String(params.value ?? "-"),
      },
    ];

    const columnsByMode: Record<MetricMode, ColDef<RankGridRow>[]> = {
      basic: basicColumns,
      ranking: rankingColumns,
      monthly: monthlyColumns,
      info: infoColumns,
    };

    return [...leadingColumns, ...columnsByMode[metricMode]];
  }, [
    addingRow,
    dirtyCellKeys,
    hasIndustryData,
    maRule,
    metricMode,
    monthlyReturnLabels,
    pageMode,
    selectedTickerType,
    isAllTickerType,
    selectedTickerTypeItem?.country_code,
    selectedTickerTypeItem?.top_n_hold,
    selectedTickerTypeItem?.currency,
    ticker_types,
  ]);

  function handleTickerTypeChange(accountId: string) {
    setSelectedAccountId(accountId);
    writeRememberedTickerType(accountId);
    if (accountId === "all") {
      setPageMode("rank");
    }
    void load({
      ticker_type: accountId,
      as_of_date: selectedAsOfDate,
      bootstrap: true,
    });
  }

  function handleMaRuleDaysChange(key: "short_ma_days" | "long_ma_days", nextDays: number) {
    if (!maRule) {
      return;
    }
    const nextRule = { ...maRule, [key]: nextDays };
    setMaRule(nextRule);
    void load({
      ticker_type: selectedTickerType,
      ma_rule_override: nextRule,
      as_of_date: selectedAsOfDate,
    });
  }

  function handleMaxPerIndustryChange(next: number) {
    setMaxPerIndustry(next);
    void load({
      ticker_type: selectedTickerType,
      ma_rule_override: maRule ?? undefined,
      as_of_date: selectedAsOfDate,
      max_per_industry: next,
    });
  }

  function handleAsOfDateChange(nextAsOfDate: string) {
    setSelectedAsOfDate(nextAsOfDate);
    void load({
      ticker_type: selectedTickerType,
      ma_rule_override: maRule ?? undefined,
      as_of_date: nextAsOfDate,
    });
  }

  function showErrorToast(message: string) {
    toast.error(`[순위] ${message}`);
  }

  function handleAddRow() {
    if (addingRow) {
      return;
    }
    addingTickerDraftRef.current = "";
    setAddingRow({
      ticker: "",
      name: "",
      listing_date: "-",
      bucket: 1,
      status: null,
      is_validating: false,
      is_validated: false,
    });
  }

  function handleBucketChanged(row: RankGridRow | undefined, bucketName: string) {
    if (!row || row.__isAddingRow) {
      return;
    }
    const nextBucketId = getBucketIdByName(bucketName);
    setRows((prev) =>
      prev.map((currentRow) =>
        getRowId(currentRow) === row.id
          ? {
            ...currentRow,
            bucket: nextBucketId,
            버킷: getBucketName(nextBucketId),
          }
          : currentRow,
      ),
    );
    setDirtyRowIds((prev) => (prev.includes(row.id) ? prev : [...prev, row.id]));
    const dirtyCellKey = buildDirtyCellKey(row.id, "버킷");
    setDirtyCellKeys((prev) => (prev.includes(dirtyCellKey) ? prev : [...prev, dirtyCellKey]));
  }

  async function handleValidateAddingTicker(tickerInput?: string) {
    const ticker = normalizeTicker(tickerInput ?? addingTickerDraftRef.current ?? addingRow?.ticker ?? "");
    if (!ticker || !selectedTickerType || !addingRow || addingRow.is_validating) {
      return;
    }

    try {
      setAddingRow((prev) => (prev ? { ...prev, ticker, is_validating: true } : null));
      const validated = await validateStockCandidate(selectedTickerType, ticker);
      addingTickerDraftRef.current = normalizeTicker(validated.ticker);
      setAddingRow((prev) =>
        prev
          ? {
            ...prev,
            ticker: normalizeTicker(validated.ticker),
            name: String(validated.name ?? "").trim(),
            listing_date: String(validated.listing_date ?? "-").trim() || "-",
            bucket: Number(validated.bucket_id ?? prev.bucket ?? 1),
            status: validated.status,
            is_validating: false,
            is_validated: validated.status !== "active",
          }
          : null,
      );
      if (validated.status === "active") {
        showErrorToast("이미 등록된 종목입니다.");
        return;
      }
      toast.success(`[순위] ${validated.name}(${validated.ticker}) 확인 완료`);
    } catch (validationError) {
      addingTickerDraftRef.current = ticker;
      setAddingRow((prev) =>
        prev
          ? {
            ...prev,
            ticker,
            is_validating: false,
            is_validated: false,
          }
          : null,
      );
      showErrorToast(validationError instanceof Error ? validationError.message : "티커 확인에 실패했습니다.");
    }
  }

  async function processAddingRow() {
    if (!addingRow || !addingRow.is_validated) {
      throw new Error("추가할 종목을 먼저 확인하세요.");
    }

    const created = await addStockCandidate(selectedTickerType, addingRow.ticker, addingRow.bucket);
    toast.success(`[순위] ${created.name}(${created.ticker}) 추가 완료`);
  }

  async function processDirtyRows() {
    const dirtyRows = rows.filter((row) => dirtyRowIds.includes(getRowId(row)));
    for (const row of dirtyRows) {
      await updateStockBucket(selectedTickerType, String(row.티커 ?? ""), Number(row.bucket ?? 1));
    }
  }

  async function handleMemoChange(ticker: string, memo: string) {
    if (!ticker) return;
    try {
      await updateStockMemo(ticker, memo);
      // 행 데이터도 갱신해 재조회 전까지 값이 유지되게 한다.
      setRows((prev) => prev.map((row) => (String(row.티커 ?? "") === ticker ? { ...row, 메모: memo } : row)));
      toast.success("메모 저장 완료");
    } catch (error) {
      showErrorToast(error instanceof Error ? error.message : "메모 저장에 실패했습니다.");
    }
  }

  function handleSaveChanges() {
    if (!selectedTickerType || (!addingRow && dirtyRowIds.length === 0)) {
      return;
    }

    startTransition(async () => {
      try {
        if (addingRow) {
          await processAddingRow();
        }
        if (dirtyRowIds.length > 0) {
          await processDirtyRows();
        }
        toast.success("[순위] 변경사항 저장 완료");
        void load({
          ticker_type: selectedTickerType,
          ma_rule_override: maRule ?? undefined,
          as_of_date: selectedAsOfDate,
          skip_session_cache: true,
        });
      } catch (saveError) {
        showErrorToast(saveError instanceof Error ? saveError.message : "변경사항 저장에 실패했습니다.");
      }
    });
  }

  /** 이동 모달 열기 — 대상 풀 목록은 서버가 국가·구분으로 걸러 준다. */
  function handleMoveSelected() {
    if (selectedTickers.length === 0) return;
    setMoveTarget("");
    setMovablePools(null);
    setMoveOpen(true);
    void (async () => {
      try {
        setMovablePools(await loadMovablePools(selectedTickerType));
      } catch (error) {
        setMovablePools([]);
        toast.error(error instanceof Error ? error.message : "옮길 수 있는 종목풀을 불러오지 못했습니다.");
      }
    })();
  }

  function handleConfirmMove() {
    const target = moveTarget.trim();
    if (!target || selectedTickers.length === 0) return;
    const selectedRows = rows.filter((row) => selectedTickers.includes(getRowId(row)));
    // 진행도는 transition **밖에서** 먼저 세운다 — 안에서 세우면 React 가 렌더를 미뤄
    // 첫 종목이 끝날 때까지(10초 안팎) 화면에 아무것도 안 나온다.
    const firstRow = selectedRows[0];
    setMoveProgress({
      done: 0,
      total: selectedRows.length,
      ticker: String(firstRow?.티커 ?? ""),
      name: String(firstRow?.종목명 ?? ""),
    });
    startTransition(async () => {
      let movedCount = 0;
      const failed: string[] = [];
      for (const [index, row] of selectedRows.entries()) {
        const ticker = String(row.티커 ?? "");
        setMoveProgress({ done: index, total: selectedRows.length, ticker, name: String(row.종목명 ?? "") });
        try {
          await moveStockToPool(selectedTickerType, target, ticker);
          movedCount += 1;
        } catch {
          failed.push(ticker);
        } finally {
          setMoveProgress({ done: index + 1, total: selectedRows.length, ticker, name: String(row.종목명 ?? "") });
        }
      }
      setMoveProgress(null);
      setMoveOpen(false);
      setSelectedTickers([]);
      if (movedCount > 0) toast.success(`[순위] ${movedCount}개 종목을 옮겼습니다.`);
      if (failed.length > 0) toast.error(`이동 실패: ${failed.join(", ")}`);
      clearCacheWarningState();
      void load({
        ticker_type: selectedTickerType,
        ma_rule_override: maRule ?? undefined,
        as_of_date: selectedAsOfDate,
        skip_session_cache: true,
      });
    });
  }

  function handleDeleteSelected() {
    if (selectedTickers.length === 0) {
      return;
    }
    setDeleteConfirmOpen(true);
  }

  function handleCloseDeleteConfirm() {
    if (isPending) {
      return;
    }
    setDeleteConfirmOpen(false);
  }

  function handleConfirmDeleteSelected() {
    if (selectedTickers.length === 0) {
      setDeleteConfirmOpen(false);
      return;
    }

    const selectedRows = rows.filter((row) => selectedTickers.includes(getRowId(row)));
    startTransition(async () => {
      try {
        for (const row of selectedRows) {
          await deleteStock(selectedTickerType, String(row.티커 ?? ""));
        }
        const deletedIdSet = new Set(selectedRows.map((row) => getRowId(row)));
        setRows((prev) => prev.filter((row) => !deletedIdSet.has(getRowId(row))));
        setSelectedTickers([]);
        setDeleteConfirmOpen(false);
        clearCacheWarningState();
        toast.success(`[순위] ${selectedRows.length}개 종목 삭제 완료`);
        void load({
          ticker_type: selectedTickerType,
          ma_rule_override: maRule ?? undefined,
          as_of_date: selectedAsOfDate,
          skip_session_cache: true,
        });
      } catch (deleteError) {
        showErrorToast(deleteError instanceof Error ? deleteError.message : "종목 삭제에 실패했습니다.");
      }
    });
  }

  const blockedMessage = useMemo(() => {
    if (!cacheBlocked) {
      return null;
    }

    const parts: string[] = ["일부 종목의 가격 캐시가 없습니다."];
    const missing = missingTickerLabels.length > 0 ? missingTickerLabels : missingTickers;
    if (missing.length > 0) {
      parts.push(`누락 ${summarizeTickers(missing)}`);
    }
    if (staleTickers.length > 0) {
      parts.push(`오래된 캐시 ${summarizeTickers(staleTickers)}`);
    }
    return parts.join(" | ");
  }, [cacheBlocked, missingTickerLabels, missingTickers, staleTickers]);

  useEffect(() => {
    if (!blockedMessage) {
      lastBlockedToastRef.current = null;
      return;
    }

    if (lastBlockedToastRef.current === blockedMessage) {
      return;
    }

    lastBlockedToastRef.current = blockedMessage;
    toast.error(`[순위] ${blockedMessage}`);
  }, [blockedMessage, toast]);

  const headerSummary = useMemo<RankHeaderSummary>(() => {
    const candidateRows = gridRows.filter((r) => !r.is_benchmark && !r.exclude_from_ranking);
    const totalCount = candidateRows.length;
    const upCount = candidateRows.filter((r) => (r["추세"] ?? 0) > 0).length;
    const upPct = totalCount > 0 ? Math.round((upCount / totalCount) * 100) : 0;
    const configuredTopN = selectedTickerTypeItem?.top_n_hold;
    const ruleSummaryParts: string[] = [];
    if (configuredTopN != null && !Number.isNaN(configuredTopN)) {
      ruleSummaryParts.push(`TOP ${formatNumber(configuredTopN, 0)}`);
    }
    ruleSummaryParts.push(...maRuleSummary);
    return {
      upCount,
      upPct,
      totalCount,
      ruleSummary: ruleSummaryParts.join(" / ") || "-",
    };
  }, [gridRows, maRuleSummary, selectedTickerTypeItem?.top_n_hold]);

  useEffect(() => {
    onHeaderSummaryChange?.(headerSummary);
  }, [headerSummary, onHeaderSummaryChange]);

  return (
    <div className="appPageStack appPageStackFill">
      {error ? (
        <div className="appBannerStack">
          <div className="bannerError alert alert-danger mb-0">{error}</div>
        </div>
      ) : null}

      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <ResponsiveFiltersSection>
              <div className="appMainHeader">
                <div className="appMainHeaderLeft rankMainHeaderLeft">
                  <label className="appLabeledField">
                    <span className="appLabeledFieldLabel">기준일</span>
                    <input
                      className="form-control"
                      type="date"
                      value={selectedAsOfDate}
                      max={getTodayDateInputValue()}
                      onChange={(event) => handleAsOfDateChange(event.target.value)}
                    />
                  </label>
                  <label className="appLabeledField">
                    <span className="appLabeledFieldLabel">종목풀</span>
                    <select
                      className="form-select"
                      value={selectedTickerType}
                      onChange={(event) => handleTickerTypeChange(event.target.value)}
                      disabled={ticker_types.length === 0}
                    >
                      {ticker_types.length === 0 ? (
                        <option value="">종목풀 불러오는 중...</option>
                      ) : (
                        ticker_types.map((account) => (
                          <option key={account.ticker_type} value={account.ticker_type}>
                            {formatPoolLabel(account)}
                          </option>
                        ))
                      )}
                    </select>
                  </label>
                  <label className="appLabeledField">
                    <span className="appLabeledFieldLabel">화면 모드</span>
                    <div className="appSegmentedToggle appSegmentedToggleCompact" role="group" aria-label="순위 화면 모드">
                      <button
                        type="button"
                        className={pageMode === "rank" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                        onClick={() => setPageMode("rank")}
                      >
                        순위모드
                      </button>
                      <button
                        type="button"
                        className={pageMode === "manage" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                        onClick={() => {
                          if (!isAllTickerType) {
                            setPageMode("manage");
                          }
                        }}
                        disabled={isAllTickerType}
                        title={isAllTickerType ? "전체 종목풀에서는 관리모드를 사용할 수 없습니다." : undefined}
                      >
                        관리모드
                      </button>
                    </div>
                  </label>
                  {/* 업종 상한 — 추천(✅) 이 한 업종에 몰리지 않게 자르는 기준.
                      기본값은 종목풀 저장값이고, 여기서 바꾼 값은 화면에서만 쓴다(저장 안 함). */}
                  {pageMode === "rank" && hasIndustryData ? (
                    <label className="appLabeledField">
                      <span className="appLabeledFieldLabel">업종 상한</span>
                      <select
                        className="form-select form-select-sm"
                        style={{ width: "auto" }}
                        title="한 업종에서 최대 몇 종목까지 ✅ 를 붙일지"
                        value={String(effectiveMaxPerIndustry)}
                        onChange={(event) => handleMaxPerIndustryChange(Number(event.target.value))}
                      >
                        {MAX_PER_INDUSTRY_CHOICES.map((option) => (
                          <option key={option} value={String(option)}>
                            {option < 0 ? "없음" : `${option}종목`}
                          </option>
                        ))}
                      </select>
                    </label>
                  ) : null}
                  {pageMode === "rank" && maRule ? (
                    <label className="appLabeledField">
                      <span className="appLabeledFieldLabel">이평선</span>
                      <div className="appMaRuleRow">
                        <MaDaysSelect
                          title="단기 이평선"
                          value={maRule.short_ma_days}
                          options={maOptions.short_ma_options}
                          onChange={(days) => handleMaRuleDaysChange("short_ma_days", days)}
                        />
                        <MaDaysSelect
                          title="장기 이평선"
                          value={maRule.long_ma_days}
                          options={maOptions.long_ma_options}
                          onChange={(days) => handleMaRuleDaysChange("long_ma_days", days)}
                        />
                      </div>
                    </label>
                  ) : null}
                  <label className="appLabeledField">
                    <span className="appLabeledFieldLabel">컬럼</span>
                    <div className="appSegmentedToggle appSegmentedToggleCompact" role="group" aria-label="컬럼 표시 방식">
                      {METRIC_MODE_OPTIONS.map(({ value, label }) => (
                        <button
                          key={value}
                          type="button"
                          className={metricMode === value ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                          onClick={() => setMetricMode(value)}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  </label>
                </div>
              </div>
            </ResponsiveFiltersSection>
          </div>

          {pageMode === "manage" ? (
            <div className="card-header appActionHeader bg-light-subtle border-top">
              <div className="appActionHeaderInner">
                <button
                  className="btn btn-primary btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                  type="button"
                  onClick={handleAddRow}
                  disabled={loading || isPending || Boolean(addingRow)}
                >
                  <IconPlus size={16} stroke={2} />
                  <span>추가</span>
                </button>
                <button
                  className="btn btn-success btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                  type="button"
                  onClick={handleSaveChanges}
                  disabled={loading || isPending || (!addingRow && dirtyRowIds.length === 0)}
                >
                  <IconDeviceFloppy size={16} stroke={2} />
                  <span>저장</span>
                </button>
                <button
                  className="btn btn-outline-secondary btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                  type="button"
                  onClick={handleMoveSelected}
                  disabled={loading || isPending || selectedTickers.length === 0 || isAllTickerType}
                  title="같은 국가·구분(개별주/ETF)의 다른 종목풀로 옮깁니다."
                >
                  <IconArrowsExchange size={16} stroke={2} />
                  <span>이동</span>
                </button>
                <button
                  className="btn btn-outline-danger btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                  type="button"
                  onClick={handleDeleteSelected}
                  disabled={loading || isPending || selectedTickers.length === 0}
                >
                  <IconTrash size={16} stroke={2} />
                  <span>삭제</span>
                </button>
              </div>
            </div>
          ) : null}

          <div className="card-body appCardBodyTight appTableCardBodyFill">
            <div className="appGridFillWrap">
              <AppAgGrid
                key={`${selectedTickerType}:${selectedAsOfDate}:${maRule?.short_ma_days}:${maRule?.long_ma_days}`}
                className="rankAgGrid"
                rowData={displayGridRows}
                columnDefs={columns}
                getRowId={(params) => params.data.id ?? ""}
                loading={loading || isPending}
                theme={rankGridTheme}
                getRowClass={(params: RowClassParams<RankGridRow>) => {
                  const classes: string[] = [];
                  // 추세 이탈(장기·단기 중 하나라도 음수) — 종목명 뒤 ❗ 와 같은 조건으로 행을 연한 회색으로.
                  if (isTrendBroken(params.data?.단기이격, params.data?.이격)) {
                    classes.push("appTrendBrokenRow");
                  }
                  if (params.data?.exclude_from_ranking) {
                    classes.push("rankFixedRow");
                  }
                  // 실제 보유 중인 종목은 배경을 녹색으로(보유 컬럼 대체). 보유가 다른 색보다 우선한다.
                  if (String(params.data?.보유 ?? "").trim()) {
                    classes.push("rankHeldRow");
                  }
                  return classes.join(" ");
                }}
                minHeight="100%"
                gridOptions={{
                  suppressMovableColumns: true,
                  rowSelection: pageMode === "manage"
                    ? {
                      mode: "multiRow",
                      checkboxes: (params) => !params.data?.__isAddingRow,
                      headerCheckbox: true,
                      hideDisabledCheckboxes: true,
                      enableClickSelection: false,
                    }
                    : undefined,
                  selectionColumnDef: pageMode === "manage"
                    ? {
                      width: 52,
                      minWidth: 52,
                      maxWidth: 52,
                      pinned: "left",
                      sortable: false,
                      resizable: false,
                      suppressMovable: true,
                      headerName: "",
                      cellClass: "stocksSelectCell",
                    }
                    : undefined,
                  onSelectionChanged: (params: { api: { getSelectedRows: () => RankGridRow[] } }) => {
                    if (pageMode !== "manage") {
                      setSelectedTickers([]);
                      return;
                    }
                    setSelectedTickers(
                      params.api
                        .getSelectedRows()
                        .map((row) => row.id)
                        .filter((rowId) => rowId !== "__adding__"),
                    );
                  },
                  onCellValueChanged: (params: {
                    data?: RankGridRow;
                    newValue?: unknown;
                    oldValue?: unknown;
                  }) => {
                    if (pageMode !== "manage" || !params.data || params.data.__isAddingRow || params.newValue === params.oldValue) {
                      return;
                    }
                    handleBucketChanged(params.data, String(params.newValue ?? ""));
                  },
                }}
              />
            </div>
          </div>
        </div>
      </section>

      <AppModal
        open={deleteConfirmOpen}
        title="종목 삭제 확인"
        subtitle="선택 종목은 즉시 영구 삭제됩니다."
        onClose={handleCloseDeleteConfirm}
        footer={(
          <>
            <button type="button" className="btn btn-outline-secondary" onClick={handleCloseDeleteConfirm} disabled={isPending}>
              취소
            </button>
            <button type="button" className="btn btn-danger" onClick={handleConfirmDeleteSelected} disabled={isPending}>
              삭제
            </button>
          </>
        )}
      >
        <div className="d-flex flex-column gap-2">
          <div className="fw-semibold">
            {selectedTickers.length === 1
              ? `${rows.find((row) => selectedTickers.includes(getRowId(row)))?.종목명 ?? ""}(${selectedTickers[0]}) 종목을 삭제합니다.`
              : `${selectedTickers.length}개 종목을 삭제합니다.`}
          </div>
          <div className="text-secondary small">삭제된 종목은 복구되지 않으며 즉시 제거됩니다.</div>
        </div>
      </AppModal>

      {/* 종목 이동 — 지금 풀에서 빼고 고른 풀에 담는다. 한 티커는 한 풀에만 있을 수 있어
          '양쪽에 두기' 가 아니라 이동이다. */}
      <AppModal
        open={moveOpen}
        title="종목풀 이동"
        subtitle={`선택한 종목 ${selectedTickers.length}개를 다른 종목풀로 옮깁니다.`}
        onClose={() => {
          if (!isPending) setMoveOpen(false);
        }}
        footer={(
          <>
            <button
              type="button"
              className="btn btn-outline-secondary"
              onClick={() => setMoveOpen(false)}
              disabled={isPending}
            >
              취소
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleConfirmMove}
              disabled={isPending || !moveTarget}
            >
              {isPending ? "옮기는 중…" : "이동"}
            </button>
          </>
        )}
      >
        <div className="d-flex flex-column gap-2">
          {movablePools === null ? (
            <div className="text-secondary small">옮길 수 있는 종목풀을 확인하는 중…</div>
          ) : movablePools.length === 0 ? (
            <div className="alert alert-warning mb-0">
              옮길 수 있는 종목풀이 없습니다.
              <div className="small mt-1">
                국가와 구분(개별주/ETF)이 같은 종목풀로만 옮길 수 있습니다.
              </div>
            </div>
          ) : (
            <>
              <label className="appLabeledField">
                <span className="appLabeledFieldLabel">옮길 종목풀</span>
                <select
                  className="field compactField"
                  value={moveTarget}
                  disabled={isPending}
                  onChange={(event) => setMoveTarget(event.target.value)}
                >
                  <option value="">종목풀 선택</option>
                  {movablePools.map((pool) => (
                    <option key={pool.ticker_type} value={pool.ticker_type}>
                      {formatPoolLabel(pool)}
                    </option>
                  ))}
                </select>
              </label>
              <div className="text-secondary small">
                기존 종목풀에서 빠지고 새 종목풀에 담깁니다. 버킷·메모·가격 캐시는 그대로 옮겨집니다.
              </div>
              <PoolAddProgressBar progress={moveProgress} />
            </>
          )}
        </div>
      </AppModal>
    </div>
  );
}
