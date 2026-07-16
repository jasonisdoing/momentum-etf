"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ColDef, GridOptions } from "ag-grid-community";
import { ColorType, LineSeries, createChart, createSeriesMarkers } from "lightweight-charts";
import type { Time } from "lightweight-charts";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { BUCKET_COLORS, BUCKET_THEME } from "@/lib/bucket-theme";
import {
  readRememberedMomentumEtfAccountId,
  writeRememberedMomentumEtfAccountId,
} from "../components/account-selection";
import { AppAgGrid } from "../components/AppAgGrid";
import { PageFrame } from "../components/PageFrame";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";

type TopPickTicker = {
  ticker: string;
  name?: string;
  ticker_type?: string;
  country_code?: string;
  is_etf?: boolean;
  bucket?: number;
  fixed_weight_pct?: number | null;
  alignment?: string | null;
};

type TopPickSettingsPayload = {
  tickers?: TopPickTicker[];
  weight_mode?: "variable" | "fixed";
  settings?: TopPickSettings;
  backtest_settings?: TopPickBacktestSettings;
  approved_weights?: TopPickWeightPreview | null;
  approved_at?: string | null;
  updated_at?: string | null;
  error?: string;
};

type TopPickWeightRow = {
  ticker: string;
  name: string;
  daily_change_pct?: number | null;
  return_1m_pct?: number | null;
  return_3m_pct?: number | null;
  return_6m_pct?: number | null;
  return_12m_pct?: number | null;
  trend_pct: number | null;
  mdd_pct?: number | null;
  sortino?: number | null;
  score: number | null;
  target_weight_pct: number | null;
};

type TopPickWeightComparisonRow = TopPickWeightRow & {
  approved_weight_pct: number | null;
  calculated_weight_pct: number | null;
  weight_diff_pct: number | null;
};

type TopPickWeightPreview = {
  as_of_date?: string;
  settings?: TopPickSettings;
  rows?: TopPickWeightRow[];
  missing_tickers?: string[];
  approved_at?: string | null;
  error?: string;
};

type TopPickSettings = {
  VARIABLE_TICKERS: number;
  FIXED_TICKERS: number;
  MAX_TICKERS: number;
  ACCOUNT_ID: string;
  START_AMOUNT_MANWON: number | null;
  START_DATE: string | null;
  POOL_TICKER_TYPE?: string | null;
  POOL_NAME?: string | null;
  SHORT_MA_DAYS?: number | null;
  MAIN_MA_DAYS?: number | null;
};

type AccountTopPickBasis = {
  startAmount: number | null;
  startDate: string | null;
};

type TopPickBacktestSettings = {
  benchmark?: TopPickTicker | null;
  months?: number;
  rebalance?: string;
  initial_amount_manwon?: number;
};

type AccountOption = {
  account_id: string;
  name: string;
  currency?: string;
};

type RankRow = {
  source_ticker_type?: string;
  티커: string;
  종목명: string;
  bucket?: number;
  추세: number | null;
  배열?: string | null;
  exclude_from_ranking?: boolean;
  is_benchmark?: boolean;
};

type RankResponse = {
  rows?: RankRow[];
  error?: string;
};

type LabSummary = {
  total_return_pct: number;
  cagr_pct: number;
  mdd_pct: number;
  sortino: number;
};

type LabPosition = TopPickTicker & {
  buy_date: string;
  late_entry: boolean;
  shares: number;
  buy_price: number | null;
  last_price: number | null;
  return_pct: number | null;
  mdd_pct: number | null;
  mdd_start: string;
  mdd_end: string;
  sortino: number | null;
  profit: number;
  profit_contribution_pct?: number | null;
  value: number;
  min_weight?: number;
  max_weight?: number;
};

function getBucketCellClass(bucketId: number | undefined): string {
  return bucketId ? `rankBucketCell rankBucketCell${bucketId}` : "rankBucketCell";
}

function getBucketName(bucketId: number | undefined): string {
  return bucketId ? BUCKET_THEME[String(bucketId)]?.name ?? "-" : "-";
}

const NAME_HIGHLIGHT_KEYWORDS: Record<string, { color: string; emoji: string }> = {
  레버리지: { color: "#d63939", emoji: "💣" },
  Geared: { color: "#d63939", emoji: "💣" },
  "3X": { color: "#d63939", emoji: "💣" },
  Ultra: { color: "#d63939", emoji: "💣" },
  액티브: { color: "#206bc4", emoji: "✋" },
};

const NAME_HIGHLIGHT_RE = new RegExp(`(${Object.keys(NAME_HIGHLIGHT_KEYWORDS).join("|")})`, "i");

function getNameHighlight(part: string): { color: string; emoji: string } | undefined {
  const lower = part.toLowerCase();
  for (const [keyword, style] of Object.entries(NAME_HIGHLIGHT_KEYWORDS)) {
    if (keyword.toLowerCase() === lower) {
      return style;
    }
  }
  return undefined;
}

function renderNameWithLeverageHighlight(name: string) {
  const parts = name.split(NAME_HIGHLIGHT_RE);
  if (parts.length === 1) {
    return name;
  }
  const emojis: string[] = [];
  const rendered = parts.map((part, index) => {
    const style = index % 2 === 1 ? getNameHighlight(part) : undefined;
    if (!style) {
      return <span key={index}>{part}</span>;
    }
    if (!emojis.includes(style.emoji)) {
      emojis.push(style.emoji);
    }
    return (
      <span key={index} style={{ color: style.color, fontWeight: 700 }}>
        {part}
      </span>
    );
  });
  return (
    <>
      {rendered}
      {emojis.length > 0 && <span> {emojis.join("")}</span>}
    </>
  );
}

type LabResult = {
  months: number;
  rebalance?: string;
  buy_date: string;
  end_date: string;
  has_late_entry?: boolean;
  initial_capital: number;
  final_value: number;
  slippage?: { total_cost: number; total_cost_pct: number };
  summary: LabSummary;
  benchmark: TopPickTicker & { summary: LabSummary };
  positions: LabPosition[];
  cash_min_weight?: number;
  cash_max_weight?: number;
  chart: {
    dates: string[];
    portfolio_value?: number[];
    benchmark_value?: number[];
    portfolio_pct: number[];
    benchmark_pct: number[];
  };
  weight_history?: TopPickWeightHistoryRow[];
  weight_items?: TopPickWeightItem[];
  error?: string;
};

type TopPickWeightHistoryRow = {
  date: string;
  [key: string]: string | number;
};

type TopPickWeightItem = {
  key: string;
  label: string;
  bucket?: number;
};

type TopPickWeightHoverDetail = {
  date: string;
  items: Array<TopPickWeightItem & { weight: number; color: string }>;
};

type TopPickReserveCandidate = TopPickTicker & {
  trend_pct: number | null;
  alignment?: string | null;
};

type TopPickSelectedGridRow = TopPickTicker & {
  row_index: number;
  trend_pct: number | null;
  alignment?: string | null;
  is_confirmed: boolean;
};

type TopPickReserveGridRow = TopPickReserveCandidate & {
  row_index: number;
};

// 코드 기본값 없음(silent default 금지). 설정값은 전적으로 DB(/top-pick-settings 저장)에서 온다.
// 로드 전에는 settings=null 이며 폼은 빈 상태로 렌더되고, 로드 후 DB 값으로 채워진다.

const MAX_TOP_PICK_SELECTED = 10;
const previewGridTheme = createAppGridTheme();
const DEFAULT_BACKTEST_SETTINGS: Required<TopPickBacktestSettings> = {
  benchmark: { ticker: "", name: "" },
  months: 12,
  rebalance: "none",
  initial_amount_manwon: 10000,
};
const BACKTEST_REBALANCE_OPTIONS: { value: string; label: string }[] = [
  { value: "none", label: "리밸런싱 없음 (보유)" },
  { value: "weekly", label: "매주 (금요일)" },
  { value: "monthly", label: "매월 (말일)" },
  { value: "quarterly", label: "분기 (분기말)" },
  { value: "yearly", label: "매년 (연말)" },
];

function normalizeTicker(value: string): string {
  return value.trim().toUpperCase();
}

function isKorTickerInput(value: string): boolean {
  return value.startsWith("KOR:") || /^[0-9A-Z]{6}$/.test(value);
}

function stripKorPrefix(value: string): string {
  return value.startsWith("KOR:") ? value.slice("KOR:".length) : value;
}

function buildTickerRows(items: TopPickTicker[] | undefined): TopPickTicker[] {
  const rows = (items ?? [])
    .filter((item) => (item?.ticker ?? "").trim() || (item?.name ?? "").trim())
    .map((item) => ({ ...item, ticker: item.ticker ?? "" }));
  return rows.length > 0 ? rows : [{ ticker: "" }];
}

function buildSettingsForRequest(
  settings: TopPickSettings | null,
  tickerCount: number,
  basis: AccountTopPickBasis,
): TopPickSettings | null {
  if (!settings) {
    return null;
  }
  const count = Math.max(1, tickerCount);
  return {
    VARIABLE_TICKERS: count,
    FIXED_TICKERS: 0,
    MAX_TICKERS: count,
    ACCOUNT_ID: settings.ACCOUNT_ID,
    START_AMOUNT_MANWON: basis.startAmount,
    START_DATE: basis.startDate,
    POOL_TICKER_TYPE: settings.POOL_TICKER_TYPE ?? null,
    POOL_NAME: settings.POOL_NAME ?? null,
    SHORT_MA_DAYS: settings.SHORT_MA_DAYS ?? null,
    MAIN_MA_DAYS: settings.MAIN_MA_DAYS ?? null,
  };
}

function formatReturnPct(value: number | null | undefined): string {
  if (value == null) {
    return "-";
  }
  return `${value.toFixed(2)}%`;
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

function formatWeightPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${value.toFixed(1)}%`;
}

function formatKrw(value: number): string {
  return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
}

function formatCompactKrw(value: number): string {
  const absValue = Math.abs(value);
  if (absValue >= 100_000_000) {
    return `${new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 1 }).format(value / 100_000_000)}억`;
  }
  if (absValue >= 10_000) {
    return `${new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 0 }).format(value / 10_000)}만`;
  }
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 0 }).format(value);
}

function signedColor(value: number): string {
  if (value > 0) return "#d63939";
  if (value < 0) return "#206bc4";
  return "#475569";
}

function rebalanceLabel(value?: string): string {
  return BACKTEST_REBALANCE_OPTIONS.find((option) => option.value === (value ?? "none"))?.label ?? "리밸런싱 없음 (보유)";
}

function getTopPickWeightColor(items: TopPickWeightItem[] | undefined, key: string): string {
  if (key === "__CASH__") {
    return BUCKET_COLORS[4];
  }
  const bucket = items?.find((item) => item.key === key)?.bucket;
  return bucket && BUCKET_COLORS[bucket - 1] ? BUCKET_COLORS[bucket - 1] : "#64748b";
}

function formatMonthAxisLabel(value: string): string {
  const [year, month] = value.split("-").map((part) => Number(part));
  if (!year || !month) return value;
  return `${year}.${String(month).padStart(2, "0")}`;
}

function getSignedClass(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) {
    return "";
  }
  return value > 0 ? "metricPositive" : "metricNegative";
}

function renderReturnPctCell(params: { value: number | null | undefined }) {
  return <span className={getSignedClass(params.value)}>{formatReturnPct(params.value)}</span>;
}

function renderAlignmentCell(params: { value: string | null | undefined }) {
  const value = params.value ?? "-";
  const color = value === "정배열" ? "#d63939" : value === "역배열" ? "#1d6fd1" : "var(--text-muted)";
  return <span style={{ color, fontWeight: 700 }}>{value}</span>;
}

function LabChart({ result }: { result: LabResult }) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const chart = createChart(container, {
      height: 320,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#5b6675" },
      grid: { vertLines: { color: "rgba(148,163,184,0.12)" }, horzLines: { color: "rgba(148,163,184,0.12)" } },
      rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.18, bottom: 0.15 } },
      timeScale: { borderVisible: false, rightOffset: 10 },
      localization: { priceFormatter: (price: number) => formatCompactKrw(price) },
      autoSize: true,
    });
    const toLine = (values: number[]) => result.chart.dates.map((date, index) => ({ time: date as Time, value: values[index] }));
    const portfolioValues = result.chart.portfolio_value ?? result.chart.portfolio_pct;
    const benchmarkValues = result.chart.benchmark_value ?? result.chart.benchmark_pct;
    const portfolioSeries = chart.addSeries(LineSeries, { color: "#2563eb", lineWidth: 2, lastValueVisible: false, priceLineVisible: false });
    const benchmarkSeries = chart.addSeries(LineSeries, { color: "#94a3b8", lineWidth: 1, lastValueVisible: false, priceLineVisible: false });
    portfolioSeries.setData(toLine(portfolioValues));
    benchmarkSeries.setData(toLine(benchmarkValues));

    const peakMarker = (values: number[], color: string, position: "aboveBar" | "belowBar") => {
      const peakIndex = values.reduce(
        (bestIndex, value, index) => (Number.isFinite(value) && value > values[bestIndex] ? index : bestIndex),
        0,
      );
      const peakValue = values[peakIndex];
      const finalValue = values[values.length - 1];
      const drawdown = peakValue > 0 ? ((finalValue / peakValue) - 1) * 100 : 0;
      const date = result.chart.dates[peakIndex];
      const displayDate = date ? date.replaceAll("-", ".") : "-";
      return {
        time: date as Time,
        position,
        color,
        shape: position === "aboveBar" ? "arrowDown" as const : "arrowUp" as const,
        text: `${displayDate}(${drawdown.toFixed(2)}%)`,
        size: 1,
      };
    };

    if (portfolioValues.length > 0) {
      createSeriesMarkers(portfolioSeries, [peakMarker(portfolioValues, "#2563eb", "aboveBar")]);
    }
    if (benchmarkValues.length > 0) {
      createSeriesMarkers(benchmarkSeries, [peakMarker(benchmarkValues, "#64748b", "belowBar")]);
    }
    chart.timeScale().fitContent();

    return () => chart.remove();
  }, [result]);

  return (
    <div ref={containerRef} style={{ width: "100%", height: 320 }} />
  );
}

function TopPickWeightHistoryChart({
  rows,
  items,
  onHoverChange,
}: {
  rows: TopPickWeightHistoryRow[];
  items: TopPickWeightItem[];
  onHoverChange: (detail: TopPickWeightHoverDetail | null) => void;
}) {
  if (!rows.length || !items.length) {
    return <div style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>비중 이력이 없습니다.</div>;
  }

  // 누적 막대는 먼저 그린 항목이 아래에 쌓이므로 5번 버킷부터 역순으로 그립니다.
  const sortedItems = [...items].sort((a, b) => {
    const aBucket = a.key === "__CASH__" ? 5 : (a.bucket ?? 0);
    const bBucket = b.key === "__CASH__" ? 5 : (b.bucket ?? 0);
    return bBucket - aBucket;
  });
  const showWeightsForDate = (date: string) => {
    const activeRow = rows.find((row) => String(row.date) === date);
    if (!activeRow) return;
    const total = items.reduce((sum, item) => sum + Number(activeRow[item.key] ?? 0), 0);
    const hoverItems = items
      .map((item) => ({
        ...item,
        weight: total > 0 ? (Number(activeRow[item.key] ?? 0) / total) * 100 : 0,
        color: getTopPickWeightColor(items, item.key),
      }))
      .filter((item) => item.weight > 0);
    onHoverChange({ date, items: hoverItems });
  };

  return (
    <div className="topPickWeightChartWrap">
      <div className="topPickWeightChartCanvas">
        <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={320}>
          <BarChart
            data={rows}
            margin={{ top: 10, right: 12, bottom: 6, left: 0 }}
            onMouseMove={(state) => {
              if (state?.activeLabel != null) showWeightsForDate(String(state.activeLabel));
            }}
            onMouseLeave={() => onHoverChange(null)}
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="date" tickFormatter={formatMonthAxisLabel} minTickGap={18} tick={{ fontSize: 12 }} />
            <YAxis tickFormatter={(value) => formatCompactKrw(Number(value))} width={52} tick={{ fontSize: 12 }} />
            <Tooltip content={() => null} />
            {sortedItems.map((item, index) => (
              <Bar
                key={item.key}
                dataKey={item.key}
                name={item.label}
                stackId="weight"
                fill={getTopPickWeightColor(items, item.key)}
                radius={index === sortedItems.length - 1 ? [3, 3, 0, 0] : [0, 0, 0, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function TopPickSettingsClient() {
  const toast = useToast();
  const [tickers, setTickers] = useState<TopPickTicker[]>(() => buildTickerRows(undefined));
  const [settings, setSettings] = useState<TopPickSettings | null>(null);
  const [accounts, setAccounts] = useState<AccountOption[]>([]);
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null);
  // 선택 계좌에 연결된 종목풀 id — 티커 조회 범위 제한에 사용(빈 목록이면 전체 검색).
  const [accountTickerTypes, setAccountTickerTypes] = useState<string[]>([]);
  const [accountMemoById, setAccountMemoById] = useState<Record<string, string>>({});
  const [accountTopPickBasisById, setAccountTopPickBasisById] = useState<Record<string, AccountTopPickBasis>>({});
  // 종목풀 id → 이름 매핑(안내 문구 표시용).
  const [poolNameById, setPoolNameById] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [running, setRunning] = useState(false);
  const [approving, setApproving] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [approvedAt, setApprovedAt] = useState<string | null>(null);
  const [approvedWeights, setApprovedWeights] = useState<TopPickWeightPreview | null>(null);
  const [preview, setPreview] = useState<TopPickWeightPreview | null>(null);
  const [previewMode, setPreviewMode] = useState<"weights" | "backtest">("weights");
  const [backtestResult, setBacktestResult] = useState<LabResult | null>(null);
  const [backtestRunning, setBacktestRunning] = useState(false);
  const [weightHoverDetail, setWeightHoverDetail] = useState<TopPickWeightHoverDetail | null>(null);
  const [backtestPeriodMonths, setBacktestPeriodMonths] = useState(String(DEFAULT_BACKTEST_SETTINGS.months));
  const [backtestRebalanceMode, setBacktestRebalanceMode] = useState(DEFAULT_BACKTEST_SETTINGS.rebalance);
  const [backtestInitialAmountManwon, setBacktestInitialAmountManwon] = useState(String(DEFAULT_BACKTEST_SETTINGS.initial_amount_manwon));
  const [reserveCandidates, setReserveCandidates] = useState<TopPickReserveCandidate[]>([]);
  const [reserveLoading, setReserveLoading] = useState(false);
  const [reserveError, setReserveError] = useState<string | null>(null);
  const [rankMetaByTicker, setRankMetaByTicker] = useState<Record<string, { trend_pct: number | null; alignment?: string | null }>>({});
  const autoRunStartedRef = useRef(false);

  const validTickers = useMemo(
    () => tickers.filter((item) => item.ticker.trim() && item.name),
    [tickers],
  );
  const selectedTickerSet = useMemo(
    () => new Set(validTickers.map((item) => normalizeTicker(item.ticker))),
    [validTickers],
  );
  const reservePoolId = accountTickerTypes[0] ?? "";

  const loadSettings = useCallback(async (accountId?: string) => {
    try {
      setLoading(true);
      // 1) 계좌 목록 먼저 — account_id 미지정이면 탑픽 첫 계좌를 명시적으로 선택(auto-resolve 의존 X).
      const [accountsResp, acctSettingsResp, poolsResp] = await Promise.all([
        fetch("/api/holdings-components/accounts", { cache: "no-store" }),
        fetch("/api/account-settings", { cache: "no-store" }),
        fetch("/api/pool-settings", { cache: "no-store" }),
      ]);
      const accountsData = (await accountsResp.json()) as AccountOption[] | { error?: string };
      const acctSettingsData = (await acctSettingsResp.json()) as {
        accounts?: {
          account_id: string;
          ticker_types?: string[];
          memo?: string;
          top_pick_start_amount_manwon?: number | null;
          top_pick_start_date?: string | null;
        }[];
        error?: string;
      };
      const poolsData = (await poolsResp.json()) as {
        pools?: { ticker_type: string; name?: string }[];
        error?: string;
      };
      const poolNameMap: Record<string, string> = {};
      for (const pool of Array.isArray(poolsData.pools) ? poolsData.pools : []) {
        poolNameMap[pool.ticker_type] = pool.name ?? pool.ticker_type;
      }
      setPoolNameById(poolNameMap);
      if (!accountsResp.ok || !Array.isArray(accountsData)) {
        throw new Error(Array.isArray(accountsData) ? "계좌 목록을 불러오지 못했습니다." : accountsData.error ?? "계좌 목록을 불러오지 못했습니다.");
      }
      setAccounts(accountsData);
      // 모든 계좌가 대상. account_id 미지정이면 기억된 계좌(전체 목록에 있으면) → 없으면 첫 계좌.
      const allIds = accountsData.map((a) => a.account_id);
      const remembered = readRememberedMomentumEtfAccountId();
      const target = accountId ?? (remembered && allIds.includes(remembered) ? remembered : allIds[0]);
      if (!target) {
        throw new Error("등록된 계좌가 없습니다.");
      }
      writeRememberedMomentumEtfAccountId(target);
      const acctList = Array.isArray(acctSettingsData.accounts) ? acctSettingsData.accounts : [];
      const memoMap: Record<string, string> = {};
      const basisMap: Record<string, AccountTopPickBasis> = {};
      for (const account of acctList) {
        memoMap[account.account_id] = String(account.memo ?? "").trim();
        basisMap[account.account_id] = {
          startAmount: account.top_pick_start_amount_manwon ?? null,
          startDate: account.top_pick_start_date ?? null,
        };
      }
      setAccountMemoById(memoMap);
      setAccountTopPickBasisById(basisMap);
      const targetAcct = acctList.find((a) => a.account_id === target);
      setAccountTickerTypes(targetAcct?.ticker_types ?? []);
      // 2) 선택 계좌 설정 로드
      const settingsResp = await fetch(
        `/api/top-pick-settings?account_id=${encodeURIComponent(target)}`,
        { cache: "no-store" },
      );
      const data = (await settingsResp.json()) as TopPickSettingsPayload;
      if (!settingsResp.ok || data.error) {
        throw new Error(data.error ?? "탑픽 설정을 불러오지 못했습니다.");
      }
      setSelectedAccount(target);
      setTickers(buildTickerRows(data.tickers));
      setSettings(data.settings ?? null);
      setUpdatedAt(data.updated_at ?? null);
      setApprovedAt(data.approved_at ?? null);
      setApprovedWeights(data.approved_weights ?? null);
      setPreview(null);
      const backtestSettings = data.backtest_settings ?? DEFAULT_BACKTEST_SETTINGS;
      setBacktestPeriodMonths(String(backtestSettings.months ?? DEFAULT_BACKTEST_SETTINGS.months));
      setBacktestRebalanceMode(backtestSettings.rebalance ?? DEFAULT_BACKTEST_SETTINGS.rebalance);
      setBacktestInitialAmountManwon(String(backtestSettings.initial_amount_manwon ?? DEFAULT_BACKTEST_SETTINGS.initial_amount_manwon));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "탑픽 설정을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    void loadSettings();
  }, [loadSettings]);

  useEffect(() => {
    if (loading || !settings || !reservePoolId) {
      setReserveCandidates([]);
      setReserveError(null);
      setRankMetaByTicker({});
      return;
    }

    const abortController = new AbortController();
    const loadReserveCandidates = async () => {
      try {
        setReserveLoading(true);
        setReserveError(null);
        const params = new URLSearchParams({
          ticker_type: reservePoolId,
        });
        const response = await fetch(`/api/rank?${params.toString()}`, {
          cache: "no-store",
          signal: abortController.signal,
        });
        const payload = (await response.json()) as RankResponse;
        if (!response.ok || payload.error) {
          throw new Error(payload.error ?? "예비 종목을 불러오지 못했습니다.");
        }
        const rankMeta: Record<string, { trend_pct: number | null; alignment?: string | null }> = {};
        for (const row of payload.rows ?? []) {
          if (row.티커) {
            rankMeta[normalizeTicker(row.티커)] = { trend_pct: row.추세, alignment: row.배열 ?? null };
          }
        }
        setRankMetaByTicker(rankMeta);
        const candidates = (payload.rows ?? [])
          .filter((row) => row.티커 && !row.exclude_from_ranking && !row.is_benchmark)
          .filter((row) => !selectedTickerSet.has(normalizeTicker(row.티커)))
          .sort((a, b) => (b.추세 ?? Number.NEGATIVE_INFINITY) - (a.추세 ?? Number.NEGATIVE_INFINITY))
          .map((row) => ({
            ticker: row.티커,
            name: row.종목명,
            ticker_type: row.source_ticker_type ?? reservePoolId,
            bucket: row.bucket,
            trend_pct: row.추세,
            alignment: row.배열 ?? null,
          }));
        setReserveCandidates(candidates);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          return;
        }
        setReserveCandidates([]);
        setRankMetaByTicker({});
        setReserveError(err instanceof Error ? err.message : "예비 종목을 불러오지 못했습니다.");
      } finally {
        setReserveLoading(false);
      }
    };

    void loadReserveCandidates();
    return () => abortController.abort();
  }, [accountTickerTypes, loading, reservePoolId, selectedTickerSet, settings]);

  // 계좌 명칭 앞 숫자 오름차순(예: "3. 연금저축" < "5. 호주 연금"). 숫자 없으면 뒤로.
  const accountNumberOf = useCallback(
    (accountId: string) => {
      const name = accounts.find((a) => a.account_id === accountId)?.name ?? accountId;
      const matched = name.match(/^\s*(\d+)/);
      return matched ? parseInt(matched[1], 10) : Number.MAX_SAFE_INTEGER;
    },
    [accounts],
  );
  // 모든 계좌를 셀렉터에 노출한다(명칭 앞 숫자 오름차순). 저장 전 계좌는 초기(빈) 상태로 로드된다.
  const sortedAllAccounts = useMemo(
    () => [...accounts].sort((a, b) => accountNumberOf(a.account_id) - accountNumberOf(b.account_id)),
    [accounts, accountNumberOf],
  );

  // 선택 계좌의 통화(환종). 호주 계좌의 ASX 표시 같은 계좌별 표시 규칙에 사용한다.
  const selectedCurrency = useMemo(
    () => accounts.find((a) => a.account_id === selectedAccount)?.currency || "KRW",
    [accounts, selectedAccount],
  );
  const selectedAccountMemo = useMemo(
    () => (selectedAccount ? accountMemoById[selectedAccount] ?? "" : ""),
    [accountMemoById, selectedAccount],
  );
  const selectedAccountTopPickBasis = useMemo(
    () => (selectedAccount ? accountTopPickBasisById[selectedAccount] ?? { startAmount: null, startDate: null } : { startAmount: null, startDate: null }),
    [accountTopPickBasisById, selectedAccount],
  );
  // 호주(AUD) 계좌의 티커는 미국 동일 심볼과 구분하기 위해 ASX: 접두사로 표시한다.
  const isAusAccount = selectedCurrency === "AUD";
  const displayTickerOf = useCallback(
    (ticker: string) =>
      isAusAccount && ticker && ticker !== "__CASH__" && !ticker.startsWith("ASX:") ? `ASX:${ticker}` : ticker,
    [isAusAccount],
  );

  const resolveTicker = async (index: number) => {
    const raw = normalizeTicker(tickers[index]?.ticker ?? "");
    if (!raw) {
      toast.error("티커를 입력해주세요.");
      return;
    }
    const duplicated = tickers.some(
      (item, itemIndex) => itemIndex !== index && normalizeTicker(item.ticker) === raw && item.name,
    );
    if (duplicated) {
      toast.error("이미 등록된 종목입니다.");
      return;
    }
    try {
      // 계좌에 연결된 종목풀이 있으면 그 범위 안에서만 조회한다.
      const poolFilter = accountTickerTypes.length
        ? `&ticker_types=${encodeURIComponent(accountTickerTypes.join(","))}`
        : "";
      const resp = await fetch(
        `/api/ticker-resolve?ticker=${encodeURIComponent(raw)}${poolFilter}`,
      );
      const data = (await resp.json()) as {
        ticker?: string;
        name?: string;
        ticker_type?: string;
        country_code?: string;
        is_etf?: boolean;
        bucket?: number;
        error?: string;
        detail?: string;
      };
      if (!resp.ok || data.error || !data.name) {
        throw new Error(data.error ?? data.detail ?? "존재하지 않는 티커입니다.");
      }
      const resolvedTicker = data.ticker ?? stripKorPrefix(raw);
      const resolvedItem: TopPickTicker = {
        ticker: resolvedTicker,
        name: data.name,
        ticker_type: data.ticker_type ?? (isKorTickerInput(raw) ? "kor_kr" : undefined),
        country_code: data.country_code ?? (isKorTickerInput(raw) ? "kor" : undefined),
        is_etf: data.is_etf ?? (isKorTickerInput(raw) ? true : undefined),
        bucket: data.bucket,
      };
      setTickers((current) =>
        current.map((item, itemIndex) =>
          itemIndex === index ? resolvedItem : item,
        ),
      );
      setPreview(null);
      toast.success(`${data.name}(${resolvedTicker}) 확인 완료`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "티커 조회에 실패했습니다.");
    }
  };

  const saveSettings = async () => {
    if (validTickers.length === 0) {
      toast.error("확인된 종목이 1개 이상 필요합니다.");
      return;
    }
    if (!settings || !settings.ACCOUNT_ID) {
      toast.error("탑픽 적용 계좌를 선택해주세요.");
      return;
    }
    if (!Number.isInteger(Number(backtestInitialAmountManwon)) || Number(backtestInitialAmountManwon) <= 0) {
      toast.error("백테스트 최초 금액(만원)은 1 이상의 정수여야 합니다.");
      return;
    }
    const requestSettings = buildSettingsForRequest(settings, validTickers.length, selectedAccountTopPickBasis);
    if (!requestSettings) {
      toast.error("탑픽 설정을 불러오지 못했습니다.");
      return;
    }
    try {
      setSaving(true);
      const resp = await fetch("/api/top-pick-settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: validTickers,
          weight_mode: "variable",
          settings: requestSettings,
          account_id: selectedAccount,
          backtest_settings: {
            months: Number(backtestPeriodMonths),
            rebalance: backtestRebalanceMode,
            initial_amount_manwon: Number(backtestInitialAmountManwon),
          },
        }),
      });
      const data = (await resp.json()) as TopPickSettingsPayload;
      if (!resp.ok || data.error) {
        throw new Error(data.error ?? "탑픽 설정 저장에 실패했습니다.");
      }
      setTickers(buildTickerRows(data.tickers));
      setSettings(data.settings ?? null);
      setUpdatedAt(data.updated_at ?? null);
      setApprovedAt(data.approved_at ?? approvedAt);
      setApprovedWeights(data.approved_weights ?? approvedWeights);
      setPreview(null);
      const backtestSettings = data.backtest_settings ?? DEFAULT_BACKTEST_SETTINGS;
      setBacktestPeriodMonths(String(backtestSettings.months ?? DEFAULT_BACKTEST_SETTINGS.months));
      setBacktestRebalanceMode(backtestSettings.rebalance ?? DEFAULT_BACKTEST_SETTINGS.rebalance);
      setBacktestInitialAmountManwon(String(backtestSettings.initial_amount_manwon ?? DEFAULT_BACKTEST_SETTINGS.initial_amount_manwon));
      toast.success("탑픽 설정 저장 완료");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "탑픽 설정 저장에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  };

  const runPreview = useCallback(async () => {
    if (validTickers.length < 3) {
      toast.error("비중 계산에는 확인된 종목이 3개 이상 필요합니다.");
      return;
    }
    if (!settings || !settings.ACCOUNT_ID) {
      toast.error("탑픽 적용 계좌를 선택해주세요.");
      return;
    }
    const requestSettings = buildSettingsForRequest(settings, validTickers.length, selectedAccountTopPickBasis);
    if (!requestSettings) {
      toast.error("탑픽 설정을 불러오지 못했습니다.");
      return;
    }
    try {
      setRunning(true);
      setPreviewMode("weights");
      const resp = await fetch("/api/top-pick-settings/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: validTickers,
          settings: requestSettings,
          weight_mode: "variable",
          backtest_settings: {
            months: Number(backtestPeriodMonths),
            rebalance: backtestRebalanceMode,
            initial_amount_manwon: Number(backtestInitialAmountManwon),
          },
        }),
      });
      const data = (await resp.json()) as TopPickWeightPreview;
      if (!resp.ok || data.error) {
        throw new Error(data.error ?? "탑픽 비중 계산에 실패했습니다.");
      }
      setPreview(data);
      setPreviewMode("weights");
      toast.success("탑픽 비중 계산 완료");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "탑픽 비중 계산에 실패했습니다.");
    } finally {
      setRunning(false);
    }
  }, [backtestInitialAmountManwon, backtestPeriodMonths, backtestRebalanceMode, selectedAccountTopPickBasis, settings, toast, validTickers]);

  useEffect(() => {
    if (
      !loading &&
      !approvedWeights &&
      validTickers.length >= 3 &&
      settings?.ACCOUNT_ID &&
      !autoRunStartedRef.current
    ) {
      autoRunStartedRef.current = true;
      void runPreview();
    }
  }, [loading, approvedWeights, validTickers, settings?.ACCOUNT_ID, runPreview]);

  const approvePreview = async () => {
    if (!preview) {
      toast.error("먼저 실행해서 계산 결과를 확인해주세요.");
      return;
    }
    const requestSettings = buildSettingsForRequest(settings, validTickers.length, selectedAccountTopPickBasis);
    if (!requestSettings) {
      toast.error("탑픽 설정을 불러오지 못했습니다.");
      return;
    }
    try {
      setApproving(true);
      const resp = await fetch("/api/top-pick-settings/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: validTickers,
          settings: requestSettings,
          account_id: selectedAccount,
          weight_mode: "variable",
          backtest_settings: {
            months: Number(backtestPeriodMonths),
            rebalance: backtestRebalanceMode,
            initial_amount_manwon: Number(backtestInitialAmountManwon),
          },
        }),
      });
      const data = (await resp.json()) as TopPickWeightPreview & { approved_at?: string; tickers?: TopPickTicker[] };
      if (!resp.ok || data.error) {
        throw new Error(data.error ?? "탑픽 비중 확인 저장에 실패했습니다.");
      }
      setApprovedWeights(data);
      setPreview(data);
      setApprovedAt(data.approved_at ?? null);
      if (data.tickers && data.tickers.length > 0) {
        setTickers(buildTickerRows(data.tickers));
      }
      toast.success("탑픽 비중 확인 저장 완료");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "탑픽 비중 확인 저장에 실패했습니다.");
    } finally {
      setApproving(false);
    }
  };

  const parseUtcDate = (val: string | null | undefined): string => {
    if (!val) return "-";
    const dateStr = val.endsWith("Z") || val.includes("+") ? val : `${val}Z`;
    return new Date(dateStr).toLocaleString("ko-KR");
  };

  const updatedLabel = parseUtcDate(updatedAt);
  const approvedLabel = parseUtcDate(approvedAt);
  const formatScoreSettingLabel = (source?: Partial<TopPickSettings>) => {
    const mainMaDays = source?.MAIN_MA_DAYS ?? "-";
    const poolName = source?.POOL_NAME ?? source?.POOL_TICKER_TYPE;
    return `${poolName ? `${poolName} · ` : ""}SMA 메인 ${mainMaDays}일 · 추세선 위 투자`;
  };

  const runBacktest = async () => {
    if (validTickers.length < 3) {
      toast.error("탑픽 백테스트에는 확인된 편입 ETF가 3개 이상 필요합니다.");
      return;
    }
    if (!Number.isInteger(Number(backtestInitialAmountManwon)) || Number(backtestInitialAmountManwon) <= 0) {
      toast.error("백테스트 최초 금액(만원)은 1 이상의 정수여야 합니다.");
      return;
    }
    const requestSettings = buildSettingsForRequest(settings, validTickers.length, selectedAccountTopPickBasis);
    if (!requestSettings) {
      toast.error("탑픽 설정을 불러오지 못했습니다.");
      return;
    }
    try {
      setBacktestRunning(true);
      setPreviewMode("backtest");
      setBacktestResult(null);
      const resp = await fetch("/api/top-pick-settings/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: validTickers,
          settings: requestSettings,
          weight_mode: "variable",
          backtest_settings: {
            months: Number(backtestPeriodMonths),
            rebalance: backtestRebalanceMode,
            initial_amount_manwon: Number(backtestInitialAmountManwon),
          },
        }),
      });
      const data = (await resp.json()) as LabResult & { detail?: string };
      if (!resp.ok || data.error) {
        throw new Error(data.error ?? data.detail ?? "탑픽 백테스트 실행에 실패했습니다.");
      }
      setBacktestResult(data);
      toast.success("탑픽 백테스트 실행 완료");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "탑픽 백테스트 실행에 실패했습니다.");
    } finally {
      setBacktestRunning(false);
    }
  };

  const comparisonColumns = useMemo<ColDef<TopPickWeightComparisonRow>[]>(
    () => [
      {
        field: "ticker",
        headerName: "티커",
        width: 110,
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: string | null | undefined }) =>
          params.value === "__CASH__" ? "CASH" : params.value ? displayTickerOf(params.value) : "-",
      },
      {
        field: "name",
        headerName: "종목명",
        minWidth: 240,
        flex: 1,
        cellClass: "topPickNameCell",
        cellRenderer: (params: { value: string | null | undefined }) => {
          const name = params.value || "-";
          return <span className="topPickNameCellText" title={name}>{name}</span>;
        },
      },
      {
        field: "daily_change_pct",
        headerName: "금일(%)",
        width: 100,
        type: "rightAligned",
        cellRenderer: renderReturnPctCell,
      },
      {
        field: "return_1m_pct",
        headerName: "1개월",
        width: 96,
        type: "rightAligned",
        cellRenderer: renderReturnPctCell,
      },
      {
        field: "return_3m_pct",
        headerName: "3개월",
        width: 96,
        type: "rightAligned",
        cellRenderer: renderReturnPctCell,
      },
      {
        field: "return_6m_pct",
        headerName: "6개월",
        width: 96,
        type: "rightAligned",
        cellRenderer: renderReturnPctCell,
      },
      {
        field: "return_12m_pct",
        headerName: "12개월",
        width: 100,
        type: "rightAligned",
        cellRenderer: renderReturnPctCell,
      },
      {
        field: "trend_pct",
        headerName: "추세",
        width: 112,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatNumber(params.value),
      },
      {
        field: "mdd_pct",
        headerName: `MDD(${backtestPeriodMonths || "-"}개월)`,
        width: 112,
        type: "rightAligned",
        cellRenderer: renderReturnPctCell,
      },
      {
        field: "sortino",
        headerName: `Sortino(${backtestPeriodMonths || "-"}개월)`,
        width: 126,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatNumber(params.value),
      },
      {
        field: "score",
        headerName: "점수",
        width: 96,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) => formatNumber(params.value),
      },
      {
        field: "approved_weight_pct",
        headerName: "저장 비중",
        width: 110,
        type: "rightAligned",
        cellStyle: { fontWeight: 800 },
        cellRenderer: (params: { value: number | null | undefined }) => formatWeightPct(params.value),
      },
      {
        field: "calculated_weight_pct",
        headerName: "계산 비중",
        width: 110,
        type: "rightAligned",
        cellStyle: { fontWeight: 800 },
        cellRenderer: (params: { value: number | null | undefined }) => formatWeightPct(params.value),
      },
      {
        field: "weight_diff_pct",
        headerName: "비중 차이",
        width: 110,
        type: "rightAligned",
        cellStyle: { fontWeight: 800 },
        cellRenderer: (params: { value: number | null | undefined }) => {
          if (params.value == null) return "-";
          const prefix = params.value > 0 ? "+" : "";
          return <span style={{ color: signedColor(params.value) }}>{prefix}{params.value.toFixed(1)}%p</span>;
        },
      },
    ],
    [backtestPeriodMonths, displayTickerOf],
  );

  const comparisonGridOptions = useMemo<GridOptions<TopPickWeightComparisonRow>>(
    () => ({
      domLayout: "autoHeight",
      suppressMovableColumns: true,
      getRowStyle: (params) => {
        if (params.data?.weight_diff_pct && params.data.weight_diff_pct !== 0) {
          return { backgroundColor: "rgba(249, 115, 22, 0.08)" };
        }
        return undefined;
      },
    }),
    [],
  );

  const comparisonRows = useMemo<TopPickWeightComparisonRow[]>(() => {
    const approvedRows = approvedWeights?.rows ?? [];
    const calculatedRows = preview?.rows ?? [];
    const approvedByTicker = new Map(approvedRows.map((row) => [row.ticker, row]));
    const calculatedByTicker = new Map(calculatedRows.map((row) => [row.ticker, row]));
    const tickersInOrder = [
      ...calculatedRows.map((row) => row.ticker),
      ...approvedRows.map((row) => row.ticker),
    ].filter((ticker, index, all) => all.indexOf(ticker) === index);

    return tickersInOrder.map((ticker) => {
      const approvedRow = approvedByTicker.get(ticker);
      const calculatedRow = calculatedByTicker.get(ticker);
      const source = calculatedRow ?? approvedRow;
      const approvedWeight = approvedRow?.target_weight_pct ?? null;
      const calculatedWeight = calculatedRow?.target_weight_pct ?? null;
      const difference = approvedWeight == null || calculatedWeight == null
        ? null
        : Number((calculatedWeight - approvedWeight).toFixed(1));
      return {
        ...source!,
        approved_weight_pct: approvedWeight,
        calculated_weight_pct: calculatedWeight,
        weight_diff_pct: difference,
      };
    });
  }, [approvedWeights?.rows, preview?.rows]);

  const backtestPositionColumns = useMemo<ColDef<LabPosition>[]>(
    () => [
      {
        field: "bucket",
        headerName: "버킷",
        width: 108,
        minWidth: 108,
        valueGetter: (params) => getBucketName(params.data?.bucket),
        cellClass: (params) => getBucketCellClass(params.data?.bucket),
      },
      {
        field: "ticker",
        headerName: "티커",
        width: 95,
        minWidth: 95,
        cellRenderer: (params: { data?: LabPosition }) => {
          const row = params.data;
          if (!row) return "-";
          if (row.ticker === "__CASH__") return <span style={{ fontWeight: 800 }}>현금</span>;
          const display = displayTickerOf(row.ticker);
          return <TickerDetailLink ticker={display} displayTicker={display} />;
        },
      },
      {
        field: "name",
        headerName: "종목명",
        minWidth: 180,
        flex: 1,
        cellClass: "topPickNameCell",
        valueGetter: (params) => {
          const row = params.data;
          if (!row) return "-";
          return row.ticker === "__CASH__" ? "" : row.name ?? row.ticker;
        },
        tooltipValueGetter: (params) => String(params.value ?? ""),
        cellRenderer: (params: { value: string | null | undefined }) => {
          const name = params.value || "-";
          return <span className="topPickNameCellText" title={name}>{name}</span>;
        },
        cellStyle: (params) => ({
          color: getTopPickWeightColor(backtestResult?.weight_items, params.data?.ticker ?? ""),
          fontWeight: 800,
        }),
      },
      {
        field: "buy_date",
        headerName: "매수일",
        width: 110,
        cellRenderer: (params: { data?: LabPosition; value?: string }) => {
          const suffix = params.data?.late_entry ? " ↩" : "";
          return `${params.value ?? "-"}${suffix}`;
        },
      },
      {
        field: "buy_price",
        headerName: "초기가",
        width: 105,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) =>
          params.value == null ? "-" : new Intl.NumberFormat("ko-KR").format(params.value),
      },
      {
        field: "last_price",
        headerName: "현재가",
        width: 105,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) =>
          params.value == null ? "-" : new Intl.NumberFormat("ko-KR").format(params.value),
      },
      {
        field: "return_pct",
        headerName: `수익률(${backtestResult?.months ?? "-"}개월)`,
        width: 150,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) => (
          <span style={{ color: params.value == null ? "#475569" : signedColor(params.value) }}>{formatReturnPct(params.value)}</span>
        ),
      },
      {
        field: "mdd_pct",
        headerName: `MDD(${backtestResult?.months ?? "-"}개월)`,
        width: 150,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) =>
          params.value == null ? "-" : <span style={{ color: signedColor(params.value) }}>{formatReturnPct(params.value)}</span>,
      },
      {
        field: "sortino",
        headerName: `Sortino(${backtestResult?.months ?? "-"}개월)`,
        width: 150,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatNumber(params.value),
      },
      {
        field: "profit",
        headerName: "수익금",
        width: 130,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) =>
          params.value == null ? "-" : <span style={{ color: signedColor(params.value) }}>{formatKrw(params.value)}</span>,
      },
      {
        field: "profit_contribution_pct",
        headerName: "수익기여",
        width: 100,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) =>
          params.value == null ? "-" : <span style={{ color: signedColor(params.value) }}>{params.value.toFixed(1)}%</span>,
      },
      {
        field: "min_weight",
        headerName: "최저 비중",
        width: 100,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) =>
          params.value == null ? "-" : `${formatNumber(params.value, 1)}%`,
      },
      {
        field: "max_weight",
        headerName: "최대 비중",
        width: 100,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) =>
          params.value == null ? "-" : `${formatNumber(params.value, 1)}%`,
      },
    ],
    [backtestResult?.months, backtestResult?.weight_items, displayTickerOf],
  );

  const backtestGridOptions = useMemo<GridOptions<LabPosition>>(
    () => ({
      domLayout: "autoHeight",
      suppressMovableColumns: true,
    }),
    [],
  );

  const backtestPositionRows = useMemo<LabPosition[]>(() => {
    if (!backtestResult) return [];
    const lastWeightRow = backtestResult.weight_history?.[backtestResult.weight_history.length - 1];
    const cashValue = Number(lastWeightRow?.__CASH__ ?? 0);
    const totalProfit = backtestResult.final_value - backtestResult.initial_capital;
    const withContribution = (position: LabPosition): LabPosition => ({
      ...position,
      profit_contribution_pct: totalProfit === 0 ? null : (position.profit / totalProfit) * 100,
    });
    return [
      withContribution({
        ticker: "__CASH__",
        name: "현금",
        bucket: 5,
        buy_date: backtestResult.buy_date,
        late_entry: false,
        shares: 0,
        buy_price: null,
        last_price: null,
        return_pct: null,
        mdd_pct: null,
        mdd_start: "",
        mdd_end: "",
        sortino: null,
        profit: 0,
        value: Number.isFinite(cashValue) ? cashValue : 0,
        min_weight: backtestResult.cash_min_weight ?? 0,
        max_weight: backtestResult.cash_max_weight ?? 0,
      }),
      ...backtestResult.positions.map(withContribution),
    ];
  }, [backtestResult]);

  const summaryChip = (label: string, value: string, color?: string) => (
    <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 88 }}>
      <span style={{ color: "var(--text-muted)", fontSize: "0.78rem", fontWeight: 600 }}>{label}</span>
      <span style={{ fontWeight: 800, fontSize: "1rem", color: color ?? "#182433" }}>{value}</span>
    </div>
  );

  const addReserveCandidate = (candidate: TopPickReserveCandidate) => {
    if (validTickers.length >= MAX_TOP_PICK_SELECTED) {
      toast.error(`선택 종목은 최대 ${MAX_TOP_PICK_SELECTED}개까지 등록할 수 있습니다.`);
      return;
    }
    if (selectedTickerSet.has(normalizeTicker(candidate.ticker))) {
      toast.error("이미 선택된 종목입니다.");
      return;
    }
    setTickers((current) => [
      ...buildTickerRows(current).filter((item) => item.ticker.trim() || item.name),
      {
        ticker: candidate.ticker,
        name: candidate.name,
        ticker_type: candidate.ticker_type,
        country_code: candidate.country_code,
        is_etf: candidate.is_etf,
        bucket: candidate.bucket,
        alignment: candidate.alignment,
      },
    ]);
    setPreview(null);
  };

  const selectedGridRows = useMemo<TopPickSelectedGridRow[]>(
    () =>
      tickers.map((item, index) => ({
        ...item,
        row_index: index + 1,
        trend_pct: rankMetaByTicker[normalizeTicker(item.ticker)]?.trend_pct ?? null,
        alignment: rankMetaByTicker[normalizeTicker(item.ticker)]?.alignment ?? item.alignment ?? null,
        is_confirmed: !!item.name,
      })),
    [rankMetaByTicker, tickers],
  );

  const reserveGridRows = useMemo<TopPickReserveGridRow[]>(
    () => reserveCandidates.map((item, index) => ({ ...item, row_index: index + 1 })),
    [reserveCandidates],
  );

  const selectedGridColumnDefs = useMemo<ColDef<TopPickSelectedGridRow>[]>(
    () => [
      {
        colId: "bucket",
        headerName: "버킷",
        minWidth: 108,
        width: 108,
        pinned: "left",
        sortable: true,
        comparator: (_a, _b, nodeA, nodeB) => Number(nodeA.data?.bucket ?? 0) - Number(nodeB.data?.bucket ?? 0),
        valueGetter: (params) => getBucketName(params.data?.bucket),
        cellClass: (params) => getBucketCellClass(params.data?.bucket),
        cellRenderer: (params: { data?: TopPickSelectedGridRow }) => <span>{getBucketName(params.data?.bucket)}</span>,
      },
      {
        field: "ticker",
        headerName: "티커",
        minWidth: 95,
        width: 95,
        pinned: "left",
        sortable: false,
        cellRenderer: (params: { data?: TopPickSelectedGridRow; value?: string }) => {
          const row = params.data;
          if (!row) return "-";
          const index = row.row_index - 1;
          if (row.is_confirmed) {
            const display = displayTickerOf(params.value ?? "");
            return <TickerDetailLink ticker={display} displayTicker={display} />;
          }
          return (
            <input
              className="topPickGridTickerInput"
              placeholder="티커"
              value={row.ticker}
              onChange={(event) => {
                setTickers((current) =>
                  current.map((currentItem, itemIndex) =>
                    itemIndex === index ? { ticker: event.target.value } : currentItem,
                  ),
                );
                setPreview(null);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  void resolveTicker(index);
                }
              }}
            />
          );
        },
      },
      {
        field: "name",
        headerName: "종목명",
        minWidth: 249,
        flex: 1.05,
        sortable: false,
        cellRenderer: (params: { value?: string }) => (
          <span className="rankNameCellText" title={params.value ?? ""}>
            {params.value ? renderNameWithLeverageHighlight(params.value) : "티커 입력 후 확인"}
          </span>
        ),
      },
      {
        field: "trend_pct",
        headerName: "추세",
        minWidth: 86,
        width: 86,
        sortable: true,
        sort: "desc",
        type: "rightAligned",
        cellRenderer: renderReturnPctCell,
      },
      {
        field: "alignment",
        headerName: "배열",
        minWidth: 78,
        width: 78,
        sortable: true,
        cellRenderer: renderAlignmentCell,
      },
      {
        colId: "action",
        headerName: "",
        width: 86,
        sortable: false,
        cellRenderer: (params: { data?: TopPickSelectedGridRow }) => {
          const row = params.data;
          if (!row) return null;
          const index = row.row_index - 1;
          return row.is_confirmed ? (
            <button
              type="button"
              className="btn btn-sm btn-outline-danger topPickGridActionButton"
              onClick={() => {
                setTickers((current) =>
                  buildTickerRows(
                    current.map((currentItem, itemIndex) =>
                      itemIndex === index ? { ticker: "" } : currentItem,
                    ),
                  ),
                );
                setPreview(null);
              }}
            >
              비우기
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-sm btn-outline-secondary topPickGridActionButton"
              disabled={!row.ticker.trim()}
              onClick={() => void resolveTicker(index)}
            >
              확인
            </button>
          );
        },
      },
    ],
    [displayTickerOf, resolveTicker],
  );

  const reserveGridColumnDefs = useMemo<ColDef<TopPickReserveGridRow>[]>(
    () => [
      {
        colId: "bucket",
        headerName: "버킷",
        minWidth: 108,
        width: 108,
        pinned: "left",
        sortable: true,
        valueGetter: (params) => getBucketName(params.data?.bucket),
        cellClass: (params) => getBucketCellClass(params.data?.bucket),
        cellRenderer: (params: { data?: TopPickReserveGridRow }) => <span>{getBucketName(params.data?.bucket)}</span>,
      },
      {
        field: "ticker",
        headerName: "티커",
        minWidth: 95,
        width: 95,
        pinned: "left",
        cellRenderer: (params: { value?: string }) => {
          const display = displayTickerOf(params.value ?? "");
          return <TickerDetailLink ticker={display} displayTicker={display} />;
        },
      },
      {
        field: "name",
        headerName: "종목명",
        minWidth: 249,
        flex: 1.05,
        cellRenderer: (params: { value?: string }) => (
          <span className="rankNameCellText" title={params.value ?? ""}>
            {params.value ? renderNameWithLeverageHighlight(params.value) : "-"}
          </span>
        ),
      },
      { field: "trend_pct", headerName: "추세", minWidth: 86, width: 86, type: "rightAligned", cellRenderer: renderReturnPctCell, sort: "desc" },
      {
        field: "alignment",
        headerName: "배열",
        minWidth: 78,
        width: 78,
        sortable: true,
        cellRenderer: renderAlignmentCell,
      },
      {
        colId: "action",
        headerName: "",
        width: 76,
        sortable: false,
        cellRenderer: (params: { data?: TopPickReserveGridRow }) => {
          const row = params.data;
          return row ? (
            <button type="button" className="btn btn-sm btn-outline-primary topPickGridActionButton" onClick={() => addReserveCandidate(row)}>
              선택
            </button>
          ) : null;
        },
      },
    ],
    [displayTickerOf, selectedTickerSet, validTickers.length],
  );

  const symbolGridOptions = useMemo<GridOptions<TopPickSelectedGridRow | TopPickReserveGridRow>>(
    () => ({
      domLayout: "autoHeight",
      headerHeight: 34,
      suppressMovableColumns: true,
      suppressHorizontalScroll: false,
    }),
    [],
  );
  const reserveGridOptions = useMemo<GridOptions<TopPickReserveGridRow>>(
    () => ({
      domLayout: "autoHeight",
      headerHeight: 34,
      suppressMovableColumns: true,
      suppressHorizontalScroll: false,
    }),
    [],
  );

  return (
    <PageFrame
      title="탑픽 설정"
      titleRight={
        <div className="appHeaderMetrics rankToolbarMeta">
          <div className="appHeaderMetric">
            <span>마지막 저장:</span>
            <span className="appHeaderMetricValue">{updatedLabel}</span>
          </div>
        </div>
      }
    >
      <div className="appPageStack">
        {/* 메인 헤더 — 카드 내부(card-header). 왼쪽은 이후 계좌 셀렉터 등 추가 예정, 오른쪽은 설정 저장 */}
        <div className="card appCard">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <label className="appLabeledField" style={{ minWidth: 200 }}>
                  <span className="appLabeledFieldLabel">계좌</span>
                  <select
                    className="form-select form-select-sm"
                    value={selectedAccount ?? ""}
                    disabled={loading || sortedAllAccounts.length === 0}
                    onChange={(event) => void loadSettings(event.target.value)}
                  >
                    {sortedAllAccounts.length === 0 ? (
                      <option value="">계좌 불러오는 중...</option>
                    ) : (
                      sortedAllAccounts.map((a) => (
                        <option key={a.account_id} value={a.account_id}>
                          {a.name}
                        </option>
                      ))
                    )}
                  </select>
                </label>
                <label className="appLabeledField topPickMemoField">
                  <span className="appLabeledFieldLabel">메모</span>
                  <span className="topPickAccountMemoText" title={selectedAccountMemo || "-"}>
                    {selectedAccountMemo || "-"}
                  </span>
                </label>
              </div>
              <div className="appMainHeaderRight" style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <button type="button" className="btn btn-sm btn-outline-primary" disabled={running} onClick={() => void runPreview()}>
                  {running ? "비중 계산 중..." : "비중 계산"}
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-success"
                  disabled={previewMode !== "weights" || !preview || approving}
                  onClick={() => void approvePreview()}
                >
                  {approving ? "확인 저장 중..." : "확인 저장"}
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-outline-dark"
                  disabled={
                    backtestRunning ||
                    validTickers.length < 3
                  }
                  onClick={() => void runBacktest()}
                >
                  {backtestRunning ? "백테스트 중..." : "백테스트"}
                </button>
                <button type="button" className="btn btn-sm btn-primary" disabled={saving} onClick={() => void saveSettings()}>
                  {saving ? "저장 중..." : "설정 저장"}
                </button>
              </div>
            </div>
          </div>
        </div>
        <div className="topPickSettingsTopLayout">
          <div className="card appCard">
            <div className="card-body">
              <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 12 }}>백테스트 설정</h2>
              <div style={{ color: "#4b5563", fontSize: "0.83rem", marginBottom: 12, lineHeight: "1.4" }}>
                비중 계산은 계좌에 연결된 종목풀의 메인 이평선을 사용합니다. {settings ? formatScoreSettingLabel(settings) : ""}
              </div>
              <div className="topPickBacktestGrid">
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">최초 금액(만원)</span>
                  <input
                    type="number"
                    className="form-control form-control-sm"
                    min={1}
                    value={backtestInitialAmountManwon}
                    onChange={(event) => {
                      setBacktestInitialAmountManwon(event.target.value);
                      setBacktestResult(null);
                    }}
                  />
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">기간(개월)</span>
                  <select
                    className="form-select form-select-sm"
                    value={backtestPeriodMonths}
                    onChange={(event) => {
                      setBacktestPeriodMonths(event.target.value);
                      setBacktestResult(null);
                    }}
                  >
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 36].map((month) => (
                      <option key={month} value={month}>
                        {month}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">리밸런싱</span>
                  <select
                    className="form-select form-select-sm"
                    value={backtestRebalanceMode}
                    onChange={(event) => {
                      setBacktestRebalanceMode(event.target.value);
                      setBacktestResult(null);
                    }}
                  >
                    {BACKTEST_REBALANCE_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            </div>
          </div>
        </div>

        <div className="topPickSymbolLayout">
          <div className="card appCard">
            <div className="card-body">
              <div className="topPickSymbolHeader">
                <div>
                  <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>선택 종목</h2>
                  <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", margin: 0 }}>
                    비중 계산과 백테스트에 사용됩니다. 최대 {MAX_TOP_PICK_SELECTED}개까지 선택합니다.
                  </p>
                </div>
                <div className="topPickSymbolHeaderRight">
                  <span className="topPickSymbolCount">{validTickers.length}개</span>
                </div>
              </div>
              <AppAgGrid<TopPickSelectedGridRow>
                className="rankAgGrid"
                rowData={selectedGridRows}
                columnDefs={selectedGridColumnDefs}
                loading={loading}
                minHeight={Math.max(260, Math.min(MAX_TOP_PICK_SELECTED, Math.max(selectedGridRows.length, 4)) * 44 + 46)}
                gridOptions={symbolGridOptions as GridOptions<TopPickSelectedGridRow>}
                theme={previewGridTheme}
                getRowClass={(params) => {
                  const trend = params.data?.trend_pct;
                  return typeof trend === "number" && trend <= 0 ? "rankNegativeTrendRow" : "";
                }}
              />
            </div>
          </div>

          <div className="card appCard">
            <div className="card-body">
              <div className="topPickSymbolHeader">
                <div>
                  <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>예비 종목</h2>
                  <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", margin: 0 }}>
                    {reservePoolId ? `${poolNameById[reservePoolId] ?? reservePoolId} · 추세 순` : "연결 종목풀 없음"}
                  </p>
                </div>
              </div>
              {reserveError ? (
                <div style={{ color: "#dc2626", fontSize: "0.88rem", marginBottom: 8 }}>{reserveError}</div>
              ) : null}
              <AppAgGrid<TopPickReserveGridRow>
                className="rankAgGrid topPickReserveGrid"
                rowData={reserveGridRows}
                columnDefs={reserveGridColumnDefs}
                loading={reserveLoading}
                minHeight={0}
                gridOptions={reserveGridOptions}
                theme={previewGridTheme}
                getRowClass={(params) => {
                  const trend = params.data?.trend_pct;
                  return typeof trend === "number" && trend <= 0 ? "rankNegativeTrendRow" : "";
                }}
              />
            </div>
          </div>
        </div>
        {previewMode === "backtest" ? (
          <div className="card appCard">
            <div className="card-body">
              <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "flex-start", marginBottom: 14 }}>
                <div>
                  <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>백테스트</h2>
                  <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", margin: 0 }}>
                    백테스트 결과와 종목별 성과를 검토합니다.
                  </p>
                </div>
              </div>
              {backtestResult ? (
                <div className="topPickBacktestResultLayout">
                  <div className="topPickBacktestTopLayout">
                    <div className="topPickBacktestResultPanel">
                      {weightHoverDetail ? (
                        <div className="topPickWeightHoverOverlay">
                          <div className="topPickWeightHoverDate">{weightHoverDetail.date}</div>
                          <div className="topPickWeightHoverTitle">종목별 비중</div>
                          <div className="topPickWeightHoverRows">
                            {weightHoverDetail.items.map((item) => (
                              <div key={item.key} className="topPickWeightHoverRow">
                                <span style={{ color: item.color }}>{item.label}</span>
                                <strong>{item.weight.toFixed(1)}%</strong>
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : null}
                      <h3 style={{ fontSize: "0.98rem", fontWeight: 800, marginBottom: 4 }}>
                        백테스트 결과
                      </h3>
                      <p style={{ color: "var(--text-muted)", fontSize: "0.82rem", marginBottom: 4 }}>
                        {backtestResult.buy_date} ~ {backtestResult.end_date} ({backtestResult.months}개월)
                      </p>
                      <p style={{ color: "var(--text-muted)", fontSize: "0.82rem", marginBottom: 12 }}>
                        초기 {formatKrw(backtestResult.initial_capital)} → 최종 {formatKrw(backtestResult.final_value)} · 리밸런싱:{" "}
                        {rebalanceLabel(backtestResult.rebalance)}
                        {backtestResult.slippage ? (
                          <>
                            {" "}· 총 슬리피지 {formatKrw(backtestResult.slippage.total_cost)} (초기 대비 {backtestResult.slippage.total_cost_pct.toFixed(2)}%)
                          </>
                        ) : null}
                      </p>
                      <div className="topPickSummaryCompare">
                        <div className="topPickSummaryGroup topPickSummaryPortfolio">
                          <div className="topPickSummaryTitle">
                            <span className="topPickSummaryLine" /> 포트폴리오
                          </div>
                          <div className="topPickSummaryMetrics">
                            {summaryChip("총수익률", `${backtestResult.summary.total_return_pct.toFixed(2)}%`, signedColor(backtestResult.summary.total_return_pct))}
                            {summaryChip("CAGR", `${backtestResult.summary.cagr_pct.toFixed(2)}%`, signedColor(backtestResult.summary.cagr_pct))}
                            {summaryChip("MDD", `${backtestResult.summary.mdd_pct.toFixed(2)}%`, "#d63939")}
                            {summaryChip("Sortino", backtestResult.summary.sortino.toFixed(2))}
                          </div>
                        </div>
                        <div className="topPickSummaryGroup topPickSummaryBenchmark">
                          <div className="topPickSummaryTitle">
                            <span className="topPickSummaryLine" /> {backtestResult.benchmark.name}
                          </div>
                          <div className="topPickSummaryMetrics">
                            {summaryChip("총수익률", `${backtestResult.benchmark.summary.total_return_pct.toFixed(2)}%`, signedColor(backtestResult.benchmark.summary.total_return_pct))}
                            {summaryChip("CAGR", `${backtestResult.benchmark.summary.cagr_pct.toFixed(2)}%`, signedColor(backtestResult.benchmark.summary.cagr_pct))}
                            {summaryChip("MDD", `${backtestResult.benchmark.summary.mdd_pct.toFixed(2)}%`, "#d63939")}
                            {summaryChip("Sortino", backtestResult.benchmark.summary.sortino.toFixed(2))}
                          </div>
                        </div>
                      </div>
                      <LabChart result={backtestResult} />
                    </div>
                    <div className="topPickBacktestResultPanel">
                      <h3 style={{ fontSize: "0.98rem", fontWeight: 800, marginBottom: 4 }}>비중 변화</h3>
                      <p style={{ color: "var(--text-muted)", fontSize: "0.82rem", marginBottom: 10 }}>
                        매주 금요일 기준 가격 변동이 반영된 종목별 평가금액
                      </p>
                      <TopPickWeightHistoryChart
                        rows={backtestResult.weight_history ?? []}
                        items={backtestResult.weight_items ?? []}
                        onHoverChange={setWeightHoverDetail}
                      />
                    </div>
                  </div>
                  <div className="topPickBacktestPerformancePanel">
                    <h3 style={{ fontSize: "0.98rem", fontWeight: 800, marginBottom: 8 }}>백테스트 종목별 성과</h3>
                    {backtestResult.has_late_entry ? (
                      <p style={{ color: "#b45309", background: "rgba(245,158,11,0.08)", fontSize: "0.8rem", padding: "6px 10px", borderRadius: 6, marginBottom: 10 }}>
                        실험 시작 이후 상장된 종목은 배정 예산을 현금으로 대기시켰다가 상장일 종가에 편입합니다.
                      </p>
                    ) : null}
                    <AppAgGrid<LabPosition>
                      rowData={backtestPositionRows}
                      columnDefs={backtestPositionColumns}
                      minHeight="auto"
                      className="topPickPreviewGrid rankAgGrid"
                      theme={previewGridTheme}
                      getRowId={(params) => params.data.ticker}
                      gridOptions={backtestGridOptions}
                    />
                  </div>
                </div>
              ) : (
                <div style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>
                  {backtestRunning ? "백테스트 실행 중..." : "백테스트 버튼을 누르면 결과가 여기에 표시됩니다."}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 12, width: "100%" }}>
            <div className="card appCard">
              <div className="card-body">
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "flex-start", marginBottom: 14 }}>
                  <div>
                    <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>비중 비교</h2>
                    <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", margin: 0 }}>
                      저장된 확정 비중과 현재 설정으로 계산된 비중을 비교합니다.
                    </p>
                  </div>
                  <div style={{ color: "var(--text-muted)", fontSize: "0.82rem", whiteSpace: "nowrap" }}>마지막 확인 저장: {approvedLabel}</div>
                </div>
                {comparisonRows.length > 0 ? (
                  <>
                    <div className="topPickComparisonMeta">
                      <div>
                        <strong>저장 기준</strong>: {approvedWeights?.as_of_date ?? "-"} · {formatScoreSettingLabel(approvedWeights?.settings)}
                      </div>
                      <div>
                        <strong>계산 기준</strong>: {preview?.as_of_date ?? "-"} · {formatScoreSettingLabel(preview?.settings)}
                      </div>
                    </div>
                    <AppAgGrid<TopPickWeightComparisonRow>
                      rowData={comparisonRows}
                      columnDefs={comparisonColumns}
                      minHeight="auto"
                      className="topPickPreviewGrid"
                      theme={previewGridTheme}
                      getRowId={(params) => params.data.ticker}
                      gridOptions={comparisonGridOptions}
                    />
                    {preview?.missing_tickers && preview.missing_tickers.length > 0 && (
                      <div style={{ color: "#b45309", fontSize: "0.85rem", marginTop: 10 }}>
                        가격 캐시 누락: {preview.missing_tickers.map(displayTickerOf).join(", ")}
                      </div>
                    )}
                  </>
                ) : (
                  <div style={{ color: "var(--text-muted)", fontSize: "0.9rem", padding: "10px 0" }}>
                    저장된 비중 또는 계산된 비중 정보가 없습니다.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
      <style jsx global>{`
        .topPickSettingsTopLayout {
          display: grid;
          grid-template-columns: minmax(0, 1fr);
          gap: 12px;
          align-items: start;
        }

        .topPickMemoField {
          min-width: 260px;
          max-width: 560px;
        }

        .topPickAccountMemoText {
          display: flex;
          align-items: center;
          min-width: 0;
          min-height: 31px;
          overflow: hidden;
          color: var(--text-normal);
          font-size: 0.9rem;
          font-weight: 700;
          line-height: 1.2;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .topPickBacktestGrid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(min(100%, 170px), 1fr));
          gap: 12px 16px;
          align-items: end;
        }

        .topPickBacktestGrid > .appLabeledField {
          width: 100%;
        }

        .topPickSymbolLayout {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 12px;
          align-items: start;
        }

        .topPickSymbolHeader {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          align-items: flex-start;
          margin-bottom: 12px;
        }

        .topPickSymbolHeaderRight {
          display: flex;
          align-items: center;
          gap: 8px;
          flex-shrink: 0;
        }

        .topPickSymbolCount {
          color: var(--text-normal);
          font-size: 0.95rem;
          font-weight: 800;
        }

        .topPickReserveGrid.appAgGridWrap {
          max-height: 474px;
          overflow-y: auto;
          overflow-x: hidden;
        }

        .topPickGridTickerInput {
          width: 100%;
          min-height: 28px;
          padding: 4px 7px;
          border: 1px solid rgba(148, 163, 184, 0.45);
          border-radius: 6px;
          color: var(--text-normal);
          font-size: 0.86rem;
        }

        .topPickGridActionButton {
          min-width: 52px;
          padding-left: 7px;
          padding-right: 7px;
        }

        .topPickBacktestResultLayout {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .topPickBacktestTopLayout {
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
          gap: 12px;
          align-items: stretch;
        }

        .topPickBacktestResultPanel {
          position: relative;
          display: flex;
          flex-direction: column;
          min-width: 0;
        }

        .topPickWeightHoverOverlay {
          position: absolute;
          inset: 0;
          z-index: 5;
          display: flex;
          flex-direction: column;
          padding: 18px 20px;
          border: 1px solid rgba(148, 163, 184, 0.42);
          border-radius: 10px;
          background: rgba(15, 23, 42, 0.96);
          box-shadow: 0 18px 40px rgba(15, 23, 42, 0.22);
          color: #f8fafc;
          pointer-events: none;
        }

        .topPickWeightHoverDate {
          font-size: 1.05rem;
          font-weight: 900;
        }

        .topPickWeightHoverTitle {
          margin-top: 3px;
          color: #94a3b8;
          font-size: 0.8rem;
          font-weight: 700;
        }

        .topPickWeightHoverRows {
          display: grid;
          grid-template-columns: minmax(0, 1fr);
          gap: 6px;
          margin-top: 12px;
        }

        .topPickWeightHoverRow {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          min-width: 0;
          font-size: 0.86rem;
        }

        .topPickWeightHoverRow span {
          overflow: hidden;
          font-weight: 800;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .topPickWeightHoverRow strong {
          flex: 0 0 auto;
          color: #f8fafc;
        }

        .topPickSummaryCompare {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px;
          margin-bottom: 10px;
        }

        .topPickSummaryGroup {
          min-width: 0;
          padding: 7px 10px;
          border: 1px solid rgba(148, 163, 184, 0.22);
          border-radius: 9px;
          background: rgba(248, 250, 252, 0.62);
        }

        .topPickSummaryPortfolio {
          border-top: 3px solid #2563eb;
        }

        .topPickSummaryBenchmark {
          border-top: 3px solid #94a3b8;
        }

        .topPickSummaryTitle {
          display: flex;
          align-items: center;
          gap: 7px;
          margin-bottom: 5px;
          color: #334155;
          font-size: 0.86rem;
          font-weight: 800;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .topPickSummaryLine {
          width: 18px;
          height: 3px;
          flex: 0 0 auto;
          border-radius: 2px;
          background: #2563eb;
        }

        .topPickSummaryBenchmark .topPickSummaryLine {
          background: #94a3b8;
        }

        .topPickSummaryMetrics {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 4px 12px;
        }

        .topPickBacktestPerformancePanel {
          min-width: 0;
          width: 100%;
        }

        .topPickWeightChartWrap {
          display: flex;
          flex-direction: column;
          flex: 1;
          min-width: 0;
          min-height: 380px;
          width: 100%;
        }

        .topPickWeightChartCanvas {
          flex: 1;
          min-height: 320px;
          min-width: 0;
        }

        .topPickPreviewGrid {
          height: auto !important;
        }

        .topPickPreviewGrid .appAgGridTheme {
          height: auto;
        }

        .topPickComparisonMeta {
          display: flex;
          flex-direction: column;
          gap: 4px;
          margin-bottom: 10px;
          color: var(--text-muted);
          font-size: 0.86rem;
        }

        .topPickComparisonMeta strong {
          color: var(--text-normal);
          font-weight: 800;
        }

        @media (max-width: 900px) {
          .topPickSettingsTopLayout,
          .topPickSymbolLayout,
          .topPickBacktestGrid,
          .topPickBacktestTopLayout {
            grid-template-columns: minmax(0, 1fr);
          }

          .topPickSummaryCompare {
            grid-template-columns: minmax(0, 1fr);
          }

        }
      `}</style>
    </PageFrame>
  );
}
