"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ColDef, GridOptions } from "ag-grid-community";
import { ColorType, LineSeries, createChart } from "lightweight-charts";
import type { Time } from "lightweight-charts";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { BUCKET_COLORS } from "@/lib/bucket-theme";
import { AppAgGrid } from "../components/AppAgGrid";
import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";

type TopPickTicker = {
  ticker: string;
  name?: string;
  ticker_type?: string;
  country_code?: string;
  is_etf?: boolean;
  nickname?: string;
};

type TopPickSettingsPayload = {
  tickers?: TopPickTicker[];
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
  return_12m_pct?: number | null;
  trend_pct: number | null;
  sortino_score: number | null;
  sortino?: number | null;
  score: number | null;
  target_weight_pct: number | null;
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
  MA_TYPE: string;
  MA_MONTHS: number;
  TREND_WEIGHT_RATIO: number;
  SORTINO_MONTHS: number;
  MIN_WEIGHT: number;
  MAX_WEIGHT: number;
  CASH_MAX_WEIGHT: number;
  ACCOUNT_ID: string;
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
  value: number;
  min_weight?: number;
  max_weight?: number;
};

type LabResult = {
  months: number;
  rebalance?: string;
  buy_date: string;
  end_date: string;
  has_late_entry?: boolean;
  initial_capital: number;
  final_value: number;
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
};

const DEFAULT_SETTINGS: TopPickSettings = {
  MA_TYPE: "SMA",
  MA_MONTHS: 6,
  TREND_WEIGHT_RATIO: 100,
  SORTINO_MONTHS: 3,
  MIN_WEIGHT: 5,
  MAX_WEIGHT: 40,
  CASH_MAX_WEIGHT: 40,
  ACCOUNT_ID: "",
};

const MA_TYPES = ["SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA", "ALMA"];
const TREND_WEIGHT_OPTIONS = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0];
const SORTINO_MONTH_OPTIONS = [1, 2, 3, 4, 5, 6];
const TOP_PICK_SLOT_COUNT = 10;
const previewGridTheme = createAppGridTheme();
const DEFAULT_BACKTEST_BENCHMARK: TopPickTicker = { ticker: "069500", name: "KODEX 200" };
const DEFAULT_BACKTEST_SETTINGS: Required<TopPickBacktestSettings> = {
  benchmark: DEFAULT_BACKTEST_BENCHMARK,
  months: 12,
  rebalance: "none",
  initial_amount_manwon: 10000,
};
const TOP_PICK_WEIGHT_COLORS = [
  ...BUCKET_COLORS.filter((_, i) => i !== 4),
  "#2563eb",
  "#7c3aed",
  "#db2777",
  "#0891b2",
  "#65a30d",
  "#f97316",
  "#b45309",
];
const BACKTEST_REBALANCE_OPTIONS: { value: string; label: string }[] = [
  { value: "none", label: "리밸런싱 없음 (보유)" },
  { value: "weekly", label: "매주 (금요일)" },
  { value: "monthly", label: "매월 (말일)" },
  { value: "quarterly", label: "분기 (분기말)" },
  { value: "yearly", label: "매년 (연말)" },
];

const inputStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "5px 8px",
  fontSize: "0.9rem",
};

function normalizeTicker(value: string): string {
  return value.trim().toUpperCase();
}

function isKorTickerInput(value: string): boolean {
  return value.startsWith("KOR:") || /^[0-9A-Z]{6}$/.test(value);
}

function stripKorPrefix(value: string): string {
  return value.startsWith("KOR:") ? value.slice("KOR:".length) : value;
}

function buildTickerSlots(items: TopPickTicker[] | undefined): TopPickTicker[] {
  const slots = Array.from({ length: TOP_PICK_SLOT_COUNT }, (_, index) => items?.[index] ?? { ticker: "" });
  return slots.map((item) => ({
    ...item,
    ticker: item.ticker ?? "",
    nickname: item.nickname ?? "",
  }));
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
    return "#94a3b8";
  }
  const nonCashItems = (items ?? []).filter((item) => item.key !== "__CASH__");
  const index = Math.max(0, nonCashItems.findIndex((item) => item.key === key));
  return TOP_PICK_WEIGHT_COLORS[index % TOP_PICK_WEIGHT_COLORS.length];
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

function LabChart({ result }: { result: LabResult }) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const chart = createChart(container, {
      height: 240,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#64748b" },
      grid: { vertLines: { color: "rgba(148,163,184,0.12)" }, horzLines: { color: "rgba(148,163,184,0.12)" } },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, rightOffset: 6 },
      localization: { priceFormatter: (price: number) => formatCompactKrw(price) },
      autoSize: true,
    });
    const toLine = (values: number[]) => result.chart.dates.map((date, index) => ({ time: date as Time, value: values[index] }));
    const portfolioValues = result.chart.portfolio_value ?? result.chart.portfolio_pct;
    const benchmarkValues = result.chart.benchmark_value ?? result.chart.benchmark_pct;
    chart.addSeries(LineSeries, { color: "#2563eb", lineWidth: 2, lastValueVisible: false, priceLineVisible: false }).setData(toLine(portfolioValues));
    chart.addSeries(LineSeries, { color: "#94a3b8", lineWidth: 1, lastValueVisible: false, priceLineVisible: false }).setData(toLine(benchmarkValues));
    chart.timeScale().fitContent();

    return () => chart.remove();
  }, [result]);

  return (
    <div>
      <div style={{ display: "flex", gap: 16, marginBottom: 6, fontSize: "0.82rem", fontWeight: 600 }}>
        <span style={{ display: "flex", alignItems: "center", gap: 5, color: "#2563eb" }}>
          <span style={{ width: 14, height: 3, background: "#2563eb", borderRadius: 2 }} /> 포트폴리오
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 5, color: "#94a3b8" }}>
          <span style={{ width: 14, height: 3, background: "#94a3b8", borderRadius: 2 }} /> {result.benchmark.name}
        </span>
      </div>
      <div ref={containerRef} style={{ width: "100%", height: 240 }} />
    </div>
  );
}

function TopPickWeightTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ name?: string; value?: number; color?: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const rows = payload.filter((item) => Number(item.value ?? 0) > 0);
  const totalValue = rows.reduce((sum, item) => sum + Number(item.value ?? 0), 0);
  return (
    <div className="topPickWeightTooltip">
      <div className="topPickWeightTooltipTitle">{label}</div>
      {rows.map((item) => (
        <div key={item.name} className="topPickWeightTooltipRow">
          <span style={{ color: item.color }}>{item.name}</span>
          <strong>{totalValue > 0 ? `${Number(((Number(item.value ?? 0) / totalValue) * 100).toFixed(1))}%` : "-"}</strong>
        </div>
      ))}
    </div>
  );
}

function TopPickWeightHistoryChart({
  rows,
  items,
}: {
  rows: TopPickWeightHistoryRow[];
  items: TopPickWeightItem[];
}) {
  if (!rows.length || !items.length) {
    return <div style={{ color: "#94a3b8", fontSize: "0.9rem" }}>비중 이력이 없습니다.</div>;
  }

  // __CASH__가 첫 번째로 그려져 가장 아래에 위치하도록 정렬합니다.
  const sortedItems = [...items].sort((a, b) => {
    if (a.key === "__CASH__") return -1;
    if (b.key === "__CASH__") return 1;
    return 0;
  });

  return (
    <div className="topPickWeightChartWrap">
      <ResponsiveContainer width="100%" height={380} minWidth={0}>
        <BarChart data={rows} margin={{ top: 10, right: 12, bottom: 6, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="date" tickFormatter={formatMonthAxisLabel} minTickGap={18} tick={{ fontSize: 12 }} />
          <YAxis tickFormatter={(value) => formatCompactKrw(Number(value))} width={52} tick={{ fontSize: 12 }} />
          <Tooltip content={<TopPickWeightTooltip />} />
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
  );
}

export function TopPickSettingsClient() {
  const toast = useToast();
  const [tickers, setTickers] = useState<TopPickTicker[]>(() => buildTickerSlots(undefined));
  const [settings, setSettings] = useState<TopPickSettings>(DEFAULT_SETTINGS);
  const [accounts, setAccounts] = useState<AccountOption[]>([]);
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
  const [backtestBenchmarkTicker, setBacktestBenchmarkTicker] = useState(DEFAULT_BACKTEST_BENCHMARK.ticker);
  const [backtestBenchmarkName, setBacktestBenchmarkName] = useState(DEFAULT_BACKTEST_BENCHMARK.name ?? "");
  const [backtestBenchmarkResolving, setBacktestBenchmarkResolving] = useState(false);
  const [backtestPeriodMonths, setBacktestPeriodMonths] = useState(String(DEFAULT_BACKTEST_SETTINGS.months));
  const [backtestRebalanceMode, setBacktestRebalanceMode] = useState(DEFAULT_BACKTEST_SETTINGS.rebalance);
  const [backtestInitialAmountManwon, setBacktestInitialAmountManwon] = useState(String(DEFAULT_BACKTEST_SETTINGS.initial_amount_manwon));
  const autoRunStartedRef = useRef(false);

  const validTickers = useMemo(
    () => tickers.filter((item) => item.ticker.trim() && item.name),
    [tickers],
  );

  const loadSettings = useCallback(async () => {
    try {
      setLoading(true);
      const [settingsResp, accountsResp] = await Promise.all([
        fetch("/api/top-pick-settings", { cache: "no-store" }),
        fetch("/api/holdings-components/accounts", { cache: "no-store" }),
      ]);
      const data = (await settingsResp.json()) as TopPickSettingsPayload;
      const accountsData = (await accountsResp.json()) as AccountOption[] | { error?: string };
      if (!accountsResp.ok || !Array.isArray(accountsData)) {
        throw new Error(Array.isArray(accountsData) ? "계좌 목록을 불러오지 못했습니다." : accountsData.error ?? "계좌 목록을 불러오지 못했습니다.");
      }
      if (!settingsResp.ok || data.error) {
        throw new Error(data.error ?? "탑픽 설정을 불러오지 못했습니다.");
      }
      setAccounts(accountsData);
      setTickers(buildTickerSlots(data.tickers));
      setSettings({ ...DEFAULT_SETTINGS, ...(data.settings ?? {}) });
      setUpdatedAt(data.updated_at ?? null);
      setApprovedAt(data.approved_at ?? null);
      setApprovedWeights(data.approved_weights ?? null);
      setPreview(null);
      const backtestSettings = data.backtest_settings ?? DEFAULT_BACKTEST_SETTINGS;
      const benchmark = backtestSettings.benchmark ?? DEFAULT_BACKTEST_BENCHMARK;
      setBacktestBenchmarkTicker(benchmark.ticker || DEFAULT_BACKTEST_BENCHMARK.ticker);
      setBacktestBenchmarkName(benchmark.name || DEFAULT_BACKTEST_BENCHMARK.name || "");
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
      const resp = isKorTickerInput(raw)
        ? await fetch(`/api/backtest-lab/resolve?ticker=${encodeURIComponent(stripKorPrefix(raw))}`)
        : await fetch(`/api/ticker-resolve?ticker=${encodeURIComponent(raw)}`);
      const data = (await resp.json()) as {
        ticker?: string;
        name?: string;
        ticker_type?: string;
        country_code?: string;
        is_etf?: boolean;
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
    if (!settings.ACCOUNT_ID) {
      toast.error("탑픽 적용 계좌를 선택해주세요.");
      return;
    }
    if (!Number.isInteger(Number(backtestInitialAmountManwon)) || Number(backtestInitialAmountManwon) <= 0) {
      toast.error("백테스트 최초 금액(만원)은 1 이상의 정수여야 합니다.");
      return;
    }
    try {
      setSaving(true);
      const resp = await fetch("/api/top-pick-settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: validTickers,
          settings,
          backtest_settings: {
            benchmark: { ticker: backtestBenchmarkTicker, name: backtestBenchmarkName },
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
      setTickers(buildTickerSlots(data.tickers));
      setSettings({ ...DEFAULT_SETTINGS, ...(data.settings ?? {}) });
      setUpdatedAt(data.updated_at ?? null);
      setApprovedAt(data.approved_at ?? approvedAt);
      setApprovedWeights(data.approved_weights ?? approvedWeights);
      setPreview(null);
      const backtestSettings = data.backtest_settings ?? DEFAULT_BACKTEST_SETTINGS;
      const benchmark = backtestSettings.benchmark ?? DEFAULT_BACKTEST_BENCHMARK;
      setBacktestBenchmarkTicker(benchmark.ticker || DEFAULT_BACKTEST_BENCHMARK.ticker);
      setBacktestBenchmarkName(benchmark.name || DEFAULT_BACKTEST_BENCHMARK.name || "");
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
    if (!settings.ACCOUNT_ID) {
      toast.error("탑픽 적용 계좌를 선택해주세요.");
      return;
    }
    try {
      setRunning(true);
      setPreviewMode("weights");
      const resp = await fetch("/api/top-pick-settings/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers: validTickers, settings }),
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
  }, [settings, toast, validTickers]);

  useEffect(() => {
    if (
      !loading &&
      !approvedWeights &&
      validTickers.length >= 3 &&
      settings.ACCOUNT_ID &&
      !autoRunStartedRef.current
    ) {
      autoRunStartedRef.current = true;
      void runPreview();
    }
  }, [loading, approvedWeights, validTickers, settings.ACCOUNT_ID, runPreview]);

  const approvePreview = async () => {
    if (!preview) {
      toast.error("먼저 실행해서 계산 결과를 확인해주세요.");
      return;
    }
    try {
      setApproving(true);
      const resp = await fetch("/api/top-pick-settings/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers: validTickers, settings }),
      });
      const data = (await resp.json()) as TopPickWeightPreview & { approved_at?: string; tickers?: TopPickTicker[] };
      if (!resp.ok || data.error) {
        throw new Error(data.error ?? "탑픽 비중 확인 저장에 실패했습니다.");
      }
      setApprovedWeights(data);
      setPreview(data);
      setApprovedAt(data.approved_at ?? null);
      if (data.tickers && data.tickers.length > 0) {
        setTickers(buildTickerSlots(data.tickers));
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
  const updateSetting = (key: keyof TopPickSettings, value: string) => {
    setPreview(null);
    setSettings((current) => ({
      ...current,
      [key]: key === "MA_TYPE" || key === "ACCOUNT_ID" ? value : Number(value),
    }));
  };

  const formatScoreSettingLabel = (source?: Partial<TopPickSettings>) => {
    const maType = source?.MA_TYPE ?? settings.MA_TYPE;
    const maMonths = source?.MA_MONTHS ?? settings.MA_MONTHS;
    const trendWeight = source?.TREND_WEIGHT_RATIO ?? settings.TREND_WEIGHT_RATIO;
    const sortinoMonths = source?.SORTINO_MONTHS ?? settings.SORTINO_MONTHS;
    return `${maType} ${maMonths}개월 · 추세 ${trendWeight}% · Sortino ${100 - trendWeight}%/${sortinoMonths}개월`;
  };

  const resolveBacktestBenchmark = async () => {
    const raw = normalizeTicker(backtestBenchmarkTicker);
    if (!raw) {
      toast.error("벤치마크 티커를 입력해주세요.");
      return;
    }
    try {
      setBacktestBenchmarkResolving(true);
      const resp = isKorTickerInput(raw)
        ? await fetch(`/api/backtest-lab/resolve?ticker=${encodeURIComponent(stripKorPrefix(raw))}`)
        : await fetch(`/api/ticker-resolve?ticker=${encodeURIComponent(raw)}`);
      const data = (await resp.json()) as { ticker?: string; name?: string; error?: string; detail?: string };
      if (!resp.ok || data.error || !data.name) {
        throw new Error(data.error ?? data.detail ?? "존재하지 않는 벤치마크입니다.");
      }
      setBacktestBenchmarkTicker(data.ticker ?? stripKorPrefix(raw));
      setBacktestBenchmarkName(data.name);
      setBacktestResult(null);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "벤치마크 조회에 실패했습니다.");
    } finally {
      setBacktestBenchmarkResolving(false);
    }
  };

  const runBacktest = async () => {
    if (validTickers.length < 3) {
      toast.error("탑픽 백테스트에는 확인된 편입 ETF가 3개 이상 필요합니다.");
      return;
    }
    if (!backtestBenchmarkTicker.trim() || !backtestBenchmarkName.trim()) {
      toast.error("벤치마크 티커를 확인해주세요.");
      return;
    }
    if (!Number.isInteger(Number(backtestInitialAmountManwon)) || Number(backtestInitialAmountManwon) <= 0) {
      toast.error("백테스트 최초 금액(만원)은 1 이상의 정수여야 합니다.");
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
          settings,
          backtest_settings: {
            months: Number(backtestPeriodMonths),
            benchmark: { ticker: backtestBenchmarkTicker, name: backtestBenchmarkName },
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

  const previewColumns = useMemo<ColDef<TopPickWeightRow>[]>(
    () => [
      {
        field: "ticker",
        headerName: "티커",
        width: 110,
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: string | null | undefined }) => (params.value === "__CASH__" ? "CASH" : params.value ?? "-"),
      },
      {
        field: "name",
        headerName: "종목명",
        minWidth: 240,
        flex: 1,
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
        field: "return_12m_pct",
        headerName: "12개월",
        width: 100,
        type: "rightAligned",
        cellRenderer: renderReturnPctCell,
      },
      {
        field: "trend_pct",
        headerName: "추세(%)",
        width: 100,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatNumber(params.value),
      },
      {
        field: "sortino",
        headerName: "Sortino",
        width: 100,
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
        field: "target_weight_pct",
        headerName: "목표비중",
        width: 110,
        type: "rightAligned",
        cellStyle: { fontWeight: 800 },
        cellRenderer: (params: { value: number | null | undefined }) => formatWeightPct(params.value),
      },
    ],
    [],
  );

  const previewGridOptions = useMemo<GridOptions<TopPickWeightRow>>(
    () => ({
      domLayout: "autoHeight",
      suppressMovableColumns: true,
    }),
    [],
  );

  const calculatedGridOptions = useMemo<GridOptions<TopPickWeightRow>>(
    () => ({
      domLayout: "autoHeight",
      suppressMovableColumns: true,
      getRowStyle: (params) => {
        if (!params.data) return undefined;
        const ticker = params.data.ticker;
        const targetW = Number((params.data.target_weight_pct ?? 0).toFixed(1));
        
        const approvedItem = approvedWeights?.rows?.find((r) => r.ticker === ticker);
        const approvedW = Number((approvedItem ? (approvedItem.target_weight_pct ?? 0) : 0).toFixed(1));
        
        if (targetW !== approvedW) {
          return { backgroundColor: "rgba(249, 115, 22, 0.08)" };
        }
        return undefined;
      },
    }),
    [approvedWeights],
  );

  const backtestPositionColumns = useMemo<ColDef<LabPosition>[]>(
    () => [
      {
        field: "name",
        headerName: "종목",
        minWidth: 220,
        flex: 1,
        cellRenderer: (params: { data?: LabPosition }) => {
          const row = params.data;
          if (!row) return "-";
          const label = row.ticker === "__CASH__" ? "현금" : `${row.name ?? row.ticker} (${row.ticker})`;
          return (
            <span style={{ color: getTopPickWeightColor(backtestResult?.weight_items, row.ticker), fontWeight: 800 }}>
              {label}
            </span>
          );
        },
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
        headerName: "수익률",
        width: 100,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) => (
          <span style={{ color: params.value == null ? "#475569" : signedColor(params.value) }}>{formatReturnPct(params.value)}</span>
        ),
      },
      {
        field: "mdd_pct",
        headerName: "MDD",
        width: 100,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) =>
          params.value == null ? "-" : <span style={{ color: "#d63939" }}>{formatReturnPct(params.value)}</span>,
      },
      {
        field: "sortino",
        headerName: "Sortino",
        width: 100,
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
    [backtestResult?.weight_items],
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
    return [
      {
        ticker: "__CASH__",
        name: "현금",
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
      },
      ...backtestResult.positions,
    ];
  }, [backtestResult]);

  const summaryChip = (label: string, value: string, color?: string) => (
    <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 88 }}>
      <span style={{ color: "#94a3b8", fontSize: "0.78rem", fontWeight: 600 }}>{label}</span>
      <span style={{ fontWeight: 800, fontSize: "1rem", color: color ?? "#182433" }}>{value}</span>
    </div>
  );

  return (
    <PageFrame title="탑픽 설정">
      <div className="appPageStack">
        <div className="appActionHeader" style={{ padding: "0.25rem 0 0" }}>
          <div className="appActionHeaderInner" style={{ justifyContent: "flex-end", gap: 10 }}>
            <span style={{ color: "#64748b", fontSize: "0.82rem", whiteSpace: "nowrap" }}>마지막 저장: {updatedLabel}</span>
            <button type="button" className="btn btn-sm btn-primary" disabled={saving} onClick={() => void saveSettings()}>
              {saving ? "저장 중..." : "설정 저장"}
            </button>
          </div>
        </div>
        <div className="topPickSettingsLayout">
          <div className="topPickSettingsLeft">
            <div className="card appCard">
              <div className="card-body">
                <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 12 }}>비중 계산 설정</h2>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
                    <label className="appLabeledField" style={{ minWidth: 110 }}>
                      <span className="appLabeledFieldLabel">추세 타입</span>
                      <select
                        className="form-select form-select-sm"
                        value={settings.MA_TYPE}
                        onChange={(event) => updateSetting("MA_TYPE", event.target.value)}
                      >
                        {MA_TYPES.map((type) => (
                          <option key={type} value={type}>
                            {type}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="appLabeledField" style={{ minWidth: 110 }}>
                      <span className="appLabeledFieldLabel">추세 개월</span>
                      <select
                        className="form-select form-select-sm"
                        value={settings.MA_MONTHS}
                        onChange={(event) => updateSetting("MA_MONTHS", event.target.value)}
                      >
                        {[1, 2, 3, 5, 6, 9, 12, 18, 24].map((month) => (
                          <option key={month} value={month}>
                            {month}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="appLabeledField" style={{ minWidth: 130 }}>
                      <span className="appLabeledFieldLabel">추세 가중치(%)</span>
                      <select
                        className="form-select form-select-sm"
                        value={settings.TREND_WEIGHT_RATIO}
                        onChange={(event) => updateSetting("TREND_WEIGHT_RATIO", event.target.value)}
                      >
                        {TREND_WEIGHT_OPTIONS.map((ratio) => (
                          <option key={ratio} value={ratio}>
                            {ratio}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="appLabeledField" style={{ minWidth: 120 }}>
                      <span className="appLabeledFieldLabel">Sortino 개월</span>
                      <select
                        className="form-select form-select-sm"
                        value={settings.SORTINO_MONTHS}
                        onChange={(event) => updateSetting("SORTINO_MONTHS", event.target.value)}
                      >
                        {SORTINO_MONTH_OPTIONS.map((month) => (
                          <option key={month} value={month}>
                            {month}
                          </option>
                        ))}
                      </select>
                    </label>
                    <div style={{ display: "flex", alignItems: "flex-end" }}>
                      <button type="button" className="btn btn-sm btn-outline-primary" disabled={running} onClick={() => void runPreview()}>
                        {running ? "비중 계산 중..." : "비중 계산"}
                      </button>
                    </div>
                  </div>

                  <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
                    <label className="appLabeledField" style={{ minWidth: 110 }}>
                      <span className="appLabeledFieldLabel">최소 비중(%)</span>
                      <input
                        type="number"
                        className="form-control form-control-sm"
                        min={1}
                        max={100}
                        step={0.1}
                        value={settings.MIN_WEIGHT}
                        onChange={(event) => updateSetting("MIN_WEIGHT", event.target.value)}
                      />
                    </label>
                    <label className="appLabeledField" style={{ minWidth: 110 }}>
                      <span className="appLabeledFieldLabel">최대 비중(%)</span>
                      <input
                        type="number"
                        className="form-control form-control-sm"
                        min={1}
                        max={100}
                        step={0.1}
                        value={settings.MAX_WEIGHT}
                        onChange={(event) => updateSetting("MAX_WEIGHT", event.target.value)}
                      />
                    </label>
                    <label className="appLabeledField" style={{ minWidth: 140 }}>
                      <span className="appLabeledFieldLabel">현금 최대 비중(%)</span>
                      <select
                        className="form-select form-select-sm"
                        value={settings.CASH_MAX_WEIGHT}
                        onChange={(event) => updateSetting("CASH_MAX_WEIGHT", event.target.value)}
                      >
                        {[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map((ratio) => (
                          <option key={ratio} value={ratio}>
                            {ratio}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="appLabeledField" style={{ minWidth: 180 }}>
                      <span className="appLabeledFieldLabel">적용 계좌</span>
                      <select
                        className="form-select form-select-sm"
                        value={settings.ACCOUNT_ID}
                        onChange={(event) => updateSetting("ACCOUNT_ID", event.target.value)}
                      >
                        <option value="">계좌 선택</option>
                        {accounts.map((account) => (
                          <option key={account.account_id} value={account.account_id}>
                            {account.name}
                          </option>
                        ))}
                      </select>
                    </label>
                    <div style={{ display: "flex", alignItems: "flex-end" }}>
                      <button
                        type="button"
                        className="btn btn-sm btn-success"
                        disabled={previewMode !== "weights" || !preview || approving}
                        onClick={() => void approvePreview()}
                      >
                        {approving ? "확인 저장 중..." : "확인 저장"}
                      </button>
                    </div>
                  </div>
                  <div style={{ color: "#4b5563", fontSize: "0.83rem", marginTop: 16, paddingLeft: 4, width: "100%", lineHeight: "1.4" }}>
                    💡 <strong>시장 연동형 현금 제어:</strong> 벤치마크 시장지수가 상승 추세(accel_up)일 때는 수익 극대화를 위해 현금 한도를 자동으로 <strong>0%</strong>로 낮추어 상승 종목에 풀 투자합니다.
                  </div>
                </div>
              </div>
            </div>
            <div className="card appCard">
              <div className="card-body">
                <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 12 }}>백테스트</h2>
                <div className="topPickBacktestGrid">
                  <label className="appLabeledField topPickBacktestBenchmark">
                    <span className="appLabeledFieldLabel">벤치마크</span>
                    <div className="topPickBacktestInline">
                      <input
                        style={{ ...inputStyle, width: 130 }}
                        value={backtestBenchmarkTicker}
                        onChange={(event) => {
                          setBacktestBenchmarkTicker(event.target.value);
                          setBacktestBenchmarkName("");
                          setBacktestResult(null);
                        }}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") {
                            event.preventDefault();
                            void resolveBacktestBenchmark();
                          }
                        }}
                      />
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-secondary"
                        disabled={backtestBenchmarkResolving}
                        onClick={() => void resolveBacktestBenchmark()}
                      >
                        {backtestBenchmarkResolving ? "확인 중..." : "확인"}
                      </button>
                      <strong className="topPickBacktestBenchmarkName">{backtestBenchmarkName || "-"}</strong>
                    </div>
                  </label>
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
                      {[6, 12, 24].map((month) => (
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
                  <div style={{ display: "flex", alignItems: "flex-end" }}>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-dark"
                      disabled={backtestRunning || validTickers.length < 3 || !backtestBenchmarkName.trim()}
                      onClick={() => void runBacktest()}
                    >
                      {backtestRunning ? "백테스트 중..." : "백테스트"}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="card appCard topPickEtfCard">
            <div className="card-body">
            <div
              style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "flex-start", marginBottom: 14 }}
            >
              <div>
                <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>편입 ETF</h2>
                <p style={{ color: "#64748b", fontSize: "0.9rem", margin: 0 }}>
                  티커를 입력하고 확인한 뒤 저장합니다. 저장된 종목명은 비중 화면에서 조회 기준으로 사용합니다.
                </p>
              </div>
            </div>

            {loading ? (
              <div style={{ color: "#64748b", padding: "8px 0" }}>불러오는 중...</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <div className="topPickSlotColumn">
                  {tickers.map((item, index) => {
                    const confirmed = !!item.name;
                    return (
                      <div key={index} className="topPickSlotRow">
                        <span className="topPickSlotNumber">{index + 1}</span>
                        <input
                          style={{
                            ...inputStyle,
                            width: "100%",
                            backgroundColor: confirmed ? "#f8fafc" : undefined,
                            color: confirmed ? "#64748b" : undefined,
                          }}
                          placeholder="티커"
                          value={item.ticker}
                          readOnly={confirmed}
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
                         <input
                          style={{ ...inputStyle, width: "100%", minWidth: 0, backgroundColor: "#f8fafc", color: "#64748b" }}
                          placeholder="종목명 (티커 입력 후 확인)"
                          value={item.name ?? ""}
                          readOnly
                        />
                        <input
                          style={{
                            ...inputStyle,
                            width: "100%",
                            minWidth: 0,
                            backgroundColor: confirmed ? undefined : "#f8fafc",
                            color: confirmed ? undefined : "#64748b",
                          }}
                          placeholder=""
                          value={item.nickname ?? ""}
                          disabled={!confirmed}
                          onChange={(event) => {
                            setTickers((current) =>
                              current.map((currentItem, itemIndex) =>
                                itemIndex === index
                                  ? { ...currentItem, nickname: event.target.value }
                                  : currentItem,
                              ),
                            );
                            setPreview(null);
                          }}
                        />
                        {confirmed ? (
                          <button
                            type="button"
                            className="btn btn-sm btn-outline-danger"
                            onClick={() => {
                              setTickers((current) =>
                                current.map((currentItem, itemIndex) => (itemIndex === index ? { ticker: "" } : currentItem)),
                              );
                              setPreview(null);
                            }}
                          >
                            비우기
                          </button>
                        ) : (
                          <button
                            type="button"
                            className="btn btn-sm btn-outline-secondary"
                            disabled={!item.ticker.trim()}
                            onClick={() => void resolveTicker(index)}
                          >
                            확인
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
            </div>
          </div>
        </div>
        {previewMode === "backtest" ? (
          <div className="card appCard">
            <div className="card-body">
              <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "flex-start", marginBottom: 14 }}>
                <div>
                  <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>백테스트 결과</h2>
                  <p style={{ color: "#64748b", fontSize: "0.9rem", margin: 0 }}>
                    백테스트 결과와 종목별 성과를 검토합니다.
                  </p>
                </div>
              </div>
              {backtestResult ? (
                <div className="topPickBacktestResultLayout">
                  <div className="topPickBacktestTopLayout">
                    <div className="topPickBacktestResultPanel">
                      <h3 style={{ fontSize: "0.98rem", fontWeight: 800, marginBottom: 4 }}>
                        결과 — {backtestResult.buy_date} ~ {backtestResult.end_date} ({backtestResult.months}개월)
                      </h3>
                      <p style={{ color: "#94a3b8", fontSize: "0.82rem", marginBottom: 12 }}>
                        초기 {formatKrw(backtestResult.initial_capital)} → 최종 {formatKrw(backtestResult.final_value)} · 리밸런싱:{" "}
                        {rebalanceLabel(backtestResult.rebalance)}
                      </p>
                      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 8 }}>
                        {summaryChip("총수익률", `${backtestResult.summary.total_return_pct.toFixed(2)}%`, signedColor(backtestResult.summary.total_return_pct))}
                        {summaryChip("CAGR", `${backtestResult.summary.cagr_pct.toFixed(2)}%`, signedColor(backtestResult.summary.cagr_pct))}
                        {summaryChip("MDD", `${backtestResult.summary.mdd_pct.toFixed(2)}%`, "#d63939")}
                        {summaryChip("Sortino", backtestResult.summary.sortino.toFixed(2))}
                      </div>
                      <div style={{ color: "#94a3b8", fontSize: "0.8rem", marginBottom: 10 }}>
                        벤치마크 {backtestResult.benchmark.name}: 총 {backtestResult.benchmark.summary.total_return_pct.toFixed(2)}% · MDD{" "}
                        {backtestResult.benchmark.summary.mdd_pct.toFixed(2)}% · Sortino {backtestResult.benchmark.summary.sortino.toFixed(2)}
                      </div>
                      <LabChart result={backtestResult} />
                    </div>
                    <div className="topPickBacktestResultPanel">
                      <h3 style={{ fontSize: "0.98rem", fontWeight: 800, marginBottom: 4 }}>비중 변화</h3>
                      <p style={{ color: "#94a3b8", fontSize: "0.82rem", marginBottom: 10 }}>
                        매주 금요일 기준 가격 변동이 반영된 종목별 평가금액
                      </p>
                      <TopPickWeightHistoryChart
                        rows={backtestResult.weight_history ?? []}
                        items={backtestResult.weight_items ?? []}
                      />
                    </div>
                  </div>
                  <div className="topPickBacktestPerformancePanel">
                    <h3 style={{ fontSize: "0.98rem", fontWeight: 800, marginBottom: 8 }}>종목별 성과</h3>
                    {backtestResult.has_late_entry ? (
                      <p style={{ color: "#b45309", background: "rgba(245,158,11,0.08)", fontSize: "0.8rem", padding: "6px 10px", borderRadius: 6, marginBottom: 10 }}>
                        실험 시작 이후 상장된 종목은 배정 예산을 현금으로 대기시켰다가 상장일 종가에 편입합니다.
                      </p>
                    ) : null}
                    <AppAgGrid<LabPosition>
                      rowData={backtestPositionRows}
                      columnDefs={backtestPositionColumns}
                      minHeight="auto"
                      className="topPickPreviewGrid"
                      theme={previewGridTheme}
                      getRowId={(params) => params.data.ticker}
                      gridOptions={backtestGridOptions}
                    />
                  </div>
                </div>
              ) : (
                <div style={{ color: "#64748b", fontSize: "0.9rem" }}>
                  {backtestRunning ? "백테스트 실행 중..." : "백테스트 버튼을 누르면 결과가 여기에 표시됩니다."}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 12, width: "100%" }}>
            {/* 1. 저장된 비중 카드 */}
            <div className="card appCard">
              <div className="card-body">
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "flex-start", marginBottom: 14 }}>
                  <div>
                    <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>저장된 비중</h2>
                    <p style={{ color: "#64748b", fontSize: "0.9rem", margin: 0 }}>
                      실제 계좌에 적용되어 운용 중인 확정 포트폴리오 비중입니다.
                    </p>
                  </div>
                  <div style={{ color: "#64748b", fontSize: "0.82rem", whiteSpace: "nowrap" }}>마지막 확인 저장: {approvedLabel}</div>
                </div>
                {approvedWeights?.rows && approvedWeights.rows.length > 0 ? (
                  <>
                    <div style={{ color: "#64748b", fontSize: "0.9rem", marginBottom: 10 }}>
                      기준일 {approvedWeights.as_of_date ?? "-"} · {formatScoreSettingLabel(approvedWeights.settings)} · 벤치마크 {backtestBenchmarkName || backtestBenchmarkTicker || "-"}
                    </div>
                    <AppAgGrid<TopPickWeightRow>
                      rowData={approvedWeights.rows.filter(r => r.ticker === "__CASH__" || validTickers.some(vt => vt.ticker === r.ticker))}
                      columnDefs={previewColumns}
                      minHeight="auto"
                      className="topPickPreviewGrid"
                      theme={previewGridTheme}
                      getRowId={(params) => params.data.ticker}
                      gridOptions={previewGridOptions}
                    />
                  </>
                ) : (
                  <div style={{ color: "#64748b", fontSize: "0.9rem", padding: "10px 0" }}>저장된 비중 정보가 없습니다.</div>
                )}
              </div>
            </div>

            {/* 2. 계산된 비중 카드 */}
            <div className="card appCard">
              <div className="card-body">
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "flex-start", marginBottom: 14 }}>
                  <div>
                    <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>계산된 비중</h2>
                    <p style={{ color: "#64748b", fontSize: "0.9rem", margin: 0 }}>
                      현재 설정 조건으로 계산된 실시간 시뮬레이션 비중입니다.
                    </p>
                  </div>
                </div>
                {preview?.rows && preview.rows.length > 0 ? (
                  <>
                    <div style={{ color: "#64748b", fontSize: "0.9rem", marginBottom: 10 }}>
                      기준일 {preview.as_of_date ?? "-"} · {formatScoreSettingLabel(preview.settings)} · 벤치마크 {backtestBenchmarkName || backtestBenchmarkTicker || "-"}
                    </div>
                    <AppAgGrid<TopPickWeightRow>
                      rowData={preview.rows.filter(r => r.ticker === "__CASH__" || validTickers.some(vt => vt.ticker === r.ticker))}
                      columnDefs={previewColumns}
                      minHeight="auto"
                      className="topPickPreviewGrid"
                      theme={previewGridTheme}
                      getRowId={(params) => params.data.ticker}
                      gridOptions={calculatedGridOptions}
                    />
                    {preview.missing_tickers && preview.missing_tickers.length > 0 && (
                      <div style={{ color: "#b45309", fontSize: "0.85rem", marginTop: 10 }}>
                        가격 캐시 누락: {preview.missing_tickers.join(", ")}
                      </div>
                    )}
                  </>
                ) : (
                  <div style={{ color: "#64748b", fontSize: "0.9rem", padding: "10px 0" }}>
                    실시간 계산을 수행하거나 설정을 변경하면 새로운 계산 비중 결과가 여기에 표시됩니다.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
      <style jsx global>{`
        .topPickSettingsLayout {
          display: grid;
          grid-template-columns: minmax(320px, 0.95fr) minmax(420px, 1.05fr);
          gap: 12px;
          align-items: start;
        }

        .topPickSettingsLeft {
          display: flex;
          flex-direction: column;
          gap: 12px;
          min-width: 0;
        }

        .topPickEtfCard {
          min-width: 0;
        }

        .topPickBacktestGrid {
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(120px, 160px) minmax(180px, 1fr);
          gap: 12px 16px;
          align-items: end;
        }

        .topPickBacktestBenchmark {
          grid-column: 1 / -1;
        }

        .topPickBacktestInline {
          display: flex;
          align-items: center;
          gap: 8px;
          min-width: 0;
        }

        .topPickBacktestBenchmarkName {
          color: #16a34a;
          font-size: 1rem;
          min-width: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .topPickSlotColumn {
          display: flex;
          flex-direction: column;
          gap: 8px;
          min-width: 0;
        }

        .topPickSlotRow {
          display: grid;
          grid-template-columns: 28px minmax(84px, 112px) 280px 280px 64px;
          gap: 8px;
          align-items: center;
          min-width: 0;
        }

        .topPickSlotNumber {
          color: #64748b;
          font-size: 0.82rem;
          font-weight: 800;
          text-align: right;
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
          display: flex;
          flex-direction: column;
          min-width: 0;
        }

        .topPickBacktestPerformancePanel {
          min-width: 0;
          width: 100%;
        }

        .topPickWeightChartWrap {
          flex: 1;
          min-width: 0;
          width: 100%;
        }

        .topPickWeightTooltip {
          min-width: 190px;
          border: 1px solid rgba(148, 163, 184, 0.28);
          border-radius: 12px;
          background: rgba(15, 23, 42, 0.92);
          box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
          color: #f8fafc;
          padding: 10px 12px;
        }

        .topPickWeightTooltipTitle {
          margin-bottom: 8px;
          font-weight: 800;
        }

        .topPickWeightTooltipRow {
          display: flex;
          justify-content: space-between;
          gap: 14px;
          font-size: 0.84rem;
          line-height: 1.7;
        }

        .topPickPreviewGrid {
          height: auto !important;
        }

        .topPickPreviewGrid .appAgGridTheme {
          height: auto;
        }

        @media (max-width: 900px) {
          .topPickSettingsLayout,
          .topPickBacktestGrid,
          .topPickBacktestTopLayout {
            grid-template-columns: minmax(0, 1fr);
          }

          .topPickBacktestBenchmark {
            grid-column: auto;
          }
        }
      `}</style>
    </PageFrame>
  );
}
