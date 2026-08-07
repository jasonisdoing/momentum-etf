"use client";

import type { ColDef } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AppAgGrid } from "../components/AppAgGrid";
import { AppLoadingProgress, type LoadingProgress } from "../components/AppLoadingProgress";
import { PageFrame } from "../components/PageFrame";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";
import { formatPoolLabel, type PoolLabelSource } from "@/lib/pool-label";
import { formatPrice } from "../../lib/price-format";

const gridTheme = createAppGridTheme();

// 종목풀 폴백 목록 — 백엔드 미기동으로 pool_options 를 못 받았을 때만 쓴다.
// 한국 개별주 풀만 지원한다 (백엔드 POOL_CONFIGS 와 같아야 한다).
// 실제 표기는 pools-rank 와 같은 공용 formatPoolLabel(이름·아이콘·순서는 백엔드 응답)이다.
const POOL_OPTIONS: readonly PoolLabelSource[] = [
  { ticker_type: "kor", name: "코스피 개별주" },
  { ticker_type: "kor_kosdaq", name: "코스닥 개별주" },
];

type Settings = {
  pool: string;
  lookback_months: number;
  top_n: number;
  slippage_pct: number;
  max_per_industry: number;
  backtest_months: number;
};

// 한 업종에서 최대 몇 종목까지 담을지 — 백엔드 MAX_PER_INDUSTRY_OPTIONS 와 같아야 한다.
const MAX_PER_INDUSTRY_OPTIONS = [1, 2, 3, 4, 5, 10] as const;

// 룩백·종목 수 선택지 — 백엔드 검증 범위(룩백 1~24, 종목 수 5~100) 안에서 자주 쓰는 값만 노출한다.
const LOOKBACK_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24];
const TOP_N_OPTIONS = [5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 100];

/** 저장된 값이 선택지에 없으면 함께 노출한다 — 빼면 셀렉트가 빈칸이 되어 무엇이 저장돼 있는지 알 수 없다. */
function withSavedValue(options: number[], saved: string | undefined): number[] {
  const value = Number(saved);
  if (!Number.isFinite(value) || options.includes(value)) return options;
  return [...options, value].sort((a, b) => a - b);
}

type PickRow = {
  rank: number;
  is_reserve: boolean;
  streak_months: number | null;
  ticker: string;
  name: string;
  sector: string;
  industry: string;
  currency: string;
  price: number | null;
  return_1m_pct: number | null;
  return_3m_pct: number | null;
  return_12m_pct: number | null;
  return_lookback_pct: number;
  rel_return_pct: number;
  win_label: string;
  slope_annual_pct: number;
  r_squared: number;
  momentum_score: number;
};

type PicksResult = {
  as_of: string;
  portfolio_month: string;
  rebalance_date: string;
  signal_date: string;
  universe_count: number;
  candidate_count: number;
  rows: PickRow[];
};

type BacktestMonthRow = {
  month: string;
  strategy_pct: number | null;
  benchmark_pct: number | null;
  reference_pct: number | null;
  excess_pp: number | null;
  holdings_count: number;
  turnover_pct: number | null;
  added: string[];
  removed: string[];
  is_pending?: boolean;
};

type BacktestDayRow = {
  date: string;
  strategy_pct: number | null;
  benchmark_pct: number | null;
  reference_pct: number | null;
};

type BacktestResult = {
  start_date: string;
  end_date: string;
  months: number;
  strategy_total_pct: number;
  benchmark_total_pct: number;
  strategy_mdd_pct: number | null;
  benchmark_mdd_pct: number | null;
  strategy_sortino: number | null;
  benchmark_sortino: number | null;
  strategy_cagr_pct: number | null;
  benchmark_cagr_pct: number | null;
  reference_cagr_pct: number | null;
  benchmark_name: string;
  benchmark_ticker: string;
  reference_name: string | null;
  reference_total_pct: number | null;
  reference_mdd_pct: number | null;
  reference_sortino: number | null;
  monthly: BacktestMonthRow[];
  daily: BacktestDayRow[];
};

type View = {
  settings: Settings;
  pool_options?: PoolLabelSource[];
  // 기간 선택지는 서버가 가격 캐시 범위로 계산해 내려준다 (종목풀 백테스트와 동일).
  month_options?: number[];
  picks: PicksResult | null;
};

// 선정 결과를 바꾸는 설정 — 이 값들이 바뀔 때만 저장 후 선정을 다시 계산한다.
// 슬리피지와 백테스트 기간은 백테스트에만 쓰이므로 선정을 다시 돌릴 이유가 없다.
const PICK_AFFECTING_KEYS = [
  "pool",
  "lookback_months",
  "top_n",
  "max_per_industry",
] as const;

function needsRepick(before: Settings | null, after: Settings): boolean {
  if (!before) return true;
  return PICK_AFFECTING_KEYS.some((key) => before[key] !== after[key]);
}

// 백테스트 표 보기 단위 — /compare 의 연간·월간·일간 구분과 같은 개념.
const VIEW_MODES = [
  { key: "yearly", label: "연간" },
  { key: "monthly", label: "월간" },
  { key: "daily", label: "일간" },
] as const;
type ViewMode = (typeof VIEW_MODES)[number]["key"];

type YearRow = {
  year: string;
  strategy_pct: number | null;
  benchmark_pct: number | null;
  reference_pct: number | null;
  strategy_partial: boolean;
  benchmark_partial: boolean;
  reference_partial: boolean;
};

/** 월별 수익률을 복리로 합성한다. 값이 하나도 없으면 null. */
function compoundPct(values: (number | null)[]): number | null {
  const usable = values.filter((v): v is number => v != null && Number.isFinite(v));
  if (usable.length === 0) return null;
  return (usable.reduce((acc, v) => acc * (1 + v / 100), 1) - 1) * 100;
}

/**
 * 월별 행을 연도별로 묶는다. 12개월이 다 차지 않은 해는 `partial` 로 표시한다
 * (/compare 의 부분 기간 `*` 표기와 같은 규칙). 예정 행은 수익률이 없어 제외한다.
 */
function toYearRows(monthly: BacktestMonthRow[]): YearRow[] {
  const byYear = new Map<string, BacktestMonthRow[]>();
  for (const row of monthly) {
    if (row.is_pending) continue;
    const year = row.month.slice(0, 4);
    byYear.set(year, [...(byYear.get(year) ?? []), row]);
  }
  const countOf = (rows: BacktestMonthRow[], key: keyof BacktestMonthRow) =>
    rows.filter((r) => r[key] != null).length;

  return [...byYear.entries()]
    .sort((a, b) => b[0].localeCompare(a[0]))
    .map(([year, rows]) => ({
      year,
      strategy_pct: compoundPct(rows.map((r) => r.strategy_pct)),
      benchmark_pct: compoundPct(rows.map((r) => r.benchmark_pct)),
      reference_pct: compoundPct(rows.map((r) => r.reference_pct)),
      strategy_partial: countOf(rows, "strategy_pct") < 12,
      benchmark_partial: countOf(rows, "benchmark_pct") < 12,
      reference_partial: countOf(rows, "reference_pct") < 12,
    }));
}

const hintStyle: React.CSSProperties = { color: "var(--text-muted)", fontSize: "var(--fs-sm)" };
const numberInputStyle: React.CSSProperties = { width: 88, textAlign: "right" };

function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return value.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatSigned(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

function signColor(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value) || value === 0) return "inherit";
  return value > 0 ? "var(--up-color, #d64545)" : "var(--down-color, #2f6fd0)";
}

/** 성과 요약 한 덩어리 — 전략·벤치마크·참고 지수를 같은 형식으로 보여준다. */
function PerformanceSummary({
  label,
  totalPct,
  cagrPct,
  mddPct,
  sortino,
}: {
  label: string;
  totalPct: number | null;
  cagrPct: number | null;
  mddPct: number | null;
  sortino: number | null;
}) {
  return (
    <span>
      {label} <b style={{ color: signColor(totalPct) }}>{formatSigned(totalPct)}</b>
      <span style={hintStyle}>
        {` (CAGR ${cagrPct != null ? formatSigned(cagrPct, 1) : "-"}`}
        {mddPct != null ? ` · MDD ${mddPct.toFixed(1)}%` : ""}
        {` · 소르티노 ${sortino != null ? sortino.toFixed(2) : "-"})`}
      </span>
    </span>
  );
}

/**
 * 진행률을 일정 간격으로 90%까지 올린다. 서버가 단일 요청으로 응답해 실제 단계를
 * 알 수 없으므로, 실제 소요 시간(수 초)에 맞춘 완만한 램프만 보여준다.
 */
function startProgressRamp(
  setProgress: (updater: (prev: LoadingProgress | null) => LoadingProgress | null) => void,
): () => void {
  const timer = window.setInterval(() => {
    setProgress((prev) => (prev ? { ...prev, percent: Math.min(90, prev.percent + 6) } : prev));
  }, 400);
  return () => window.clearInterval(timer);
}

export function SteadyMomentumClient() {
  const toast = useToast();
  const [view, setView] = useState<View | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [picking, setPicking] = useState(false);
  const [backtesting, setBacktesting] = useState(false);
  const [backtest, setBacktest] = useState<BacktestResult | null>(null);
  const [pickProgress, setPickProgress] = useState<LoadingProgress | null>(null);
  const [pickFailed, setPickFailed] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [backtestProgress, setBacktestProgress] = useState<LoadingProgress | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("monthly");
  const autoPickedRef = useRef(false);

  // 설정 입력 초안 (문자열로 보관해 입력 중 상태를 그대로 둔다)
  const [draft, setDraft] = useState<Record<string, string>>({});
  // 초안 초기값은 자리만 잡는다 — 설정을 받은 뒤 applyView 가 항상 덮어쓰며,
  // 설정을 못 받으면 폼 자체를 그리지 않으므로 이 값이 화면에 보이는 경우는 없다.
  const [draftPool, setDraftPool] = useState<string>("");
  const [draftBacktestMonths, setDraftBacktestMonths] = useState<number>(0);
  const [draftMaxPerIndustry, setDraftMaxPerIndustry] = useState<number>(0);

  const applyView = useCallback((data: View) => {
    setView(data);
    setDraftPool(data.settings.pool);
    setDraftBacktestMonths(data.settings.backtest_months);
    setDraftMaxPerIndustry(data.settings.max_per_industry);
    setDraft({
      lookback_months: String(data.settings.lookback_months),
      top_n: String(data.settings.top_n),
      slippage_pct: String(data.settings.slippage_pct),
    });
  }, []);

  const load = useCallback(async (): Promise<boolean> => {
    setLoading(true);
    try {
      const resp = await fetch("/api/strategy-sm", { cache: "no-store" });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "설정을 불러오지 못했습니다.");
      setLoadError(null);
      applyView(payload as View);
      return true;
    } catch (error) {
      // 설정을 못 받으면 값을 지어내지 않는다 — 폼을 그리지 않고 실패만 알린다.
      // (기본값을 그렸다가 그대로 저장되면 저장돼 있던 설정이 덮어써진다.)
      const message = error instanceof Error ? error.message : "설정을 불러오지 못했습니다.";
      setLoadError(message);
      toast.error(message);
      return false;
    } finally {
      setLoading(false);
    }
  }, [applyView, toast]);

  const runPicks = useCallback(async () => {
    setPicking(true);
    setPickFailed(false);
    setPickProgress({ percent: 10, message: "월 확정 포트폴리오 계산 중" });
    const stopRamp = startProgressRamp(setPickProgress);
    try {
      const resp = await fetch("/api/strategy-sm/picks", { method: "POST" });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "선정에 실패했습니다.");
      setPickProgress({ percent: 100, message: "선정 결과 반영 중" });
      setView((prev) => (prev ? { ...prev, picks: payload as PicksResult } : prev));
    } catch (error) {
      setPickFailed(true);
      toast.error(error instanceof Error ? error.message : "선정에 실패했습니다.");
    } finally {
      stopRamp();
      setPicking(false);
      setPickProgress(null);
    }
  }, [toast]);

  // 진입 시 저장된 설정으로 선정을 한 번 자동 실행한다 (가격 캐시 기반이라 수 초).
  useEffect(() => {
    void (async () => {
      const ok = await load();
      if (ok && !autoPickedRef.current) {
        autoPickedRef.current = true;
        await runPicks();
      }
    })();
  }, [load, runPicks]);

  const saveSettings = useCallback(async () => {
    const lookback = Number(draft.lookback_months);
    const topN = Number(draft.top_n);
    const slippage = Number(draft.slippage_pct);
    if (![lookback, topN, slippage].every((v) => Number.isFinite(v))) {
      toast.error("설정 값이 올바르지 않습니다.");
      return;
    }
    setSaving(true);
    try {
      const resp = await fetch("/api/strategy-sm", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          settings: {
            pool: draftPool,
            lookback_months: lookback,
            top_n: topN,
            slippage_pct: slippage,
            max_per_industry: draftMaxPerIndustry,
            backtest_months: draftBacktestMonths,
          },
        }),
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "설정을 저장하지 못했습니다.");
      const saved = payload as View;
      const repick = needsRepick(view?.settings ?? null, saved.settings);
      // 선정에 영향이 없는 변경(슬리피지·백테스트 기간)이면 기존 선정 결과를 그대로 둔다.
      applyView({ ...saved, picks: repick ? null : (view?.picks ?? null) });
      // 백테스트는 어느 설정이 바뀌든 결과가 달라지므로 비운다.
      setBacktest(null);
      toast.success(repick ? "설정을 저장했습니다. 선정을 다시 계산합니다." : "설정을 저장했습니다.");
      if (repick) await runPicks();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "설정을 저장하지 못했습니다.");
    } finally {
      setSaving(false);
    }
  }, [
    applyView,
    draft,
    draftBacktestMonths,
    draftMaxPerIndustry,
    draftPool,
    runPicks,
    toast,
    view,
  ]);

  const runBacktest = useCallback(async () => {
    // 기간도 저장된 설정을 따른다 — 미저장 상태에서는 실행 버튼이 막혀 있다.
    const months = view?.settings.backtest_months;
    if (months == null) return;
    setBacktesting(true);
    setBacktestProgress({ percent: 10, message: "월별 리밸런싱 시뮬레이션 중" });
    const stopRamp = startProgressRamp(setBacktestProgress);
    try {
      const resp = await fetch("/api/strategy-sm/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // 일간 탭에서 실행할 때만 일별 행을 요청한다 (응답이 수천 행이라 무겁다).
        body: JSON.stringify({ months, include_daily: viewMode === "daily" }),
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "백테스트에 실패했습니다.");
      setBacktestProgress({ percent: 100, message: "결과 반영 중" });
      setBacktest(payload as BacktestResult);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "백테스트에 실패했습니다.");
    } finally {
      stopRamp();
      setBacktesting(false);
      setBacktestProgress(null);
    }
  }, [toast, view?.settings.backtest_months, viewMode]);

  // 저장하지 않은 입력이 있으면 실행 결과가 화면 값과 어긋난다 — 저장을 먼저 요구한다.
  const isDirty = useMemo(() => {
    if (!view) return false;
    const saved = view.settings;
    return (
      draftPool !== saved.pool ||
      draftBacktestMonths !== saved.backtest_months ||
      draftMaxPerIndustry !== saved.max_per_industry ||
      draft.lookback_months !== String(saved.lookback_months) ||
      draft.top_n !== String(saved.top_n) ||
      draft.slippage_pct !== String(saved.slippage_pct)
    );
  }, [draft, draftBacktestMonths, draftMaxPerIndustry, draftPool, view]);

  const lookbackMonths = view?.settings.lookback_months ?? null;

  const pickColumns = useMemo<ColDef<PickRow>[]>(() => {
    // 설정을 받기 전에는 컬럼(룩백 개월 머리글)을 만들 수 없다.
    if (lookbackMonths == null) return [];
    return [
      { headerName: "순위", field: "rank", width: 60, type: "numericColumn" },
      {
        headerName: "연속",
        field: "streak_months",
        headerTooltip: "이번 포트폴리오까지 몇 달 연속 편입됐는지 (신규 = 이번 달 첫 편입, 최대 12개월 추적)",
        width: 68,
        valueFormatter: (p) =>
          p.value == null ? "-" : p.value <= 1 ? "신규" : p.value >= 12 ? "12+" : `${p.value}개월`,
        cellStyle: (p) => ({
          color: p.value != null && p.value <= 1 && !p.data?.is_reserve ? "var(--up-color, #d64545)" : "inherit",
        }),
      },
      {
        headerName: "티커",
        field: "ticker",
        // 미국 티커 전용 화면이라 최장 5자(BRK-B)면 충분하다.
        // `/pools-rank` 는 한국 6자리 코드와 `ASX:` 접두사까지 담아야 해서 더 넓다.
        width: 82,
        cellRenderer: (p: { value: string | null | undefined }) => <TickerDetailLink ticker={p.value} />,
      },
      {
        headerName: "종목명",
        field: "name",
        // 이 표에서 유일한 flex 컬럼 — 남는 폭을 종목명이 가져간다.
        // 상한(maxWidth)을 두지 않아 넓은 화면에서 계속 늘어난다.
        flex: 1,
        minWidth: 220,
        // `/pools-rank` 와 같은 셀 스타일 — 2줄까지 보이고 넘치면 말줄임.
        cellRenderer: (p: { value?: string | null }) => (
          <span className="appNameCellText" title={String(p.value ?? "")}>
            {String(p.value ?? "-")}
          </span>
        ),
      },
      {
        headerName: "섹터",
        field: "sector",
        headerTooltip: "지수 구성종목 메타",
        // yfinance 값 최장 22자. 좁은 화면에서는 줄어들고 말줄임 처리된다.
        width: 170,
        minWidth: 110,
        cellClass: "appTextEllipsisCell",
        tooltipValueGetter: (p) => p.value || undefined,
        valueFormatter: (p) => p.value || "-",
      },
      {
        headerName: "업종",
        field: "industry",
        headerTooltip: "지수 구성종목 메타",
        // 실제로 쓰이는 값은 `Software - Infrastructure`(25자) 정도까지다.
        // 최장 40자짜리는 드물어 말줄임에 맡기고, 폭은 섹터와 비슷하게 잡는다.
        width: 200,
        minWidth: 110,
        cellClass: "appTextEllipsisCell",
        tooltipValueGetter: (p) => p.value || undefined,
        valueFormatter: (p) => p.value || "-",
      },
      {
        headerName: "현재가",
        field: "price",
        headerTooltip: "가격 캐시의 최신 종가 (통화는 종목풀 기준)",
        width: 100,
        type: "numericColumn",
        valueFormatter: (p) => formatPrice(p.value, p.data?.currency),
      },
      {
        headerName: "1개월(%)",
        field: "return_1m_pct",
        headerTooltip: "판정일 기준 1개월 수익률",
        width: 92,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value, 1),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
      {
        headerName: `${lookbackMonths}개월(%)`,
        field: "return_lookback_pct",
        headerTooltip: "룩백 구간 수익률 — 룩백(개월) 설정을 따른다",
        width: 92,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value, 1),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
      {
        headerName: `상대${lookbackMonths}개월(%)`,
        field: "rel_return_pct",
        headerTooltip: "룩백 구간의 벤치마크 대비 초과 수익률",
        width: 108,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value, 1),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
      {
        headerName: "12개월(%)",
        field: "return_12m_pct",
        headerTooltip: "판정일 기준 12개월 수익률",
        width: 98,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value, 1),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
      {
        headerName: "월승률",
        field: "win_label",
        headerTooltip: "룩백 구간의 월별 상대수익 중 벤치마크를 이긴 달",
        width: 78,
      },
      {
        headerName: "상대기울기(%)",
        field: "slope_annual_pct",
        headerTooltip: "상대 가격선(종가÷벤치마크) 회귀의 연율화 기울기",
        width: 112,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value, 1),
      },
      {
        headerName: "R²",
        field: "r_squared",
        headerTooltip: "상대 가격선이 회귀선에 얼마나 잘 맞는지 (1에 가까울수록 매끄러운 추세)",
        width: 82,
        type: "numericColumn",
        valueFormatter: (p) => (p.value != null ? p.value.toFixed(3) : "-"),
      },
      {
        headerName: "점수(이격%)",
        field: "momentum_score",
        headerTooltip: "장기 이평선 이격(%) — 순위 화면과 같은 신호(이평선 일수는 종목풀 설정)",
        width: 92,
        type: "numericColumn",
        valueFormatter: (p) => formatNumber(p.value, 1),
        cellStyle: () => ({ fontWeight: 700 }),
      },
    ];
  }, [lookbackMonths]);

  const backtestColumns = useMemo<ColDef<BacktestMonthRow>[]>(() => {
    if (!backtest) return [];
    const columns: ColDef<BacktestMonthRow>[] = [
      {
        headerName: "월",
        field: "month",
        width: 116,
        valueFormatter: (p) => (p.data?.is_pending ? `${p.value} (예정)` : String(p.value ?? "")),
        cellStyle: () => ({ fontWeight: 700 }),
      },
      {
        headerName: "전략(%)",
        field: "strategy_pct",
        width: 92,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
      {
        headerName: `${backtest.benchmark_ticker}(%)`,
        headerTooltip: `벤치마크 ${backtest.benchmark_name}`,
        field: "benchmark_pct",
        width: 96,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
    ];
    if (backtest.reference_name) {
      columns.push({
        headerName: `${backtest.reference_name}(%)`,
        headerTooltip: "참고 지수 — 유사 컨셉 ETF (벤치마크가 아니며 선정에 관여하지 않는다)",
        field: "reference_pct",
        width: 104,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      });
    }
    columns.push(
      { headerName: "종목 수", field: "holdings_count", width: 74, type: "numericColumn" },
      {
        headerName: "교체율(%)",
        field: "turnover_pct",
        headerTooltip: "직전 달 대비 교체된 종목 비중",
        width: 84,
        type: "numericColumn",
        valueFormatter: (p) => (p.value == null ? "-" : formatNumber(p.value)),
      },
      {
        headerName: "편입",
        field: "added",
        flex: 1,
        minWidth: 200,
        wrapText: true,
        autoHeight: true,
        cellClass: "steadyWrapCell",
        valueFormatter: (p) => (p.value?.length ? p.value.join(", ") : "-"),
        cellStyle: () => ({ color: "var(--up-color, #d64545)" }),
      },
      {
        headerName: "편출",
        field: "removed",
        flex: 1,
        minWidth: 200,
        wrapText: true,
        autoHeight: true,
        cellClass: "steadyWrapCell",
        valueFormatter: (p) => (p.value?.length ? p.value.join(", ") : "-"),
        cellStyle: () => ({ color: "var(--down-color, #2f6fd0)" }),
      },
    );
    return columns;
  }, [backtest]);

  const dailyColumns = useMemo<ColDef<BacktestDayRow>[]>(() => {
    if (!backtest) return [];
    const pctColumn = (headerName: string, field: keyof BacktestDayRow, headerTooltip?: string): ColDef<BacktestDayRow> => ({
      headerName,
      field,
      headerTooltip,
      flex: 1,
      minWidth: 110,
      type: "numericColumn",
      valueFormatter: (p) => formatSigned(p.value),
      cellStyle: (p) => ({ color: signColor(p.value), fontWeight: field === "strategy_pct" ? 700 : 400 }),
    });
    const columns: ColDef<BacktestDayRow>[] = [
      { headerName: "날짜", field: "date", width: 128, cellStyle: () => ({ fontWeight: 700 }) },
      pctColumn("전략(%)", "strategy_pct", "보유 종목 동일가중 일간 변동률 (교체일에는 리밸런싱 비용 반영)"),
      pctColumn(`${backtest.benchmark_ticker}(%)`, "benchmark_pct", `벤치마크 ${backtest.benchmark_name}`),
    ];
    if (backtest.reference_name) {
      columns.push(
        pctColumn(
          `${backtest.reference_name}(%)`,
          "reference_pct",
          "참고 지수 — 유사 컨셉 ETF (벤치마크가 아니며 선정에 관여하지 않는다)",
        ),
      );
    }
    return columns;
  }, [backtest]);

  const yearRows = useMemo<YearRow[]>(() => (backtest ? toYearRows(backtest.monthly) : []), [backtest]);

  const yearColumns = useMemo<ColDef<YearRow>[]>(() => {
    if (!backtest) return [];
    // 부분 기간은 /compare 와 같은 규칙으로 값 뒤에 `*` 를 붙인다.
    const pctColumn = (
      headerName: string,
      field: "strategy_pct" | "benchmark_pct" | "reference_pct",
      partialField: "strategy_partial" | "benchmark_partial" | "reference_partial",
      headerTooltip?: string,
    ): ColDef<YearRow> => ({
      headerName,
      field,
      headerTooltip,
      flex: 1,
      minWidth: 120,
      type: "numericColumn",
      valueFormatter: (p) =>
        p.value == null ? "-" : `${formatSigned(p.value)}${p.data?.[partialField] ? "*" : ""}`,
      cellStyle: (p) => ({ color: signColor(p.value), fontWeight: field === "strategy_pct" ? 700 : 400 }),
      tooltipValueGetter: (p) =>
        p.data?.[partialField] ? "12개월이 다 차지 않은 해 — 있는 달만 합성한 부분 기간" : undefined,
    });

    const columns: ColDef<YearRow>[] = [
      { headerName: "연도", field: "year", width: 104, cellStyle: () => ({ fontWeight: 700 }) },
      pctColumn("전략(%)", "strategy_pct", "strategy_partial"),
      pctColumn(
        `${backtest.benchmark_ticker}(%)`,
        "benchmark_pct",
        "benchmark_partial",
        `벤치마크 ${backtest.benchmark_name}`,
      ),
    ];
    if (backtest.reference_name) {
      columns.push(
        pctColumn(
          `${backtest.reference_name}(%)`,
          "reference_pct",
          "reference_partial",
          "참고 지수 — 유사 컨셉 ETF (벤치마크가 아니며 선정에 관여하지 않는다)",
        ),
      );
    }
    return columns;
  }, [backtest]);

  if (loading && !view) {
    return (
      <PageFrame title="Steady Momentum" fullWidth>
        <div style={{ ...hintStyle, padding: 20 }}>불러오는 중…</div>
      </PageFrame>
    );
  }
  if (!view) {
    // 설정을 못 받은 상태 — 값을 지어내 폼을 그리지 않고 실패와 재시도만 제공한다.
    return (
      <PageFrame title="Steady Momentum" fullWidth>
        <div className="card appCard">
          <div className="card-body" style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ fontWeight: 700 }}>설정을 불러오지 못했습니다.</span>
            <span style={hintStyle}>{loadError ?? "원인을 알 수 없습니다."}</span>
            <button type="button" className="btn btn-sm btn-primary" onClick={() => void load()} disabled={loading}>
              {loading ? "다시 시도 중…" : "다시 시도"}
            </button>
          </div>
        </div>
      </PageFrame>
    );
  }

  const selectedCount = view.picks?.rows.filter((row) => !row.is_reserve).length ?? 0;
  const reserveCount = view.picks?.rows.filter((row) => row.is_reserve).length ?? 0;

  return (
    <PageFrame title="Steady Momentum" fullWidth>
      <div className="appPageStack">
        {/* ① 변수 설정 */}
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">종목풀</span>
                  <select
                    className="form-select form-select-sm"
                    value={draftPool}
                    onChange={(e) => setDraftPool(e.target.value)}
                  >
                    {(view.pool_options?.length ? view.pool_options : POOL_OPTIONS).map((pool) => (
                      <option key={pool.ticker_type} value={pool.ticker_type}>
                        {formatPoolLabel(pool)}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">룩백(개월)</span>
                  <select
                    className="form-select form-select-sm"
                    style={{ width: 80 }}
                    value={draft.lookback_months ?? ""}
                    onChange={(e) => setDraft((d) => ({ ...d, lookback_months: e.target.value }))}
                  >
                    {withSavedValue(LOOKBACK_OPTIONS, draft.lookback_months).map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">종목 수</span>
                  <select
                    className="form-select form-select-sm"
                    style={{ width: 80 }}
                    value={draft.top_n ?? ""}
                    onChange={(e) => setDraft((d) => ({ ...d, top_n: e.target.value }))}
                  >
                    {withSavedValue(TOP_N_OPTIONS, draft.top_n).map((n) => (
                      <option key={n} value={n}>
                        {n}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">업종별 최대 보유</span>
                  <select
                    className="form-select form-select-sm"
                    style={{ width: 80 }}
                    value={draftMaxPerIndustry}
                    onChange={(e) => setDraftMaxPerIndustry(Number(e.target.value))}
                  >
                    {MAX_PER_INDUSTRY_OPTIONS.map((count) => (
                      <option key={count} value={count}>
                        {count}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">슬리피지(%)</span>
                  <input
                    className="form-control form-control-sm"
                    style={numberInputStyle}
                    value={draft.slippage_pct ?? ""}
                    onChange={(e) => setDraft((d) => ({ ...d, slippage_pct: e.target.value }))}
                    inputMode="decimal"
                  />
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">백테스트 기간</span>
                  <select
                    className="form-select form-select-sm"
                    style={{ width: 104 }}
                    value={draftBacktestMonths}
                    onChange={(e) => setDraftBacktestMonths(Number(e.target.value))}
                  >
                    {(view.month_options ?? [view.settings.backtest_months]).map((m) => (
                      <option key={m} value={m}>
                        {m}개월
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="appMainHeaderRight">
                {isDirty ? (
                  <span style={{ ...hintStyle, color: "var(--up-color, #d64545)", fontWeight: 700 }}>
                    저장하지 않은 변경
                  </span>
                ) : null}
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  onClick={() => void saveSettings()}
                  disabled={saving || !isDirty}
                >
                  {saving ? "저장 중…" : "저장"}
                </button>
              </div>
            </div>
            <div style={hintStyle}>
              장기 이평선 이격 상위(순위 화면과 같은 신호, 단기 이격 음수 제외)로 선정 · 고정 종목 제외
            </div>
          </div>
        </div>

        {/* ② 현재 선정 종목 */}
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <span style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>현재 선정 종목</span>
                {view.picks ? (
                  <span style={{ ...hintStyle, fontSize: "var(--fs-sm)" }}>
                    <b style={{ color: "inherit" }}>{view.picks.portfolio_month} 포트폴리오</b> ·{" "}
                    {view.picks.rebalance_date} 종가 교체 (판정 {view.picks.signal_date}) · 유니버스{" "}
                    {view.picks.universe_count} → 후보 {view.picks.candidate_count} → 선정 {selectedCount}
                    {reserveCount > 0 ? ` (+차순위 ${reserveCount})` : ""} · 다음 교체 전까지 결과가 바뀌지 않습니다
                  </span>
                ) : (
                  <span style={{ ...hintStyle, fontSize: "var(--fs-sm)" }}>
                    {pickFailed
                      ? "선정 결과를 불러오지 못했습니다. 설정을 저장하거나 새로고침하세요."
                      : "이번 달 확정 포트폴리오를 계산하고 있습니다."}
                  </span>
                )}
              </div>
            </div>
            {picking ? <AppLoadingProgress title="선정 계산 중..." progress={pickProgress} /> : null}
            {view.picks && !picking ? (
              // autoHeight — 그리드가 행 수만큼만 높이를 차지해 하단 낭비가 없다.
              <AppAgGrid<PickRow>
                rowData={view.picks.rows}
                columnDefs={pickColumns}
                theme={gridTheme}
                minHeight={0}
                height="auto"
                gridOptions={{ domLayout: "autoHeight" }}
                getRowClass={(p) => (p.data?.is_reserve ? "steadyReserveRow" : "")}
                getRowId={(p) => p.data.ticker}
              />
            ) : null}
          </div>
        </div>

        {/* ③ 백테스트 */}
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <span style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>백테스트</span>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">보기</span>
                  <div className="appSegmentedToggle appSegmentedToggleCompact" role="group" aria-label="백테스트 보기 단위">
                    {VIEW_MODES.map((mode) => (
                      <button
                        key={mode.key}
                        type="button"
                        className={
                          viewMode === mode.key
                            ? "btn appSegmentedToggleButton is-active"
                            : "btn appSegmentedToggleButton"
                        }
                        onClick={() => setViewMode(mode.key)}
                      >
                        {mode.label}
                      </button>
                    ))}
                  </div>
                </label>
                <span style={hintStyle}>
                  {view.settings.backtest_months}개월 · 월간 리밸런싱 · 현재 종목풀 기준(생존 편향 있음)
                </span>
              </div>
              <div className="appMainHeaderRight">
                {isDirty ? <span style={hintStyle}>설정을 저장해야 실행할 수 있습니다</span> : null}
                <button
                  type="button"
                  className="btn btn-sm btn-dark"
                  onClick={() => void runBacktest()}
                  disabled={backtesting || isDirty}
                >
                  {backtesting ? "실행 중…" : "실행"}
                </button>
              </div>
            </div>
            {backtesting ? <AppLoadingProgress title="백테스트 실행 중..." progress={backtestProgress} /> : null}
            {backtest && !backtesting ? (
              <>
                <div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: "var(--fs-sm)", padding: "2px 0 8px" }}>
                  <span>
                    {backtest.start_date} ~ {backtest.end_date}
                  </span>
                  <PerformanceSummary
                    label="전략"
                    totalPct={backtest.strategy_total_pct}
                    cagrPct={backtest.strategy_cagr_pct}
                    mddPct={backtest.strategy_mdd_pct}
                    sortino={backtest.strategy_sortino}
                  />
                  <PerformanceSummary
                    label={`${backtest.benchmark_name}(${backtest.benchmark_ticker})`}
                    totalPct={backtest.benchmark_total_pct}
                    cagrPct={backtest.benchmark_cagr_pct}
                    mddPct={backtest.benchmark_mdd_pct}
                    sortino={backtest.benchmark_sortino}
                  />
                  {backtest.reference_name && backtest.reference_total_pct != null ? (
                    <PerformanceSummary
                      label={`${backtest.reference_name} (참고)`}
                      totalPct={backtest.reference_total_pct}
                      cagrPct={backtest.reference_cagr_pct}
                      mddPct={backtest.reference_mdd_pct}
                      sortino={backtest.reference_sortino}
                    />
                  ) : null}
                  <span>
                    초과{" "}
                    <b style={{ color: signColor(backtest.strategy_total_pct - backtest.benchmark_total_pct) }}>
                      {formatSigned(backtest.strategy_total_pct - backtest.benchmark_total_pct)}p
                    </b>
                  </span>
                </div>
                {viewMode === "daily" && backtest.daily.length === 0 ? (
                  <span style={{ ...hintStyle, fontSize: "var(--fs-sm)" }}>
                    일간은 따로 계산합니다. 이 탭에서 실행을 누르면 일별 성과가 표시됩니다.
                  </span>
                ) : viewMode === "daily" ? (
                  // 월간·연간과 같이 autoHeight — 카드 안에서 스크롤하지 않고 브라우저 스크롤로 본다.
                  <AppAgGrid<BacktestDayRow>
                    rowData={backtest.daily}
                    columnDefs={dailyColumns}
                    theme={gridTheme}
                    minHeight={0}
                    height="auto"
                    gridOptions={{ domLayout: "autoHeight" }}
                    getRowId={(p) => p.data.date}
                  />
                ) : viewMode === "monthly" ? (
                  <AppAgGrid<BacktestMonthRow>
                    rowData={backtest.monthly}
                    columnDefs={backtestColumns}
                    theme={gridTheme}
                    minHeight={0}
                    height="auto"
                    gridOptions={{ domLayout: "autoHeight" }}
                    getRowClass={(p) => (p.data?.is_pending ? "steadyPendingRow" : "")}
                    getRowId={(p) => p.data.month}
                  />
                ) : (
                  <>
                    <AppAgGrid<YearRow>
                      rowData={yearRows}
                      columnDefs={yearColumns}
                      theme={gridTheme}
                      minHeight={0}
                      height="auto"
                      gridOptions={{ domLayout: "autoHeight" }}
                      getRowId={(p) => p.data.year}
                    />
                    {yearRows.some(
                      (row) => row.strategy_partial || row.benchmark_partial || row.reference_partial,
                    ) ? (
                      <span style={{ ...hintStyle, fontSize: "var(--fs-sm)" }}>
                        * 12개월이 다 차지 않은 해 — 있는 달만 합성한 부분 기간입니다.
                      </span>
                    ) : null}
                  </>
                )}
              </>
            ) : !backtesting ? (
              <span style={{ ...hintStyle, fontSize: "var(--fs-sm)" }}>
                실행을 누르면 월별 성과가 표시됩니다. 기간은 위 변수 설정에서 바꿉니다.
              </span>
            ) : null}
          </div>
        </div>
      </div>
    </PageFrame>
  );
}
