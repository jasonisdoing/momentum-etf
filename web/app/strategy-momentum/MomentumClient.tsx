"use client";

import type { ColDef } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AppAgGrid } from "../components/AppAgGrid";
import { AppLoadingProgress, type LoadingProgress } from "../components/AppLoadingProgress";
import { BacktestSummary } from "../components/BacktestSummary";
import { NavTabs } from "../components/NavTabs";
import { PageFrame } from "../components/PageFrame";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";
import { formatDateWithWeekday } from "@/lib/datetime";
import { formatPoolLabel, type PoolLabelSource } from "@/lib/pool-label";
import { formatKorMarketCap } from "@/lib/market-cap-format";
import {
  INDUSTRY_COLUMN_MIN_WIDTH,
  INDUSTRY_COLUMN_WIDTH,
  formatSignedPct,
  marketBadgeCellStyle,
  renderHighDrawdownCell,
  renderIndustryCell,
  signColor,
} from "@/lib/grid-cells";
import { isTrendBroken, renderStockNameCell } from "@/lib/name-highlight";
import { formatPrice } from "../../lib/price-format";

const gridTheme = createAppGridTheme();

// 풀별로 따로 저장되는 설정 — 풀 셀렉트를 바꾸면 그 풀의 저장분으로 폼이 전환된다.
// 슬리피지는 종목풀 설정을, 백테스트 기간은 실행할 때 고른 값을 쓴다 — 여기 없다.
type PoolSettings = {
  top_n: number;
  max_per_industry: number;
  short_ma_days: number;
  long_ma_days: number;
};

type Settings = PoolSettings & { pool: string };

/** 저장된 값이 선택지에 없으면 함께 노출한다 — 빼면 셀렉트가 빈칸이 되어 무엇이 저장돼 있는지 알 수 없다. */
function withSavedValue(options: number[], saved: string | undefined): number[] {
  const value = Number(saved);
  if (!Number.isFinite(value) || options.includes(value)) return options;
  return [...options, value].sort((a, b) => a - b);
}

type PickRow = {
  rank: number | null;
  // 다음 주 예상 순위 — 현재 가격 기준으로 같은 선정 규칙을 돌린 순위 (자격 미달은 null).
  expected_rank: number | null;
  is_reserve: boolean;
  // 현재 표(선정+후보) 밖인데 다음 주 편입이 예상되는 종목 — 하단 별도 행.
  is_expected_only: boolean;
  // 주중 매도 — 보유 자격(장기>0 & 단기≥0) 상실로 매도됨(체결 완료) / 다음 시가 매도 예정.
  is_exited?: boolean;
  is_exit_pending?: boolean;
  exit_date?: string | null;
  streak_weeks: number | null;
  next_week_expected: boolean;
  ticker: string;
  name: string;
  // 종목의 소속 마켓(KOSPI/KOSDAQ) — 한국 통합 풀 구분 표시용, 없으면 빈 값.
  market: string;
  industry: string;
  currency: string;
  price: number | null;
  monthly_returns: Record<string, number | null>;
  daily_change_pct: number | null;
  high_drawdown_pct: number | null;
  market_cap_eok: number | null;
  signal_short_pct: number | null;
  signal_long_pct: number | null;
  current_short_pct: number | null;
  current_long_pct: number | null;
};

type PicksResult = {
  as_of: string;
  portfolio_week: string;
  rebalance_date: string;
  signal_date: string;
  universe_count: number;
  candidate_count: number;
  // 풀의 국가·통화 — 마켓·시가총액 컬럼 표시와 티커 표기(ASX: 등)를 정한다.
  country: string;
  currency: string;
  monthly_return_labels: string[];
  rows: PickRow[];
};

// 월간 행은 집계만 담는다 — 매매 내역(편입·편출·교체율·보유 수)은 주간 행이 담당한다.
type BacktestMonthRow = {
  month: string;
  strategy_pct: number | null;
  benchmark_pct: number | null;
  reference_pct: number | null;
};

// 주간 행 — 달력 주 단위. 기준일은 그 주 마지막 거래일, 편입·편출은 그 주 체결분.
type BacktestWeekRow = {
  week_end: string;
  strategy_pct: number | null;
  benchmark_pct: number | null;
  reference_pct: number | null;
  holdings_count: number;
  turnover_pct: number | null;
  added: string[];
  removed: string[];
  // 다음 교체일에 실행될 예정 행 — 아직 수익률 없음.
  is_pending?: boolean;
};

// 체결 행 — 편입~편출 한 쌍. exit_date 가 없으면 아직 보유 중이다.
type BacktestTradeRow = {
  ticker: string;
  name: string;
  entry_date: string;
  entry_price: number;
  exit_date: string | null;
  exit_price: number | null;
  return_pct: number | null;
  days: number;
  reason: string;
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
  weekly: BacktestWeekRow[];
  daily: BacktestDayRow[];
  trades: BacktestTradeRow[];
};

// 풀 옵션 — 공용 라벨 소스에 국가·통화·풀 성격(stock/etf)이 붙는다.
type PoolOption = PoolLabelSource & {
  country_code?: string;
  currency?: string;
  pool_kind?: string | null;
};

type View = {
  settings: Settings;
  // 풀별 저장 설정 맵 — 셀렉트 전환 시 즉시 그 풀의 값으로 폼을 채운다.
  settings_by_pool?: Record<string, PoolSettings>;
  pool_options?: PoolOption[];
  // 전략 전용 이평선 — momentum_settings 에 저장되며 종목풀 설정과 무관하다.
  ma_rule?: {
    short_ma_days: number;
    long_ma_days: number;
    ma_day_options: number[];
  };
  // 기간 선택지는 서버가 가격 캐시 범위로 계산해 내려준다 (종목풀 백테스트와 동일).
  month_options?: number[];
  // 셀렉트 선택지 — 백엔드 상수가 단일 소스(프론트에 복사본을 두지 않는다).
  constraints?: {
    top_n_options: number[];
    max_per_industry_options: number[];
  };
  picks: PicksResult | null;
};

// 선정 결과를 바꾸는 설정 — 이 값들이 바뀔 때만 저장 후 선정을 다시 계산한다.
// 슬리피지와 백테스트 기간은 백테스트에만 쓰이므로 선정을 다시 돌릴 이유가 없다.
const PICK_AFFECTING_KEYS = [
  "pool",
  "top_n",
  "max_per_industry",
] as const;

function needsRepick(before: Settings | null, after: Settings): boolean {
  if (!before) return true;
  return PICK_AFFECTING_KEYS.some((key) => before[key] !== after[key]);
}

// 백테스트 표 보기 단위 — /compare 의 연간·월간·일간 구분에 주간을 더한 것.
const VIEW_MODES = [
  { key: "yearly", label: "연간" },
  { key: "monthly", label: "월간" },
  { key: "weekly", label: "주간" },
  { key: "daily", label: "일간" },
  { key: "trades", label: "체결" },
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

// 부호 %·색 헬퍼는 공용 모듈(@/lib/grid-cells)을 쓴다.
const formatSigned = formatSignedPct;

/** 성과 요약 한 덩어리 — 전략·벤치마크·참고 지수를 같은 형식으로 보여준다. */
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

export function MomentumClient() {
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
  // 기본 보기는 주간 — 매매 내역(편입·편출)이 실제 체결 주에 붙는 표라 주간 점검의 기준이다.
  const [viewMode, setViewMode] = useState<ViewMode>("weekly");
  const autoPickedRef = useRef(false);

  // 설정 입력 초안 (문자열로 보관해 입력 중 상태를 그대로 둔다)
  const [draft, setDraft] = useState<Record<string, string>>({});
  // 초안 초기값은 자리만 잡는다 — 설정을 받은 뒤 applyView 가 항상 덮어쓰며,
  // 설정을 못 받으면 폼 자체를 그리지 않으므로 이 값이 화면에 보이는 경우는 없다.
  const [draftPool, setDraftPool] = useState<string>("");
  const [draftMaxPerIndustry, setDraftMaxPerIndustry] = useState<number>(0);
  // 이평선 초안 — 전략 전용 값(momentum_settings)으로 풀별 저장되며 종목풀 설정과 무관하다.
  const [draftMaRule, setDraftMaRule] = useState<{ short: number; long: number } | null>(null);

  // 풀별 설정을 폼 초안에 채운다 — 풀 셀렉트 전환과 응답 반영이 같은 경로를 쓴다.
  const fillDrafts = useCallback((values: PoolSettings) => {
    setDraftMaxPerIndustry(values.max_per_industry);
    setDraft({
      top_n: String(values.top_n),
    });
    setDraftMaRule({ short: values.short_ma_days, long: values.long_ma_days });
  }, []);

  const applyView = useCallback(
    (data: View) => {
      setView(data);
      setDraftPool(data.settings.pool);
      fillDrafts(data.settings);
    },
    [fillDrafts],
  );

  const isNewPoolDraft = view != null && draftPool !== "" && view.settings_by_pool?.[draftPool] == null;

  const load = useCallback(async (): Promise<boolean> => {
    setLoading(true);
    try {
      const resp = await fetch("/api/strategy-momentum", { cache: "no-store" });
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
      const resp = await fetch("/api/strategy-momentum/picks", { method: "POST" });
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

  // 저장 공용 경로 — 폼 저장 버튼과 풀 전환(저장된 풀의 설정 자동 적용)이 함께 쓴다.
  const persistSettings = useCallback(
    async (settings: Settings, successMessage: string) => {
      setSaving(true);
      try {
        const maRuleChanged =
          view?.ma_rule != null &&
          (settings.short_ma_days !== view.ma_rule.short_ma_days ||
            settings.long_ma_days !== view.ma_rule.long_ma_days);
        const resp = await fetch("/api/strategy-momentum", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ settings }),
        });
        const payload = await resp.json();
        if (!resp.ok) throw new Error(payload?.error ?? "설정을 저장하지 못했습니다.");
        const saved = payload as View;
        // 이평선이 바뀌면 이격 점수가 달라지므로 무조건 재선정한다.
        const repick = maRuleChanged || needsRepick(view?.settings ?? null, saved.settings);
        // 선정에 영향이 없는 변경(슬리피지·백테스트 기간)이면 기존 선정 결과를 그대로 둔다.
        applyView({ ...saved, picks: repick ? null : (view?.picks ?? null) });
        // 백테스트는 어느 설정이 바뀌든 결과가 달라지므로 비운다.
        setBacktest(null);
        toast.success(repick ? `${successMessage} 선정을 다시 계산합니다.` : successMessage);
        if (repick) await runPicks();
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "설정을 저장하지 못했습니다.");
      } finally {
        setSaving(false);
      }
    },
    [applyView, runPicks, toast, view],
  );

  const saveSettings = useCallback(async () => {
    const topN = Number(draft.top_n);
    if (!Number.isFinite(topN) || draftMaRule == null) {
      toast.error("설정 값이 올바르지 않습니다.");
      return;
    }
    await persistSettings(
      {
        pool: draftPool,
        top_n: topN,
        max_per_industry: draftMaxPerIndustry,
        // 이평선은 전략 전용 값 — 설정의 일부로 풀별 저장한다(종목풀 설정과 무관).
        short_ma_days: draftMaRule.short,
        long_ma_days: draftMaRule.long,
      },
      "설정을 저장했습니다.",
    );
  }, [draft, draftMaRule, draftMaxPerIndustry, draftPool, persistSettings, toast]);

  // 풀 셀렉트 변경 — 그 풀의 저장 설정이 있으면 **즉시 전환·저장·재선정**한다
  // (전환은 초안이 아니라 컨텍스트 스위치다). 저장분이 없는 풀(첫 설정)만 초안으로
  // 남기고, 현재 폼 값을 시작점으로 보여준 뒤 저장을 요구한다.
  const handlePoolChange = useCallback(
    (pool: string) => {
      setDraftPool(pool);
      const saved = view?.settings_by_pool?.[pool];
      if (saved) {
        fillDrafts(saved);
        void persistSettings({ pool, ...saved }, "풀을 전환했습니다.");
      }
    },
    [fillDrafts, persistSettings, view],
  );

  const [backtestMonths, setBacktestMonths] = useState<number>(12);

  const runBacktest = useCallback(async () => {
    const months = backtestMonths;
    setBacktesting(true);
    setBacktestProgress({ percent: 10, message: "월별 리밸런싱 시뮬레이션 중" });
    const stopRamp = startProgressRamp(setBacktestProgress);
    try {
      const resp = await fetch("/api/strategy-momentum/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // 한 번 실행으로 연간·월간·주간·일간을 모두 만든다 — 탭 전환 시 재실행이 없도록.
        body: JSON.stringify({ months, include_daily: true }),
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
  }, [toast, backtestMonths]);

  // 저장하지 않은 입력이 있으면 실행 결과가 화면 값과 어긋난다 — 저장을 먼저 요구한다.
  const isDirty = useMemo(() => {
    if (!view) return false;
    const saved = view.settings;
    return (
      draftPool !== saved.pool ||
      draftMaxPerIndustry !== saved.max_per_industry ||
      draft.top_n !== String(saved.top_n) ||
      (draftMaRule != null &&
        view.ma_rule != null &&
        (draftMaRule.short !== view.ma_rule.short_ma_days ||
          draftMaRule.long !== view.ma_rule.long_ma_days))
    );
  }, [draft, draftMaRule, draftMaxPerIndustry, draftPool, view]);

  const monthlyLabels = view?.picks?.monthly_return_labels ?? [];
  // 선정 결과 풀의 국가 — 마켓·시가총액 컬럼 표시와 티커 표기(ASX:)를 정한다.
  const picksCountry = view?.picks?.country ?? "";
  // 업종 UI 노출 — 종목풀 설정의 풀 성격(pool_kind) 토글이 1순위:
  // 개별주(stock)면 표시, ETF 면 숨김. 미설정(구 문서)이면 선정 행에 값이 있는지로
  // 추정한다(pools-rank 와 같은 패턴). 업종을 모르는 종목엔 상한이 원래 미적용이라
  // 숨겨도 동작은 그대로다.
  const poolKind = view?.pool_options?.find((option) => option.ticker_type === view?.settings.pool)?.pool_kind ?? "";
  const hasIndustryData =
    poolKind === "stock"
      ? true
      : poolKind === "etf"
        ? false
        : view?.picks
          ? view.picks.rows.some((row) => row.industry)
          : true;
  const pickColumns = useMemo<ColDef<PickRow>[]>(() => {
    return [
      {
        headerName: "순위",
        field: "rank",
        headerTooltip:
          "판정일 기준 순위 — 선정 1~N, 차순위 그 아래. 매도예정 = 자격 상실로 다음 시가 매도, 매도 = 주중에 이미 매도됨",
        width: 68,
        type: "numericColumn",
        cellDataType: "text",
        cellRenderer: (p: { value?: number | null; data?: PickRow }) => {
          if (p.data?.is_exited) return <span style={{ color: "var(--text-muted)" }}>매도</span>;
          if (p.data?.is_exit_pending) return <span style={{ color: "#d62828", fontWeight: 700 }}>매도예정</span>;
          return <span>{p.value == null ? "-" : String(p.value)}</span>;
        },
      },
      {
        headerName: "예상",
        field: "expected_rank",
        headerTooltip:
          "다음 주 예상 순위 — 오늘까지의 가격으로 같은 선정 규칙을 돌린 순위 (편입 예상 1~N, 그 아래는 점수순). 자격 미달은 '-'",
        width: 56,
        type: "numericColumn",
        valueFormatter: (p) => (p.value == null ? "-" : String(p.value)),
      },
      {
        headerName: "연속",
        field: "streak_weeks",
        headerTooltip: "이번 포트폴리오까지 몇 주 연속 편입됐는지 (신규 = 이번 주 첫 편입, 최대 12주 추적)",
        width: 60,
        valueFormatter: (p) => (p.value == null ? "-" : p.value <= 1 ? "신규" : p.value >= 12 ? "12+" : `${p.value}주`),
        cellStyle: (p) => ({
          color: p.value != null && p.value <= 1 && !p.data?.is_reserve ? "var(--up-color, #d64545)" : "inherit",
        }),
      },
      {
        headerName: "다음주",
        field: "next_week_expected",
        headerTooltip:
          "오늘까지의 가격(실시간 반영)으로 같은 규칙을 돌렸을 때의 다음 주 예상 — 유지(보유 중·계속 편입) / 신규(새로 편입) / -(편출 예상). 확정은 교체일 직전 판정일 종가",
        width: 64,
        // boolean 필드는 AG Grid 가 체크박스로 자동 렌더링하므로 텍스트로 강제한다.
        cellDataType: "text",
        cellRenderer: (p: { value?: boolean; data?: PickRow }) => {
          if (!p.value) return <span style={{ color: "var(--text-muted)" }}>-</span>;
          // 이번 달 보유(선정분)면 계속 편입 예상 = 유지, 차순위·표 밖 예상 행은 새로 편입 예상 = 신규.
          const label = p.data && !p.data.is_reserve && !p.data.is_expected_only ? "유지" : "신규";
          return <span style={{ color: "#2f9e44", fontWeight: 700 }}>{label}</span>;
        },
      },
      {
        headerName: "고점",
        field: "high_drawdown_pct",
        headerTooltip: "캐시 전 기간 최고가 대비 마지막 종가(%) — pools-rank 고점과 같은 규칙, 0 = 신고점",
        width: 80,
        type: "rightAligned",
        // `/pools-rank` 고점 컬럼과 같은 공용 렌더러 — 0 이면 ⭐신고점.
        cellRenderer: (p: { value?: number | null }) => renderHighDrawdownCell(p.value, 1),
      },
      // 마켓(KOSPI/KOSDAQ)은 한국 풀에서만 의미가 있다 — 통합 풀 구분 표시용.
      ...(picksCountry === "kor"
        ? [
            {
              headerName: "마켓",
              field: "market",
              headerTooltip: "종목의 소속 마켓 — 코스피+코스닥 통합 풀 구분용",
              width: 80,
              // `/pools-rank` 마켓 컬럼과 같은 공용 배지 스타일.
              cellStyle: (p) => marketBadgeCellStyle(p.value),
            } as ColDef<PickRow>,
          ]
        : []),
      {
        headerName: "티커",
        field: "ticker",
        // 한국 6자리 코드 + 상세 링크 아이콘이 잘리지 않는 폭.
        width: 96,
        // 호주 풀은 미국 동일 심볼과 구분되게 ASX: 접두를 붙인다 (pools-rank 와 동일).
        cellRenderer: (p: { value: string | null | undefined }) => {
          const raw = String(p.value ?? "-");
          const display = picksCountry === "au" && raw !== "-" && !raw.startsWith("ASX:") ? `ASX:${raw}` : raw;
          return <TickerDetailLink ticker={display} displayTicker={display} />;
        },
      },
      {
        headerName: "종목명",
        field: "name",
        // 이 표에서 유일한 flex 컬럼 — 남는 폭을 종목명이 가져간다.
        // 상한(maxWidth)을 두지 않아 넓은 화면에서 계속 늘어난다.
        flex: 1,
        minWidth: 220,
        // 종목명 셀 표준(`@/lib/name-highlight`) — 말줄임·레버리지 강조·추세 이탈 배지가 전 화면 공통이다.
        // 이탈 판정은 `현재-단기/장기` 기준이다. 선정 당시가 아니라 지금 상태를 보여준다.
        cellRenderer: (p: { value?: string | null; data?: PickRow }) =>
          renderStockNameCell(p.value, {
            trendBroken: isTrendBroken(p.data?.current_short_pct, p.data?.current_long_pct),
          }),
      },
      // 업종 데이터가 아예 없는 풀(ETF 모음 등)에서는 빈 컬럼을 숨긴다.
      ...(hasIndustryData
        ? [
            {
              headerName: "업종",
              field: "industry",
              headerTooltip: "한국은 네이버 분류, 미국은 지수 구성종목의 yfinance 분류",
              width: INDUSTRY_COLUMN_WIDTH,
              minWidth: INDUSTRY_COLUMN_MIN_WIDTH,
              cellRenderer: (p: { value?: string }) => renderIndustryCell(p.value),
            } as ColDef<PickRow>,
          ]
        : []),
      {
        headerName: "일간(%)",
        field: "daily_change_pct",
        headerTooltip: "실시간 일간 등락률 (실시간 실패 시 캐시 종가 기준)",
        width: 88,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value, 2),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
      {
        headerName: "현재가",
        field: "price",
        headerTooltip: "실시간 현재가 (실시간 실패 시 가격 캐시 최신 종가, 통화는 종목풀 기준)",
        width: 100,
        type: "numericColumn",
        valueFormatter: (p) => formatPrice(p.value, p.data?.currency),
      },
      // 시가총액 소스(네이버)가 한국 전용이라 한국 풀에서만 보여준다.
      ...(picksCountry === "kor"
        ? [
            {
              headerName: "시가총액",
              field: "market_cap_eok",
              headerTooltip: "네이버 시가총액 (/kor-market-stock 과 같은 소스, 10분 캐시)",
              width: 120,
              type: "numericColumn",
              valueFormatter: (p) => formatKorMarketCap(p.value),
            } as ColDef<PickRow>,
          ]
        : []),
      // 월별 수익률 — pools-rank 월별과 같은 계산(전월 말 종가 대비, 이번 달은 마지막
      // 종가까지)의 최근 6개월. 라벨은 서버가 내려주고 헤더는 (%) 없이 표시한다.
      ...monthlyLabels.map(
        (label): ColDef<PickRow> => ({
          headerName: label.replace("(%)", ""),
          colId: label,
          headerTooltip: "전월 말 종가 대비 수익률 (이번 달은 실시간 현재가까지) — pools-rank 월별과 같은 계산",
          width: 92,
          type: "numericColumn",
          valueGetter: (p) => p.data?.monthly_returns?.[label] ?? null,
          valueFormatter: (p) => formatSigned(p.value, 1),
          cellStyle: (p) => ({ color: signColor(p.value) }),
        }),
      ),
      {
        headerName: "판정일-단기",
        field: "signal_short_pct",
        headerTooltip: "판정일 종가 기준 단기 이평선 이격 — 이번 달 선정에 쓰인 값 (음수면 후보 제외)",
        width: 108,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value, 1),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
      {
        headerName: "판정일-장기",
        field: "signal_long_pct",
        headerTooltip: "판정일 종가 기준 장기 이평선 이격 = 이번 달 선정 점수",
        width: 108,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value, 1),
        // 선정 점수라 볼드 유지, 색은 부호 색 규칙.
        cellStyle: (p) => ({ fontWeight: 700, color: signColor(p.value) }),
      },
      {
        headerName: "현재-단기",
        field: "current_short_pct",
        headerTooltip: "오늘까지의 가격(실시간 반영) 기준 단기 이격 — 다음달 예상 판정에 쓰이는 값",
        width: 100,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value, 1),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
      {
        headerName: "현재-장기",
        field: "current_long_pct",
        headerTooltip: "오늘까지의 가격(실시간 반영) 기준 장기 이격 — 다음달 예상 판정에 쓰이는 값",
        width: 100,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value, 1),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
    ];
    // 월별 라벨·국가·업종 유무가 선정 응답에 실려 온다 — 바뀌면(월 전환·풀 전환) 컬럼도 다시 만든다.
  }, [hasIndustryData, monthlyLabels, picksCountry]);

  // 월간 표 — 연간과 같은 집계형(월/전략/벤치/참고). 매매 내역은 주간 표가 담당한다.
  const backtestColumns = useMemo<ColDef<BacktestMonthRow>[]>(() => {
    if (!backtest) return [];
    const columns: ColDef<BacktestMonthRow>[] = [
      {
        headerName: "월",
        field: "month",
        width: 116,
        cellStyle: () => ({ fontWeight: 700 }),
      },
      {
        headerName: "전략(%)",
        field: "strategy_pct",
        width: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value),
        cellStyle: (p) => ({ color: signColor(p.value), fontWeight: 700 }),
      },
      {
        headerName: "벤치마크(%)",
        headerTooltip: `벤치마크 ${backtest.benchmark_name}(${backtest.benchmark_ticker})`,
        field: "benchmark_pct",
        width: 140,
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
      pctColumn("벤치마크(%)", "benchmark_pct", `${backtest.benchmark_name}(${backtest.benchmark_ticker})`),
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

  const tradeColumns = useMemo<ColDef<BacktestTradeRow>[]>(
    () => [
      { headerName: "티커", field: "ticker", width: 96 },
      { headerName: "종목명", field: "name", flex: 1, minWidth: 180 },
      { headerName: "편입일", field: "entry_date", width: 116 },
      {
        headerName: "매수가",
        field: "entry_price",
        width: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatPrice(p.value as number, view?.picks?.currency),
      },
      {
        headerName: "청산일",
        field: "exit_date",
        width: 116,
        valueFormatter: (p) => (p.value ? String(p.value) : "-"),
      },
      {
        headerName: "청산가",
        field: "exit_price",
        headerTooltip: "보유중 행은 마지막 종가",
        width: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatPrice(p.value as number, view?.picks?.currency),
      },
      {
        headerName: "수익률(%)",
        field: "return_pct",
        width: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value),
        cellStyle: (p) => ({ color: signColor(p.value), fontWeight: 700 }),
      },
      { headerName: "보유일", field: "days", width: 84, type: "numericColumn" },
      { headerName: "사유", field: "reason", width: 110 },
    ],
    [view?.picks?.currency],
  );

  const yearRows = useMemo<YearRow[]>(() => (backtest ? toYearRows(backtest.monthly) : []), [backtest]);

  // 주간 표 — 매매 일지. 주 수익률 + 그 주에 체결된 편입·편출과 주말 보유 수·교체율.
  const weeklyColumns = useMemo<ColDef<BacktestWeekRow>[]>(() => {
    if (!backtest) return [];
    const pctColumn = (headerName: string, field: keyof BacktestWeekRow, headerTooltip?: string): ColDef<BacktestWeekRow> => ({
      headerName,
      field,
      headerTooltip,
      width: 120,
      type: "numericColumn",
      valueFormatter: (p) => formatSigned(p.value as number | null),
      cellStyle: (p) => ({ color: signColor(p.value as number | null), fontWeight: field === "strategy_pct" ? 700 : 400 }),
    });
    const columns: ColDef<BacktestWeekRow>[] = [
      {
        headerName: "기준일",
        field: "week_end",
        headerTooltip: "그 주 마지막 거래일 — 수익률은 그 주 성과, 편입·편출은 그 주에 체결된 매매",
        width: 148,
        valueFormatter: (p) => formatDateWithWeekday(String(p.value ?? "")),
        cellStyle: () => ({ fontWeight: 700 }),
      },
      pctColumn("전략(%)", "strategy_pct", "그 주 보유 포트폴리오의 수익률 (교체 비용 반영)"),
      pctColumn("벤치마크(%)", "benchmark_pct", `${backtest.benchmark_name}(${backtest.benchmark_ticker})`),
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
    columns.push(
      {
        headerName: "종목 수",
        field: "holdings_count",
        headerTooltip: "다음 교체 직전까지 들고 가는 종목 수 (주중 매도분 제외)",
        width: 74,
        type: "numericColumn",
      },
      {
        headerName: "교체율(%)",
        field: "turnover_pct",
        headerTooltip: "이 교체에서 편입된 슬롯 비중 (편입 수 ÷ 종목 수 설정)",
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
        cellClass: "momentumWrapCell",
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
        cellClass: "momentumWrapCell",
        valueFormatter: (p) => (p.value?.length ? p.value.join(", ") : "-"),
        cellStyle: () => ({ color: "var(--down-color, #2f6fd0)" }),
      },
    );
    return columns;
  }, [backtest]);

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
        "벤치마크(%)",
        "benchmark_pct",
        "benchmark_partial",
        `${backtest.benchmark_name}(${backtest.benchmark_ticker})`,
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
      <PageFrame title="모멘텀 전략" fullWidth>
        <div style={{ ...hintStyle, padding: 20 }}>불러오는 중…</div>
      </PageFrame>
    );
  }
  if (!view) {
    // 설정을 못 받은 상태 — 값을 지어내 폼을 그리지 않고 실패와 재시도만 제공한다.
    return (
      <PageFrame title="모멘텀 전략" fullWidth>
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

  // 보유 수 — 판정일 선정 − 주중 매도(체결 완료).
  const selectedCount = view.picks?.rows.filter((row) => !row.is_reserve && !row.is_exited).length ?? 0;
  const reserveCount = view.picks?.rows.filter((row) => row.is_reserve).length ?? 0;

  return (
    <PageFrame title="모멘텀 전략" fullWidth>
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
                    onChange={(e) => handlePoolChange(e.target.value)}
                  >
                    {/* 설정을 못 받으면 폼 자체를 그리지 않으므로(로드 실패 화면) 폴백 목록이 필요 없다. */}
                    {(view.pool_options ?? []).map((pool) => (
                      <option key={pool.ticker_type} value={pool.ticker_type}>
                        {formatPoolLabel(pool)}
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
                    {withSavedValue(view.constraints?.top_n_options ?? [], draft.top_n).map((n) => (
                      <option key={n} value={n}>
                        {n}
                      </option>
                    ))}
                  </select>
                </label>
                {/* 업종 데이터가 없는 풀(ETF 모음 등)은 상한이 무의미하므로 셀렉트를 숨긴다 (저장값은 유지). */}
                {hasIndustryData ? (
                  <label className="appLabeledField">
                    <span className="appLabeledFieldLabel">업종별 최대 보유</span>
                    <select
                      className="form-select form-select-sm"
                      style={{ width: 80 }}
                      value={draftMaxPerIndustry}
                      onChange={(e) => setDraftMaxPerIndustry(Number(e.target.value))}
                    >
                      {withSavedValue(view.constraints?.max_per_industry_options ?? [], String(draftMaxPerIndustry)).map((count) => (
                        <option key={count} value={count}>
                          {count}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                {draftMaRule != null && view.ma_rule != null ? (
                  <>
                    <label className="appLabeledField">
                      <span className="appLabeledFieldLabel">이평선</span>
                      <span className="appMaRuleRow">
                        <select
                          className="form-select appMaRuleSelect"
                          value={draftMaRule.short}
                          onChange={(e) => setDraftMaRule((r) => r && { ...r, short: Number(e.target.value) })}
                        >
                          {view.ma_rule.ma_day_options.map((d) => (
                            <option key={d} value={d}>
                              단기 {d}일
                            </option>
                          ))}
                        </select>
                        <select
                          className="form-select appMaRuleSelect"
                          value={draftMaRule.long}
                          onChange={(e) => setDraftMaRule((r) => r && { ...r, long: Number(e.target.value) })}
                        >
                          {view.ma_rule.ma_day_options.map((d) => (
                            <option key={d} value={d}>
                              장기 {d}일
                            </option>
                          ))}
                        </select>
                      </span>
                    </label>
                  </>
                ) : null}
              </div>
              <div className="appMainHeaderRight">
                {isNewPoolDraft ? (
                  <span style={{ ...hintStyle, color: "var(--up-color, #d64545)", fontWeight: 700 }}>
                    이 풀은 첫 설정 — 저장해야 선정·백테스트가 실행됩니다
                  </span>
                ) : isDirty ? (
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
                    <b style={{ color: "inherit" }}>{formatDateWithWeekday(view.picks.portfolio_week)} 포트폴리오</b> ·
                    체결 {view.picks.rebalance_date} (판정 {view.picks.signal_date}) · {view.picks.universe_count} →{" "}
                    {view.picks.candidate_count} → {selectedCount}
                    {reserveCount > 0 ? ` (+${reserveCount})` : ""}
                  </span>
                ) : (
                  <span style={{ ...hintStyle, fontSize: "var(--fs-sm)" }}>
                    {pickFailed
                      ? "선정 결과를 불러오지 못했습니다. 설정을 저장하거나 새로고침하세요."
                      : "계산 중…"}
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
                getRowClass={(p) => {
                  // 추세 이탈은 종목명 뒤 ❗ 와 같은 조건으로 행을 연한 회색으로 눌러 둔다.
                  const classes: string[] = [];
                  if (p.data?.is_reserve) classes.push("momentumReserveRow");
                  // 주중 매도 예정 — 판정만 끝나고 체결 전 (백테스트 예정 행과 같은 스타일).
                  if (p.data?.is_exit_pending) classes.push("momentumPendingRow");
                  // 주중 매도 완료 — 더는 보유가 아니다.
                  if (p.data?.is_exited) classes.push("appTrendBrokenRow");
                  if (isTrendBroken(p.data?.current_short_pct, p.data?.current_long_pct)) {
                    classes.push("appTrendBrokenRow");
                  }
                  return classes.join(" ");
                }}
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
              </div>
              <div className="appMainHeaderRight">
                {isDirty ? <span style={hintStyle}>설정을 저장해야 실행할 수 있습니다</span> : null}
                <select
                  className="form-select form-select-sm"
                  style={{ width: "auto" }}
                  value={String(backtestMonths)}
                  disabled={backtesting}
                  onChange={(event) => setBacktestMonths(Number(event.target.value))}
                >
                  {(view.month_options ?? [12]).map((m) => (
                    <option key={m} value={m}>
                      {m}개월
                    </option>
                  ))}
                </select>
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
                <BacktestSummary
                  startDate={backtest.start_date}
                  endDate={backtest.end_date}
                  strategy={{
                    label: "전략",
                    totalPct: backtest.strategy_total_pct,
                    cagrPct: backtest.strategy_cagr_pct,
                    mddPct: backtest.strategy_mdd_pct,
                    sortino: backtest.strategy_sortino,
                  }}
                  benchmark={{
                    label: `${backtest.benchmark_name}(${backtest.benchmark_ticker})`,
                    totalPct: backtest.benchmark_total_pct,
                    cagrPct: backtest.benchmark_cagr_pct,
                    mddPct: backtest.benchmark_mdd_pct,
                    sortino: backtest.benchmark_sortino,
                  }}
                  extra={
                    backtest.reference_name && backtest.reference_total_pct != null
                      ? {
                          label: `${backtest.reference_name} (참고)`,
                          totalPct: backtest.reference_total_pct,
                          cagrPct: backtest.reference_cagr_pct,
                          mddPct: backtest.reference_mdd_pct,
                          sortino: backtest.reference_sortino,
                        }
                      : null
                  }
                />
                <NavTabs
                  items={VIEW_MODES}
                  value={viewMode}
                  onChange={setViewMode}
                  label="백테스트 보기 단위"
                  style={{ marginBottom: 10 }}
                />
                {viewMode === "trades" ? (
                  <AppAgGrid<BacktestTradeRow>
                    rowData={backtest.trades}
                    columnDefs={tradeColumns}
                    theme={gridTheme}
                    minHeight={0}
                    height="auto"
                    gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                    getRowClass={(p) => (p.data?.exit_date ? "" : "momentumPendingRow")}
                  />
                ) : viewMode === "weekly" ? (
                  <AppAgGrid<BacktestWeekRow>
                    rowData={backtest.weekly}
                    columnDefs={weeklyColumns}
                    theme={gridTheme}
                    minHeight={0}
                    height="auto"
                    gridOptions={{ domLayout: "autoHeight" }}
                    getRowClass={(p) => (p.data?.is_pending ? "momentumPendingRow" : "")}
                    getRowId={(p) => p.data.week_end}
                  />
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
