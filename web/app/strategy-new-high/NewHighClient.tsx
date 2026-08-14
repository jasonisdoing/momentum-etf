"use client";

import type { ColDef } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useState } from "react";

import { HoldingChart, type HoldingChartData } from "./HoldingChart";
import { AppAgGrid } from "../components/AppAgGrid";
import { NavTabs } from "../components/NavTabs";
import { PageFrame } from "../components/PageFrame";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";
import {
  INDUSTRY_COLUMN_MIN_WIDTH,
  INDUSTRY_COLUMN_WIDTH,
  STOCK_NAME_COLUMN_MIN_WIDTH,
  formatSignedPct,
  renderIndustryCell,
  signColor,
  tradeValueMultStyle,
} from "@/lib/grid-cells";
import { formatKstDateTime } from "@/lib/datetime";
import { renderStockNameCell } from "@/lib/name-highlight";
import { formatPoolLabel, type PoolLabelSource } from "@/lib/pool-label";

const gridTheme = createAppGridTheme();
const hintStyle: React.CSSProperties = { color: "var(--text-muted)", fontSize: "var(--fs-sm)" };

type Settings = {
  pool: string;
  top_n: number;
  stop_loss_pct: number;
  exit_ma_days: number;
  slippage_pct: number;
  backtest_months: number;
  /** 신호가 자리보다 많을 때의 진입 우선순위. */
  entry_priority: "value_surge" | "market_cap";
  /** 진입 자격 — 거래대금 급증 배수 하한. null 이면 조건 없음. */
  min_value_mult: number | null;
  /** 한 업종 최대 보유 종목 수. null 이면 제한 없음. */
  max_per_industry: number | null;
  /** 슬랙 알람 — 켠 풀만 장중 감시 배치가 진입·매도 예정 변화를 발송한다. */
  slack_enabled: boolean;
};

const ENTRY_PRIORITY_LABEL: Record<Settings["entry_priority"], string> = {
  value_surge: "거래대금 급증",
  market_cap: "시가총액",
};

type PoolOption = PoolLabelSource & { country_code?: string; currency?: string; pool_kind?: string | null };
// PoolLabelSource 가 ticker_type·name·icon·order 를 갖는다 — 번호는 order 에서 나온다.

type Constraints = {
  top_n_options: number[];
  stop_loss_options: number[];
  exit_ma_options: number[];
  entry_priority_options: Settings["entry_priority"][];
  min_value_mult_options: (number | null)[];
  max_per_industry_options: (number | null)[];
  month_options: number[];
  /** 신고가 창 — "52주" 같은 문구를 이 값에서 만든다(화면에 숫자를 박지 않는다). */
  high_window_weeks: number;
};

type View = {
  settings: Settings;
  /** 저장 이력이 없는 풀로 전환할 때 채울 값 (백엔드 기본값). */
  default_settings: Omit<Settings, "pool">;
  settings_by_pool?: Record<string, Partial<Settings>>;
  pool_options?: PoolOption[];
  constraints: Constraints;
};

type PositionRow = {
  ticker: string;
  name: string;
  industry: string;
  /** 직전 거래일 종가 대비 등락률 — 다른 화면과 같은 기준. */
  change_pct: number | null;
  /** 오늘 기준 시가총액(순위 화면과 같은 소스). 과거 이력이 없어 백테스트에는 쓰지 않는다. */
  market_cap: number | null;
  trade_value: number | null;
  /** 돌파했더라도 거짓이면 사지 않는다(배수 하한 미달). */
  qualifies: boolean;
  price: number;
  /** 진입 판정에 쓰는 직전 52주 최고 '종가'. */
  prior_high: number;
  /** 관례상의 52주 신고가(장중 고가) — 참고 표시용. */
  prior_high_intraday: number | null;
  /** 0 이상이면 돌파, 음수면 최고 종가까지 남은 거리(%). 상태 판정 기준. */
  gap_pct: number;
  /** 장중 고가 대비 거리(%). 판정에는 쓰지 않는다. */
  gap_high_pct: number | null;
  /** 장중에 최고 종가선을 건드렸으나 종가는 그 아래로 밀린 상태. */
  touched: boolean;
  /** 이미 보유 중 — 다시 사지 않는다. */
  is_held: boolean;
  /** 돌파·자격을 다 통과했지만 업종 상한에 밀려 이번에는 사지 않는다. */
  industry_blocked: boolean;
  value_mult: number | null;
};

type Holding = {
  ticker: string;
  name: string;
  industry: string;
  /** 직전 거래일 종가 대비 등락률 — 진입 후보 표와 같은 값. */
  change_pct: number | null;
  entry_date: string;
  entry_price: number;
  price: number;
  return_pct: number;
  days: number;
  /** 오늘 편입된 종목 */
  is_new: boolean;
  /** hold = 계속 보유, sell = 내일 시가에 청산 */
  status: "hold" | "sell";
  exit_reason: string | null;
};

/** 보유 표에 함께 그리는 행. 아직 안 산 종목은 매수가·수익률이 없다. */
type PlanRow = {
  ticker: string;
  name: string;
  industry: string;
  change_pct: number | null;
  /** 현재 시세 — 이탈 행도 지금 값이다(청산가는 exit_price). */
  price: number | null;
  /** 청산가 — 오늘 이탈한 행에만 있다. */
  exit_price: number | null;
  entry_date: string | null;
  entry_price: number | null;
  return_pct: number | null;
  plan: "hold" | "sell" | "buy" | "exited";
  days: number | null;
  is_new: boolean;
  exit_reason: string | null;
};

type Positions = {
  as_of: string;
  /** 진입·청산이 체결되는 거래일. 캘린더가 답하지 못하면 null. */
  next_session: string | null;
  holdings: Holding[];
  planned_entries: PositionRow[];
  exited_today: Trade[];
  pool: string;
  universe_count: number;
  window_weeks: number;
  min_value_mult: number | null;
  /** 최근 거래일 목록 — 기준일 셀렉트가 쓴다. */
  available_dates: { date: string; candidate_count: number; breakout_count: number }[];
  /** 가격 캐시가 마지막으로 갱신된 시각(ISO). 배치가 안 돌았으면 null. */
  refreshed_at: string | null;
  /** 진행 중인 세션의 실시간 시세를 얹었는지. 참이면 종가가 아직 확정 전이다. */
  live: boolean;
  /** 장전(동시호가) 구간 — 판정에는 안 쓰고 현재가·등락률만 얹었다. */
  pre_market: boolean;
  /** 주기 갱신을 걸 시점인지(장중이거나 개장이 가까운 장전). 개장 시각은 시장마다 달라 백엔드가 판단한다. */
  auto_refresh: boolean;
  quote_at: string | null;
  breakouts: PositionRow[];
  candidates: PositionRow[];
};

type Trade = {
  ticker: string;
  name: string;
  industry: string;
  change_pct?: number | null;
  /** 청산 후 현재 시세 — 현재 상태 표에서만 쓴다(백테스트 체결 목록에는 없다). */
  price?: number | null;
  entry_date: string;
  entry_price: number;
  exit_date: string;
  exit_price: number;
  return_pct: number;
  days: number;
  reason: string;
};

type Backtest = {
  start_date: string;
  end_date: string;
  months: number;
  strategy_total_pct: number;
  strategy_cagr_pct: number;
  strategy_mdd_pct: number;
  strategy_sortino: number | null;
  benchmark_total_pct: number;
  benchmark_cagr_pct: number;
  benchmark_mdd_pct: number;
  benchmark_sortino: number | null;
  benchmark_name: string;
  trade_count: number;
  win_rate_pct: number | null;
  avg_win_pct: number | null;
  avg_loss_pct: number | null;
  stop_count: number;
  exit_ma_count: number;
  trades: Trade[];
  /** 일별 누적 수익률(%) — 월간·연간은 이 값에서 만든다. */
  daily: { date: string; strategy_pct: number; benchmark_pct: number }[];
};

// 장중 자동 갱신 주기. 계산이 수 초 걸려 더 짧게 잡으면 요청이 겹친다.
const LIVE_REFRESH_MS = 5 * 60 * 1000;

/** 화면 전환 — 지금 상태를 보는 탭과 과거 성과를 보는 탭. */
const MAIN_TABS = [
  { key: "current", label: "현재 상태" },
  { key: "backtest", label: "백테스트" },
] as const;
type MainTab = (typeof MAIN_TABS)[number]["key"];

/** 현재 상태 안쪽 탭. 차트는 보유 종목 수만큼 그리므로 열 때만 그린다. */
const CURRENT_TABS = [
  { key: "list", label: "종목" },
  { key: "chart", label: "차트" },
] as const;
type CurrentTab = (typeof CURRENT_TABS)[number]["key"];

const VIEW_MODES = [
  { key: "yearly", label: "연간" },
  { key: "monthly", label: "월간" },
  { key: "daily", label: "일간" },
  { key: "trades", label: "체결" },
] as const;
type ViewMode = (typeof VIEW_MODES)[number]["key"];

type PeriodRow = { period: string; strategy_pct: number; benchmark_pct: number };

/** 누적(%) 시계열을 기간별 수익률로 자른다. 구간 양끝의 누적값 비로 계산한다. */
function toPeriodRows(daily: Backtest["daily"], keyLength: number): PeriodRow[] {
  if (daily.length === 0) return [];
  const lastByPeriod = new Map<string, { strategy: number; benchmark: number }>();
  const order: string[] = [];
  for (const point of daily) {
    const key = point.date.slice(0, keyLength);
    if (!lastByPeriod.has(key)) order.push(key);
    lastByPeriod.set(key, { strategy: point.strategy_pct, benchmark: point.benchmark_pct });
  }
  // 첫 구간의 기준은 시작 시점(누적 0%)이다.
  let prev = { strategy: 0, benchmark: 0 };
  const rows: PeriodRow[] = [];
  for (const key of order) {
    const current = lastByPeriod.get(key)!;
    const step = (now: number, before: number) => ((1 + now / 100) / (1 + before / 100) - 1) * 100;
    rows.push({
      period: key,
      strategy_pct: step(current.strategy, prev.strategy),
      benchmark_pct: step(current.benchmark, prev.benchmark),
    });
    prev = current;
  }
  return rows.reverse();
}

/** 상태 단계 — 색·설명을 여기 한 곳에서만 정한다.
 *
 * 거리 척도(관찰 → 근접 → 임박 → 돌파)는 회색·초록·주황·빨강으로 온도가 올라가고,
 * 척도 밖의 두 상태(터치 후 밀림·돌파 미달)는 계열이 다른 색을 써서 구분한다.
 * 파랑 계열은 쓰지 않는다 — 이 앱에서 파랑은 하락을 뜻해 헷갈린다.
 */
const STAGE_STYLE = {
  breakout: { color: "#d62828", text: "장중 현재가가 직전 최고 종가 이상이면 돌파중, 장 마감 후 종가가 그 이상이면 돌파성공입니다. 다음 거래일 시가에 매수합니다." },
  blocked: { color: "#0c8599", text: "최고 종가는 넘었지만 사지 않는 종목입니다. (미달)은 거래대금 급증 하한에 못 미친 것이고, (상한)은 그 업종을 이미 상한만큼 들고 있어 다음 순위에 자리를 넘긴 것입니다. 날짜 목록의 돌파 수에서도 빠집니다." },
  pullback: { color: "#7048e8", text: "장중 고가는 최고 종가선에 닿았지만 종가가 다시 아래로 내려온 상태입니다." },
  imminent: { color: "#e8590c", text: "최고 종가까지 3% 이내로 남은 종목입니다." },
  near: { color: "#2f9e44", text: "최고 종가까지 3% 초과 7% 이내로 남은 종목입니다." },
  watch: { color: "#868e96", text: "7%보다 많이 남았지만 후보 범위인 12% 이내에 들어온 종목입니다." },
  held: { color: "#495057", text: "이미 보유 중이라 다시 사지 않습니다. 신고가를 계속 갱신하면 돌파 신호가 매일 나오므로 목록에는 남습니다." },
} as const;

function describeStage(row: PositionRow, live: boolean): { label: string; color: string } {
  // 보유 중이면 상태보다 '못 산다'는 사실이 먼저다.
  if (row.is_held) return { label: "보유중", color: STAGE_STYLE.held.color };
  if (row.gap_pct >= 0) {
    // 장중에는 종가가 아직 안 나왔으니 '돌파중' — 마감 뒤 확정되면 '돌파성공'.
    const label = live ? "돌파중" : "돌파성공";
    if (!row.qualifies) return { label: `${label}(미달)`, color: STAGE_STYLE.blocked.color };
    // 자격은 통과했는데 업종 상한에 밀린 종목 — 그냥 '돌파성공'으로 두면 살 것처럼 보인다.
    if (row.industry_blocked) return { label: `${label}(상한)`, color: STAGE_STYLE.blocked.color };
    return { label, color: STAGE_STYLE.breakout.color };
  }
  // 장중에 선을 건드렸다가 밀린 것은 그냥 '임박'과 성격이 달라 따로 표시한다.
  if (row.touched) return { label: "터치 후 밀림", color: STAGE_STYLE.pullback.color };
  if (row.gap_pct > -3) return { label: "임박", color: STAGE_STYLE.imminent.color };
  if (row.gap_pct > -7) return { label: "근접", color: STAGE_STYLE.near.color };
  return { label: "관찰", color: STAGE_STYLE.watch.color };
}

/** 상태 단계 설명 — 진입 후보를 펼치면 보여준다. 색·문구는 STAGE_STYLE 이 단일 소스다. */
const STAGE_GUIDE: { label: string; key: keyof typeof STAGE_STYLE }[] = [
  { label: "돌파중 / 돌파성공", key: "breakout" },
  { label: "돌파(미달) · 돌파(상한)", key: "blocked" },
  { label: "터치 후 밀림", key: "pullback" },
  { label: "임박", key: "imminent" },
  { label: "근접", key: "near" },
  { label: "관찰", key: "watch" },
  { label: "보유중", key: "held" },
];

/** 일간(%)·현재가 컬럼 — 보유 종목 표와 진입 후보 표가 같은 정의를 쓴다. */
function dailyChangeColumn<T extends { change_pct?: number | null }>(): ColDef<T> {
  return {
    field: "change_pct" as ColDef<T>["field"],
    headerName: "일간(%)",
    width: 96,
    type: "numericColumn",
    valueFormatter: (p) => (p.value == null ? "-" : formatSignedPct(p.value as number, 2)),
    cellStyle: (p) => ({ color: signColor(p.value as number) }),
  };
}

/** 시가총액 표기 — 조/억 단위. 자리수가 커 그대로 두면 표가 밀린다. */
function formatMarketCap(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "-";
  const jo = value / 1_0000_0000_0000;
  if (jo >= 1) return `${jo.toFixed(1)}조`;
  const eok = value / 1_0000_0000;
  return `${eok.toLocaleString("ko-KR", { maximumFractionDigits: 0 })}억`;
}

const WEEKDAYS = ["일", "월", "화", "수", "목", "금", "토"];

/** `2026-08-12` → `2026-08-12 (수)`.
 *  UTC 로 파싱하면 하루 밀릴 수 있어 로컬 자정으로 읽는다. */
function formatDateWithWeekday(date: string): string {
  const parsed = new Date(`${date}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return date;
  return `${date} (${WEEKDAYS[parsed.getDay()]})`;
}

function toDateKey(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

/** 체결일을 사람이 읽는 말로. 장 시작 전에는 캐시 기준 '다음 거래일' 이 곧 오늘이다.
 *  값이 없으면 날짜를 지어내지 않고 '다음 거래일' 로 둔다. */
function formatFillDay(nextSession: string | null | undefined): string {
  if (!nextSession) return "다음 거래일";
  const today = new Date();
  if (nextSession === toDateKey(today)) return "오늘";
  const tomorrow = new Date(today);
  tomorrow.setDate(today.getDate() + 1);
  if (nextSession === toDateKey(tomorrow)) return "내일";
  return nextSession.slice(5).replace("-", "/");
}

function formatPrice(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return value.toLocaleString("ko-KR", { maximumFractionDigits: 2 });
}

export function NewHighClient() {
  const toast = useToast();
  const [view, setView] = useState<View | null>(null);
  const [positions, setPositions] = useState<Positions | null>(null);
  const [backtest, setBacktest] = useState<Backtest | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [backtesting, setBacktesting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backtestError, setBacktestError] = useState<string | null>(null);
  const [draft, setDraft] = useState<Settings | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("monthly");
  const [notifying, setNotifying] = useState(false);
  const [mainTab, setMainTab] = useState<MainTab>("current");
  const [currentTab, setCurrentTab] = useState<CurrentTab>("list");
  // 기준일 — 빈 값이면 최신 거래일. 과거 날짜를 고르면 그 시점 상태를 재현한다.
  const [asOf, setAsOf] = useState<string>("");
  const [candidatesOpen, setCandidatesOpen] = useState(false);
  const [charts, setCharts] = useState<HoldingChartData[] | null>(null);
  const [chartsLoading, setChartsLoading] = useState(false);
  const [chartsError, setChartsError] = useState<string | null>(null);

  const constraints = view?.constraints ?? null;
  // 창 길이가 바뀌면 문구도 따라 바뀐다 — "52주" 를 문자열로 박지 않는다.
  const windowLabel = constraints ? `${constraints.high_window_weeks}주` : "";
  // 진입·청산 체결일 — 장 시작 전에는 '오늘', 마감 뒤 캐시가 갱신되면 '내일' 이 된다.
  const fillDay = formatFillDay(positions?.next_session);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const response = await fetch("/api/strategy-new-high", { cache: "no-store" });
        const payload = (await response.json()) as View & { error?: string };
        if (!response.ok) throw new Error(payload.error ?? "설정을 불러오지 못했습니다.");
        if (!alive) return;
        setView(payload);
        setDraft(payload.settings);
      } catch (loadError) {
        if (alive) setError(loadError instanceof Error ? loadError.message : "설정을 불러오지 못했습니다.");
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const runPositions = useCallback(async (settings: Settings, date = "") => {
    setRunning(true);
    try {
      const response = await fetch("/api/strategy-new-high/positions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ settings, as_of: date || null }),
      });
      const payload = (await response.json()) as Positions & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "돌파 종목을 불러오지 못했습니다.");
      setPositions(payload);
    } catch (runError) {
      toast.error(runError instanceof Error ? runError.message : "돌파 종목을 불러오지 못했습니다.");
    } finally {
      setRunning(false);
    }
  }, [toast]);

  // 설정을 받으면 곧바로 오늘 상태를 채운다 — 빈 화면을 먼저 보여주지 않는다.
  useEffect(() => {
    if (view?.settings && !positions && !running) void runPositions(view.settings);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view]);

  const persistSettings = useCallback(
    async (settings: Settings, message: string) => {
      try {
        const response = await fetch("/api/strategy-new-high", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ settings }),
        });
        const payload = (await response.json()) as View & { error?: string };
        if (!response.ok) throw new Error(payload.error ?? "설정을 저장하지 못했습니다.");
        setView(payload);
        setDraft(payload.settings);
        // 백테스트는 어느 설정이 바뀌든 결과가 달라지므로 비운다 (SM 과 같은 규칙).
        setBacktest(null);
        setBacktestError(null);
        toast.success(message);
        setAsOf("");
        void runPositions(payload.settings, "");
      } catch (saveError) {
        toast.error(saveError instanceof Error ? saveError.message : "설정을 저장하지 못했습니다.");
      }
    },
    [runPositions, toast],
  );

  /** 풀을 바꾸면 그 풀에 저장된 설정으로 전환한다. 저장 이력이 없으면 **기본값**으로 채운다 —
   *  직전 풀의 값을 물려받으면 다른 풀의 설정이 섞여 풀별로 보관하는 의미가 없어진다. */
  const handlePoolChange = useCallback(
    (pool: string) => {
      if (!view) return;
      const saved = view.settings_by_pool?.[pool];
      void persistSettings(
        { ...view.default_settings, ...(saved ?? {}), pool },
        saved ? "종목풀을 전환했습니다." : "저장 이력이 없어 기본값으로 시작합니다.",
      );
    },
    [persistSettings, view],
  );

  const handleBacktest = useCallback(async () => {
    if (!draft) return;
    setBacktesting(true);
    setBacktestError(null);
    try {
      const response = await fetch("/api/strategy-new-high/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ months: draft.backtest_months, settings: draft }),
      });
      const payload = (await response.json()) as Backtest & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "백테스트에 실패했습니다.");
      setBacktest(payload);
    } catch (runError) {
      const message = runError instanceof Error ? runError.message : "백테스트에 실패했습니다.";
      setBacktestError(message);
      toast.error(message);
    } finally {
      setBacktesting(false);
    }
  }, [draft, toast]);

  // 장중에는 실시간 시세가 움직이므로 주기적으로 다시 받는다.
  // 과거 날짜를 보는 중이거나 장이 닫혀 있으면 갱신할 것이 없어 타이머를 걸지 않는다.
  useEffect(() => {
    if (!positions?.auto_refresh || asOf || mainTab !== "current" || !draft) return;
    const id = window.setInterval(() => void runPositions(draft, ""), LIVE_REFRESH_MS);
    return () => window.clearInterval(id);
  }, [positions?.auto_refresh, asOf, mainTab, draft, runPositions]);

  // 백테스트 탭을 처음 열거나 설정이 바뀌어 결과가 비워졌을 때만 돌린다.
  useEffect(() => {
    if (mainTab === "backtest" && !backtest && !backtesting && !backtestError) {
      void handleBacktest();
    }
  }, [mainTab, backtest, backtesting, backtestError, handleBacktest]);

  // 업종 컬럼 노출 여부 — 표시 중인 결과의 풀 성격(pool_kind)이 1순위(개별주=표시, ETF=숨김),
  // 미설정 풀은 행 값 유무로 추정 (pools-rank·strategy-sm 과 같은 기준).
  const hasIndustryData = useMemo(() => {
    const pool = positions?.pool ?? view?.settings.pool ?? "";
    const poolKind = String(view?.pool_options?.find((option) => option.ticker_type === pool)?.pool_kind ?? "");
    if (poolKind === "stock") return true;
    if (poolKind === "etf") return false;
    return [...(positions?.breakouts ?? []), ...(positions?.candidates ?? [])].some(
      (row) => String(row.industry ?? "").trim() !== "",
    );
  }, [positions, view?.settings.pool, view?.pool_options]);

  const positionColumns = useMemo<ColDef<PositionRow>[]>(
    () => [
      {
        headerName: "상태",
        width: 124,
        minWidth: 110,
        cellStyle: { display: "flex", alignItems: "center", justifyContent: "center" },
        valueGetter: (p) => p.data?.gap_pct ?? null,
        cellRenderer: (p: { data?: PositionRow }) => {
          if (!p.data) return null;
          const stage = describeStage(p.data, positions?.live ?? false);
          return <strong style={{ color: stage.color }}>{stage.label}</strong>;
        },
      },
      {
        field: "ticker",
        headerName: "티커",
        width: 108,
        cellRenderer: (p: { value?: string }) => <TickerDetailLink ticker={p.value} />,
      },
      {
        field: "name",
        headerName: "종목명",
        flex: 1,
        minWidth: STOCK_NAME_COLUMN_MIN_WIDTH,
        cellRenderer: (p: { value?: string | null }) => renderStockNameCell(p.value),
      },
      // 업종 컬럼 노출 여부 — 종목풀 설정의 풀 성격(pool_kind)이 1순위(개별주=표시, ETF=숨김),
      // 미설정 풀은 행 값 유무로 추정 (pools-rank·strategy-sm 과 같은 기준).
      {
        field: "industry",
        headerName: "업종",
        hide: !hasIndustryData,
        width: INDUSTRY_COLUMN_WIDTH,
        minWidth: INDUSTRY_COLUMN_MIN_WIDTH,
        headerTooltip: "한국은 네이버 분류, 미국은 지수 구성종목의 yfinance 분류",
        cellRenderer: (p: { value?: string }) => renderIndustryCell(p.value),
      },
      dailyChangeColumn<PositionRow>(),
      {
        field: "price",
        headerName: "현재가",
        width: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatPrice(p.value as number),
      },
      {
        field: "gap_high_pct",
        headerName: "고가 대비",
        width: 108,
        type: "numericColumn",
        headerTooltip:
          `관례상의 ${windowLabel} 신고가(장중 고가) 대비. 참고용이며 진입 판정에는 쓰지 않는다.`,
        tooltipValueGetter: (p) =>
          p.data?.prior_high_intraday != null
            ? `${windowLabel} 고가 ${formatPrice(p.data.prior_high_intraday)}`
            : "",
        valueFormatter: (p) => (p.value == null ? "-" : formatSignedPct(p.value as number, 2)),
        cellStyle: (p) => ({ color: signColor(p.value as number) }),
      },
      {
        field: "gap_pct",
        headerName: "종가 대비",
        width: 108,
        type: "numericColumn",
        headerTooltip:
          `직전 ${windowLabel} 최고 '종가' 대비. 이 값이 0 이상이면 돌파로 보고 진입한다.`,
        tooltipValueGetter: (p) =>
          p.data ? `${windowLabel} 최고 종가 ${formatPrice(p.data.prior_high)}` : "",
        valueFormatter: (p) => formatSignedPct(p.value as number, 2),
        cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 700 }),
      },
      {
        field: "market_cap",
        headerName: "시가총액",
        width: 116,
        type: "numericColumn",
        headerTooltip: "신호가 자리보다 많을 때 이 순서로 담는다(우선순위=시가총액일 때).",
        valueFormatter: (p) => formatMarketCap(p.value as number | null),
      },
      {
        field: "value_mult",
        headerName: "거래대금",
        width: 104,
        type: "numericColumn",
        headerTooltip: "20일 평균 거래대금 대비 배수 — 하한 이상이어야 진입한다. 굵은 글씨가 자격 통과다.",
        valueFormatter: (p) => (p.value == null ? "-" : `${(p.value as number).toFixed(1)}배`),
        // 농도는 배수 크기, 굵기는 진입 자격. 자격 하한은 설정값이라 농도 단계와 다를 수 있다.
        cellStyle: (p) => tradeValueMultStyle(p.value as number | null, (p.data as PositionRow | undefined)?.qualifies),
      },
    ],
    // live 가 빠지면 장전에 만든 컬럼이 그대로 남아 장중에도 '돌파성공' 으로 보인다.
    [windowLabel, hasIndustryData, positions?.live],
  );

  const periodColumns = useMemo<ColDef<PeriodRow>[]>(
    () => [
      { field: "period", headerName: viewMode === "yearly" ? "연도" : viewMode === "monthly" ? "월" : "일자", width: 132 },
      {
        field: "strategy_pct",
        headerName: "전략",
        flex: 1,
        minWidth: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatSignedPct(p.value as number, 2),
        cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 700 }),
      },
      {
        field: "benchmark_pct",
        headerName: backtest?.benchmark_name ?? "벤치마크",
        flex: 1,
        minWidth: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatSignedPct(p.value as number, 2),
        cellStyle: (p) => ({ color: signColor(p.value as number) }),
      },
      {
        headerName: "초과",
        flex: 1,
        minWidth: 100,
        type: "numericColumn",
        valueGetter: (p) => (p.data ? p.data.strategy_pct - p.data.benchmark_pct : null),
        valueFormatter: (p) => (p.value == null ? "-" : `${formatSignedPct(p.value as number, 2)}p`),
        cellStyle: (p) => ({ color: signColor(p.value as number) }),
      },
    ],
    [viewMode, backtest?.benchmark_name],
  );

  const periodRows = useMemo<PeriodRow[]>(() => {
    if (!backtest) return [];
    if (viewMode === "yearly") return toPeriodRows(backtest.daily, 4);
    if (viewMode === "monthly") return toPeriodRows(backtest.daily, 7);
    if (viewMode === "daily") return toPeriodRows(backtest.daily, 10);
    return [];
  }, [backtest, viewMode]);

  // 보유 + 내일 매도 + 내일 매수를 한 표로 합친다 — 이 표만 보고 주문을 낼 수 있게.
  const planRows = useMemo<PlanRow[]>(() => {
    if (!positions) return [];
    const held: PlanRow[] = positions.holdings.map((h) => ({
      ticker: h.ticker, name: h.name, industry: h.industry, change_pct: h.change_pct, price: h.price,
      exit_price: null,
      entry_date: h.entry_date, entry_price: h.entry_price, return_pct: h.return_pct,
      plan: h.status, days: h.days, is_new: h.is_new, exit_reason: h.exit_reason,
    }));
    const buys: PlanRow[] = positions.planned_entries.map((row) => ({
      ticker: row.ticker, name: row.name, industry: row.industry, change_pct: row.change_pct, price: row.price,
      exit_price: null,
      entry_date: null, entry_price: null, return_pct: null,
      plan: "buy", days: null, is_new: false, exit_reason: null,
    }));
    // 오늘 이미 청산된 종목 — 현재가는 지금 시세, 청산가는 따로 담는다.
    const exited: PlanRow[] = positions.exited_today.map((t) => ({
      ticker: t.ticker, name: t.name, industry: t.industry, change_pct: t.change_pct ?? null,
      price: t.price ?? null, exit_price: t.exit_price,
      entry_date: t.entry_date, entry_price: t.entry_price, return_pct: t.return_pct,
      plan: "exited", days: t.days, is_new: false, exit_reason: t.reason,
    }));
    // 보유(매도 예정 포함)가 위, 아직 안 산 것, 이미 끝난 것 순.
    // 같은 묶음 안에서는 **오래 들고 있는 것이 위** — 편입일이 이른 순이다.
    const rank = { hold: 0, sell: 0, buy: 1, exited: 2 } as const;
    return [...held, ...buys, ...exited].sort(
      (a, b) => rank[a.plan] - rank[b.plan] || (a.entry_date ?? "").localeCompare(b.entry_date ?? ""),
    );
  }, [positions]);

  // 차트를 그릴 대상 — 이미 나간 종목은 뺀다. 표와 같은 순서로 그린다.
  const chartRows = useMemo(() => planRows.filter((row) => row.plan !== "exited"), [planRows]);
  // 풀·기준일·구성이 바뀌면 이전 차트는 버린다.
  const chartKey = useMemo(
    () => `${positions?.pool ?? ""}|${positions?.as_of ?? ""}|${chartRows.map((row) => row.ticker).join(",")}`,
    [positions?.pool, positions?.as_of, chartRows],
  );

  useEffect(() => {
    setCharts(null);
    setChartsError(null);
  }, [chartKey]);

  // 차트 탭을 열 때만 받는다 — 보유 종목 수만큼 일봉을 실어 오므로 목록 탭에서는 낭비다.
  useEffect(() => {
    if (currentTab !== "chart" || !draft || !positions || charts || chartsLoading || chartsError) return;
    if (chartRows.length === 0) {
      setCharts([]);
      return;
    }
    setChartsLoading(true);
    void (async () => {
      try {
        const response = await fetch("/api/strategy-new-high/charts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            settings: draft,
            tickers: chartRows.map((row) => row.ticker),
            as_of: asOf || null,
          }),
        });
        const payload = (await response.json()) as { charts?: HoldingChartData[]; error?: string };
        if (!response.ok) throw new Error(payload.error ?? "차트를 불러오지 못했습니다.");
        setCharts(payload.charts ?? []);
      } catch (chartError) {
        const message = chartError instanceof Error ? chartError.message : "차트를 불러오지 못했습니다.";
        setChartsError(message);
        toast.error(message);
      } finally {
        setChartsLoading(false);
      }
    })();
  }, [currentTab, draft, positions, charts, chartsLoading, chartsError, chartRows, asOf, toast]);

  const holdingColumns = useMemo<ColDef<PlanRow>[]>(
    () => [
      {
        headerName: "상태",
        width: 110,
        minWidth: 100,
        cellStyle: { display: "flex", alignItems: "center", justifyContent: "center" },
        valueGetter: (p) => p.data?.plan ?? "",
        cellRenderer: (p: { data?: PlanRow }) => {
          if (!p.data) return null;
          if (p.data.plan === "buy") return <strong style={{ color: "#d62828" }}>진입 예정</strong>;
          if (p.data.plan === "sell") {
            return <strong style={{ color: "#1971c2" }}>매도 예정{p.data.exit_reason ? ` (${p.data.exit_reason})` : ""}</strong>;
          }
          if (p.data.plan === "exited") {
            return <span style={{ color: "var(--text-muted)" }}>이탈{p.data.exit_reason ? ` (${p.data.exit_reason})` : ""}</span>;
          }
          return <span>{p.data.is_new ? "진입" : `${p.data.days}일`}</span>;
        },
      },
      {
        field: "ticker",
        headerName: "티커",
        width: 108,
        cellRenderer: (p: { value?: string }) => <TickerDetailLink ticker={p.value} />,
      },
      {
        field: "name",
        headerName: "종목명",
        flex: 1,
        minWidth: STOCK_NAME_COLUMN_MIN_WIDTH,
        cellRenderer: (p: { value?: string | null }) => renderStockNameCell(p.value),
      },
      {
        field: "industry",
        headerName: "업종",
        hide: !hasIndustryData,
        width: INDUSTRY_COLUMN_WIDTH,
        minWidth: INDUSTRY_COLUMN_MIN_WIDTH,
        headerTooltip: "한국은 네이버 분류, 미국은 지수 구성종목의 yfinance 분류",
        cellRenderer: (p: { value?: string }) => renderIndustryCell(p.value),
      },
      dailyChangeColumn<PlanRow>(),
      {
        field: "price",
        headerName: "현재가",
        width: 110,
        type: "numericColumn",
        headerTooltip: "이탈한 종목도 지금 시세다 — 판 뒤의 흐름을 청산가와 견줘 볼 수 있다.",
        valueFormatter: (p) => (p.value == null ? "-" : formatPrice(p.value as number)),
      },
      {
        field: "entry_date",
        headerName: "편입일",
        width: 116,
        // 아직 안 산 종목은 편입일이 없다 — 내일 시가에 정해진다.
        valueFormatter: (p) => (p.value ? String(p.value) : "-"),
      },
      {
        field: "entry_price",
        headerName: "매수가",
        width: 110,
        type: "numericColumn",
        headerTooltip: `진입 예정 종목은 ${fillDay} 시가에 체결되므로 아직 값이 없다.`,
        valueFormatter: (p) => (p.value == null ? "-" : formatPrice(p.value as number)),
      },
      {
        field: "exit_price",
        headerName: "청산가",
        width: 110,
        type: "numericColumn",
        headerTooltip: "오늘 이탈한 종목의 체결가. 아직 들고 있는 종목은 값이 없다.",
        valueFormatter: (p) => (p.value == null ? "-" : formatPrice(p.value as number)),
      },
      {
        field: "return_pct",
        headerName: "수익률",
        width: 108,
        type: "numericColumn",
        headerTooltip: "아직 청산 전이라 매도 슬리피지는 빠져 있다.",
        valueFormatter: (p) => (p.value == null ? "-" : formatSignedPct(p.value as number, 2)),
        cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 700 }),
      },
    ],
    [hasIndustryData, fillDay],
  );

  const tradeColumns = useMemo<ColDef<Trade>[]>(
    () => [
      { field: "exit_date", headerName: "청산일", width: 116 },
      {
        field: "ticker",
        headerName: "티커",
        width: 108,
        cellRenderer: (p: { value?: string }) => <TickerDetailLink ticker={p.value} />,
      },
      {
        field: "name",
        headerName: "종목명",
        flex: 1,
        minWidth: STOCK_NAME_COLUMN_MIN_WIDTH,
        cellRenderer: (p: { value?: string | null }) => renderStockNameCell(p.value),
      },
      {
        field: "industry",
        headerName: "업종",
        hide: !hasIndustryData,
        width: INDUSTRY_COLUMN_WIDTH,
        minWidth: INDUSTRY_COLUMN_MIN_WIDTH,
        headerTooltip: "한국은 네이버 분류, 미국은 지수 구성종목의 yfinance 분류",
        cellRenderer: (p: { value?: string }) => renderIndustryCell(p.value),
      },
      { field: "entry_date", headerName: "진입일", width: 116 },
      {
        field: "entry_price",
        headerName: "진입가",
        width: 108,
        type: "numericColumn",
        headerTooltip: "돌파 다음 거래일의 실제 시가. 슬리피지는 가격이 아니라 비용이라 손익률에만 반영한다.",
        valueFormatter: (p) => formatPrice(p.value as number),
      },
      {
        field: "exit_price",
        headerName: "청산가",
        width: 108,
        type: "numericColumn",
        valueFormatter: (p) => formatPrice(p.value as number),
      },
      { field: "days", headerName: "보유일", width: 92, type: "numericColumn" },
      {
        field: "reason",
        headerName: "사유",
        width: 88,
        cellStyle: { display: "flex", alignItems: "center", justifyContent: "center" },
      },
      {
        field: "return_pct",
        headerName: "손익률",
        width: 108,
        type: "numericColumn",
        headerTooltip: "진입·청산 슬리피지가 반영된 값이라 (청산가 ÷ 진입가 − 1) 보다 그만큼 낮다.",
        valueFormatter: (p) => formatSignedPct(p.value as number, 2),
        cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 700 }),
      },
    ],
    [hasIndustryData],
  );

  if (loading) return <PageFrame title="신고가 돌파" fullWidth><div className="appPageStack">불러오는 중…</div></PageFrame>;
  if (error || !view || !draft || !constraints) {
    return (
      <PageFrame title="신고가 돌파" fullWidth>
        <div className="alert alert-danger">{error ?? "설정을 불러오지 못했습니다."}</div>
      </PageFrame>
    );
  }

  return (
    <PageFrame title="신고가 돌파" fullWidth>
      <div className="appPageStack">
        <section className="appSection">
          <div className="card appCard">
            <div className="card-body appCardBodyTight">
              <div style={{ display: "flex", flexWrap: "wrap", gap: 16, alignItems: "flex-end" }}>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">종목풀</span>
                  <select
                    className="form-select form-select-sm"
                    value={draft.pool}
                    onChange={(event) => handlePoolChange(event.target.value)}
                  >
                    {(view.pool_options ?? []).map((option) => (
                      <option key={option.ticker_type} value={option.ticker_type}>
                        {formatPoolLabel(option)}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">보유 종목수</span>
                  <select
                    className="form-select form-select-sm"
                    value={String(draft.top_n)}
                    onChange={(event) => setDraft({ ...draft, top_n: Number(event.target.value) })}
                  >
                    {constraints.top_n_options.map((n) => (
                      <option key={n} value={n}>{n}종목</option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">손절선</span>
                  <select
                    className="form-select form-select-sm"
                    value={String(draft.stop_loss_pct)}
                    onChange={(event) => setDraft({ ...draft, stop_loss_pct: Number(event.target.value) })}
                  >
                    {constraints.stop_loss_options.map((n) => (
                      <option key={n} value={n}>{n}%</option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">이탈 이평선</span>
                  <select
                    className="form-select form-select-sm"
                    value={String(draft.exit_ma_days)}
                    onChange={(event) => setDraft({ ...draft, exit_ma_days: Number(event.target.value) })}
                  >
                    {constraints.exit_ma_options.map((n) => (
                      <option key={n} value={n}>{n}일</option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">진입 우선순위</span>
                  <div className="appSegmentedToggle appSegmentedToggleCompact" role="group" aria-label="진입 우선순위">
                    {constraints.entry_priority_options.map((key) => (
                      <button
                        key={key}
                        type="button"
                        className={draft.entry_priority === key ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                        onClick={() => void persistSettings({ ...draft, entry_priority: key }, "진입 우선순위를 바꿨습니다.")}
                      >
                        {ENTRY_PRIORITY_LABEL[key]}
                      </button>
                    ))}
                  </div>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">거래대금 급증 하한</span>
                  <select
                    className="form-select form-select-sm"
                    // '없음'(null)은 빈 문자열로 실어 보낸다 — select 의 value 는 문자열만 받는다.
                    value={draft.min_value_mult == null ? "" : String(draft.min_value_mult)}
                    onChange={(event) => setDraft({
                      ...draft,
                      min_value_mult: event.target.value === "" ? null : Number(event.target.value),
                    })}
                  >
                    {constraints.min_value_mult_options.map((value) => (
                      <option key={String(value)} value={value == null ? "" : String(value)}>
                        {value == null ? "없음" : `${value}배`}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">업종 상한</span>
                  <select
                    className="form-select form-select-sm"
                    value={draft.max_per_industry == null ? "" : String(draft.max_per_industry)}
                    onChange={(event) => setDraft({
                      ...draft,
                      max_per_industry: event.target.value === "" ? null : Number(event.target.value),
                    })}
                  >
                    {constraints.max_per_industry_options.map((value) => (
                      <option key={String(value)} value={value == null ? "" : String(value)}>
                        {value == null ? "제한 없음" : `${value}종목`}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">백테스트 기간</span>
                  <select
                    className="form-select form-select-sm"
                    value={String(draft.backtest_months)}
                    onChange={(event) => setDraft({ ...draft, backtest_months: Number(event.target.value) })}
                  >
                    {constraints.month_options.map((n) => (
                      <option key={n} value={n}>{n}개월</option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">슬랙 알람</span>
                  <div className="form-check form-switch" style={{ paddingLeft: "2.6em", marginBottom: 0, minHeight: 31, display: "flex", alignItems: "center" }}>
                    <input
                      className="form-check-input"
                      type="checkbox"
                      role="switch"
                      id="newHighSlackToggle"
                      style={{ width: "2.2em", height: "1.2em" }}
                      checked={draft.slack_enabled}
                      onChange={(event) => setDraft({ ...draft, slack_enabled: event.target.checked })}
                    />
                    <span style={{ fontWeight: 700, marginLeft: 8, fontSize: "var(--fs-sm)", color: draft.slack_enabled ? "inherit" : "var(--text-muted)" }}>
                      {draft.slack_enabled ? "켜짐" : "꺼짐"}
                    </span>
                    <button
                      className="btn btn-sm btn-outline-secondary"
                      type="button"
                      style={{ marginLeft: 10, whiteSpace: "nowrap" }}
                      disabled={notifying}
                      title="저장된 설정 기준으로 현재 진입·매도 예정 전체를 즉시 발송합니다 (토글과 무관)"
                      onClick={() => {
                        setNotifying(true);
                        void (async () => {
                          try {
                            const response = await fetch("/api/strategy-new-high/notify", {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({ pool: draft.pool }),
                            });
                            const payload = (await response.json()) as {
                              sent?: boolean; entries?: number; sells?: number; error?: string;
                            };
                            if (!response.ok || payload.error) throw new Error(payload.error ?? "슬랙 발송에 실패했습니다.");
                            toast.success(`슬랙 발송 완료 — 진입 예정 ${payload.entries ?? 0} · 매도 예정 ${payload.sells ?? 0}`);
                          } catch (notifyError) {
                            toast.error(notifyError instanceof Error ? notifyError.message : "슬랙 발송에 실패했습니다.");
                          } finally {
                            setNotifying(false);
                          }
                        })();
                      }}
                    >
                      {notifying ? "발송 중…" : "지금 발송(테스트)"}
                    </button>
                  </div>
                </label>
                <button
                  className="btn btn-primary"
                  type="button"
                  onClick={() => void persistSettings(draft, "설정을 저장했습니다.")}
                >
                  저장
                </button>
              </div>
              <div style={{ ...hintStyle, marginTop: 10, lineHeight: 1.6 }}>
                종가가 <strong>직전 {windowLabel} 최고가</strong>를 넘으면 다음 거래일 시가에 매수합니다.
                {" "}진입가 대비 <strong>{draft.stop_loss_pct}%</strong> 또는{" "}
                <strong>{draft.exit_ma_days}일선 종가 하회</strong> 중 먼저 걸리는 쪽에서 청산합니다.
                {" "}목표가는 두지 않습니다 — 오르는 종목은 계속 보유합니다.
                {draft.min_value_mult != null
                  ? ` 단, 20일 평균 대비 거래대금이 ${draft.min_value_mult}배 이상 늘어난 돌파만 삽니다.`
                  : ""}
                {" "}자리가 모자라면 <strong>{ENTRY_PRIORITY_LABEL[draft.entry_priority]}</strong>이 큰 순으로 담습니다.
                {draft.max_per_industry != null
                  ? ` 한 업종은 최대 ${draft.max_per_industry}종목까지만 담아, 상한에 걸리면 다음 순위가 그 자리를 채웁니다.`
                  : ""}
              </div>
            </div>
          </div>
        </section>

        <section className="appSection">
          <div className="card appCard">
            <div className="card-header appCardHeader">
              <NavTabs items={MAIN_TABS} value={mainTab} onChange={setMainTab} variant="card" label="화면 전환" />
              <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
              {mainTab === "current" && positions ? (
                <span style={hintStyle}>
                  {positions.live || positions.pre_market
                    ? `${formatKstDateTime(positions.quote_at)} 시세 · ${positions.live ? "장중" : "장전"}`
                    : `${formatKstDateTime(positions.refreshed_at)} 갱신`}
                </span>
              ) : null}
              {mainTab === "current" && positions ? (
                <select
                  className="form-select form-select-sm"
                  style={{ width: "auto" }}
                  value={asOf}
                  onChange={(event) => {
                    setAsOf(event.target.value);
                    if (draft) void runPositions(draft, event.target.value);
                  }}
                >
                  <option value="">
                    {positions.auto_refresh
                      ? `오늘 (${positions.live ? "장중" : "장전"} 자동 갱신)`
                      : `오늘 (${formatDateWithWeekday(positions.as_of)})`}
                  </option>
                  {positions.available_dates.map((d) => (
                    <option key={d.date} value={d.date}>
                      {formatDateWithWeekday(d.date)} ({d.candidate_count}종목
                      {d.breakout_count ? ` · 돌파 ${d.breakout_count}` : ""})
                    </option>
                  ))}
                </select>
              ) : null}
              {/* 기준일·풀·종목수는 셀렉트와 표에 이미 있어 중복이라 뺐다. */}
              <span style={hintStyle}>
                {mainTab === "current"
                  ? running
                    ? "계산 중…"
                    : ""
                  : backtest
                    ? `${backtest.start_date} ~ ${backtest.end_date} · ${backtest.months}개월`
                    : backtesting
                      ? "실행 중…"
                      : `${draft.backtest_months}개월`}
              </span>
              </span>
            </div>
            {mainTab === "current" ? (
              <div className="card-body appCardBodyTight">
                <NavTabs
                  items={CURRENT_TABS}
                  value={currentTab}
                  onChange={setCurrentTab}
                  label="현재 상태 보기"
                  style={{ marginBottom: 12 }}
                />
                {currentTab === "list" ? (
                  <>
                    <div style={{ ...hintStyle, fontWeight: 700, margin: "4px 0 6px" }}>
                      보유 종목 ({positions?.holdings.length ?? 0}개)
                      {positions?.holdings.some((h) => h.status === "sell")
                        ? ` · ${fillDay} 매도 ${positions.holdings.filter((h) => h.status === "sell").length}`
                        : ""}
                      {positions?.planned_entries.length
                        ? ` · ${fillDay} 매수 ${positions.planned_entries.length}`
                        : ""}
                      {positions?.exited_today.length ? ` · 오늘 이탈 ${positions.exited_today.length}` : ""}
                    </div>
                    <AppAgGrid<PlanRow>
                      rowData={planRows}
                      columnDefs={holdingColumns}
                      loading={running}
                      theme={gridTheme}
                      minHeight={0}
                      height="auto"
                      gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                    />
                    <button
                      type="button"
                      onClick={() => setCandidatesOpen((open) => !open)}
                      style={{
                        ...hintStyle, fontWeight: 700, margin: "16px 0 6px", padding: 0,
                        background: "none", border: "none", cursor: "pointer",
                        display: "inline-flex", alignItems: "center", gap: 6,
                      }}
                    >
                      <span>{candidatesOpen ? "▾" : "▸"}</span>
                      진입 후보 ({(positions?.breakouts.length ?? 0) + (positions?.candidates.length ?? 0)}개)
                    </button>
                    {candidatesOpen ? (
                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
                          gap: "10px 20px",
                          padding: "12px 14px",
                          marginBottom: 10,
                          borderRadius: 8,
                          background: "var(--bs-secondary-bg, #f1f5f9)",
                        }}
                      >
                        {STAGE_GUIDE.map((stage) => (
                          <div key={stage.label} style={{ fontSize: "var(--fs-sm)", lineHeight: 1.5 }}>
                            <strong style={{ color: STAGE_STYLE[stage.key].color }}>{stage.label}</strong>
                            <div style={{ color: "var(--text-muted)" }}>{STAGE_STYLE[stage.key].text}</div>
                          </div>
                        ))}
                      </div>
                    ) : null}
                    <AppAgGrid<PositionRow>
                      rowData={[...(positions?.breakouts ?? []), ...(positions?.candidates ?? [])]}
                      columnDefs={positionColumns}
                      loading={running}
                      theme={gridTheme}
                      minHeight={0}
                      height="auto"
                      gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                    />
                  </>
                ) : chartsLoading || (!charts && !chartsError) ? (
                  <div style={{ ...hintStyle, padding: "24px 0", textAlign: "center" }}>차트를 불러오는 중…</div>
                ) : chartsError ? (
                  <div className="alert alert-danger">{chartsError}</div>
                ) : charts && charts.length > 0 ? (
                  <>
                    <div style={{ ...hintStyle, margin: "4px 0 10px" }}>
                      최근 6개월 일봉입니다. 종가가 <strong style={{ color: "#7048e8" }}>직전 최고 종가선</strong>을
                      넘으면 진입, <strong style={{ color: "#f08c00" }}>{draft.exit_ma_days}일선</strong>을 하회하면
                      청산합니다. 진입한 종목은 매수가에 점선이 그려집니다.
                    </div>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))",
                        gap: "18px 20px",
                      }}
                    >
                      {charts.map((item) => {
                        const row = chartRows.find((candidate) => candidate.ticker === item.ticker);
                        return (
                          <HoldingChart
                            key={item.ticker}
                            chart={item}
                            entryDate={row?.entry_date}
                            entryPrice={row?.entry_price}
                            maDays={draft.exit_ma_days}
                          />
                        );
                      })}
                    </div>
                  </>
                ) : (
                  <div style={{ ...hintStyle, padding: "24px 0", textAlign: "center" }}>
                    보유 중이거나 진입 예정인 종목이 없습니다.
                  </div>
                )}
              </div>
            ) : (
              <div className="card-body appCardBodyTight">
                {backtestError ? <div className="alert alert-danger">{backtestError}</div> : null}
                {!backtest ? (
                  <div style={{ ...hintStyle, padding: "24px 0", textAlign: "center" }}>
                    {backtesting
                      ? "체결 내역을 계산하고 있습니다. 구간이 길면 1분 이상 걸립니다."
                      : "잠시 후 결과가 표시됩니다."}
                  </div>
                ) : (
                <>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 20, marginBottom: 12 }}>
                  <span>
                    전략{" "}
                    <b style={{ color: signColor(backtest.strategy_total_pct) }}>
                      {formatSignedPct(backtest.strategy_total_pct, 1)}
                    </b>
                    <span style={hintStyle}>
                      {` (CAGR ${formatSignedPct(backtest.strategy_cagr_pct, 1)} · MDD ${backtest.strategy_mdd_pct.toFixed(1)}%`}
                      {` · 소르티노 ${backtest.strategy_sortino != null ? backtest.strategy_sortino.toFixed(2) : "-"})`}
                    </span>
                  </span>
                  <span>
                    {backtest.benchmark_name}{" "}
                    <b style={{ color: signColor(backtest.benchmark_total_pct) }}>
                      {formatSignedPct(backtest.benchmark_total_pct, 1)}
                    </b>
                    <span style={hintStyle}>
                      {` (CAGR ${formatSignedPct(backtest.benchmark_cagr_pct, 1)} · MDD ${backtest.benchmark_mdd_pct.toFixed(1)}%)`}
                    </span>
                  </span>
                  <span style={hintStyle}>
                    {`거래 ${backtest.trade_count}건 · 승률 ${backtest.win_rate_pct ?? "-"}%`}
                    {` · 평균이익 ${backtest.avg_win_pct != null ? formatSignedPct(backtest.avg_win_pct, 1) : "-"}`}
                    {` · 평균손실 ${backtest.avg_loss_pct != null ? formatSignedPct(backtest.avg_loss_pct, 1) : "-"}`}
                    {` · 손절 ${backtest.stop_count} / 이탈 ${backtest.exit_ma_count}`}
                  </span>
                </div>
                <NavTabs
                  items={VIEW_MODES}
                  value={viewMode}
                  onChange={setViewMode}
                  label="백테스트 보기 단위"
                  style={{ marginBottom: 10 }}
                />
                {viewMode === "trades" ? (
                  // autoHeight — 카드 안에서 스크롤하지 않고 브라우저 스크롤로 본다.
                  <AppAgGrid<Trade>
                    rowData={backtest.trades}
                    columnDefs={tradeColumns}
                    theme={gridTheme}
                    minHeight={0}
                    height="auto"
                    gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                  />
                ) : (
                  <AppAgGrid<PeriodRow>
                    rowData={periodRows}
                    columnDefs={periodColumns}
                    theme={gridTheme}
                    minHeight={0}
                    height="auto"
                    gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                  />
                )}
                </>
                )}
              </div>
            )}
          </div>
        </section>
      </div>
    </PageFrame>
  );
}
