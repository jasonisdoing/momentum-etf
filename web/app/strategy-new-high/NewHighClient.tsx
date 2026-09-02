"use client";

import type { ColDef } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useState } from "react";
import { IconCheck } from "@tabler/icons-react";

import { formatMarketCapWon } from "@/lib/market-cap-format";
import { type HoldingChartData } from "../components/HoldingChart";
import { StrategyHoldingCharts } from "../components/StrategyHoldingCharts";
import { AppAgGrid } from "../components/AppAgGrid";
import { MonthsSelect } from "../components/MonthsSelect";
import { AppLoadingProgress, startProgressRamp, type LoadingProgress } from "../components/AppLoadingProgress";
import { BacktestSummary } from "../components/BacktestSummary";
import { BacktestTradeStats } from "../components/BacktestTradeStats";
import { useRealtimeQuotes } from "../components/useRealtimeQuotes";
import { StrategyNotes } from "../components/StrategyNotes";
import { StrategyTuning, type TuningResult } from "../components/StrategyTuning";
import { NavTabs } from "../components/NavTabs";
import { PageFrame } from "../components/PageFrame";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { MaDaysSelect } from "../components/MaDaysSelect";
import { UnsavedChangesBadge } from "../components/UnsavedChangesBadge";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";
import {
  industryColumn,
  STOCK_NAME_COLUMN_MIN_WIDTH,
  adrColumn,
  formatSignedPct,
  maExitGapColumn,
  signColor,
  tradeValueMultColumn,
  marketCapRankColumn,
  stockMemoColumn,
} from "@/lib/grid-cells";
import { formatDateWithWeekday, formatKstDateTime } from "@/lib/datetime";
import { poolHasIndustry, poolHasMarketCap } from "@/lib/pool-industry";
import { renderStockNameCell } from "@/lib/name-highlight";
import { updateStockMemo } from "@/lib/stocks-store";
import { readRememberedTickerType, writeRememberedTickerType } from "../components/account-selection";
import { formatPoolLabel, type PoolLabelSource } from "@/lib/pool-label";

const gridTheme = createAppGridTheme();
const hintStyle: React.CSSProperties = { color: "var(--text-muted)", fontSize: "var(--fs-sm)" };

type Settings = {
  pool: string;
  /** 종목풀 설정의 보유종목 수 — 이 화면에서는 표시·계산에만 쓴다. */
  top_n: number;
  stop_loss_pct: number | null;
  exit_ma_days: number;
  /** 진입 자격 — 거래대금 배수 하한(20일 평균 대비). null 이면 조건 없음. */
  min_value_mult: number | null;
  /** ADR 하한 — 전일 시장 ADR 이 미만이면 그날 신규 진입 차단. null = 없음. */
  adr_floor?: number | null;
};

type PoolOption = PoolLabelSource & { country_code?: string; currency?: string; pool_kind?: string | null };
// PoolLabelSource 가 ticker_type·name·icon·order 를 갖는다 — 번호는 order 에서 나온다.

type Constraints = {
  adr_floor_options?: (number | null)[];
  stop_loss_options: (number | null)[];
  exit_ma_options: number[];
  min_value_mult_options: (number | null)[];
  month_options: number[];
  /** 튜닝 전용 기간 — ADR 축이 항상 있으므로 ADR 이력이 덮는 범위만 온다. */
  tuning_month_options?: number[];
  /** 신고가 창 — "52주" 같은 문구를 이 값에서 만든다(화면에 숫자를 박지 않는다). */
  high_window_weeks: number;
};

type View = {
  /** 선택지 밖 저장값을 첫 선택지로 보정한 내역 — 있으면 화면이 저장을 요구한다. */
  coerced?: string[];
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
  /** 지금 계좌에 실제로 들고 있는지 — 전략상 보유(is_held)와 뜻이 다르다. */
  account_held?: boolean;
  /** 직전 거래일 종가 대비 등락률 — 다른 화면과 같은 기준. */
  change_pct: number | null;
  /** 오늘 기준 시가총액(순위 화면과 같은 소스). 과거 이력이 없어 백테스트에는 쓰지 않는다. */
  market_cap: number | null;
  market_cap_rank: number | null;
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
  /** 전략이 판정에 쓰는 배수(KRX 정규시장 확정). */
  value_mult: number | null;
  /** 장중 시간 환산 배수(누적 ÷ 장 경과율). 장중에만 값이 있고 괄호로 보여준다. */
  value_mult_live?: number | null;
  /** 종목에 붙는 메모 — 순위·모멘텀·자산 관리 화면과 같은 값. */
  memo?: string;
};

type Holding = {
  ticker: string;
  name: string;
  industry: string;
  /** 시장 전체 시총 순위 — 진입 후보 표와 같은 값(배치 B 가 메타 캐시에 적어 둔다). */
  market_cap_rank?: number | null;
  /** 직전 거래일 종가 대비 등락률 — 진입 후보 표와 같은 값. */
  change_pct: number | null;
  entry_date: string;
  entry_price: number;
  price: number;
  return_pct: number;
  days: number;
  /** 20일 평균 대비 거래대금 배수 — 진입 후보 표와 같은 값. */
  value_mult?: number | null;
  value_mult_live?: number | null;
  /** 오늘 편입된 종목 */
  is_new: boolean;
  /** hold = 계속 보유, sell = 내일 시가에 청산 */
  status: "hold" | "sell";
  exit_reason: string | null;
  /** 종목에 붙는 메모 — 순위·모멘텀·자산 관리 화면과 같은 값. */
  memo?: string;
  exit_ma_gap_pct?: number | null;
  exit_ma?: number | null;
};

/** 보유 표에 함께 그리는 행. 아직 안 산 종목은 매수가·수익률이 없다. */
type PlanRow = {
  ticker: string;
  name: string;
  industry: string;
  market_cap_rank?: number | null;
  change_pct: number | null;
  /** 현재 시세 — 이탈 행도 지금 값이다(청산가는 exit_price). */
  price: number | null;
  /** 청산가 — 오늘 이탈한 행에만 있다. */
  exit_price: number | null;
  entry_date: string | null;
  entry_price: number | null;
  return_pct: number | null;
  plan: "hold" | "sell" | "buy" | "exited";
  value_mult?: number | null;
  value_mult_live?: number | null;
  days: number | null;
  is_new: boolean;
  exit_reason: string | null;
  /** 종목에 붙는 메모 — 순위·모멘텀·자산 관리 화면과 같은 값. */
  memo?: string;
  /** 이탈 이평선까지 남은 여유(%) — (현재가 ÷ 이탈선 − 1) × 100. 0 에 가까울수록 매도 임박. */
  exit_ma_gap_pct?: number | null;
  /** 이탈 이평선 값 — 여유율 툴팁에 함께 보여준다. */
  exit_ma?: number | null;
};

type Positions = {
  as_of: string;
  /** 표시용 시세 갱신에 쓰는 국가 코드(시세 소스가 국가별로 다르다). */
  country: string;
  /** 진입·청산이 체결되는 거래일. 캘린더가 답하지 못하면 null. */
  next_session: string | null;
  holdings: Holding[];
  planned_entries: PositionRow[];
  /** ADR 게이트 — 하한 미설정이면 null. blocked=True 면 오늘 신규 진입 없음. */
  adr_gate?: { market: string | null; floor: number; value: number | null; blocked: boolean } | null;
  exited_today: Trade[];
  pool: string;
  universe_count: number;
  window_weeks: number;
  min_value_mult: number | null;
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
  /** 시장 전체 시총 순위 — 운용 현황 표에서만 쓴다(백테스트 체결 목록에는 없다). */
  market_cap_rank?: number | null;
  change_pct?: number | null;
  /** 청산 후 현재 시세 — 운용 현황 표에서만 쓴다(백테스트 체결 목록에는 없다). */
  price?: number | null;
  /** 거래대금 배수 — 운용 현황 표에서만 쓴다(백테스트 체결 목록에는 없다). */
  value_mult?: number | null;
  value_mult_live?: number | null;
  /** 종목 메모 — 운용 현황 표에서만 쓴다(백테스트 체결 목록에는 없다). */
  memo?: string;
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
  /** 청산 사유별 건수 — 전략마다 사유가 달라 서버가 준 그대로 쓴다. */
  reason_counts?: Record<string, number>;
  trades: Trade[];
  /** 일별 누적 수익률(%) — 월간·연간은 이 값에서 만든다. */
  daily: { date: string; strategy_pct: number; benchmark_pct: number; adr?: number | null }[];
};

// 장중 자동 갱신 주기. 계산이 수 초 걸려 더 짧게 잡으면 요청이 겹친다.
const LIVE_REFRESH_MS = 5 * 60 * 1000;

/** 운용 현황 안쪽 탭. 차트는 보유 종목 수만큼 그리므로 열 때만 그린다. */
const CURRENT_TABS = [
  { key: "list", label: "종목" },
  { key: "chart", label: "차트" },
] as const;
type CurrentTab = (typeof CURRENT_TABS)[number]["key"];

const VIEW_MODES = [
  { key: "yearly", label: "연간" },
  { key: "monthly", label: "월간" },
  { key: "weekly", label: "주간" },
  { key: "daily", label: "일간" },
  { key: "trades", label: "체결" },
] as const;
type ViewMode = (typeof VIEW_MODES)[number]["key"];

type PeriodRow = {
  period: string;
  strategy_pct: number;
  benchmark_pct: number;
  /** 구간 마지막 날의 시장 ADR — 일간·주간(판정일) 컬럼이 쓴다. */
  adr: number | null;
  /** 구간 최저 시장 ADR — 월간·연간 컬럼이 쓴다. */
  adr_min: number | null;
};

/** 그 날짜가 속한 주의 월요일 — 주간 묶음 키. 로컬 기준으로 조립한다(UTC 파싱은 하루 밀린다). */
function weekKeyOf(date: string): string {
  const parsed = new Date(`${date}T00:00:00`);
  parsed.setDate(parsed.getDate() - ((parsed.getDay() + 6) % 7));
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${parsed.getFullYear()}-${pad(parsed.getMonth() + 1)}-${pad(parsed.getDate())}`;
}

/** 누적(%) 시계열을 기간별 수익률로 자른다. 구간 양끝의 누적값 비로 계산한다.
 *  주간은 묶음 키(월요일)와 표시 라벨(그 주 마지막 거래일)이 달라 따로 담는다. */
function toPeriodRows(daily: Backtest["daily"], keyOf: (date: string) => string, labelByLastDate = false): PeriodRow[] {
  if (daily.length === 0) return [];
  const lastByPeriod = new Map<
    string,
    { strategy: number; benchmark: number; lastDate: string; adr: number | null; adrMin: number | null }
  >();
  const order: string[] = [];
  for (const point of daily) {
    const key = keyOf(point.date);
    if (!lastByPeriod.has(key)) order.push(key);
    const knownMin = lastByPeriod.get(key)?.adrMin ?? null;
    const adr = point.adr ?? null;
    lastByPeriod.set(key, {
      strategy: point.strategy_pct,
      benchmark: point.benchmark_pct,
      lastDate: point.date,
      adr,
      adrMin: adr == null ? knownMin : knownMin == null ? adr : Math.min(knownMin, adr),
    });
  }
  // 첫 구간의 기준은 시작 시점(누적 0%)이다.
  let prev = { strategy: 0, benchmark: 0 };
  const rows: PeriodRow[] = [];
  for (const key of order) {
    const current = lastByPeriod.get(key)!;
    const step = (now: number, before: number) => ((1 + now / 100) / (1 + before / 100) - 1) * 100;
    rows.push({
      period: labelByLastDate ? current.lastDate : key,
      strategy_pct: step(current.strategy, prev.strategy),
      benchmark_pct: step(current.benchmark, prev.benchmark),
      adr: current.adr,
      adr_min: current.adrMin,
    });
    prev = current;
  }
  return rows.reverse();
}

/** 상태 단계 — 색·설명을 여기 한 곳에서만 정한다.
 *
 * 거리 척도(근접 → 임박 → 돌파)는 초록·주황·빨강으로 온도가 올라가고,
 * 척도 밖의 두 상태(터치 후 밀림·돌파 미달)는 계열이 다른 색을 써서 구분한다.
 * 파랑 계열은 쓰지 않는다 — 이 앱에서 파랑은 하락을 뜻해 헷갈린다.
 */
const STAGE_STYLE = {
  breakout: { color: "#d62828", text: "현재가(마감 후에는 종가)가 직전 최고 종가 이상이고 거래대금 하한 기준까지 달성한 종목입니다. 한국 종목은 장중에 실시간 거래대금을 장 경과 시간 비례 목표와 비교해 판정합니다." },
  blocked: { color: "#0c8599", text: "최고 종가에 도달은 했지만 거래대금 하한을 넘지 못한 종목입니다. 한국 풀은 장중에 하한을 장 경과 시간에 비례해 낮춰 판정합니다 — 그 시점 목표 페이스를 넘으면 돌파성공으로 바뀝니다. 날짜 목록의 돌파 수에서도 빠집니다." },
  pullback: { color: "#7048e8", text: "장중 고가는 최고 종가선에 닿았지만 종가가 다시 아래로 내려온 상태입니다." },
  imminent: { color: "#e8590c", text: "최고 종가까지 3% 이내로 남은 종목입니다." },
  near: { color: "#2f9e44", text: "최고 종가까지 3% 초과 7% 이내로 남은 종목입니다." },
  held: { color: "#495057", text: "전략이 이미 들고 있는 종목이 신고가를 다시 넘은 상태입니다. 돌파 신호는 매일 나오지만 재매수하지 않아 목록에만 남습니다." },
} as const;

function describeStage(row: PositionRow): { label: string; color: string } {
  // 전략이 이미 들고 있으면 상태보다 '못 산다'는 사실이 먼저다.
  // (실계좌 보유 여부는 별도 '보유' 컬럼이 담당한다 — 뜻이 달라 섞지 않는다.)
  if (row.is_held) return { label: "신고가 갱신", color: STAGE_STYLE.held.color };
  if (row.gap_pct >= 0) {
    // (미달)은 어차피 매수 대상이 아니라 장중/마감 구분 없이 한 라벨로 합친다.
    if (!row.qualifies) return { label: "돌파(미달)", color: STAGE_STYLE.blocked.color };
    // 장중/마감 구분 없이 '돌파성공' 하나 — 거래대금 기준(한국은 시간 비례)까지 통과한 상태다.
    return { label: "돌파성공", color: STAGE_STYLE.breakout.color };
  }
  // 장중에 선을 건드렸다가 밀린 것은 그냥 '임박'과 성격이 달라 따로 표시한다.
  if (row.touched) return { label: "터치 후 밀림", color: STAGE_STYLE.pullback.color };
  if (row.gap_pct > -3) return { label: "임박", color: STAGE_STYLE.imminent.color };
  return { label: "근접", color: STAGE_STYLE.near.color };
}

/** 상태 컬럼 폭 — 보유 종목·진입 후보 표가 같은 값을 쓴다(폭이 다르면 어긋나 보인다). */
const STATUS_COLUMN_WIDTH = 124;
const STATUS_COLUMN_MIN_WIDTH = 110;

/** 상태 단계 설명 — 진입 후보를 펼치면 보여준다. 색·문구는 STAGE_STYLE 이 단일 소스다. */
const STAGE_GUIDE: { label: string; key: keyof typeof STAGE_STYLE }[] = [
  { label: "돌파성공", key: "breakout" },
  { label: "돌파(미달)", key: "blocked" },
  { label: "터치 후 밀림", key: "pullback" },
  { label: "임박", key: "imminent" },
  { label: "근접", key: "near" },
  { label: "신고가 갱신", key: "held" },
];

/** 접이식 전략 설명 — 운용 현황·백테스트 섹션 상단(기본 접힘). */
const CURRENT_NOTES = [
  {
    title: "진입",
    body:
      "종가가 직전 52주 최고 종가를 넘고 거래대금 배수(20일 평균 대비)가 하한 이상이면 다음 거래일 시가에 매수합니다. " +
      "빈 슬롯만큼, 거래대금 배수 순으로 담습니다. " +
      "ADR 하한을 설정하면 전일 시장 ADR(20일 상승/하락 종목수 비율 — 시장은 종목풀 설정의 시장 레짐 지수)이 " +
      "하한 미만인 날은 신규 진입을 하지 않습니다(보유 종목은 아래 청산 규칙이 그대로 관리).",
  },
  {
    title: "청산",
    body:
      "종가가 손절선(진입가 대비) 아래거나 이탈 이동평균을 하회하면 다음 거래일 시가에 전량 매도합니다. " +
      "부분 매도와 목표가(익절)는 없습니다.",
  },
  {
    title: "보유 중",
    body:
      "보유 중에는 비중을 건드리지 않습니다 — 오르는 종목은 커진 채 갑니다. " +
      "슬롯이 꽉 차면 새 돌파가 와도 교체하지 않습니다.",
  },
  {
    title: "장중 표시",
    body:
      "장중에는 현재가를 잠정 종가로 넣어 돌파·이탈을 미리 판정합니다 — 종가 확정 전에는 결과가 뒤집힐 수 있습니다. " +
      "현재가·일간(%)은 60초마다 갱신됩니다.",
  },
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
  /** 종목 메모 저장 — 계좌가 아니라 종목에 붙는다(순위·모멘텀·자산 관리 화면과 같은 값·같은 API). */
  const saveMemo = useCallback(
    async (ticker: string, memo: string) => {
      try {
        await updateStockMemo(ticker, memo);
        toast.success("메모 저장 완료");
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "메모 저장에 실패했습니다.");
      }
    },
    [toast],
  );
  const [view, setView] = useState<View | null>(null);
  const [positions, setPositions] = useState<Positions | null>(null);
  const [backtest, setBacktest] = useState<Backtest | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [backtesting, setBacktesting] = useState(false);
  // 백테스트 기간 — 저장하지 않고 실행할 때 고른다.
  const [backtestMonths, setBacktestMonths] = useState<number>(12);
  const [backtestProgress, setBacktestProgress] = useState<LoadingProgress | null>(null);
  const [positionsProgress, setPositionsProgress] = useState<LoadingProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backtestError, setBacktestError] = useState<string | null>(null);
  const [draft, setDraft] = useState<Settings | null>(null);
  const [saving, setSaving] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("monthly");
  const [currentTab, setCurrentTab] = useState<CurrentTab>("list");
  // 기준일 — 빈 값이면 최신 거래일. 과거 날짜를 고르면 그 시점 상태를 재현한다.
  const [candidatesOpen, setCandidatesOpen] = useState(false);
  const [charts, setCharts] = useState<HoldingChartData[] | null>(null);
  const [chartsLoading, setChartsLoading] = useState(false);
  const [chartsError, setChartsError] = useState<string | null>(null);
  // 차트 기간(개월) — 백엔드 config.HOLDING_CHART_MONTHS 가 단일 소스. 응답에서 받아 문구에 쓴다.
  const [chartMonths, setChartMonths] = useState<number | null>(null);

  const constraints = view?.constraints ?? null;
  // 창 길이가 바뀌면 문구도 따라 바뀐다 — "52주" 를 문자열로 박지 않는다.
  const windowLabel = constraints ? `${constraints.high_window_weeks}주` : "";
  // 진입·청산 체결일 — 장 시작 전에는 '오늘', 마감 뒤 캐시가 갱신되면 '내일' 이 된다.
  const fillDay = formatFillDay(positions?.next_session);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        // 마지막으로 고른 풀은 브라우저에 기억한다(다른 화면들과 같은 공용 키).
        const remembered = readRememberedTickerType();
        const query = remembered ? `?pool=${encodeURIComponent(remembered)}` : "";
        const response = await fetch(`/api/strategy-new-high${query}`, { cache: "no-store" });
        const payload = (await response.json()) as View & { error?: string };
        if (!response.ok) throw new Error(payload.error ?? "설정을 불러오지 못했습니다.");
        if (!alive) return;
        setView(payload);
        setDraft(payload.settings);
        if (payload.coerced?.length) {
          toast.warning(`저장값이 선택지에 없어 보정했습니다: ${payload.coerced.join(", ")} — 확인 후 저장하세요.`);
        }
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

  const runPositions = useCallback(async (settings: Settings) => {
    setRunning(true);
    setPositionsProgress({ percent: 10, message: "돌파·보유 상태를 계산하는 중" });
    const stopRamp = startProgressRamp(setPositionsProgress);
    try {
      const response = await fetch("/api/strategy-new-high/positions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ settings }),
      });
      const payload = (await response.json()) as Positions & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "돌파 종목을 불러오지 못했습니다.");
      setPositions(payload);
    } catch (runError) {
      toast.error(runError instanceof Error ? runError.message : "돌파 종목을 불러오지 못했습니다.");
    } finally {
      stopRamp();
      setRunning(false);
      setPositionsProgress(null);
    }
  }, [toast]);

  // 설정을 받으면 곧바로 오늘 상태를 채운다 — 빈 화면을 먼저 보여주지 않는다.
  useEffect(() => {
    if (view?.settings && !positions && !running) void runPositions(view.settings);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view]);

  const persistSettings = useCallback(
    async (settings: Settings, message: string) => {
      setSaving(true);
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
        void runPositions(payload.settings);
      } catch (saveError) {
        toast.error(saveError instanceof Error ? saveError.message : "설정을 저장하지 못했습니다.");
      } finally {
        setSaving(false);
      }
    },
    [runPositions, toast],
  );

  // 저장하지 않은 입력이 있으면 아래 결과가 화면 값과 어긋난다 — 저장을 먼저 요구한다(SM 과 같은 규칙).
  // 저장 응답을 그대로 초안에 넣으므로 두 객체의 키 구성은 항상 같다.
  // 업종 컬럼 노출 여부 — 표시 중인 결과의 풀 성격(pool_kind)이 1순위(개별주=표시, ETF=숨김),
  // 미설정 풀은 행 값 유무로 추정 (pools-rank·strategy-momentum 과 같은 기준).
  // 업종 컬럼·업종 상한 노출 — 판정은 전 화면 공용(`@/lib/pool-industry`).
  const selectedPoolOption = useMemo(() => {
    const pool = positions?.pool ?? view?.settings.pool ?? "";
    return view?.pool_options?.find((option) => option.ticker_type === pool);
  }, [positions?.pool, view?.settings.pool, view?.pool_options]);
  const hasIndustryData = poolHasIndustry(selectedPoolOption);
  const hasMarketCap = poolHasMarketCap(selectedPoolOption);

  const effectiveDraft = draft;

  const isDirty = useMemo(() => {
    const saved = view?.settings;
    if (!effectiveDraft || !saved) return false;
    // 서버가 선택지 밖 저장값을 보정해 보낸 경우도 '저장 필요'로 본다 (저장하면 coerced 가 비워진다).
    if (view?.coerced?.length) return true;
    return (Object.keys(effectiveDraft) as (keyof Settings)[]).some((key) => effectiveDraft[key] !== saved[key]);
  }, [effectiveDraft, view?.settings, view?.coerced]);

  /** 풀을 바꾸면 그 풀에 저장된 설정으로 전환한다. 저장 이력이 없으면 **기본값**으로 채운다 —
   *  직전 풀의 값을 물려받으면 다른 풀의 설정이 섞여 풀별로 보관하는 의미가 없어진다. */
  const handlePoolChange = useCallback(
    (pool: string) => {
      if (!view) return;
      writeRememberedTickerType(pool);
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
    setBacktestProgress({ percent: 10, message: "체결 내역을 계산하는 중" });
    const stopRamp = startProgressRamp(setBacktestProgress);
    try {
      const response = await fetch("/api/strategy-new-high/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ months: backtestMonths, settings: draft }),
      });
      const payload = (await response.json()) as Backtest & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "백테스트에 실패했습니다.");
      setBacktestProgress({ percent: 100, message: "결과 반영 중" });
      setBacktest(payload);
    } catch (runError) {
      const message = runError instanceof Error ? runError.message : "백테스트에 실패했습니다.";
      setBacktestError(message);
      toast.error(message);
    } finally {
      stopRamp();
      setBacktesting(false);
      setBacktestProgress(null);
    }
  }, [draft, toast, backtestMonths]);

  // 장중에는 실시간 시세가 움직이므로 주기적으로 다시 받는다.
  // 과거 날짜를 보는 중이거나 장이 닫혀 있으면 갱신할 것이 없어 타이머를 걸지 않는다.
  useEffect(() => {
    if (!positions?.auto_refresh || !draft) return;
    const id = window.setInterval(() => void runPositions(draft), LIVE_REFRESH_MS);
    return () => window.clearInterval(id);
  }, [positions?.auto_refresh, draft, runPositions]);

  const positionColumns = useMemo<ColDef<PositionRow>[]>(
    () => [
      {
        headerName: "상태",
        width: STATUS_COLUMN_WIDTH,
        minWidth: STATUS_COLUMN_MIN_WIDTH,
        cellStyle: { display: "flex", alignItems: "center", justifyContent: "center" },
        valueGetter: (p) => p.data?.gap_pct ?? null,
        cellRenderer: (p: { data?: PositionRow }) => {
          if (!p.data) return null;
          const stage = describeStage(p.data);
          return <strong style={{ color: stage.color }}>{stage.label}</strong>;
        },
      },
      // 실계좌 보유는 컬럼 대신 행 배경(appHeldRow, 녹색)으로 표시한다 — 시장 화면과 같은 표준.
      // 시총은 개별주에만 있는 값이라 업종과 판정이 다르다(`@/lib/pool-industry`).
      marketCapRankColumn<PositionRow>("market_cap_rank", !hasMarketCap),
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
      // 종목 메모 — 순위·모멘텀·자산 관리 화면과 같은 값(종목에 붙는다). 셀을 벗어나면 저장.
      stockMemoColumn<PositionRow>({
        field: "memo",
        onSave: (row, memo) => void saveMemo(row.ticker, memo),
      }),
      // 업종 컬럼 노출 여부 — 종목풀 설정의 풀 성격(pool_kind)이 1순위(개별주=표시, ETF=숨김),
      // 미설정 풀은 행 값 유무로 추정 (pools-rank·strategy-momentum 과 같은 기준).
      industryColumn<PositionRow>({ hide: !hasIndustryData }),
      dailyChangeColumn<PositionRow>(),
      {
        field: "price",
        headerName: "현재가",
        width: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatPrice(p.value as number),
      },
      tradeValueMultColumn<PositionRow>({
        qualifies: (row) => row?.qualifies,
        headerTooltip:
          "20일 평균 거래대금 대비 배수 — 하한 이상이어야 진입한다(장중 한국은 시간 비례 하한). 굵은 글씨가 자격 통과다. " +
          "괄호는 장중 시간 비율로 환산한 배수 — 지금 페이스대로면 하루 기준 몇 배인지입니다.",
      }),
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
        valueFormatter: (p) => formatMarketCapWon(p.value as number | null),
      },
    ],
    [windowLabel, hasIndustryData],
  );

  // ADR 값이 하나도 없으면(레짐 시장 없는 풀) 컬럼을 숨긴다 — 모멘텀 화면과 같은 규칙.
  const hasAdr = useMemo(() => (backtest?.daily ?? []).some((row) => row.adr != null), [backtest]);

  const periodColumns = useMemo<ColDef<PeriodRow>[]>(
    () => [
      {
        field: "period",
        headerName:
          viewMode === "yearly" ? "연도" : viewMode === "monthly" ? "월" : viewMode === "weekly" ? "주" : "일자",
        width: 148,
        valueFormatter: (p) =>
          viewMode === "weekly" || viewMode === "daily"
            ? formatDateWithWeekday(String(p.value ?? ""))
            : String(p.value ?? ""),
      },
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
      // ADR — 일간=당일, 주간=판정일(주 마지막 거래일), 월간·연간=기간 최저. 모멘텀 표와 같은 구성.
      viewMode === "daily"
        ? adrColumn<PeriodRow>({ headerName: "ADR", headerTooltip: "그날의 시장 ADR(20일 등락비율)", hide: !hasAdr, getter: (row) => row.adr })
        : viewMode === "weekly"
          ? adrColumn<PeriodRow>({ headerName: "판정일 ADR", headerTooltip: "그 주 마지막 거래일의 시장 ADR — 다음 거래일 진입 게이트를 결정한 값", hide: !hasAdr, getter: (row) => row.adr })
          : adrColumn<PeriodRow>({ headerName: "최저 ADR", headerTooltip: `${viewMode === "yearly" ? "그 해" : "그 달"}의 시장 ADR 최저값`, hide: !hasAdr, getter: (row) => row.adr_min }),
    ],
    [viewMode, backtest?.benchmark_name, hasAdr],
  );

  const periodRows = useMemo<PeriodRow[]>(() => {
    if (!backtest) return [];
    if (viewMode === "weekly") return toPeriodRows(backtest.daily, weekKeyOf, true);
    const keyLength = viewMode === "yearly" ? 4 : viewMode === "monthly" ? 7 : viewMode === "daily" ? 10 : 0;
    return keyLength ? toPeriodRows(backtest.daily, (date) => date.slice(0, keyLength)) : [];
  }, [backtest, viewMode]);

  // 보유 + 내일 매도 + 내일 매수를 한 표로 합친다 — 이 표만 보고 주문을 낼 수 있게.
  // 표시용 현재가·일간(%)만 60초마다 갱신한다 — 돌파 판정은 종가 기준이라 다시 계산하지 않는다.
  const quoteTickers = useMemo(
    () =>
      [
        ...(positions?.holdings ?? []).map((row) => row.ticker),
        ...(positions?.planned_entries ?? []).map((row) => row.ticker),
        ...(positions?.breakouts ?? []).map((row) => row.ticker),
        ...(positions?.candidates ?? []).map((row) => row.ticker),
      ].filter((ticker, index, all) => all.indexOf(ticker) === index),
    [positions],
  );
  const quotes = useRealtimeQuotes(positions?.country ?? "", quoteTickers);
  /** 시세가 들어온 종목만 현재가·등락률을 덮어쓴다(나머지 값은 판정 결과 그대로). */
  const withQuote = useCallback(
    <T extends { ticker: string; price: number | null; change_pct: number | null }>(row: T): T => {
      const quote = quotes[row.ticker];
      return quote ? { ...row, price: quote.price, change_pct: quote.change_pct } : row;
    },
    [quotes],
  );

  const planRows = useMemo<PlanRow[]>(() => {
    if (!positions) return [];
    const held: PlanRow[] = positions.holdings.map((h) => ({
      ticker: h.ticker, name: h.name, industry: h.industry, market_cap_rank: h.market_cap_rank ?? null,
      change_pct: h.change_pct, price: h.price,
      value_mult: h.value_mult ?? null, value_mult_live: h.value_mult_live ?? null,
      exit_price: null,
      entry_date: h.entry_date, entry_price: h.entry_price, return_pct: h.return_pct,
      plan: h.status, days: h.days, is_new: h.is_new, exit_reason: h.exit_reason,
      memo: h.memo,
      exit_ma_gap_pct: h.exit_ma_gap_pct, exit_ma: h.exit_ma,
    }));
    const buys: PlanRow[] = positions.planned_entries.map((row) => ({
      ticker: row.ticker, name: row.name, industry: row.industry, market_cap_rank: row.market_cap_rank ?? null,
      change_pct: row.change_pct, price: row.price,
      value_mult: row.value_mult ?? null, value_mult_live: row.value_mult_live ?? null,
      exit_price: null,
      entry_date: null, entry_price: null, return_pct: null,
      // 아직 안 샀다 — null 로 두면 보유일 칸과 차트 배지가 통째로 비어 진입 전인지 알 수 없다.
      plan: "buy", days: 0, is_new: false, exit_reason: null,
      memo: row.memo,
    }));
    // 오늘 이미 청산된 종목 — 현재가는 지금 시세, 청산가는 따로 담는다.
    const exited: PlanRow[] = positions.exited_today.map((t) => ({
      ticker: t.ticker, name: t.name, industry: t.industry, market_cap_rank: t.market_cap_rank ?? null,
      change_pct: t.change_pct ?? null,
      price: t.price ?? null, exit_price: t.exit_price,
      value_mult: t.value_mult ?? null, value_mult_live: t.value_mult_live ?? null,
      entry_date: t.entry_date, entry_price: t.entry_price, return_pct: t.return_pct,
      plan: "exited", days: t.days, is_new: false, exit_reason: t.reason,
      memo: t.memo,
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
  // 사용자가 고른 풀(draft.pool)을 앞에 둔다 — `positions` 는 새 결과가 와야 바뀌므로,
  // 그것만 보면 풀을 바꾼 뒤 결과가 오기까지 **이전 풀의 차트가 그대로 남는다**.
  const chartKey = useMemo(
    () => `${draft?.pool ?? ""}|${positions?.pool ?? ""}|${positions?.as_of ?? ""}|${chartRows.map((row) => row.ticker).join(",")}`,
    [draft?.pool, positions?.pool, positions?.as_of, chartRows],
  );

  useEffect(() => {
    setCharts(null);
    setChartsError(null);
  }, [chartKey]);

  // 차트 탭을 열 때만 받는다 — 보유 종목 수만큼 일봉을 실어 오므로 목록 탭에서는 낭비다.
  useEffect(() => {
    if (currentTab !== "chart" || !draft || !positions || charts || chartsLoading || chartsError) return;
    // 재계산 중이면 목록이 옛 풀 것이거나 비어 있다 — 여기서 확정하지 않는다.
    if (running || saving) return;
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
          }),
        });
        const payload = (await response.json()) as { charts?: HoldingChartData[]; months?: number; error?: string };
        if (!response.ok) throw new Error(payload.error ?? "차트를 불러오지 못했습니다.");
        setCharts(payload.charts ?? []);
        setChartMonths(payload.months ?? null);
      } catch (chartError) {
        const message = chartError instanceof Error ? chartError.message : "차트를 불러오지 못했습니다.";
        setChartsError(message);
        toast.error(message);
      } finally {
        setChartsLoading(false);
      }
    })();
  }, [currentTab, draft, positions, charts, chartsLoading, chartsError, chartRows, toast]);

  const holdingColumns = useMemo<ColDef<PlanRow>[]>(
    () => [
      {
        headerName: "상태",
        // 진입 후보 표의 상태와 같은 폭 — 두 표가 나란히 있어 폭이 다르면 어긋나 보인다.
        // "매도 예정(예상) (손절)" 같은 긴 문구는 잘릴 수 있어 전체를 툴팁으로 둔다.
        width: STATUS_COLUMN_WIDTH,
        minWidth: STATUS_COLUMN_MIN_WIDTH,
        cellStyle: { display: "flex", alignItems: "center", justifyContent: "center" },
        valueGetter: (p) => p.data?.plan ?? "",
        cellRenderer: (p: { data?: PlanRow }) => {
          if (!p.data) return null;
          // 장중 판정은 오늘 종가로 확정되기 전이라 (예상) — 종가 확정 후에는 꼬리표가 빠진다.
          const tag = positions?.live ? "(예상)" : "";
          const reason = p.data.exit_reason ? ` (${p.data.exit_reason})` : "";
          if (p.data.plan === "buy") return <strong style={{ color: "#d62828" }}>진입 예정{tag}</strong>;
          if (p.data.plan === "sell") {
            const label = `매도 예정${tag}${reason}`;
            return <strong style={{ color: "#1971c2", whiteSpace: "nowrap" }} title={label}>{label}</strong>;
          }
          if (p.data.plan === "exited") {
            const label = `이탈${reason}`;
            return <span style={{ color: "var(--text-muted)", whiteSpace: "nowrap" }} title={label}>{label}</span>;
          }
          return <span>{p.data.is_new ? "진입" : `${p.data.days}일`}</span>;
        },
      },
      marketCapRankColumn<PlanRow>("market_cap_rank", !hasMarketCap),
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
      // 종목 메모 — 순위·모멘텀·자산 관리 화면과 같은 값(종목에 붙는다). 셀을 벗어나면 저장.
      stockMemoColumn<PlanRow>({
        field: "memo",
        onSave: (row, memo) => void saveMemo(row.ticker, memo),
      }),
      industryColumn<PlanRow>({ hide: !hasIndustryData }),
      dailyChangeColumn<PlanRow>(),
      {
        field: "price",
        headerName: "현재가",
        width: 110,
        type: "numericColumn",
        headerTooltip: "이탈한 종목도 지금 시세다 — 판 뒤의 흐름을 청산가와 견줘 볼 수 있다.",
        valueFormatter: (p) => (p.value == null ? "-" : formatPrice(p.value as number)),
      },
      // 표준 배치(일간(%) → 현재가 → 거래대금) — 순위·진입 후보 표와 같은 공용 컬럼.
      tradeValueMultColumn<PlanRow>(),
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
      maExitGapColumn<PlanRow>({
        field: "exit_ma_gap_pct",
        maDays: draft?.exit_ma_days,
        getMaValue: (row) => row?.exit_ma,
        formatMaValue: (value) => formatPrice(value),
      }),
    ],
    [hasIndustryData, fillDay, positions?.live, draft?.exit_ma_days],
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
      industryColumn<Trade>({ hide: !hasIndustryData }),
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

  if (loading) return <PageFrame title="신고가 전략" fullWidth><div className="appPageStack">불러오는 중…</div></PageFrame>;
  if (error || !view || !draft || !constraints) {
    return (
      <PageFrame title="신고가 전략" fullWidth>
        <div className="alert alert-danger">{error ?? "설정을 불러오지 못했습니다."}</div>
      </PageFrame>
    );
  }

  return (
    <PageFrame title="신고가 전략" fullWidth>
      <div className="appPageStack">
        <section className="appSection">
          <div className="card appCard">
            <div className="card-body appCardBodyTight">
              {/* 메인 헤더 — 주 제어(셀렉터·토글). CRUD 버튼은 아래 보조 액션 헤더로 뺀다. */}
              <div className="appMainHeader">
                <div className="appMainHeaderLeft">
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
                    <span className="appLabeledFieldLabel">이탈 이평선</span>
                    <MaDaysSelect
                      value={draft.exit_ma_days}
                      options={constraints.exit_ma_options}
                      onChange={(days) => setDraft({ ...draft, exit_ma_days: days })}
                    />
                  </label>
                  <label className="appLabeledField">
                    <span className="appLabeledFieldLabel">거래대금 하한</span>
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
                    <span className="appLabeledFieldLabel">ADR 하한</span>
                    <select
                      className="form-select form-select-sm"
                      value={draft.adr_floor == null ? "" : String(draft.adr_floor)}
                      onChange={(event) => setDraft({
                        ...draft,
                        adr_floor: event.target.value === "" ? null : Number(event.target.value),
                      })}
                      title="전일 시장 ADR(20일 등락비율)이 이 값 미만이면 그날은 신규 진입을 하지 않는다. 보유 종목은 손절선·이탈 이평선이 그대로 관리한다. 시장은 종목풀 설정의 시장 레짐 지수를 따른다."
                    >
                      {(constraints.adr_floor_options ?? []).map((value) => (
                        <option key={String(value)} value={value == null ? "" : String(value)}>
                          {value == null ? "없음" : String(value)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="appLabeledField">
                    <span className="appLabeledFieldLabel">손절선</span>
                    <select
                      className="form-select form-select-sm"
                      value={draft.stop_loss_pct == null ? "" : String(draft.stop_loss_pct)}
                      onChange={(event) => setDraft({
                        ...draft,
                        stop_loss_pct: event.target.value === "" ? null : Number(event.target.value),
                      })}
                      title="진입가 대비 이 낙폭에 닿으면 다음 시가에 판다. '없음'이면 손절 없이 이탈 이평선만으로 청산한다."
                    >
                      {constraints.stop_loss_options.map((value) => (
                        <option key={String(value)} value={value == null ? "" : String(value)}>
                          {value == null ? "없음" : `${value}%`}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                {/* CRUD 버튼이 하나뿐이라 별도 줄을 두지 않고 메인 헤더 오른쪽에 둔다. */}
                <div className="appMainHeaderRight">
                  <UnsavedChangesBadge show={isDirty} />
                  <button
                    type="button"
                    className="btn btn-success btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                    onClick={() => void persistSettings(effectiveDraft ?? draft, "설정을 저장했습니다.")}
                    disabled={saving || !isDirty}
                  >
                    <IconCheck size={16} />
                    <span>{saving ? "저장 중…" : "저장"}</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="appSection">
          <div className="card appCard">
            <div className="card-header appCardHeader">
              <span style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>운용 현황</span>
              <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
                {positions ? (
                  <span style={hintStyle}>
                    {positions.live || positions.pre_market
                      ? `${formatKstDateTime(positions.quote_at)} 시세 · ${positions.live ? "장중" : "장전"}`
                      : `${formatKstDateTime(positions.refreshed_at)} 갱신`}
                  </span>
                ) : null}
                {/* 기준일 셀렉트를 없앤 뒤에도 "언제 기준인지"는 남긴다 — 장중이면 자동 갱신 중이라는 표시. */}
                {positions ? (
                  <span style={hintStyle}>
                    {positions.auto_refresh
                      ? `오늘 (${positions.live ? "장중" : "장전"} 자동 갱신)`
                      : `오늘 (${formatDateWithWeekday(positions.as_of)})`}
                  </span>
                ) : null}
                <span style={hintStyle}>{running ? "계산 중…" : ""}</span>
              </span>
            </div>
            <div className="card-body appCardBodyTight">
              <StrategyNotes items={CURRENT_NOTES} />
              <NavTabs
                items={CURRENT_TABS}
                value={currentTab}
                onChange={setCurrentTab}
                label="운용 현황 보기"
                style={{ marginBottom: 12 }}
              />
              {running && !positions ? (
                <AppLoadingProgress title="운용 현황 계산 중..." progress={positionsProgress} />
              ) : currentTab === "list" ? (
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
                    {positions?.adr_gate ? (
                      positions.adr_gate.blocked ? (
                        <b style={{ color: "#d9480f" }}>
                          {" "}· ADR 게이트 발동 — {positions.adr_gate.market} {positions.adr_gate.value ?? "-"} &lt;{" "}
                          {positions.adr_gate.floor}, 오늘 신규 진입 없음
                        </b>
                      ) : (
                        <span>
                          {" "}· ADR {positions.adr_gate.market} {positions.adr_gate.value ?? "-"} (하한 {positions.adr_gate.floor})
                        </span>
                      )
                    ) : null}
                  </div>
                  <AppAgGrid<PlanRow>
                    rowData={planRows.map(withQuote)}
                    columnDefs={holdingColumns}
                    loading={running}
                    theme={gridTheme}
                    minHeight={0}
                    height="auto"
                    gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                  />
                  {/* 진입 후보 표 — 항상 펼쳐 둔다. 상태 설명만 표 아래에서 접고 펼친다. */}
                  <div style={{ ...hintStyle, fontWeight: 700, margin: "16px 0 6px" }}>
                    진입 후보 ({(positions?.breakouts.length ?? 0) + (positions?.candidates.length ?? 0)}개)
                  </div>
                  <AppAgGrid<PositionRow>
                    rowData={[...(positions?.breakouts ?? []), ...(positions?.candidates ?? [])].map(withQuote)}
                    columnDefs={positionColumns}
                    loading={running}
                    theme={gridTheme}
                    minHeight={0}
                    height="auto"
                    // 실계좌 보유 종목은 행 배경 녹색 — 시장 화면과 같은 표준(보유 컬럼 대체).
                    getRowClass={(params) => (params.data?.account_held ? "appHeldRow" : "")}
                    gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                  />
                  <button
                    type="button"
                    onClick={() => setCandidatesOpen((open) => !open)}
                    style={{
                      ...hintStyle, fontWeight: 700, margin: "12px 0 6px", padding: 0,
                      background: "none", border: "none", cursor: "pointer",
                      display: "inline-flex", alignItems: "center", gap: 6,
                    }}
                  >
                    <span>{candidatesOpen ? "▾" : "▸"}</span>
                    진입 후보 설명
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
                </>
              ) : (
                <StrategyHoldingCharts
                  charts={charts}
                  loading={chartsLoading || running || saving}
                  error={chartsError}
                  emptyMessage="보유 중이거나 진입 예정인 종목이 없습니다."
                  months={chartMonths}
                  hint={
                    <>
                      종가가 직전 최고 종가를 넘으면 진입,{" "}
                      <strong style={{ color: "#12b886" }}>MA{draft.exit_ma_days}</strong>를 하회하면 청산합니다.
                      진입한 종목은 매수가와 함께 Buy 화살표가 표시됩니다.
                    </>
                  }
                  chartProps={(item) => {
                    const row = chartRows.find((candidate) => candidate.ticker === item.ticker);
                    return {
                      strategyLabel: "신고가",
                      entryDate: row?.entry_date,
                      entryPrice: row?.entry_price,
                      returnPct: row?.return_pct,
                      days: row?.days,
                    };
                  }}
                />
              )}
            </div>
          </div>
        </section>

        {/* 백테스트 — 운용 현황 아래에 나란히 둔다. */}
        <section className="appSection">
          <div className="card appCard">
            <div className="card-header appCardHeader">
              <span style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>백테스트</span>
              <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <MonthsSelect
                  value={backtestMonths}
                  options={constraints.month_options}
                  disabled={backtesting}
                  onChange={setBacktestMonths}
                />
                <button
                  type="button"
                  className="btn btn-sm btn-dark"
                  onClick={() => void handleBacktest()}
                  disabled={backtesting}
                >
                  {backtesting ? "실행 중…" : "실행"}
                </button>
              </span>
            </div>
            <div className="card-body appCardBodyTight">
              {backtestError ? <div className="alert alert-danger">{backtestError}</div> : null}
              {backtesting ? (
                <AppLoadingProgress title="백테스트 실행 중..." progress={backtestProgress} />
              ) : !backtest ? (
                <div style={{ ...hintStyle, padding: "24px 0", textAlign: "center" }}>
                  실행을 누르면 결과가 표시됩니다.
                </div>
              ) : (
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
                      label: backtest.benchmark_name,
                      totalPct: backtest.benchmark_total_pct,
                      cagrPct: backtest.benchmark_cagr_pct,
                      mddPct: backtest.benchmark_mdd_pct,
                      sortino: backtest.benchmark_sortino,
                    }}
                  />
                  <BacktestTradeStats stats={backtest} style={{ marginBottom: 12 }} />
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
          </div>
        </section>

        {/* 튜닝 — 백테스트 아래. 축 밖의 설정은 현재 화면 값으로 고정하고 범위 조합을 전부 돌린다. */}
        <StrategyTuning
          // 풀이 바뀌면 재마운트 — 이전 풀의 튜닝 결과가 남지 않게 한다(백테스트와 같은 시점).
          key={draft.pool}
          monthOptions={constraints.tuning_month_options ?? constraints.month_options}
          defaultMonths={backtestMonths}
          // 튜닝도 백테스트와 같이 **화면 초안** 기준이라(fixedLabel 참고) 실행 조건을 같게 둔다.
          disabled={backtesting}
          secondsPerCombo={0.04}
          extraSeconds={50}
          fixedLabel={`현재 화면 값 기준 (종목 수 ${draft.top_n} 공통 고정)`}
          current={{
            stop_loss_pct: draft.stop_loss_pct,
            exit_ma_days: draft.exit_ma_days,
            min_value_mult: draft.min_value_mult ?? null,
            adr_floor: draft.adr_floor ?? null,
          }}
          axes={[
            // 축 값 = 상단 셀렉트 선택지(서버 상수) — 순서·이름도 상단 설정과 같다.
            // 종목 수(공통 고정)·업종 상한(폐기)은 축이 아니다 — 튜닝은 시장의 반응 속도와 급증·손절 기준만 잰다.
            { key: "exit_ma_days", label: "이탈 이평선", values: constraints.exit_ma_options.map((n) => ({ value: n, label: `${n}일` })) },
            {
              key: "min_value_mult",
              label: "거래대금 하한",
              values: constraints.min_value_mult_options.map((n) => (n == null ? { value: null, label: "없음" } : { value: n, label: `${n}배` })),
            },
            {
              key: "adr_floor",
              label: "ADR 하한",
              values: (constraints.adr_floor_options ?? []).map((n) => (n == null ? { value: null, label: "없음" } : { value: n, label: String(n) })),
            },
            {
              key: "stop_loss_pct",
              label: "손절선",
              values: constraints.stop_loss_options.map((n) => ({ value: n, label: n == null ? "없음" : `${n}%` })),
            },
          ]}
          onApply={async (params) => {
            // 조합을 상단 폼에 넣고 그대로 저장한다 (저장 응답이 폼·운용 현황을 갱신한다).
            await persistSettings(
              {
                ...draft,
                stop_loss_pct: params.stop_loss_pct == null ? null : Number(params.stop_loss_pct),
                exit_ma_days: Number(params.exit_ma_days),
                min_value_mult: params.min_value_mult == null ? null : Number(params.min_value_mult),
                adr_floor: params.adr_floor == null ? null : Number(params.adr_floor),
              },
              "튜닝 조합을 적용해 저장했습니다.",
            );
          }}
          run={async (months, ranges) => {
            const response = await fetch("/api/strategy-new-high/tuning", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ months, settings: effectiveDraft ?? draft, ranges }),
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.error ?? "튜닝에 실패했습니다.");
            return payload as TuningResult;
          }}
        />
      </div>
    </PageFrame>
  );
}
