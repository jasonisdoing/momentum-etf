"use client";

import type { ColDef } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useState } from "react";
import { IconCheck } from "@tabler/icons-react";

import {
  formatAccountLabel,
  type AccountOptionBase,
} from "../components/AccountSelect";
import {
  readRememberedMomentumEtfAccountId,
  writeRememberedMomentumEtfAccountId,
} from "../components/account-selection";
import { AppAgGrid } from "../components/AppAgGrid";
import { MonthsSelect } from "../components/MonthsSelect";
import { isTrendBroken, renderStockNameCell } from "@/lib/name-highlight";
import { updateStockMemo } from "@/lib/stocks-store";
import {
  AppLoadingProgress,
  startProgressRamp,
  type LoadingProgress,
} from "../components/AppLoadingProgress";
import { BacktestSummary } from "../components/BacktestSummary";
import { BacktestTradeStats } from "../components/BacktestTradeStats";
import { useRealtimeQuotes } from "../components/useRealtimeQuotes";
import { StrategyNotes } from "../components/StrategyNotes";
import { NavTabs } from "../components/NavTabs";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { UnsavedChangesBadge } from "../components/UnsavedChangesBadge";
import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";
import { formatDateWithWeekday, formatKstDateTime } from "@/lib/datetime";
import {
  STOCK_NAME_COLUMN_MIN_WIDTH,
  stockMemoColumn,
  formatSignedPct,
  signColor,
} from "@/lib/grid-cells";
import { formatPoolLabel, type PoolLabelSource } from "@/lib/pool-label";

const gridTheme = createAppGridTheme();

/** 접이식 전략 설명 — 운용 현황·백테스트 섹션 상단(기본 접힘). */
const CURRENT_NOTES = [
  {
    title: "구성",
    body:
      "A·B 두 슬리브(전략 + 종목풀)의 저장 설정을 그대로 합쳐 한 계좌로 운용합니다. " +
      "슬리브 몫은 월초 배분(위 입력칸)에서 각자 흘러간 비율을 백테스트 곡선에서 역산합니다. " +
      "현금 몫은 두 전략에 주지 않고 늘 비워 두는 부분입니다.",
  },
  {
    title: "목표 비중",
    body:
      "종목 목표 = 슬리브 몫 × 슬리브 안 실제 비중입니다. 진입할 때 1/N을 배정하고 이후 시세대로 흘러간 값이라 " +
      "고정 비율이 아니며, 두 전략이 같은 종목을 담으면 합산합니다. 입출금이 없으면 지시가 거의 나오지 않습니다.",
  },
  {
    title: "액션",
    body:
      "목표와 보유의 차이가 0.5%p 이상이면 매일 지시로 나옵니다. 목표가 흘러간 비중을 따라가므로 " +
      "평소에는 차이가 없고, 지시가 나오는 건 입출금·교체·진입·이탈·월초 이관 같은 실변화뿐입니다. " +
      "교체 확정분은 그 슬리브의 교체일 시가 그룹으로 묶입니다.",
  },
  {
    title: "월초 배분 복구",
    body:
      "슬리브 재조정은 현금 우선으로 이관합니다 — 종목은 그대로 두고 장부상 현금만 옮깁니다. " +
      "한 슬리브의 주식만으로 그 슬리브 몫을 넘을 때만 초과분 매도 지시가 나옵니다.",
  },
];

const hintStyle: React.CSSProperties = {
  color: "var(--text-muted)",
  fontSize: "var(--fs-sm)",
};

/** 통화가 다른 계좌는 목표 금액·주수를 낼 수 없어 셀렉터에서 걸러낸다. */
type AccountOption = AccountOptionBase & {
  currency?: string;
  /** 이 계좌의 슬리브 둘 — (전략, 종목풀) 쌍. 이 화면에서 고른다. */
  a_strategy: string;
  b_strategy: string;
  /** 사용자가 붙인 표시 이름 — 비면 전략 이름을 쓴다. */
  a_name: string;
  b_name: string;
  /** 화면에 실제로 쓸 이름 — 위 이름이 있으면 그것, 없으면 전략 이름. */
  a_strategy_label: string;
  b_strategy_label: string;
  a_pool: string;
  b_pool: string;
  a_pool_label?: PoolLabelSource | null;
  b_pool_label?: PoolLabelSource | null;
  /** 조합이 다 정해졌는지 — 아니면 계산 대신 "고르세요" 를 보여준다. */
  mix_ready: boolean;
  country_code: string;
  /** 오늘의 액션 슬랙 알람 — 새 지시·수량 증가 시 발송. */
  mix_slack_enabled?: boolean;
  /** 합성 배분(%) — 모멘텀·신고가·비워 두는 현금. 셋의 합이 100 이다. */
  mix_a_pct: number;
  mix_b_pct: number;
  mix_cash_pct: number;
};
type StrategyOption = { value: string; label: string };
type MixPoolOption = PoolLabelSource & { country_code?: string | null };

type Meta = {
  /** 합성을 운용하는 계좌 — 계좌 설정에서 '합성' 을 켠 계좌가 온다. */
  accounts: AccountOption[];
  month_options: number[];
  strategy_options: StrategyOption[];
  pool_options: MixPoolOption[];
  /** 기본 선택 계좌 — 목록의 첫 계좌. */
  account_id: string;
};

type View = {
  computed_at: string;
  account_id: string;
  a_pool: string;
  b_pool: string;
  months: number;
  start_date: string;
  end_date: string;
  benchmark_name: string;
  benchmark_ticker: string;
  strategy_total_pct: number;
  strategy_cagr_pct: number | null;
  strategy_mdd_pct: number | null;
  strategy_sortino: number | null;
  benchmark_total_pct: number;
  benchmark_cagr_pct: number | null;
  benchmark_mdd_pct: number | null;
  benchmark_sortino: number | null;
  /** 일별 누적(%) — 연간·월간·주간·일간 표를 이 시계열에서 만든다 (신고가 화면과 동일 방식). */
  /** 일별 누적(%) — 합성·슬리브별 단독·벤치마크. 슬리브 값은 그 날짜 데이터가 없으면 null. */
  daily: {
    date: string;
    strategy_pct: number;
    a_pct: number | null;
    b_pct: number | null;
    benchmark_pct: number;
  }[];
  /** 체결 목록 — 두 전략 합본. exit_date 가 없으면 보유중. */
  trades: MixTradeRow[];
  /** 체결 통계 — 백엔드 공용 계산(utils/trade_stats.py). 청산분만 센다. */
  trade_count: number;
  win_rate_pct: number | null;
  avg_win_pct: number | null;
  avg_loss_pct: number | null;
  reason_counts?: Record<string, number>;
};

type MixTradeRow = {
  strategy: string;
  ticker: string;
  name: string;
  entry_date: string;
  entry_price: number | null;
  exit_date: string | null;
  exit_price: number | null;
  return_pct: number | null;
  days: number | null;
  reason: string;
};

type Holding = {
  ticker: string;
  name: string;
  sources: ("a" | "b")[];
  weight_pct: number;
  /** 슬리브별 몫(%) — 합이 목표비중이다. 전량 매도 행에는 없다. */
  a_weight?: number;
  b_weight?: number;
  price: number | null;
  change_pct: number | null;
  a_status: string | null;
  b_status: string | null;
  /** 전략 수익률(이론값) — 모멘텀: 연속 시작 교체일 시가 대비 · 신고가: 진입가 대비. */
  a_return_pct?: number | null;
  b_return_pct?: number | null;
  /** 보유 기간 — 전략마다 단위가 달라 백엔드가 완성된 문자열로 내려준다("3주"·"12일"·"신규"). */
  a_held_label?: string | null;
  b_held_label?: string | null;
  /** 적용 계좌가 있을 때만 온다 — 계좌 총자산 기준 목표 금액·주수와 현재 보유. */
  target_amount?: number | null;
  target_quantity?: number | null;
  held_quantity?: number | null;
  held_value?: number | null;
  current_weight_pct?: number;
  trade_quantity?: number | null;
  /** 주중 이탈 예상(오늘 종가 확정 시) — 매매수량·상태에 예상으로 겹쳐 보여준다. */
  is_exit_forecast?: boolean;
  forecast_trade_quantity?: number | null;
  /** 이탈 후 남을 목표수량 — 목표수량 칸에 (예상)으로 겹쳐 쓴다. */
  forecast_target_quantity?: number | null;
  /** 종목풀 설정 이평선 기준 이격(%) — 종목명 옆 추세 이탈 배지(❗)에 쓴다. */
  current_short_pct?: number | null;
  current_long_pct?: number | null;
  /** 목표 포트폴리오에 없는 보유 종목 — 목표 비중 0% 행으로 표 하단에 온다. */
  is_sell_all?: boolean;
};

/** 적용 계좌의 실제 상태 — 계산 기준 계좌의 원장(portfolio_master)에서 온다. */
type AccountState = {
  account_id: string;
  cash_balance: number;
  stock_value: number;
  total_assets: number;
  /** 목표에 없는 보유 종목 = 전량 매도 대상. */
  sell_all: {
    ticker: string;
    name: string;
    quantity: number;
    value: number | null;
  }[];
};

type Positions = {
  computed_at: string;
  account_id: string;
  a_pool: string;
  b_pool: string;
  /** 표시용 시세 갱신에 쓰는 국가 코드(시세 소스가 국가별로 다르다). */
  country: string;
  as_of: string;
  live: boolean;
  /** 다음 거래일 — 모든 체결이 시가라 액션 묶음의 날짜가 된다. */
  next_trading_day: string | null;
  /** 계산 기준 계좌의 원장 상태. 없으면 비중만 보여준다. */
  account: AccountState | null;
  /** 과거 날짜 셀렉트용 — 신고가 화면과 같은 날짜 목록. */
  available_dates: string[];
  summary: {
    stock_pct: number;
    cash_pct: number;
    /** 총 현금 중 두 전략에 주지 않고 비워 둔 몫(%). 나머지는 빈 슬롯에서 생긴다. */
    reserved_cash_pct: number;
    /** 월초에 되돌릴 배분(%) — 화면 헤더에서 저장한 값. */
    base_weights: { a_pct: number; b_pct: number; cash_pct: number };
    /** slots_used = 목표가 찬 슬롯, held_count = 지금 실제로 들고 있는 종목 수. */
    a: {
      /** 슬리브 몫(%) — 월초 배분에서 흘러간 비율. */
      alloc_pct: number;
      slots_used: number;
      held_count: number;
      top_n: number;
      cash_pct: number;
    };
    b: {
      alloc_pct: number;
      slots_used: number;
      held_count: number;
      top_n: number;
      cash_pct: number;
    };
  };
  holdings: Holding[];
  actions: {
    /** 슬리브별 액션 — 슬롯 키(a/b)로 담긴다. 그 전략에 없는 항목은 빈 목록/null 이다. */
    slots: Record<string, SlotActions>;
    sleeve_rebalance_today: boolean;
    /** 오늘의 액션 — 서버가 조립한 체결일 묶음(화면·슬랙 알람 공용 단일 소스). */
    groups: ActionGroup[];
    /** 다음주 교체 가정 미리보기 — 오늘의 액션과 같은 조립을 다음주 가정 목표로 돌린 결과.
     *  종목 교체가 없으면 오늘의 액션과 똑같다. 과거 조회면 null. */
    next_week_preview: {
      fill_date: string | null;
      groups: ActionGroup[];
    } | null;
  };
};

/** 슬리브 하나의 오늘 액션 — 전략에 따라 비는 항목이 있다. */
type SlotActions = {
  label: string;
  live: boolean;
  /** 다음 거래일 시가 매도(확정) — 자격 상실·이탈·손절. */
  sells: { ticker: string; name: string; reason: string; return_pct: number | null }[];
  /** 장중 판정 기준 이탈 예상 — 오늘 종가로 확정된다. 화면 전용(알람 제외). */
  exit_forecast: { ticker: string; name: string; reason: string }[];
  /** 다음 거래일 시가에 새로 담는 것. */
  entries: {
    ticker: string;
    name: string;
    price: number | null;
    change_pct: number | null;
    value_mult: number | null;
  }[];
  /** 주기적 교체가 있는 전략만 — 판정은 끝났고 체결만 남았다. 없으면 null. */
  rebalance: {
    is_filled: boolean;
    fill_date: string | null;
    signal_date: string | null;
    portfolio_week: string | null;
    buys: { ticker: string; name: string; price: number | null }[];
    sells: { ticker: string; name: string }[];
  } | null;
};

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
  /** 합성(두 슬리브를 한 계좌에서 굴린 결과). */
  strategy_pct: number;
  /** 각 전략을 **혼자** 굴렸을 때 — 합성이 단독보다 나은지 바로 읽으라고 함께 둔다. */
  a_pct: number | null;
  b_pct: number | null;
  benchmark_pct: number;
  /** 합성 − 벤치마크. */
  excess_pp: number;
};

/** 그 날짜가 속한 주의 월요일 — 주간 묶음 키. 로컬 기준으로 조립한다(UTC 파싱은 하루 밀린다). */
function weekKeyOf(date: string): string {
  const parsed = new Date(`${date}T00:00:00`);
  parsed.setDate(parsed.getDate() - ((parsed.getDay() + 6) % 7));
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${parsed.getFullYear()}-${pad(parsed.getMonth() + 1)}-${pad(parsed.getDate())}`;
}

/** 누적(%) 시계열을 기간별 수익률로 자른다 — 신고가 화면과 같은 계산.
 *  주간은 묶음 키(월요일)와 표시 라벨(그 주 마지막 거래일)이 달라 따로 담는다. */
function toPeriodRows(
  daily: View["daily"],
  keyOf: (date: string) => string,
  labelByLastDate = false,
): PeriodRow[] {
  if (daily.length === 0) return [];
  type Snapshot = {
    strategy: number;
    a: number | null;
    b: number | null;
    benchmark: number;
    lastDate: string;
  };
  const lastByPeriod = new Map<string, Snapshot>();
  const order: string[] = [];
  for (const point of daily) {
    const key = keyOf(point.date);
    if (!lastByPeriod.has(key)) order.push(key);
    lastByPeriod.set(key, {
      strategy: point.strategy_pct,
      a: point.a_pct,
      b: point.b_pct,
      benchmark: point.benchmark_pct,
      lastDate: point.date,
    });
  }
  // 첫 구간의 기준은 시작 시점(누적 0%)이다.
  let prev: Snapshot = { strategy: 0, a: 0, b: 0, benchmark: 0, lastDate: "" };
  const rows: PeriodRow[] = [];
  // 누적(%) 두 점 사이의 구간 수익률. 어느 한쪽이라도 값이 없으면 null — 0 으로 채우면
  // 데이터가 없는 구간이 '보합' 으로 보인다.
  const step = (now: number | null, before: number | null): number | null =>
    now == null || before == null ? null : ((1 + now / 100) / (1 + before / 100) - 1) * 100;
  for (const key of order) {
    const current = lastByPeriod.get(key)!;
    const strategy = step(current.strategy, prev.strategy) ?? 0;
    const benchmark = step(current.benchmark, prev.benchmark) ?? 0;
    rows.push({
      period: labelByLastDate ? current.lastDate : key,
      strategy_pct: strategy,
      a_pct: step(current.a, prev.a),
      b_pct: step(current.b, prev.b),
      benchmark_pct: benchmark,
      excess_pp: strategy - benchmark,
    });
    prev = current;
  }
  return rows.reverse();
}

function formatPrice(value: number | null | undefined): string {
  if (value == null) return "-";
  return value.toLocaleString("ko-KR", { maximumFractionDigits: 2 });
}

function formatAmount(value: number | null | undefined): string {
  if (value == null) return "-";
  return value.toLocaleString("ko-KR", { maximumFractionDigits: 0 });
}

/** 행이 어느 슬리브에서 왔는지 — 라벨은 계좌가 고른 전략명으로 채운다(고정 문구 금지). */
const SLOT_KEYS = ["a", "b"] as const;

/** 보유 표 행 — 현금 행도 같은 표에 넣는다 (비중 합이 100%임을 한눈에 보이게). */
type PositionRow = Holding & {
  is_cash?: boolean;
  /** 종목 메모 — 계좌가 아니라 종목에 붙는다(자산 관리·순위 화면과 같은 값). */
  memo?: string;
  amount: number | null;
  shares: number | null;
};

/** 리밸런싱 밴드(%p) — 목표 비중과 이만큼 미만으로 벌어진 종목은 오늘의 액션에서 뺀다.
 *  장중 가격이 조금만 움직여도 목표수량이 ±1주씩 흔들려 목록이 계속 바뀌기 때문이다.
 *  표에는 정확한 목표수량이 그대로 남는다(표는 대조용, 액션은 실행용). */

/** 오늘의 액션 한 줄. 같은 체결 시점끼리 묶고 묶음 안에서는 매도 → 매수 순서다. */
type ActionItem = {
  key: string;
  side: "sell" | "buy";
  title: string;
  text: string;
  /** 정렬용 — 표(티커 순)와 같은 기준으로 세운다. */
  ticker: string;
};
type ActionGroup = { key: string; title: string; items: ActionItem[] };

/** 합성 전략 — SM·신고가를 저장된 배분으로 함께 운용하는 화면.
 *  운용 현황 탭은 오늘 보유해야 할 종목과 현금 비중·오늘의 액션을,
 *  백테스트 탭은 매월 배분 복구 리밸런싱 합성 성과를 보여준다. 전략 설정은 각 전략 화면의 저장값을 쓰고,
 *  배분은 이 화면 헤더에서 정한다. */
export function StrategyMixClient() {
  const toast = useToast();
  /** 종목 메모 저장 — 순위·모멘텀 화면과 같은 API(종목에 붙는다). */
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
  // 계좌 하나만 고른다 — 슬리브별 종목풀은 계좌 설정에 붙어 있다.
  const [accountId, setAccountId] = useState<string>("");
  const [accountOptions, setAccountOptions] = useState<AccountOption[]>([]);
  // 조합 셀렉트 선택지(전략·종목풀) — 계좌 목록과 같은 응답으로 온다.
  const [meta, setMeta] = useState<Meta | null>(null);
  const [months, setMonths] = useState<number>(12);
  const [monthOptions, setMonthOptions] = useState<number[]>([]);

  // 계산 기준은 계좌다 — 슬리브별 풀은 서버가 계좌 설정에서 꺼낸다(계좌 설정이 단일 소스).
  const selectedAccount = accountOptions.find((option) => option.account_id === accountId) ?? null;
  /** 슬롯(a/b) → 「A. 모멘텀」처럼 슬롯 기호 + 계좌가 고른 전략 이름.
   *  두 슬리브가 같은 전략일 수 있으므로 전략 이름만으로는 구분되지 않는다. */
  const slotLabel = (slot: string) => {
    const strategy =
      slot === "a" ? selectedAccount?.a_strategy_label : slot === "b" ? selectedAccount?.b_strategy_label : null;
    if (slot !== "a" && slot !== "b") return slot;
    const mark = slot.toUpperCase();
    return strategy ? `${mark}. ${strategy}` : mark;
  };
  const labelA = slotLabel("a");
  const labelB = slotLabel("b");

  // 운용 현황 탭.
  const [positions, setPositions] = useState<Positions | null>(null);
  const [positionsLoading, setPositionsLoading] = useState(false);
  const [positionsError, setPositionsError] = useState<string | null>(null);
  const [positionsProgress, setPositionsProgress] =
    useState<LoadingProgress | null>(null);
  /** 계좌를 저장하면 목표 금액이 달라지므로 운용 현황를 다시 계산한다. */
  const [positionsReloadKey, setPositionsReloadKey] = useState(0);
  /** 다음주 교체 가정 — 기본 접힘 (참고 정보). */
  const [nextWeekOpen, setNextWeekOpen] = useState(false);

  // 백테스트 탭.
  const [view, setView] = useState<View | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("monthly");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backtestProgress, setBacktestProgress] =
    useState<LoadingProgress | null>(null);

  // 진입 시에는 계좌 목록만 받는다 — 계산은 탭·버튼이 시작한다.
  useEffect(() => {
    let alive = true;
    void (async () => {
      try {
        const response = await fetch("/api/strategy-mix/meta", {
          cache: "no-store",
        });
        const payload = (await response.json()) as Meta & { error?: string };
        if (!response.ok || payload.error)
          throw new Error(
            payload.error ?? "합성 운용 계좌 목록을 불러오지 못했습니다.",
          );
        if (!alive) return;
        const accounts = payload.accounts ?? [];
        setMeta(payload);
        setAccountOptions(accounts);
        setMonthOptions(payload.month_options);
        // 마지막으로 고른 계좌는 브라우저에 기억한다(다른 화면의 계좌 셀렉터와 같은 공용 키).
        const remembered = readRememberedMomentumEtfAccountId();
        const initial =
          remembered && accounts.some((option) => option.account_id === remembered)
            ? remembered
            : payload.account_id;
        setAccountId(initial);
      } catch (metaError) {
        if (alive)
          setError(
            metaError instanceof Error
              ? metaError.message
              : "합성 운용 계좌 목록을 불러오지 못했습니다.",
          );
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // 운용 현황 — 계좌가 정해지면 자동 계산한다 (매일 보는 운영 화면이라 버튼을 두지 않는다).
  useEffect(() => {
    if (!accountId) return;
    let alive = true;
    setPositionsLoading(true);
    setPositionsError(null);
    setPositionsProgress({
      percent: 10,
      message: "두 전략의 운용 현황를 계산하는 중",
    });
    const stopRamp = startProgressRamp(setPositionsProgress);
    void (async () => {
      try {
        const params = new URLSearchParams({ account_id: accountId });
        const response = await fetch(
          `/api/strategy-mix/positions?${params.toString()}`,
          {
            cache: "no-store",
          },
        );
        const payload = (await response.json()) as Positions & {
          error?: string;
        };
        if (!response.ok || payload.error)
          throw new Error(
            payload.error ?? "합성 운영 상태를 불러오지 못했습니다.",
          );
        if (alive) setPositions(payload);
      } catch (positionsFetchError) {
        if (alive)
          setPositionsError(
            positionsFetchError instanceof Error
              ? positionsFetchError.message
              : "합성 운영 상태를 불러오지 못했습니다.",
          );
      } finally {
        stopRamp();
        if (alive) {
          setPositionsLoading(false);
          setPositionsProgress(null);
        }
      }
    })();
    return () => {
      alive = false;
      stopRamp();
    };
  }, [accountId, positionsReloadKey]);


  const runBacktest = useCallback(async () => {
    if (!accountId) return;
    setLoading(true);
    setError(null);
    setBacktestProgress({
      percent: 10,
      message: "두 전략의 백테스트를 계산하는 중",
    });
    const stopRamp = startProgressRamp(setBacktestProgress);
    try {
      const response = await fetch(
        `/api/strategy-mix?account_id=${encodeURIComponent(accountId)}&months=${months}`,
        {
          cache: "no-store",
        },
      );
      const payload = (await response.json()) as View & { error?: string };
      if (!response.ok || payload.error)
        throw new Error(
          payload.error ?? "합성 백테스트를 불러오지 못했습니다.",
        );
      setBacktestProgress({ percent: 100, message: "결과 반영 중" });
      setView(payload);
    } catch (runError) {
      const message =
        runError instanceof Error
          ? runError.message
          : "합성 백테스트를 불러오지 못했습니다.";
      setError(message);
      toast.error(message);
    } finally {
      stopRamp();
      setLoading(false);
      setBacktestProgress(null);
    }
  }, [accountId, months, toast]);

  // 목표 금액의 기준 = 적용 계좌의 총자산(주식 평가액 + 현금). 계좌가 없으면 비중만 보여준다.
  const totalAsset = positions?.account?.total_assets ?? null;

  // 표시용 현재가·일간(%)만 60초마다 갱신한다 — 목표·매매수량은 5분 캐시된 판정 결과다.
  const quotes = useRealtimeQuotes(
    positions?.country ?? "",
    (positions?.holdings ?? []).map((row) => row.ticker),
  );

  const positionRows = useMemo<PositionRow[]>(() => {
    if (!positions) return [];
    // 현금은 빈 슬롯마다 나누지 않고 한 행으로 맨 위에 둔다 — 계좌 현금은 한 덩어리다.
    const cashRow: PositionRow = {
      ticker: "__cash__",
      name: "현금",
      sources: [],
      weight_pct: positions.summary.cash_pct,
      // 현금도 슬리브별로 나눠 보여준다 — 합이 목표비중(현금 전체)이다.
      a_weight: positions.summary.a.cash_pct,
      b_weight: positions.summary.b.cash_pct,
      price: null,
      change_pct: null,
      a_status: null,
      b_status: null,
      is_cash: true,
      current_weight_pct:
        totalAsset && positions.account
          ? (positions.account.cash_balance / totalAsset) * 100
          : undefined,
      held_value: positions.account?.cash_balance ?? null,
      amount:
        totalAsset == null
          ? null
          : (totalAsset * positions.summary.cash_pct) / 100,
      shares: null,
    };
    // 목표 종목 행은 백엔드가 종목 단위로 합쳐 계산한 값을 그대로 쓴다.
    // 현금만 맨 위에 두고 나머지는 티커 순 — 계좌·증권사 화면과 같은 순서로 대조하기 쉽게.
    return [
      cashRow,
      ...positions.holdings
        .map((holding) => {
          // 표시용 가격만 실시간으로 덮어쓴다 — 목표·매매수량은 판정 결과 그대로.
          const quote = quotes[holding.ticker];
          return {
            ...holding,
            price: quote ? quote.price : holding.price,
            change_pct: quote ? quote.change_pct : holding.change_pct,
            amount: holding.target_amount ?? null,
            shares: holding.target_quantity ?? null,
          };
        })
        .sort((a, b) => a.ticker.localeCompare(b.ticker)),
    ];
  }, [positions, quotes, totalAsset]);

  const positionColumns = useMemo<ColDef<PositionRow>[]>(() => {
    const columns: ColDef<PositionRow>[] = [
      {
        field: "ticker",
        headerName: "티커",
        width: 108,
        cellRenderer: (p: { value?: string; data?: PositionRow }) =>
          p.data?.is_cash ? (
            <span>-</span>
          ) : (
            <TickerDetailLink ticker={p.value} />
          ),
      },
      {
        field: "name",
        headerName: "종목명",
        flex: 1.4,
        minWidth: STOCK_NAME_COLUMN_MIN_WIDTH,
        // 굵기는 주지 않는다 — 다른 화면의 종목명과 같은 무게로 보여야 표가 한 벌로 읽힌다.
        cellStyle: (p) => (p.data?.is_cash ? { color: "var(--text-muted)" } : null),
        // 종목명 표기는 순위·전략 화면과 같은 공용 렌더러를 쓴다 — 레버리지 강조(💣)와
        // 긴 이름 2줄 줄임이 여기서만 빠져 있었다. 현금 행은 종목이 아니라 그대로 둔다.
        cellRenderer: (p: { value?: string | null; data?: PositionRow }) =>
          p.data?.is_cash ? (
            <span>{p.value ?? "-"}</span>
          ) : (
            renderStockNameCell(p.value, {
              trendBroken: isTrendBroken(p.data?.current_short_pct, p.data?.current_long_pct),
            })
          ),
      },
      // 종목 메모 — 순위·모멘텀·자산 관리 화면과 같은 값(종목에 붙는다).
      // 현금 행은 종목이 아니라 편집 대상이 아니다.
      stockMemoColumn<PositionRow>({
        field: "memo",
        editable: (row) => !row?.is_cash,
        onSave: (row, memo) => void saveMemo(row.ticker, memo),
      }),
      {
        field: "sources",
        headerName: "전략",
        width: 110,
        // 두 전략이 같은 종목을 담으면 한 행에 둘 다 표시된다 (비중은 합산).
        valueFormatter: (p) => {
          const sources = (p.value as string[]) ?? [];
          return sources.length === 0
            ? "-"
            : sources.map((source) => slotLabel(source)).join("·");
        },
      },
      // 아래 순서·명칭은 /assets 계좌 보유 표와 맞춘다 (일간→현재가→비중→목표비중→목표수량→수량).
      {
        field: "change_pct",
        headerName: "일간(%)",
        width: 92,
        type: "numericColumn",
        valueFormatter: (p) => formatSignedPct(p.value as number),
        cellStyle: (p) => ({
          color: signColor(p.value as number),
          fontWeight: 600,
        }),
      },
      {
        field: "price",
        headerName: "현재가",
        width: 104,
        type: "numericColumn",
        valueFormatter: (p) => formatPrice(p.value as number),
      },
      // 비중 = 계좌 평가액 기준 현재 비중. 목표와의 차이가 곧 조정할 양이다.
      {
        field: "current_weight_pct",
        headerName: "비중",
        width: 88,
        type: "numericColumn",
        valueFormatter: (p) =>
          p.value == null ? "-" : `${(p.value as number).toFixed(2)}%`,
        cellStyle: (p) => {
          const current = p.value as number | null;
          const target = p.data?.weight_pct;
          if (current == null || target == null) return null;
          // 목표에서 1%p 넘게 벌어진 행만 눌러 표시한다 — 반올림 오차로 전부 색이 붙는 걸 막는다.
          return Math.abs(current - target) > 1
            ? { color: "var(--text-muted)" }
            : null;
        },
      },
      // 슬리브별 몫 — 두 값의 합이 목표비중이다. 그 슬리브에 없는 종목은 '-'.
      {
        field: "a_weight",
        headerName: labelA,
        width: 104,
        type: "numericColumn",
        valueFormatter: (p) =>
          p.value == null || (p.value as number) === 0 ? "-" : `${(p.value as number).toFixed(2)}%`,
        cellStyle: { color: "var(--text-muted)" },
      },
      {
        field: "b_weight",
        headerName: labelB,
        width: 104,
        type: "numericColumn",
        valueFormatter: (p) =>
          p.value == null || (p.value as number) === 0 ? "-" : `${(p.value as number).toFixed(2)}%`,
        cellStyle: { color: "var(--text-muted)" },
      },
      {
        colId: "held_for",
        headerName: "보유일",
        width: 96,
        headerTooltip: `${labelA} · ${labelB} 슬리브의 보유 기간 (진입 예정은 -)`,
        cellStyle: { color: "var(--text-muted)", textAlign: "center" },
        valueGetter: (p) => {
          if (!p.data) return "";
          const bits: string[] = [];
          if (p.data.a_held_label) bits.push(p.data.a_held_label);
          if (p.data.b_held_label) bits.push(p.data.b_held_label);
          return bits.join(" · ") || "-";
        },
      },
      {
        colId: "strategy_return",
        headerName: "전략수익률",
        width: 104,
        type: "numericColumn",
        headerTooltip:
          "전략 이론값(계좌 실손익 아님) — 각 전략이 잡은 편입가 대비. 두 슬리브 보유면 몫 가중 평균.",
        valueGetter: (p) => {
          if (!p.data) return null;
          const parts: { w: number; r: number }[] = [];
          if (p.data.a_return_pct != null && (p.data.a_weight ?? 0) > 0)
            parts.push({ w: p.data.a_weight ?? 0, r: p.data.a_return_pct });
          if (p.data.b_return_pct != null && (p.data.b_weight ?? 0) > 0)
            parts.push({ w: p.data.b_weight ?? 0, r: p.data.b_return_pct });
          if (!parts.length) {
            // 몫이 0(매도 예정 등)이어도 수익률 자체는 보여준다.
            const single = p.data.a_return_pct ?? p.data.b_return_pct;
            return single ?? null;
          }
          const total = parts.reduce((a, x) => a + x.w, 0);
          return parts.reduce((a, x) => a + (x.r * x.w) / total, 0);
        },
        valueFormatter: (p) => (p.value == null ? "-" : formatSignedPct(p.value as number, 2)),
        cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 600 }),
        tooltipValueGetter: (p) => {
          if (!p.data) return "";
          const bits: string[] = [];
          if (p.data.a_return_pct != null) bits.push(`${labelA} ${formatSignedPct(p.data.a_return_pct, 2)}`);
          if (p.data.b_return_pct != null) bits.push(`${labelB} ${formatSignedPct(p.data.b_return_pct, 2)}`);
          return bits.join(" · ");
        },
      },
      {
        field: "weight_pct",
        headerName: "목표비중",
        headerTooltip: `${labelA} + ${labelB} 슬리브 몫의 합`,
        width: 88,
        type: "numericColumn",
        valueFormatter: (p) =>
          p.value == null ? "-" : `${(p.value as number).toFixed(2)}%`,
        cellStyle: { fontWeight: 600 },
      },
    ];
    // 계좌 총자산을 알 때만 매매 지시 컬럼을 붙인다 — 목표와 실제 보유의 차이가 주문 수량이다.
    if (totalAsset != null) {
      columns.push(
        {
          field: "shares",
          headerName: "목표수량",
          headerTooltip:
            "목표비중 × 총자산 ÷ 현재가. 주중 이탈·손절이 예상되는 종목은 이탈 후 남을 목표를 (예상)으로 보여준다.",
          width: 104,
          type: "numericColumn",
          valueFormatter: (p) => {
            // 예상 이벤트(주중 이탈·손절)가 있는 행만 예상 목표로 겹쳐 쓴다.
            // 가격 변동으로 목표와 조금 어긋나는 것은 예상이 아니라 그대로 둔다.
            const forecast = p.data?.forecast_target_quantity;
            if (p.data?.is_exit_forecast && forecast != null) {
              return `${forecast.toLocaleString("ko-KR")} (예상)`;
            }
            return p.value == null ? "-" : (p.value as number).toLocaleString("ko-KR");
          },
          cellStyle: (p) =>
            p.data?.is_exit_forecast && p.data?.forecast_target_quantity != null
              ? { color: "var(--down-color, #2f6fd0)", fontWeight: 600, opacity: 0.65 }
              : null,
        },
        {
          field: "held_quantity",
          headerName: "수량",
          width: 80,
          type: "numericColumn",
          valueFormatter: (p) =>
            p.value == null ? "-" : (p.value as number).toLocaleString("ko-KR"),
        },
        {
          field: "trade_quantity",
          headerName: "매매수량",
          headerTooltip: "목표수량 − 수량. +는 매수, −는 매도",
          width: 92,
          type: "numericColumn",
          valueFormatter: (p) => {
            const forecast = p.data?.forecast_trade_quantity;
            if (p.data?.is_exit_forecast && forecast != null) {
              return `${forecast.toLocaleString("ko-KR")} (예상)`;
            }
            const value = p.value as number | null;
            if (value == null) return "-";
            if (value === 0) return "0";
            return `${value > 0 ? "+" : ""}${value.toLocaleString("ko-KR")}`;
          },
          // 매수(+)·매도(−) 색은 사이트 공용 기준을 따른다 — 한국 관례로 매수 빨강·매도 파랑.
          // 주중 이탈 예상은 확정이 아니므로 흐리게 구분한다.
          cellStyle: (p) => {
            if (p.data?.is_exit_forecast && p.data?.forecast_trade_quantity != null) {
              return { color: "var(--down-color, #2f6fd0)", fontWeight: 600, opacity: 0.65 };
            }
            return { color: signColor(p.value as number), fontWeight: 700, opacity: 1 };
          },
        },
        {
          field: "held_value",
          headerName: "평가 금액",
          width: 120,
          type: "numericColumn",
          valueFormatter: (p) => formatAmount(p.value as number),
        },
        {
          field: "amount",
          headerName: "목표 금액",
          width: 120,
          type: "numericColumn",
          valueFormatter: (p) => formatAmount(p.value as number),
        },
      );
    }
    columns.push({
      colId: "status",
      headerName: "상태",
      flex: 1.6,
      minWidth: 260,
      valueGetter: (p) => {
        if (!p.data) return "";
        if (p.data.is_cash) return "미배분 현금";
        if (p.data.is_sell_all) return "전량 매도 (목표에 없음)";
        const parts: string[] = [];
        if (p.data.a_status) parts.push(`${labelA} ${p.data.a_status}`);
        if (p.data.b_status) parts.push(`${labelB} ${p.data.b_status}`);
        if (p.data.is_exit_forecast) parts.push("매도 예정(예상)");
        return parts.join(" · ");
      },
      cellStyle: (p) => {
        const text = String(p.value ?? "");
        if (text.includes("매도 예정"))
          return { color: "var(--down-color, #2f6fd0)", fontWeight: 600 };
        if (text.includes("진입 예정") || text.includes("매수 예정"))
          return { color: "var(--up-color, #d64545)", fontWeight: 600 };
        return null;
      },
    });
    return columns;
  }, [totalAsset, labelA, labelB, saveMemo]);

  const periodRows = useMemo<PeriodRow[]>(() => {
    if (!view || viewMode === "trades") return [];
    if (viewMode === "weekly") return toPeriodRows(view.daily, weekKeyOf, true);
    const keyLength =
      viewMode === "yearly" ? 4 : viewMode === "monthly" ? 7 : 10;
    return toPeriodRows(view.daily, (date) => date.slice(0, keyLength));
  }, [view, viewMode]);

  const tradeColumns = useMemo<ColDef<MixTradeRow>[]>(
    () => [
      {
        headerName: "전략",
        field: "strategy",
        width: 110,
        valueFormatter: (p) => slotLabel(String(p.value ?? "")),
        cellStyle: (p) => ({
          fontWeight: 700,
          color: p.value === "a" ? "#1c7ed6" : "#e8590c",
        }),
      },
      { headerName: "티커", field: "ticker", width: 96 },
      {
        headerName: "종목명",
        field: "name",
        flex: 1,
        minWidth: STOCK_NAME_COLUMN_MIN_WIDTH,
        cellRenderer: (p: { value?: string | null }) => renderStockNameCell(p.value),
      },
      { headerName: "편입일", field: "entry_date", width: 116 },
      {
        headerName: "매수가",
        field: "entry_price",
        width: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatPrice(p.value as number),
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
        headerTooltip: "보유중 행은 현재가",
        width: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatPrice(p.value as number),
      },
      {
        headerName: "수익률(%)",
        field: "return_pct",
        width: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatSignedPct(p.value as number),
        cellStyle: (p) => ({
          color: signColor(p.value as number),
          fontWeight: 700,
        }),
      },
      { headerName: "보유일", field: "days", width: 84, type: "numericColumn" },
      { headerName: "사유", field: "reason", width: 110 },
    ],
    [],
  );

  const periodColumns = useMemo<ColDef<PeriodRow>[]>(() => {
    const pct = (
      field: keyof PeriodRow,
      headerName: string,
      suffix = "%",
      headerTooltip?: string,
    ): ColDef<PeriodRow> => ({
      field,
      headerName,
      headerTooltip,
      flex: 1,
      minWidth: 108,
      type: "numericColumn",
      valueFormatter: (p) =>
        p.value == null
          ? "-"
          : `${(p.value as number) >= 0 ? "+" : ""}${(p.value as number).toFixed(2)}${suffix}`,
      cellStyle: (p) => ({
        color: signColor(p.value as number),
        fontWeight: 600,
      }),
    });
    return [
      {
        field: "period",
        headerName:
          viewMode === "yearly"
            ? "연도"
            : viewMode === "monthly"
              ? "월"
              : viewMode === "weekly"
                ? "주"
                : "일자",
        width: 148,
        valueFormatter: (p) =>
          viewMode === "weekly" || viewMode === "daily"
            ? formatDateWithWeekday(String(p.value ?? ""))
            : String(p.value ?? ""),
        cellStyle: { fontWeight: 700 },
      },
      pct(
        "strategy_pct",
        "전략통합",
        "%",
        "두 전략을 한 계좌에서 함께 굴린 결과 — 매월 첫 거래일에 배분을 되돌린다(현금 우선 이관).",
      ),
      // 각 전략을 혼자 굴렸을 때 — 합성이 단독보다 나은지 같은 줄에서 비교한다.
      // 이관이 없는 곡선이라 전략통합은 두 값의 단순 평균과 일치하지 않는다.
      pct("a_pct", labelA, "%", `${labelA} 슬리브만 혼자 굴렸을 때 — 그 전략 화면의 백테스트와 같은 값.`),
      pct("b_pct", labelB, "%", `${labelB} 슬리브만 혼자 굴렸을 때 — 그 전략 화면의 백테스트와 같은 값.`),
      pct("benchmark_pct", view?.benchmark_name ?? "벤치마크", "%", "계좌의 벤치마크."),
      pct("excess_pp", "초과", "%p", "전략통합 − 벤치마크."),
    ];
  }, [viewMode, view?.benchmark_name, labelA, labelB]);

  // 요약 바에 쓰는 계좌 표기 — 셀렉터와 같은 형식.
  const accountLabel = useMemo(() => {
    const id = positions?.account?.account_id ?? "";
    if (!id) return "";
    const found = accountOptions.find((option) => option.account_id === id);
    return found ? formatAccountLabel(found) : id;
  }, [positions?.account?.account_id, accountOptions]);

  const actions = positions?.actions ?? null;

  // 헤더 설정 — 합성 배분(%) 3칸과 슬랙 알람을 한 버튼으로 저장한다(계좌 설정에 보관).
  const [slackEnabled, setSlackEnabled] = useState(false);
  const [aStrategy, setAStrategy] = useState("");
  const [aPool, setAPool] = useState("");
  const [bStrategy, setBStrategy] = useState("");
  const [bPool, setBPool] = useState("");
  // 슬리브 표시 이름 — 비우면 전략 이름을 쓴다(「A. 모멘텀」 ↔ 「A. 상승 개별주」).
  const [aName, setAName] = useState("");
  const [bName, setBName] = useState("");
  const [aPct, setAPct] = useState("50");
  const [bPct, setBPct] = useState("50");
  const [cashPct, setCashPct] = useState("0");
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [slackTesting, setSlackTesting] = useState(false);
  useEffect(() => {
    setSlackEnabled(Boolean(selectedAccount?.mix_slack_enabled));
    setAStrategy(selectedAccount?.a_strategy ?? "");
    setAPool(selectedAccount?.a_pool ?? "");
    setBStrategy(selectedAccount?.b_strategy ?? "");
    setBPool(selectedAccount?.b_pool ?? "");
    setAName(selectedAccount?.a_name ?? "");
    setBName(selectedAccount?.b_name ?? "");
    setAPct(selectedAccount ? String(selectedAccount.mix_a_pct) : "50");
    setBPct(selectedAccount ? String(selectedAccount.mix_b_pct) : "50");
    setCashPct(selectedAccount ? String(selectedAccount.mix_cash_pct) : "0");
  }, [
    selectedAccount,
    selectedAccount?.account_id,
    selectedAccount?.mix_slack_enabled,
    selectedAccount?.mix_a_pct,
    selectedAccount?.mix_b_pct,
    selectedAccount?.mix_cash_pct,
  ]);

  // 합계 — 100 이 아니면 저장을 막는다. 모자란 만큼을 현금으로 채우면 사용자가 의도한
  // 배분이 조용히 바뀌므로 보정하지 않고 그대로 알린다.
  const weightSum = useMemo(() => {
    const parts = [aPct, bPct, cashPct].map((value) => Number(value));
    return parts.some((value) => !Number.isFinite(value)) ? null : Math.round(parts.reduce((a, b) => a + b, 0) * 100) / 100;
  }, [aPct, bPct, cashPct]);
  const weightOk = weightSum !== null && Math.abs(weightSum - 100) <= 0.01;
  const settingsDirty = Boolean(
    selectedAccount &&
      (slackEnabled !== Boolean(selectedAccount.mix_slack_enabled) ||
        aStrategy !== selectedAccount.a_strategy ||
        aPool !== selectedAccount.a_pool ||
        bStrategy !== selectedAccount.b_strategy ||
        bPool !== selectedAccount.b_pool ||
        aName !== (selectedAccount.a_name ?? "") ||
        bName !== (selectedAccount.b_name ?? "") ||
        Number(aPct) !== selectedAccount.mix_a_pct ||
        Number(bPct) !== selectedAccount.mix_b_pct ||
        Number(cashPct) !== selectedAccount.mix_cash_pct),
  );

  // 배분 입력 3칸 — 라벨·상태만 다르고 나머지는 같아 한 곳에 모아 둔다.
  const strategyLabelOf = (value: string) =>
    meta?.strategy_options.find((option) => option.value === value)?.label ?? value;
  const mixWeightFields = [
    {
      key: "a",
      label: `A ${aName.trim() || strategyLabelOf(aStrategy)}`.trim(),
      value: aPct,
      set: setAPct,
      hint: "A 슬리브에 배분할 몫(%).",
    },
    {
      key: "b",
      label: `B ${bName.trim() || strategyLabelOf(bStrategy)}`.trim(),
      value: bPct,
      set: setBPct,
      hint: "B 슬리브에 배분할 몫(%).",
    },
    {
      key: "cash",
      label: "현금",
      value: cashPct,
      set: setCashPct,
      hint: "두 전략에 주지 않고 늘 비워 두는 몫(%). 빈 슬롯에서 생기는 현금은 여기에 더해진다.",
    },
  ];

  const saveHeaderSettings = async () => {
    if (!selectedAccount || !weightOk) return;
    try {
      setSettingsSaving(true);
      const resp = await fetch("/api/account-settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          account_id: selectedAccount.account_id,
          values: {
            mix_slack_enabled: slackEnabled,
            mix_a_strategy: aStrategy || null,
            mix_a_pool: aPool || null,
            mix_a_name: aName.trim() || null,
            mix_b_strategy: bStrategy || null,
            mix_b_pool: bPool || null,
            mix_b_name: bName.trim() || null,
            mix_a_pct: Number(aPct),
            mix_b_pct: Number(bPct),
            mix_cash_pct: Number(cashPct),
          },
        }),
      });
      const data = (await resp.json()) as { error?: string; detail?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? data.detail ?? "저장에 실패했습니다.");
      // 저장된 값을 목록에 바로 반영한다 — 다시 받아오지 않아도 입력칸·변경 표시가 맞아진다.
      setAccountOptions((previous) =>
        previous.map((option) =>
          option.account_id === selectedAccount.account_id
            ? {
                ...option,
                mix_slack_enabled: slackEnabled,
                a_strategy: aStrategy,
                a_pool: aPool,
                b_strategy: bStrategy,
                b_pool: bPool,
                a_name: aName.trim(),
                b_name: bName.trim(),
                // 라벨도 즉시 반영 — 이름을 지우면 전략 이름으로 돌아간다.
                a_strategy_label: aName.trim() || strategyLabelOf(aStrategy),
                b_strategy_label: bName.trim() || strategyLabelOf(bStrategy),
                mix_ready: Boolean(aStrategy && aPool && bStrategy && bPool),
                mix_a_pct: Number(aPct),
                mix_b_pct: Number(bPct),
                mix_cash_pct: Number(cashPct),
              }
            : option,
        ),
      );
      toast.success("합성 설정 저장 완료");
      // 배분이 바뀌면 목표 비중·오늘의 액션이 달라진다 — 운용 현황을 다시 계산한다.
      setPositionsReloadKey((k) => k + 1);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "합성 설정 저장에 실패했습니다.");
    } finally {
      setSettingsSaving(false);
    }
  };

  const sendSlackTest = async () => {
    if (!selectedAccount) return;
    try {
      setSlackTesting(true);
      const resp = await fetch("/api/strategy-mix/slack-test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account_id: selectedAccount.account_id }),
      });
      const data = (await resp.json()) as { sent?: boolean; items?: number; error?: string };
      if (!resp.ok || data.error || !data.sent) throw new Error(data.error ?? "발송에 실패했습니다.");
      toast.success(`슬랙 발송 완료 (지시 ${data.items ?? 0}건)`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "슬랙 테스트 발송에 실패했습니다.");
    } finally {
      setSlackTesting(false);
    }
  };

  // 오늘의 액션 — 조립은 서버(`_build_action_groups`)가 한다. 슬랙 알람과 같은 결과를
  // 쓰기 위한 단일 소스라, 화면은 받은 그대로 그리기만 한다.
  const actionGroups = actions?.groups ?? [];
  const nextWeekPreview = actions?.next_week_preview ?? null;

  const hasActions =
    actionGroups.length > 0 || Boolean(actions?.sleeve_rebalance_today);

  return (
    <PageFrame title="합성 전략" fullWidth>
      <div className="appPageStack">
        <section className="appSection">
          <div className="card appCard">
            <div className="card-body appCardBodyTight">
              {/* 메인 헤더 — 셀렉터·모드 전환 같은 주 제어. */}
              <div className="appMainHeader">
                <div className="appMainHeaderLeft">
                  {/* 계좌를 고르고, 그 계좌의 슬리브별 종목풀을 옆에 보여준다.
                      연결은 계좌 설정에서 만든다 — 여기서는 고르기만 한다. */}
                  <label className="appLabeledField" style={{ marginBottom: 0 }}>
                    <span className="appLabeledFieldLabel">계좌</span>
                    <select
                      className="form-select form-select-sm"
                      style={{ width: "auto" }}
                      value={accountId}
                      disabled={loading || positionsLoading || accountOptions.length === 0}
                      onChange={(event) => {
                        const next = event.target.value;
                        setAccountId(next);
                        writeRememberedMomentumEtfAccountId(next);
                        setPositions(null);
                        setView(null);
                      }}
                    >
                      {accountOptions.map((option) => (
                        <option key={option.account_id} value={option.account_id}>
                          {formatAccountLabel(option)}
                        </option>
                      ))}
                    </select>
                  </label>
                  {selectedAccount
                    ? (["a", "b"] as const).map((slot) => {
                        const strategy = slot === "a" ? aStrategy : bStrategy;
                        const pool = slot === "a" ? aPool : bPool;
                        const setStrategy = slot === "a" ? setAStrategy : setBStrategy;
                        const setPool = slot === "a" ? setAPool : setBPool;
                        const name = slot === "a" ? aName : bName;
                        const setName = slot === "a" ? setAName : setBName;
                        // 계좌와 국가가 같은 풀만 — 거래 달력·통화가 갈리면 합성이 성립하지 않는다.
                        const pools = (meta?.pool_options ?? []).filter(
                          (option) => (option.country_code ?? "") === selectedAccount.country_code,
                        );
                        return (
                          <label key={slot} className="appLabeledField" style={{ marginBottom: 0 }}>
                            <span className="appLabeledFieldLabel">{slot.toUpperCase()}</span>
                            <span style={{ display: "inline-flex", gap: 4 }}>
                              <select
                                className="form-select form-select-sm"
                                style={{ width: "auto" }}
                                value={strategy}
                                disabled={settingsSaving}
                                onChange={(event) => setStrategy(event.target.value)}
                              >
                                <option value="">전략</option>
                                {(meta?.strategy_options ?? []).map((option) => (
                                  <option key={option.value} value={option.value}>
                                    {option.label}
                                  </option>
                                ))}
                              </select>
                              <select
                                className="form-select form-select-sm"
                                style={{ width: "auto" }}
                                value={pool}
                                disabled={settingsSaving}
                                onChange={(event) => setPool(event.target.value)}
                              >
                                <option value="">종목풀</option>
                                {pools.map((option) => (
                                  <option key={option.ticker_type} value={option.ticker_type}>
                                    {formatPoolLabel(option)}
                                  </option>
                                ))}
                              </select>
                              {/* 표시 이름 — 비우면 전략 이름을 쓴다. 같은 전략을 두 슬롯에
                                  올릴 수 있어 이름 없이는 화면에서 둘을 구분할 수 없다. */}
                              <input
                                className="form-control form-control-sm"
                                style={{ width: 120 }}
                                type="text"
                                maxLength={20}
                                value={name}
                                disabled={settingsSaving}
                                placeholder={strategyLabelOf(strategy) || "이름"}
                                title="비우면 전략 이름으로 보입니다."
                                onChange={(event) => setName(event.target.value)}
                              />
                            </span>
                          </label>
                        );
                      })
                    : (
                      <span style={hintStyle}>계좌 설정에서 «합성» 을 켠 계좌가 여기 보입니다.</span>
                    )}
                  {selectedAccount ? (
                    <>
                      {mixWeightFields.map((field) => (
                        <label key={field.key} className="appLabeledField" style={{ marginBottom: 0 }}>
                          <span className="appLabeledFieldLabel">{field.label}</span>
                          <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                            <input
                              className="form-control form-control-sm"
                              style={{ width: 72, textAlign: "right" }}
                              type="number"
                              min={0}
                              max={100}
                              step={1}
                              value={field.value}
                              disabled={settingsSaving}
                              onChange={(event) => field.set(event.target.value)}
                              title={field.hint}
                            />
                            <span style={hintStyle}>%</span>
                          </span>
                        </label>
                      ))}
                      <span style={{ ...hintStyle, color: weightOk ? "var(--text-muted)" : "#d62828", fontWeight: weightOk ? 400 : 700 }}>
                        합계 {weightSum === null ? "-" : `${weightSum}%`}
                        {weightOk ? " ✓" : " (100% 필요)"}
                      </span>
                      <label className="appLabeledField" style={{ marginBottom: 0 }}>
                        <span className="appLabeledFieldLabel">슬랙 알람</span>
                        <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                          <div className="form-check form-switch" style={{ marginBottom: 0 }}>
                            <input
                              className="form-check-input"
                              type="checkbox"
                              role="switch"
                              checked={slackEnabled}
                              disabled={settingsSaving}
                              onChange={(e) => setSlackEnabled(e.target.checked)}
                              title="오늘의 액션에 새 지시나 수량 증가가 생기면 슬랙으로 보낸다 (장중 10분 간격 감시). 저장 버튼을 눌러야 반영된다."
                            />
                          </div>
                          <span style={hintStyle}>{slackEnabled ? "켜짐" : "꺼짐"}</span>
                          <button
                            type="button"
                            className="btn btn-sm btn-outline-secondary"
                            disabled={slackTesting}
                            onClick={() => void sendSlackTest()}
                          >
                            {slackTesting ? "발송 중…" : "지금 발송(테스트)"}
                          </button>
                        </span>
                      </label>
                    </>
                  ) : null}
                </div>
                {/* 저장은 세 전략 화면 모두 메인 헤더 오른쪽 끝에 둔다. */}
                {selectedAccount ? (
                  <div className="appMainHeaderRight">
                    <UnsavedChangesBadge show={settingsDirty} />
                    <button
                      type="button"
                      className="btn btn-success btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                      disabled={settingsSaving || !weightOk || !settingsDirty}
                      onClick={() => void saveHeaderSettings()}
                      title={weightOk ? "배분과 슬랙 알람을 함께 저장한다." : "배분 합계가 100%가 아니면 저장할 수 없다."}
                    >
                      <IconCheck size={16} />
                      <span>{settingsSaving ? "저장 중…" : "저장"}</span>
                    </button>
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </section>

        <section className="appSection">
          <div className="card appCard">
            <div className="card-header appCardHeader">
              <span style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>
                운용 현황
              </span>
              {/* 과거 날짜 셀렉트는 두지 않는다 — 계좌 수량이 현재 기준이라 과거 목표와
                  섞이면 액션·수량이 틀린 값이 된다 (판정 기록은 각 전략 화면에서 본다). */}
              <span style={{ ...hintStyle, fontWeight: 500 }}>
                {positions ? formatDateWithWeekday(positions.as_of) : ""}
              </span>
            </div>
            <div className="card-body appCardBodyTight">
              <StrategyNotes items={CURRENT_NOTES} />
              {positionsError ? (
                <div className="alert alert-danger" style={{ marginBottom: 0 }}>
                  {positionsError}
                </div>
              ) : positionsLoading || !positions ? (
                <AppLoadingProgress
                  title="운용 현황 계산 중..."
                  progress={positionsProgress}
                />
              ) : (
                <div
                  style={{ display: "flex", flexDirection: "column", gap: 14 }}
                >
                  {/* ① 요약 바 — 오늘 주식·현금을 얼마씩 둬야 하는지. */}
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 18,
                      flexWrap: "wrap",
                    }}
                  >
                    <span style={{ fontSize: "var(--fs-lg)", fontWeight: 800 }}>
                      목표 주식 {positions.summary.stock_pct.toFixed(1)}% · 현금{" "}
                      {positions.summary.cash_pct.toFixed(1)}%
                      {positions.account && totalAsset
                        ? ` (현재 ${((positions.account.stock_value / totalAsset) * 100).toFixed(1)}% · ${(
                          (positions.account.cash_balance / totalAsset) *
                          100
                        ).toFixed(1)}%)`
                        : ""}
                    </span>
                    {/* 적용 계좌 — 목표 금액의 기준이 되는 실제 잔고. */}
                    {positions.account ? (
                      <span
                        style={{ fontSize: "var(--fs-base)", fontWeight: 700 }}
                      >
                        {accountLabel} 총자산{" "}
                        {formatAmount(positions.account.total_assets)}
                        <span
                          style={{
                            ...hintStyle,
                            marginLeft: 8,
                            fontWeight: 500,
                          }}
                        >
                          (주식 {formatAmount(positions.account.stock_value)} ·
                          현금 {formatAmount(positions.account.cash_balance)})
                        </span>
                      </span>
                    ) : (
                      <span style={hintStyle}>
                        적용 계좌 없음 — 계좌를 저장하면 목표 금액·주수가
                        나옵니다
                      </span>
                    )}
                    <span style={hintStyle}>
                      {labelA} 목표 {positions.summary.a.slots_used}/
                      {positions.summary.a.top_n} (보유{" "}
                      {positions.summary.a.held_count}) · {labelB} 목표{" "}
                      {positions.summary.b.slots_used}/
                      {positions.summary.b.top_n} (보유{" "}
                      {positions.summary.b.held_count})
                    </span>
                    {/* 미체결 교체 안내 — 교체가 있는 슬리브마다 한 줄. 계좌에 둘일 수 있다. */}
                    {SLOT_KEYS.map((slot) => {
                      const rebalance = actions?.slots?.[slot]?.rebalance;
                      if (!rebalance || rebalance.is_filled || !rebalance.fill_date) return null;
                      return (
                        <span key={slot} style={hintStyle}>
                          {slotLabel(slot)} {rebalance.portfolio_week} 포트폴리오 · 체결{" "}
                          {rebalance.fill_date} (판정 {rebalance.signal_date})
                        </span>
                      );
                    })}
                    {actions?.sleeve_rebalance_today ? (
                      <span
                        style={{
                          fontSize: "var(--fs-sm)",
                          fontWeight: 700,
                          color: "#d9480f",
                          background: "rgba(247, 103, 7, 0.12)",
                          padding: "4px 10px",
                          borderRadius: 999,
                        }}
                      >
                        오늘은 매월 첫 거래일 — 슬리브 배분 복구는 현금으로 이관
                        (주식 매도 지시는 현금이 모자랄 때만 나옵니다)
                      </span>
                    ) : null}
                  </div>

                  {/* ② 보유 목록 — 현금까지 한 표로, 비중 합 100%. */}
                  <AppAgGrid<PositionRow>
                    rowData={positionRows}
                    columnDefs={positionColumns}
                    theme={gridTheme}
                    minHeight="auto"
                    getRowId={(p) => p.data.ticker}
                    // 아직 체결 전인 행(진입 예정)과 곧 나갈 행(매도 예정)은 확정 보유와
                    // 구분되게 회색으로 눌러 둔다 — 추세 이탈 행과 같은 공용 클래스.
                    getRowClass={(params) => {
                      if (params.data?.is_sell_all) return "appTrendBrokenRow";
                      const status = `${params.data?.a_status ?? ""} ${params.data?.b_status ?? ""}`;
                      return status.includes("예정") ? "appTrendBrokenRow" : "";
                    }}
                    gridOptions={{
                      domLayout: "autoHeight",
                      suppressMovableColumns: true,
                    }}
                  />

                  {/* ③ 배분 — 합계 · 모멘텀 · 신고가 세 줄. 각 줄에 배정 금액과 주식·현금. */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    {(() => {
                      const slotA = positions.summary.a;
                      const slotB = positions.summary.b;
                      const amount = (pct: number) => (totalAsset == null ? null : (totalAsset * pct) / 100);
                      // 슬리브 몫은 월초 배분에서 각자 흘러간 비율(alloc_pct)이다.
                      // 그 안에서 채운 슬롯이 주식·빈 슬롯이 현금이다.
                      const rows = [
                        ...SLOT_KEYS.map((slot) => {
                          const summary = slot === "a" ? slotA : slotB;
                          return {
                            key: slot,
                            label: slotLabel(slot),
                            allocPct: summary.alloc_pct,
                            stockPct: summary.alloc_pct - summary.cash_pct,
                            cashPct: summary.cash_pct,
                            slots: `${summary.slots_used}/${summary.top_n}`,
                          };
                        }),
                      ];
                      return rows.map((row) => (
                        <div key={row.key} style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                          <strong style={{ minWidth: 52 }}>{row.label}</strong>
                          <span style={hintStyle}>
                            {totalAsset != null ? `${formatAmount(amount(row.allocPct))} · ` : ""}
                            주식 {row.stockPct.toFixed(1)}%
                            {totalAsset != null ? ` ${formatAmount(amount(row.stockPct))}` : ""} · 현금{" "}
                            {row.cashPct.toFixed(1)}%
                            {totalAsset != null ? ` ${formatAmount(amount(row.cashPct))}` : ""} · 슬롯 {row.slots}
                          </span>
                        </div>
                      ));
                    })()}
                  </div>

                  {/* ④ 오늘의 액션 — 체결 시점별 묶음, 각 묶음은 매도 → 매수 순서. */}
                  <div>
                    <div style={{ fontWeight: 700, marginBottom: 6 }}>
                      오늘의 액션
                      <span style={{ ...hintStyle, marginLeft: 8, fontWeight: 500 }}>
                        목표 비중과 슬롯 크기의 15%p(최소 0.5%p) 이상 차이만 지시로 표시 — 가격
                        변동(드리프트)으로는 지시가 없고, 큰 단위 입출금·교체·진입·이탈·월초 이관 때
                        나옵니다 · 모멘텀 교체 확정분은 교체일 그룹
                      </span>
                    </div>
                    {!hasActions ? (
                      <div style={hintStyle}>
                        오늘은 할 일이 없습니다 — 보유 목록을 그대로 유지하세요.
                      </div>
                    ) : (
                      <div
                        style={{
                          display: "flex",
                          flexDirection: "column",
                          gap: 10,
                        }}
                      >
                        {actionGroups.map((group, groupIndex) => (
                          <div key={group.key}>
                            <div style={{ fontWeight: 700, marginBottom: 4 }}>
                              {groupIndex + 1}. {group.title}
                              <span
                                style={{
                                  ...hintStyle,
                                  marginLeft: 8,
                                  fontWeight: 500,
                                }}
                              >
                                매도{" "}
                                {
                                  group.items.filter(
                                    (item) => item.side === "sell",
                                  ).length
                                }
                                건 · 매수{" "}
                                {
                                  group.items.filter(
                                    (item) => item.side === "buy",
                                  ).length
                                }
                                건
                              </span>
                            </div>
                            <ul
                              style={{
                                margin: 0,
                                paddingLeft: 18,
                                display: "flex",
                                flexDirection: "column",
                                gap: 4,
                              }}
                            >
                              {group.items.map((item) => (
                                <li key={item.key}>
                                  <strong
                                    style={{
                                      color:
                                        item.side === "sell"
                                          ? "var(--down-color, #2f6fd0)"
                                          : "var(--up-color, #d64545)",
                                    }}
                                  >
                                    {item.title}
                                  </strong>{" "}
                                  — {item.text}
                                </li>
                              ))}
                            </ul>
                          </div>
                        ))}
                        {actions?.sleeve_rebalance_today ? (
                          <div>
                            <strong>슬리브 리밸런싱</strong> — 매월 첫 거래일입니다.
                            {positions?.summary?.base_weights
                              ? ` ${labelA} ${positions.summary.base_weights.a_pct}% · ${labelB} ${positions.summary.base_weights.b_pct}% · 현금 ${positions.summary.base_weights.cash_pct}% 로 다시 맞추세요.`
                              : " 저장된 배분으로 다시 맞추세요."}
                          </div>
                        ) : null}
                      </div>
                    )}
                  </div>

                  {/* ⑤ 다음주 교체 가정 — 지금 순위 그대로 확정될 때의 예상 (주기적 교체가 있는 슬리브만).
                      기본 접힘 — 금요일쯤 궁금할 때 열어 보는 참고 정보라 평소에는 줄만 차지한다. */}
                  {nextWeekPreview ? (
                    <div>
                      <button
                        type="button"
                        onClick={() => setNextWeekOpen((value) => !value)}
                        style={{
                          fontWeight: 700,
                          margin: 0,
                          padding: 0,
                          background: "none",
                          border: "none",
                          cursor: "pointer",
                          color: "inherit",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: 6,
                        }}
                      >
                        <span>{nextWeekOpen ? "▾" : "▸"}</span>
                        다음주 교체 가정
                        <span style={{ ...hintStyle, fontWeight: 500 }}>
                          매도{" "}
                          {nextWeekPreview.groups.reduce(
                            (n, g) =>
                              n + g.items.filter((i) => i.side === "sell").length,
                            0,
                          )}
                          건 · 매수{" "}
                          {nextWeekPreview.groups.reduce(
                            (n, g) =>
                              n + g.items.filter((i) => i.side === "buy").length,
                            0,
                          )}
                          건 (예상)
                        </span>
                      </button>
                      {nextWeekOpen ? (
                        <div style={{ marginTop: 6 }}>
                          <div style={{ ...hintStyle, marginBottom: 6 }}>
                            지금 순위가 그대로 다음주 교체로 확정된다고 가정하고,
                            오늘의 액션과 같은 방식으로 조립한 월요일 예상입니다 —
                            종목 교체가 없으면 오늘의 액션과 똑같습니다. 판정은
                            이번주 마지막 거래일 종가로 확정되므로 그때까지 바뀔 수
                            있고, 수량은 현재가·현재 총자산 기준 추정치입니다.
                          </div>
                          {nextWeekPreview.groups.length === 0 ? (
                            <div style={hintStyle}>
                              예상되는 매매가 없습니다 — 보유 목록이 그대로
                              유지됩니다.
                            </div>
                          ) : (
                            nextWeekPreview.groups.map((group) => (
                              <div key={`nw-${group.key}`}>
                                <div style={{ fontWeight: 700, marginBottom: 4 }}>
                                  {group.title} (예상)
                                </div>
                                <ul
                                  style={{
                                    margin: 0,
                                    paddingLeft: 18,
                                    display: "flex",
                                    flexDirection: "column",
                                    gap: 4,
                                  }}
                                >
                                  {group.items.map((item) => (
                                    <li key={`nw-${item.key}`}>
                                      <strong
                                        style={{
                                          color:
                                            item.side === "sell"
                                              ? "var(--down-color, #2f6fd0)"
                                              : "var(--up-color, #d64545)",
                                        }}
                                      >
                                        {item.title}
                                      </strong>{" "}
                                      — {item.text}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            ))
                          )}
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* 백테스트 — 운용 현황 아래에 나란히 둔다. */}
        <section className="appSection">
          <div className="card appCard">
            <div className="card-header appCardHeader">
              <span style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>
                백테스트
              </span>
              <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <MonthsSelect value={months} options={monthOptions} disabled={loading} onChange={setMonths} />
                <button
                  className="btn btn-sm btn-dark"
                  type="button"
                  disabled={loading || !accountId}
                  onClick={() => void runBacktest()}
                >
                  {loading ? "실행 중…" : "실행"}
                </button>
              </span>
            </div>
            <div className="card-body appCardBodyTight">
              {error ? (
                <div className="alert alert-danger" style={{ marginBottom: 0 }}>
                  {error}
                </div>
              ) : loading ? (
                <AppLoadingProgress
                  title="백테스트 실행 중..."
                  progress={backtestProgress}
                />
              ) : view ? (
                <>
                  <BacktestSummary
                    startDate={view.start_date}
                    endDate={view.end_date}
                    strategy={{
                      label: "전략",
                      totalPct: view.strategy_total_pct,
                      cagrPct: view.strategy_cagr_pct,
                      mddPct: view.strategy_mdd_pct,
                      sortino: view.strategy_sortino,
                    }}
                    benchmark={{
                      label: view.benchmark_ticker
                        ? `${view.benchmark_name}(${view.benchmark_ticker})`
                        : view.benchmark_name,
                      totalPct: view.benchmark_total_pct,
                      cagrPct: view.benchmark_cagr_pct,
                      mddPct: view.benchmark_mdd_pct,
                      sortino: view.benchmark_sortino,
                    }}
                  />
                  <BacktestTradeStats stats={view} style={{ marginBottom: 12 }} />
                  <NavTabs
                    items={VIEW_MODES}
                    value={viewMode}
                    onChange={setViewMode}
                    label="합성 백테스트 보기 단위"
                    style={{ marginBottom: 10 }}
                  />
                  {viewMode === "trades" ? (
                    <AppAgGrid<MixTradeRow>
                      rowData={view.trades}
                      columnDefs={tradeColumns}
                      theme={gridTheme}
                      minHeight="auto"
                      gridOptions={{
                        domLayout: "autoHeight",
                        suppressMovableColumns: true,
                      }}
                      getRowClass={(p) =>
                        p.data?.exit_date ? "" : "momentumPendingRow"
                      }
                    />
                  ) : (
                    <AppAgGrid<PeriodRow>
                      rowData={periodRows}
                      columnDefs={periodColumns}
                      theme={gridTheme}
                      minHeight="auto"
                      getRowId={(p) => p.data.period}
                      gridOptions={{
                        domLayout: "autoHeight",
                        suppressMovableColumns: true,
                      }}
                    />
                  )}
                </>
              ) : (
                <div
                  style={{
                    ...hintStyle,
                    padding: "32px 0",
                    textAlign: "center",
                  }}
                >
                  실행을 누르면 결과가 표시됩니다.
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </PageFrame>
  );
}
