"use client";

import type { ColDef } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useState } from "react";

import AccountSelect, { formatAccountLabel, type AccountOptionBase } from "../components/AccountSelect";
import { AppAgGrid } from "../components/AppAgGrid";
import { AppLoadingProgress, startProgressRamp, type LoadingProgress } from "../components/AppLoadingProgress";
import { BacktestSummary } from "../components/BacktestSummary";
import { NavTabs } from "../components/NavTabs";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";
import { formatDateWithWeekday, formatKstDateTime } from "@/lib/datetime";
import { STOCK_NAME_COLUMN_MIN_WIDTH, formatSignedPct, signColor } from "@/lib/grid-cells";
import { formatPoolLabel, type PoolLabelSource } from "@/lib/pool-label";

const gridTheme = createAppGridTheme();
const hintStyle: React.CSSProperties = { color: "var(--text-muted)", fontSize: "var(--fs-sm)" };

type AccountOption = AccountOptionBase;

type Meta = {
  pool: string;
  pool_options: PoolLabelSource[];
  month_options: number[];
  accounts: AccountOption[];
  /** 풀별 저장 설정 — 풀을 바꾸면 그 풀의 계좌로 전환한다. */
  settings_by_pool: Record<string, { account_id: string | null }>;
};

type View = {
  computed_at: string;
  pool: string;
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
  daily: { date: string; strategy_pct: number; benchmark_pct: number }[];
  /** 체결 목록 — 두 전략 합본. exit_date 가 없으면 보유중. */
  trades: MixTradeRow[];
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
  sources: ("sm" | "nh")[];
  weight_pct: number;
  price: number | null;
  change_pct: number | null;
  sm_status: string | null;
  nh_status: string | null;
  /** 적용 계좌가 있을 때만 온다 — 계좌 총자산 기준 목표 금액·주수와 현재 보유. */
  target_amount?: number | null;
  target_quantity?: number | null;
  held_quantity?: number | null;
  held_value?: number | null;
  current_weight_pct?: number;
  trade_quantity?: number | null;
  /** 목표 포트폴리오에 없는 보유 종목 — 목표 비중 0% 행으로 표 하단에 온다. */
  is_sell_all?: boolean;
};

/** 적용 계좌의 실제 상태 — 계좌가 연결된 풀에서만 온다. */
type AccountState = {
  account_id: string;
  cash_balance: number;
  stock_value: number;
  total_assets: number;
  /** 목표에 없는 보유 종목 = 전량 매도 대상. */
  sell_all: { ticker: string; name: string; quantity: number; value: number | null }[];
};

type Positions = {
  computed_at: string;
  pool: string;
  as_of: string;
  live: boolean;
  /** 적용 계좌를 저장한 풀에서만 온다. 없으면 비중만 보여준다. */
  account: AccountState | null;
  /** 과거 날짜 셀렉트용 — 신고가 화면과 같은 날짜 목록. */
  available_dates: string[];
  summary: {
    stock_pct: number;
    cash_pct: number;
    /** slots_used = 목표가 찬 슬롯, held_count = 지금 실제로 들고 있는 종목 수. */
    sm: { slots_used: number; held_count: number; top_n: number; cash_pct: number };
    nh: { slots_used: number; held_count: number; top_n: number; cash_pct: number };
  };
  holdings: Holding[];
  actions: {
    /** 모멘텀 주중 매도 예정 — 보유 자격(장단기 이평선) 상실, 다음 거래일 시가 매도. */
    sm_sells: { ticker: string; name: string; reason: string }[];
    /** 진입 예정 — 빈 슬롯을 채울 종목이라 목표 보유와 같은 비중·매매 지시를 갖는다. */
    nh_entries: {
      ticker: string;
      name: string;
      price: number | null;
      change_pct: number | null;
      value_mult: number | null;
      weight_pct: number;
      target_amount?: number | null;
      target_quantity?: number | null;
      held_quantity?: number | null;
      trade_quantity?: number | null;
    }[];
    nh_sells: { ticker: string; name: string; return_pct: number | null; reason: string }[];
    /** 확정된 다음 교체 — 판정은 끝났고 체결만 남았다. */
    sm_rebalance: {
      is_filled: boolean;
      fill_date: string | null;
      signal_date: string | null;
      portfolio_week: string | null;
      buys: { ticker: string; name: string; price: number | null }[];
      sells: { ticker: string; name: string }[];
    };
    sleeve_rebalance_today: boolean;
  };
};

const VIEW_MODES = [
  { key: "yearly", label: "연간" },
  { key: "monthly", label: "월간" },
  { key: "weekly", label: "주간" },
  { key: "daily", label: "일간" },
  { key: "trades", label: "체결" },
] as const;
type ViewMode = (typeof VIEW_MODES)[number]["key"];

type PeriodRow = { period: string; strategy_pct: number; benchmark_pct: number; excess_pp: number };

/** 그 날짜가 속한 주의 월요일 — 주간 묶음 키. 로컬 기준으로 조립한다(UTC 파싱은 하루 밀린다). */
function weekKeyOf(date: string): string {
  const parsed = new Date(`${date}T00:00:00`);
  parsed.setDate(parsed.getDate() - ((parsed.getDay() + 6) % 7));
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${parsed.getFullYear()}-${pad(parsed.getMonth() + 1)}-${pad(parsed.getDate())}`;
}

/** 누적(%) 시계열을 기간별 수익률로 자른다 — 신고가 화면과 같은 계산.
 *  주간은 묶음 키(월요일)와 표시 라벨(그 주 마지막 거래일)이 달라 따로 담는다. */
function toPeriodRows(daily: View["daily"], keyOf: (date: string) => string, labelByLastDate = false): PeriodRow[] {
  if (daily.length === 0) return [];
  const lastByPeriod = new Map<string, { strategy: number; benchmark: number; lastDate: string }>();
  const order: string[] = [];
  for (const point of daily) {
    const key = keyOf(point.date);
    if (!lastByPeriod.has(key)) order.push(key);
    lastByPeriod.set(key, { strategy: point.strategy_pct, benchmark: point.benchmark_pct, lastDate: point.date });
  }
  // 첫 구간의 기준은 시작 시점(누적 0%)이다.
  let prev = { strategy: 0, benchmark: 0 };
  const rows: PeriodRow[] = [];
  for (const key of order) {
    const current = lastByPeriod.get(key)!;
    const step = (now: number, before: number) => ((1 + now / 100) / (1 + before / 100) - 1) * 100;
    const strategy = step(current.strategy, prev.strategy);
    const benchmark = step(current.benchmark, prev.benchmark);
    rows.push({
      period: labelByLastDate ? current.lastDate : key,
      strategy_pct: strategy,
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

const SOURCE_LABEL: Record<string, string> = { sm: "모멘텀", nh: "신고가" };

/** 보유 표 행 — 현금 행도 같은 표에 넣는다 (비중 합이 100%임을 한눈에 보이게). */
type PositionRow = Holding & { is_cash?: boolean; amount: number | null; shares: number | null };

/** 오늘의 액션 한 줄. 같은 체결 시점끼리 묶고 묶음 안에서는 매도 → 매수 순서다. */
type ActionItem = { key: string; side: "sell" | "buy"; title: string; text: string };
type ActionGroup = { key: string; title: string; items: ActionItem[] };

/** 합성전략 — SM·신고가를 50:50으로 함께 운용하는 화면.
 *  현재 상태 탭은 오늘 보유해야 할 종목과 현금 비중·오늘의 액션을,
 *  백테스트 탭은 매월 50:50 리밸런싱 합성 성과를 보여준다. 설정은 각 전략 화면의 저장값을 그대로 쓴다. */
export function StrategyMixClient() {
  const toast = useToast();
  const [pool, setPool] = useState<string>("");
  const [poolOptions, setPoolOptions] = useState<PoolLabelSource[]>([]);
  const [months, setMonths] = useState<number>(12);
  const [monthOptions, setMonthOptions] = useState<number[]>([]);
  const [accountOptions, setAccountOptions] = useState<AccountOption[]>([]);
  // 풀별 저장 설정 — 저장 전 초안과 비교해 저장 버튼 활성화를 정한다.
  const [savedByPool, setSavedByPool] = useState<Record<string, { account_id: string | null }>>({});
  const [accountId, setAccountId] = useState<string>("");
  const [saving, setSaving] = useState(false);

  // 현재 상태 탭.
  const [positions, setPositions] = useState<Positions | null>(null);
  const [positionsLoading, setPositionsLoading] = useState(false);
  const [positionsError, setPositionsError] = useState<string | null>(null);
  const [positionsProgress, setPositionsProgress] = useState<LoadingProgress | null>(null);
  /** 과거 날짜 조회 — 빈 값이면 오늘. */
  const [asOf, setAsOf] = useState<string>("");
  /** 계좌를 저장하면 목표 금액이 달라지므로 현재 상태를 다시 계산한다. */
  const [positionsReloadKey, setPositionsReloadKey] = useState(0);

  // 백테스트 탭.
  const [view, setView] = useState<View | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("monthly");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backtestProgress, setBacktestProgress] = useState<LoadingProgress | null>(null);

  // 진입 시에는 풀 목록만 받는다 — 계산은 탭·버튼이 시작한다.
  useEffect(() => {
    let alive = true;
    void (async () => {
      try {
        const response = await fetch("/api/strategy-mix/meta", { cache: "no-store" });
        const payload = (await response.json()) as Meta & { error?: string };
        if (!response.ok || payload.error) throw new Error(payload.error ?? "종목풀 목록을 불러오지 못했습니다.");
        if (!alive) return;
        setPoolOptions(payload.pool_options);
        setPool(payload.pool);
        setMonthOptions(payload.month_options);
        setAccountOptions(payload.accounts ?? []);
        setSavedByPool(payload.settings_by_pool ?? {});
        setAccountId((payload.settings_by_pool ?? {})[payload.pool]?.account_id ?? "");
      } catch (metaError) {
        if (alive) setError(metaError instanceof Error ? metaError.message : "종목풀 목록을 불러오지 못했습니다.");
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // 현재 상태 — 풀이 정해지면 자동 계산한다 (매일 보는 운영 화면이라 버튼을 두지 않는다).
  useEffect(() => {
    if (!pool) return;
    let alive = true;
    setPositionsLoading(true);
    setPositionsError(null);
    setPositionsProgress({ percent: 10, message: "두 전략의 현재 상태를 계산하는 중" });
    const stopRamp = startProgressRamp(setPositionsProgress);
    void (async () => {
      try {
        const params = new URLSearchParams({ pool });
        if (asOf) params.set("as_of", asOf);
        const response = await fetch(`/api/strategy-mix/positions?${params.toString()}`, {
          cache: "no-store",
        });
        const payload = (await response.json()) as Positions & { error?: string };
        if (!response.ok || payload.error) throw new Error(payload.error ?? "합성 운영 상태를 불러오지 못했습니다.");
        if (alive) setPositions(payload);
      } catch (positionsFetchError) {
        if (alive)
          setPositionsError(
            positionsFetchError instanceof Error ? positionsFetchError.message : "합성 운영 상태를 불러오지 못했습니다.",
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
  }, [pool, asOf, positionsReloadKey]);

  // 저장 전 초안과 저장분이 다른지 — 다르면 저장 버튼이 열린다.
  const isDirty = useMemo(() => (savedByPool[pool]?.account_id ?? "") !== accountId, [savedByPool, pool, accountId]);

  const saveSettings = useCallback(async () => {
    if (!pool) return;
    setSaving(true);
    try {
      const response = await fetch("/api/strategy-mix/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pool, account_id: accountId || null }),
      });
      const payload = (await response.json()) as { error?: string };
      if (!response.ok || payload.error) throw new Error(payload.error ?? "설정을 저장하지 못했습니다.");
      setSavedByPool((prev) => ({ ...prev, [pool]: { account_id: accountId || null } }));
      setPositionsReloadKey((key) => key + 1);
      toast.success("설정을 저장했습니다.");
    } catch (saveError) {
      toast.error(saveError instanceof Error ? saveError.message : "설정을 저장하지 못했습니다.");
    } finally {
      setSaving(false);
    }
  }, [pool, accountId, toast]);

  const runBacktest = useCallback(async () => {
    if (!pool) return;
    setLoading(true);
    setError(null);
    setBacktestProgress({ percent: 10, message: "두 전략의 백테스트를 계산하는 중" });
    const stopRamp = startProgressRamp(setBacktestProgress);
    try {
      const response = await fetch(`/api/strategy-mix?pool=${encodeURIComponent(pool)}&months=${months}`, {
        cache: "no-store",
      });
      const payload = (await response.json()) as View & { error?: string };
      if (!response.ok || payload.error) throw new Error(payload.error ?? "합성 백테스트를 불러오지 못했습니다.");
      setBacktestProgress({ percent: 100, message: "결과 반영 중" });
      setView(payload);
    } catch (runError) {
      const message = runError instanceof Error ? runError.message : "합성 백테스트를 불러오지 못했습니다.";
      setError(message);
      toast.error(message);
    } finally {
      stopRamp();
      setLoading(false);
      setBacktestProgress(null);
    }
  }, [pool, months, toast]);

  // 목표 금액의 기준 = 적용 계좌의 총자산(주식 평가액 + 현금). 계좌가 없으면 비중만 보여준다.
  const totalAsset = positions?.account?.total_assets ?? null;

  const positionRows = useMemo<PositionRow[]>(() => {
    if (!positions) return [];
    // 현금은 빈 슬롯마다 나누지 않고 한 행으로 맨 위에 둔다 — 계좌 현금은 한 덩어리다.
    const cashRow: PositionRow = {
      ticker: "__cash__",
      name: "현금",
      sources: [],
      weight_pct: positions.summary.cash_pct,
      price: null,
      change_pct: null,
      sm_status: null,
      nh_status: null,
      is_cash: true,
      current_weight_pct:
        totalAsset && positions.account ? (positions.account.cash_balance / totalAsset) * 100 : undefined,
      held_value: positions.account?.cash_balance ?? null,
      amount: totalAsset == null ? null : (totalAsset * positions.summary.cash_pct) / 100,
      shares: null,
    };
    // 목표 종목 행은 백엔드가 종목 단위로 합쳐 계산한 값을 그대로 쓴다.
    return [
      cashRow,
      ...positions.holdings.map((holding) => ({
        ...holding,
        amount: holding.target_amount ?? null,
        shares: holding.target_quantity ?? null,
      })),
    ];
  }, [positions, totalAsset]);

  const positionColumns = useMemo<ColDef<PositionRow>[]>(() => {
    const columns: ColDef<PositionRow>[] = [
      {
        field: "ticker",
        headerName: "티커",
        width: 108,
        cellRenderer: (p: { value?: string; data?: PositionRow }) =>
          p.data?.is_cash ? <span>-</span> : <TickerDetailLink ticker={p.value} />,
      },
      {
        field: "name",
        headerName: "종목명",
        flex: 1.4,
        minWidth: STOCK_NAME_COLUMN_MIN_WIDTH,
        cellStyle: (p) => ({ fontWeight: 700, ...(p.data?.is_cash ? { color: "var(--text-muted)" } : null) }),
      },
      {
        field: "sources",
        headerName: "전략",
        width: 110,
        // 두 전략이 같은 종목을 담으면 한 행에 둘 다 표시된다 (비중은 합산).
        valueFormatter: (p) => {
          const sources = (p.value as string[]) ?? [];
          return sources.length === 0 ? "-" : sources.map((s) => SOURCE_LABEL[s] ?? s).join("·");
        },
      },
      // 아래 순서·명칭은 /assets 계좌 보유 표와 맞춘다 (일간→현재가→비중→목표비중→목표수량→수량).
      {
        field: "change_pct",
        headerName: "일간(%)",
        width: 92,
        type: "numericColumn",
        valueFormatter: (p) => formatSignedPct(p.value as number),
        cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 600 }),
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
        valueFormatter: (p) => (p.value == null ? "-" : `${(p.value as number).toFixed(2)}%`),
        cellStyle: (p) => {
          const current = p.value as number | null;
          const target = p.data?.weight_pct;
          if (current == null || target == null) return null;
          // 목표에서 1%p 넘게 벌어진 행만 눌러 표시한다 — 반올림 오차로 전부 색이 붙는 걸 막는다.
          return Math.abs(current - target) > 1 ? { color: "var(--text-muted)" } : null;
        },
      },
      {
        field: "weight_pct",
        headerName: "목표비중",
        width: 88,
        type: "numericColumn",
        valueFormatter: (p) => (p.value == null ? "-" : `${(p.value as number).toFixed(2)}%`),
        cellStyle: { fontWeight: 600 },
      },
    ];
    // 계좌를 연결한 풀에서만 매매 지시 컬럼을 붙인다 — 목표와 실제 보유의 차이가 주문 수량이다.
    if (totalAsset != null) {
      columns.push(
        {
          field: "shares",
          headerName: "목표수량",
          headerTooltip: "목표비중 × 총자산 ÷ 현재가",
          width: 88,
          type: "numericColumn",
          valueFormatter: (p) => (p.value == null ? "-" : (p.value as number).toLocaleString("ko-KR")),
        },
        {
          field: "held_quantity",
          headerName: "수량",
          width: 80,
          type: "numericColumn",
          valueFormatter: (p) => (p.value == null ? "-" : (p.value as number).toLocaleString("ko-KR")),
        },
        {
          field: "trade_quantity",
          headerName: "매매수량",
          headerTooltip: "목표수량 − 수량. +는 매수, −는 매도",
          width: 92,
          type: "numericColumn",
          valueFormatter: (p) => {
            const value = p.value as number | null;
            if (value == null) return "-";
            if (value === 0) return "0";
            return `${value > 0 ? "+" : ""}${value.toLocaleString("ko-KR")}`;
          },
          cellStyle: (p) => {
            const value = p.value as number | null;
            if (value == null || value === 0) return null;
            return { color: value > 0 ? "#2f9e44" : "#d62828", fontWeight: 700 };
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
        if (p.data.sm_status) parts.push(`모멘텀 ${p.data.sm_status}`);
        if (p.data.nh_status) parts.push(`신고가 ${p.data.nh_status}`);
        return parts.join(" / ");
      },
      cellStyle: (p) => {
        const text = String(p.value ?? "");
        if (text.includes("매도 예정")) return { color: "#d62828", fontWeight: 600 };
        if (text.includes("진입 예정") || text.includes("매수 예정")) return { color: "#2f9e44", fontWeight: 600 };
        return null;
      },
    });
    return columns;
  }, [totalAsset]);

  const periodRows = useMemo<PeriodRow[]>(() => {
    if (!view || viewMode === "trades") return [];
    if (viewMode === "weekly") return toPeriodRows(view.daily, weekKeyOf, true);
    const keyLength = viewMode === "yearly" ? 4 : viewMode === "monthly" ? 7 : 10;
    return toPeriodRows(view.daily, (date) => date.slice(0, keyLength));
  }, [view, viewMode]);

  const tradeColumns = useMemo<ColDef<MixTradeRow>[]>(
    () => [
      {
        headerName: "전략",
        field: "strategy",
        width: 92,
        cellStyle: (p) => ({ fontWeight: 700, color: p.value === "모멘텀" ? "#1c7ed6" : "#e8590c" }),
      },
      { headerName: "티커", field: "ticker", width: 96 },
      { headerName: "종목명", field: "name", flex: 1, minWidth: 180 },
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
        cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 700 }),
      },
      { headerName: "보유일", field: "days", width: 84, type: "numericColumn" },
      { headerName: "사유", field: "reason", width: 110 },
    ],
    [],
  );

  const periodColumns = useMemo<ColDef<PeriodRow>[]>(() => {
    const pct = (field: keyof PeriodRow, headerName: string, suffix = "%"): ColDef<PeriodRow> => ({
      field,
      headerName,
      flex: 1,
      minWidth: 120,
      type: "numericColumn",
      valueFormatter: (p) =>
        p.value == null ? "-" : `${(p.value as number) >= 0 ? "+" : ""}${(p.value as number).toFixed(2)}${suffix}`,
      cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 600 }),
    });
    return [
      {
        field: "period",
        headerName: viewMode === "yearly" ? "연도" : viewMode === "monthly" ? "월" : viewMode === "weekly" ? "주" : "일자",
        width: 148,
        valueFormatter: (p) =>
          viewMode === "weekly" || viewMode === "daily"
            ? formatDateWithWeekday(String(p.value ?? ""))
            : String(p.value ?? ""),
        cellStyle: { fontWeight: 700 },
      },
      pct("strategy_pct", "전략"),
      pct("benchmark_pct", view?.benchmark_name ?? "벤치마크"),
      pct("excess_pp", "초과", "%p"),
    ];
  }, [viewMode, view?.benchmark_name]);

  // 요약 바에 쓰는 계좌 표기 — 셀렉터와 같은 형식.
  const accountLabel = useMemo(() => {
    const id = positions?.account?.account_id ?? "";
    if (!id) return "";
    const found = accountOptions.find((option) => option.account_id === id);
    return found ? formatAccountLabel(found) : id;
  }, [positions?.account?.account_id, accountOptions]);

  const actions = positions?.actions ?? null;

  // 오늘의 액션 — 체결 시점으로 묶고 묶음 안에서는 매도 → 매수 순서로 세운다.
  // 매도가 끝나야 매수 대금이 생기고, 모멘텀 교체는 교체일 시가에만 체결되기 때문이다.
  const actionGroups = useMemo<ActionGroup[]>(() => {
    if (!positions || !actions) return [];
    const rowByTicker = new Map(positions.holdings.map((row) => [row.ticker, row]));
    const label = (ticker: string, name: string, quantity?: number | null) => {
      const base = `${name}(${ticker})`;
      return quantity == null || quantity === 0
        ? base
        : `${base} ${Math.abs(quantity).toLocaleString("ko-KR")}주`;
    };

    const rebalance = actions.sm_rebalance;
    const rebalanceBuys = new Set(rebalance.buys.map((row) => row.ticker));
    const rebalanceSells = new Set(rebalance.sells.map((row) => row.ticker));
    const entryTickers = new Set(actions.nh_entries.map((row) => row.ticker));
    const sellPending = new Set([...actions.sm_sells, ...actions.nh_sells].map((row) => row.ticker));

    const now: ActionItem[] = [];
    // ① 매도 — 목표에 없는 보유(교체일에 파는 것은 ②로 미룬다) → 매도 예정 → 비중 초과분.
    for (const row of positions.account?.sell_all ?? []) {
      if (rebalanceSells.has(row.ticker)) continue;
      now.push({
        key: `sell-all-${row.ticker}`,
        side: "sell",
        title: "전량 매도",
        text: `${label(row.ticker, row.name, row.quantity)}${row.value != null ? ` (${formatAmount(row.value)})` : ""} · 목표에 없는 보유 종목`,
      });
    }
    for (const row of actions.sm_sells) {
      now.push({
        key: `sm-sell-${row.ticker}`,
        side: "sell",
        title: "모멘텀 매도 예정",
        text: `${label(row.ticker, row.name, rowByTicker.get(row.ticker)?.held_quantity)} (${row.reason}) · 슬롯은 다음 교체까지 현금`,
      });
    }
    for (const row of actions.nh_sells) {
      now.push({
        key: `nh-sell-${row.ticker}`,
        side: "sell",
        title: "신고가 매도 예정",
        text: `${label(row.ticker, row.name, rowByTicker.get(row.ticker)?.held_quantity)} (${row.reason}${row.return_pct != null ? `, ${formatSignedPct(row.return_pct)}` : ""})`,
      });
    }
    for (const row of positions.holdings) {
      const trade = row.trade_quantity;
      if (row.is_sell_all || trade == null || trade >= 0) continue;
      if (rebalanceSells.has(row.ticker) || sellPending.has(row.ticker)) continue;
      now.push({
        key: `trim-${row.ticker}`,
        side: "sell",
        title: "비중 조정 매도",
        text: `${label(row.ticker, row.name, trade)} · 목표 ${row.weight_pct.toFixed(2)}%`,
      });
    }
    // ① 매수 — 신고가 진입 예정 → 비중 부족분(모멘텀 교체 매수는 ②).
    for (const row of actions.nh_entries) {
      now.push({
        key: `entry-${row.ticker}`,
        side: "buy",
        title: "신고가 진입 예정",
        text: `${label(row.ticker, row.name, rowByTicker.get(row.ticker)?.trade_quantity)}${row.price != null ? ` (현재가 ${formatPrice(row.price)})` : ""}`,
      });
    }
    for (const row of positions.holdings) {
      const trade = row.trade_quantity;
      if (row.is_sell_all || trade == null || trade <= 0) continue;
      if (rebalanceBuys.has(row.ticker) || entryTickers.has(row.ticker)) continue;
      now.push({
        key: `add-${row.ticker}`,
        side: "buy",
        title: "비중 조정 매수",
        text: `${label(row.ticker, row.name, trade)} · 목표 ${row.weight_pct.toFixed(2)}%`,
      });
    }

    // ② 모멘텀 교체 — 판정은 끝났고 교체일 시가에만 체결된다.
    const rebalanceItems: ActionItem[] = [];
    for (const row of rebalance.sells) {
      rebalanceItems.push({
        key: `reb-sell-${row.ticker}`,
        side: "sell",
        title: "교체 매도",
        text: label(row.ticker, row.name, rowByTicker.get(row.ticker)?.held_quantity),
      });
    }
    for (const row of rebalance.buys) {
      const target = rowByTicker.get(row.ticker);
      rebalanceItems.push({
        key: `reb-buy-${row.ticker}`,
        side: "buy",
        title: "교체 매수",
        text: `${label(row.ticker, row.name, target?.trade_quantity ?? target?.target_quantity)}${row.price != null ? ` (현재가 ${formatPrice(row.price)})` : ""}`,
      });
    }

    const groups: ActionGroup[] = [];
    if (now.length > 0) groups.push({ key: "now", title: "지금 · 다음 거래일 시가", items: now });
    if (rebalanceItems.length > 0 && !rebalance.is_filled) {
      groups.push({
        key: "rebalance",
        title: `${rebalance.fill_date} 시가 · 모멘텀 교체`,
        items: rebalanceItems,
      });
    }
    return groups;
  }, [positions, actions]);

  const hasActions = actionGroups.length > 0 || Boolean(actions?.sleeve_rebalance_today);

  return (
    <PageFrame title="합성전략" fullWidth>
      <div className="appPageStack">
        <section className="appSection">
          <div className="card appCard">
            <div className="card-body appCardBodyTight">
              {/* 메인 헤더 — 셀렉터·모드 전환 같은 주 제어. */}
              <div className="appMainHeader">
                <div className="appMainHeaderLeft">
                <label className="appLabeledField" style={{ marginBottom: 0 }}>
                  <span className="appLabeledFieldLabel">종목풀</span>
                  <select
                    className="form-select form-select-sm"
                    value={pool}
                    disabled={loading || positionsLoading || poolOptions.length === 0}
                    onChange={(event) => {
                      const next = event.target.value;
                      setPool(next);
                      setAccountId(savedByPool[next]?.account_id ?? "");
                      setPositions(null);
                      setView(null);
                      setAsOf("");
                    }}
                  >
                    {poolOptions.map((option) => (
                      <option key={option.ticker_type} value={option.ticker_type}>
                        {formatPoolLabel(option)}
                      </option>
                    ))}
                  </select>
                </label>
                <AccountSelect
                  label="적용 계좌"
                  accounts={accountOptions}
                  value={accountId}
                  onChange={setAccountId}
                  disabled={saving}
                  emptyLabel="선택 안 함"
                  style={{ width: "auto" }}
                  labelStyle={{ marginBottom: 0 }}
                />
                <button
                  className="btn btn-sm btn-primary"
                  type="button"
                  disabled={saving || !pool || !isDirty}
                  onClick={() => void saveSettings()}
                >
                  {saving ? "저장 중…" : "저장"}
                </button>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="appSection">
          <div className="card appCard">
            <div className="card-header appCardHeader">
              <span style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>현재 상태</span>
              {/* 기준일 셀렉트 — 신고가 화면과 같은 자리(카드 헤더 오른쪽). */}
              <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <select
                    className="form-select form-select-sm"
                    style={{ width: "auto" }}
                    value={asOf}
                    disabled={positionsLoading}
                    onChange={(event) => setAsOf(event.target.value)}
                  >
                    <option value="">
                      오늘{positions && !asOf ? ` (${formatDateWithWeekday(positions.as_of)})` : ""}
                    </option>
                    {(positions?.available_dates ?? []).map((date) => (
                      <option key={date} value={date}>
                        {formatDateWithWeekday(date)}
                      </option>
                    ))}
                  </select>
              </span>
            </div>
            <div className="card-body appCardBodyTight">
              {positionsError ? (
                  <div className="alert alert-danger" style={{ marginBottom: 0 }}>{positionsError}</div>
                ) : positionsLoading || !positions ? (
                  <AppLoadingProgress title="현재 상태 계산 중..." progress={positionsProgress} />
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                    {/* ① 요약 바 — 오늘 주식·현금을 얼마씩 둬야 하는지. */}
                    <div style={{ display: "flex", alignItems: "center", gap: 18, flexWrap: "wrap" }}>
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
                        <span style={{ fontSize: "var(--fs-base)", fontWeight: 700 }}>
                          {accountLabel} 총자산 {formatAmount(positions.account.total_assets)}
                          <span style={{ ...hintStyle, marginLeft: 8, fontWeight: 500 }}>
                            (주식 {formatAmount(positions.account.stock_value)} · 현금{" "}
                            {formatAmount(positions.account.cash_balance)})
                          </span>
                        </span>
                      ) : (
                        <span style={hintStyle}>적용 계좌 없음 — 계좌를 저장하면 목표 금액·주수가 나옵니다</span>
                      )}
                      <span style={hintStyle}>
                        모멘텀 목표 {positions.summary.sm.slots_used}/{positions.summary.sm.top_n} (보유{" "}
                        {positions.summary.sm.held_count}) · 신고가 목표 {positions.summary.nh.slots_used}/
                        {positions.summary.nh.top_n} (보유 {positions.summary.nh.held_count})
                      </span>
                      {actions && !actions.sm_rebalance.is_filled && actions.sm_rebalance.fill_date ? (
                        <span style={hintStyle}>
                          모멘텀 {actions.sm_rebalance.portfolio_week} 포트폴리오 · 체결{" "}
                          {actions.sm_rebalance.fill_date} (판정 {actions.sm_rebalance.signal_date})
                        </span>
                      ) : null}
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
                          오늘은 매월 첫 거래일 — 두 슬리브를 50:50으로 리밸런싱하세요
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
                        const status = `${params.data?.sm_status ?? ""} ${params.data?.nh_status ?? ""}`;
                        return status.includes("예정") ? "appTrendBrokenRow" : "";
                      }}
                      gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                    />

                    {/* ③ 오늘의 액션 — 체결 시점별 묶음, 각 묶음은 매도 → 매수 순서. */}
                    <div>
                      <div style={{ fontWeight: 700, marginBottom: 6 }}>오늘의 액션</div>
                      {!hasActions ? (
                        <div style={hintStyle}>오늘은 할 일이 없습니다 — 보유 목록을 그대로 유지하세요.</div>
                      ) : (
                        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                          {actionGroups.map((group, groupIndex) => (
                            <div key={group.key}>
                              <div style={{ fontWeight: 700, marginBottom: 4 }}>
                                {groupIndex + 1}. {group.title}
                                <span style={{ ...hintStyle, marginLeft: 8, fontWeight: 500 }}>
                                  매도 {group.items.filter((item) => item.side === "sell").length}건 · 매수{" "}
                                  {group.items.filter((item) => item.side === "buy").length}건
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
                                    <strong style={{ color: item.side === "sell" ? "#d62828" : "#2f9e44" }}>
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
                              <strong>슬리브 리밸런싱</strong> — 매월 첫 거래일입니다. 모멘텀·신고가 슬리브를 각각 50%로
                              다시 맞추세요.
                            </div>
                          ) : null}
                        </div>
                      )}
                    </div>
                  </div>
                )}
            </div>
          </div>
        </section>

        {/* 백테스트 — 현재 상태 아래에 나란히 둔다. */}
        <section className="appSection">
          <div className="card appCard">
            <div className="card-header appCardHeader">
              <span style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>백테스트</span>
              <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <select
                  className="form-select form-select-sm"
                  style={{ width: "auto" }}
                  value={String(months)}
                  disabled={loading || monthOptions.length === 0}
                  onChange={(event) => setMonths(Number(event.target.value))}
                >
                  {monthOptions.map((n) => (
                    <option key={n} value={n}>{n}개월</option>
                  ))}
                </select>
                <button
                  className="btn btn-sm btn-dark"
                  type="button"
                  disabled={loading || !pool}
                  onClick={() => void runBacktest()}
                >
                  {loading ? "실행 중…" : "실행"}
                </button>
              </span>
            </div>
            <div className="card-body appCardBodyTight">
              {error ? (
                <div className="alert alert-danger" style={{ marginBottom: 0 }}>{error}</div>
              ) : loading ? (
                <AppLoadingProgress title="백테스트 실행 중..." progress={backtestProgress} />
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
                      gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                      getRowClass={(p) => (p.data?.exit_date ? "" : "momentumPendingRow")}
                    />
                  ) : (
                    <AppAgGrid<PeriodRow>
                      rowData={periodRows}
                      columnDefs={periodColumns}
                      theme={gridTheme}
                      minHeight="auto"
                      getRowId={(p) => p.data.period}
                      gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                    />
                  )}
                </>
              ) : (
                <div style={{ ...hintStyle, padding: "32px 0", textAlign: "center" }}>
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
