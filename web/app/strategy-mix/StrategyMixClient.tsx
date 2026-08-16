"use client";

import type { ColDef } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppAgGrid } from "../components/AppAgGrid";
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

type Meta = {
  pool: string;
  pool_options: PoolLabelSource[];
  month_options: number[];
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
};

type Positions = {
  computed_at: string;
  pool: string;
  as_of: string;
  live: boolean;
  /** 과거 날짜 셀렉트용 — 신고가 화면과 같은 날짜 목록. */
  available_dates: string[];
  summary: {
    stock_pct: number;
    cash_pct: number;
    sm: { slots_used: number; top_n: number; cash_pct: number };
    nh: { slots_used: number; top_n: number; cash_pct: number };
  };
  holdings: Holding[];
  actions: {
    /** 모멘텀 주중 매도 예정 — 보유 자격(장단기 이평선) 상실, 다음 거래일 시가 매도. */
    sm_sells: { ticker: string; name: string; reason: string }[];
    nh_entries: { ticker: string; name: string; price: number | null; change_pct: number | null; value_mult: number | null }[];
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

const SOURCE_LABEL: Record<string, string> = { sm: "모멘텀", nh: "신고가" };

/** 보유 표 행 — 현금 행도 같은 표에 넣는다 (비중 합이 100%임을 한눈에 보이게). */
type PositionRow = Holding & { is_cash?: boolean; amount: number | null; shares: number | null };

/** 합성전략 — SM·신고가를 50:50으로 함께 운용하는 화면.
 *  현재 상태 탭은 오늘 보유해야 할 종목과 현금 비중·오늘의 액션을,
 *  백테스트 탭은 매월 50:50 리밸런싱 합성 성과를 보여준다. 설정은 각 전략 화면의 저장값을 그대로 쓴다. */
export function StrategyMixClient() {
  const toast = useToast();
  const [pool, setPool] = useState<string>("");
  const [poolOptions, setPoolOptions] = useState<PoolLabelSource[]>([]);
  const [months, setMonths] = useState<number>(12);
  const [monthOptions, setMonthOptions] = useState<number[]>([]);

  // 현재 상태 탭.
  const [positions, setPositions] = useState<Positions | null>(null);
  const [positionsLoading, setPositionsLoading] = useState(false);
  const [positionsError, setPositionsError] = useState<string | null>(null);
  const [totalAssetText, setTotalAssetText] = useState<string>("");
  /** 과거 날짜 조회 — 빈 값이면 오늘. */
  const [asOf, setAsOf] = useState<string>("");

  // 백테스트 탭.
  const [view, setView] = useState<View | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("monthly");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
        if (alive) setPositionsLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [pool, asOf]);

  const runBacktest = useCallback(async () => {
    if (!pool) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/strategy-mix?pool=${encodeURIComponent(pool)}&months=${months}`, {
        cache: "no-store",
      });
      const payload = (await response.json()) as View & { error?: string };
      if (!response.ok || payload.error) throw new Error(payload.error ?? "합성 백테스트를 불러오지 못했습니다.");
      setView(payload);
    } catch (runError) {
      const message = runError instanceof Error ? runError.message : "합성 백테스트를 불러오지 못했습니다.";
      setError(message);
      toast.error(message);
    } finally {
      setLoading(false);
    }
  }, [pool, months, toast]);

  // 입력 단위는 만원 — 환산은 원 단위로 한다.
  const totalAsset = useMemo(() => {
    const parsed = Number(totalAssetText.replaceAll(",", ""));
    return Number.isFinite(parsed) && parsed > 0 ? parsed * 10_000 : null;
  }, [totalAssetText]);

  const positionRows = useMemo<PositionRow[]>(() => {
    if (!positions) return [];
    const toAmount = (weightPct: number) => (totalAsset == null ? null : (totalAsset * weightPct) / 100);
    const rows: PositionRow[] = positions.holdings.map((holding) => {
      const amount = toAmount(holding.weight_pct);
      return {
        ...holding,
        amount,
        shares: amount != null && holding.price ? Math.floor(amount / holding.price) : null,
      };
    });
    // 빈 슬롯은 슬리브별로 채워 슬롯 수(예: 8+8)가 표에서 그대로 보이게 한다.
    // 빈 슬롯은 각 전략의 진입 예정 종목부터 채우고(다음 시가 매수), 남으면 현금 슬롯이다.
    const nhWeight = 50 / positions.summary.nh.top_n;
    const nhFree = positions.summary.nh.top_n - positions.summary.nh.slots_used;
    const entries = positions.actions.nh_entries.slice(0, nhFree);
    for (const entry of entries) {
      const amount = toAmount(nhWeight);
      rows.push({
        ticker: entry.ticker,
        name: entry.name,
        sources: ["nh"],
        weight_pct: nhWeight,
        price: entry.price,
        change_pct: entry.change_pct,
        sm_status: null,
        nh_status: "진입 예정 (다음 시가 매수)",
        amount,
        shares: amount != null && entry.price ? Math.floor(amount / entry.price) : null,
      });
    }
    for (const [source, sleeve, used] of [
      ["sm", positions.summary.sm, positions.summary.sm.slots_used],
      ["nh", positions.summary.nh, positions.summary.nh.slots_used + entries.length],
    ] as const) {
      const slotWeight = 50 / sleeve.top_n;
      for (let slot = used; slot < sleeve.top_n; slot += 1) {
        rows.push({
          ticker: `__cash_${source}_${slot}__`,
          name: "현금",
          sources: [source],
          weight_pct: slotWeight,
          price: null,
          change_pct: null,
          sm_status: null,
          nh_status: null,
          is_cash: true,
          amount: toAmount(slotWeight),
          shares: null,
        });
      }
    }
    return rows;
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
        // 겹치는 종목도 슬리브별 행으로 각각 오므로 전략은 항상 하나다.
        valueFormatter: (p) => {
          const sources = (p.value as string[]) ?? [];
          return sources.length === 0 ? "-" : SOURCE_LABEL[sources[0]] ?? sources[0];
        },
      },
      {
        field: "weight_pct",
        headerName: "목표 비중",
        width: 110,
        type: "numericColumn",
        valueFormatter: (p) => (p.value == null ? "-" : `${(p.value as number).toFixed(2)}%`),
        cellStyle: { fontWeight: 600 },
      },
      {
        field: "price",
        headerName: "현재가",
        width: 120,
        type: "numericColumn",
        valueFormatter: (p) => formatPrice(p.value as number),
      },
      {
        field: "change_pct",
        headerName: "일간(%)",
        width: 100,
        type: "numericColumn",
        valueFormatter: (p) => formatSignedPct(p.value as number),
        cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 600 }),
      },
    ];
    if (totalAsset != null) {
      columns.push(
        {
          field: "amount",
          headerName: "목표 금액",
          width: 140,
          type: "numericColumn",
          valueFormatter: (p) =>
            p.value == null ? "-" : (p.value as number).toLocaleString("ko-KR", { maximumFractionDigits: 0 }),
        },
        {
          field: "shares",
          headerName: "주수",
          width: 90,
          type: "numericColumn",
          valueFormatter: (p) => (p.value == null ? "-" : (p.value as number).toLocaleString("ko-KR")),
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
        if (p.data.is_cash) return "빈 슬롯 (현금)";
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

  const actions = positions?.actions ?? null;
  const hasActions =
    actions != null &&
    (actions.sm_sells.length > 0 ||
      actions.nh_entries.length > 0 ||
      actions.nh_sells.length > 0 ||
      actions.sm_rebalance.buys.length > 0 ||
      actions.sm_rebalance.sells.length > 0 ||
      actions.sleeve_rebalance_today);

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
                      setPool(event.target.value);
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
                <label className="appLabeledField" style={{ marginBottom: 0 }}>
                  <span className="appLabeledFieldLabel">총자산 (금액·주수 환산용, 단위 만원)</span>
                  <input
                    className="form-control form-control-sm"
                    style={{ width: 180 }}
                    inputMode="numeric"
                    placeholder="예: 1,000"
                    value={totalAssetText}
                    onChange={(event) => setTotalAssetText(event.target.value)}
                  />
                </label>
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
                  <div style={{ ...hintStyle, textAlign: "center", padding: "48px 0" }}>
                    두 전략의 현재 상태를 계산하고 있습니다… (수십 초 걸릴 수 있습니다)
                  </div>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                    {/* ① 요약 바 — 오늘 주식·현금을 얼마씩 둬야 하는지. */}
                    <div style={{ display: "flex", alignItems: "center", gap: 18, flexWrap: "wrap" }}>
                      <span style={{ fontSize: "var(--fs-lg)", fontWeight: 800 }}>
                        주식 {positions.summary.stock_pct.toFixed(1)}% · 현금 {positions.summary.cash_pct.toFixed(1)}%
                      </span>
                      <span style={hintStyle}>
                        모멘텀 슬리브 {positions.summary.sm.slots_used}/{positions.summary.sm.top_n}종목 보유 (현금{" "}
                        {positions.summary.sm.cash_pct.toFixed(1)}%) · 신고가 슬리브 {positions.summary.nh.slots_used}/
                        {positions.summary.nh.top_n}종목 보유 (현금 {positions.summary.nh.cash_pct.toFixed(1)}%)
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
                      // 같은 종목이 두 슬리브에 겹칠 수 있어 티커만으로는 행이 유일하지 않다.
                      getRowId={(p) => `${p.data.sources[0] ?? "cash"}:${p.data.ticker}`}
                      // 아직 체결 전인 행(진입 예정)과 곧 나갈 행(매도 예정)은 확정 보유와
                      // 구분되게 회색으로 눌러 둔다 — 추세 이탈 행과 같은 공용 클래스.
                      getRowClass={(params) => {
                        const status = `${params.data?.sm_status ?? ""} ${params.data?.nh_status ?? ""}`;
                        return status.includes("예정") ? "appTrendBrokenRow" : "";
                      }}
                      gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                    />

                    {/* ③ 오늘의 액션 — 사고팔 것과 다음달 교체 예상. */}
                    <div>
                      <div style={{ fontWeight: 700, marginBottom: 6 }}>오늘의 액션</div>
                      {!hasActions ? (
                        <div style={hintStyle}>오늘은 할 일이 없습니다 — 보유 목록을 그대로 유지하세요.</div>
                      ) : (
                        <ul style={{ margin: 0, paddingLeft: 18, display: "flex", flexDirection: "column", gap: 4 }}>
                          {actions!.sm_sells.map((row) => (
                            <li key={`sm-sell-${row.ticker}`}>
                              <strong style={{ color: "#d62828" }}>모멘텀 매도 예정</strong> — {row.name} ({row.reason})
                              · 다음 시가에 매도, 슬롯은 다음 교체까지 현금
                            </li>
                          ))}
                          {actions!.nh_sells.map((row) => (
                            <li key={`sell-${row.ticker}`}>
                              <strong style={{ color: "#d62828" }}>신고가 매도 예정</strong> — {row.name} ({row.reason}
                              {row.return_pct != null ? `, ${formatSignedPct(row.return_pct)}` : ""}) · 다음 시가에 매도
                            </li>
                          ))}
                          {actions!.nh_entries.map((row) => (
                            <li key={`entry-${row.ticker}`}>
                              <strong style={{ color: "#2f9e44" }}>신고가 진입 예정</strong> — {row.name}
                              {row.price != null ? ` (현재가 ${formatPrice(row.price)})` : ""} · 다음 시가에 매수
                            </li>
                          ))}
                          {actions!.sm_rebalance.buys.length > 0 ? (
                            <li>
                              <strong style={{ color: "#2f9e44" }}>모멘텀 교체 매수</strong> —{" "}
                              {actions!.sm_rebalance.buys.map((r) => r.name).join(", ")} ·{" "}
                              {actions!.sm_rebalance.fill_date} 시가
                            </li>
                          ) : null}
                          {actions!.sm_rebalance.sells.length > 0 ? (
                            <li>
                              <strong style={{ color: "#d62828" }}>모멘텀 교체 매도</strong> —{" "}
                              {actions!.sm_rebalance.sells.map((r) => r.name).join(", ")} ·{" "}
                              {actions!.sm_rebalance.fill_date} 시가
                            </li>
                          ) : null}
                          {actions!.sleeve_rebalance_today ? (
                            <li>
                              <strong>슬리브 리밸런싱</strong> — 매월 첫 거래일입니다. 모멘텀·신고가 슬리브를 각각 50%로
                              다시 맞추세요.
                            </li>
                          ) : null}
                        </ul>
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
                <div style={{ ...hintStyle, textAlign: "center", padding: "48px 0" }}>
                  두 전략의 백테스트를 계산하고 있습니다… (수 분 걸릴 수 있습니다)
                </div>
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
