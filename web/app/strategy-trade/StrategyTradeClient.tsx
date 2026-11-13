"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ColDef, RowDragEndEvent } from "ag-grid-community";

import { gridRowDragColumn, gridSelectionColumn, stockMemoColumn, stockNameColumn, stockRowClass, tickerColumn } from "@/lib/grid-cells";
import type { HoldingsGroup } from "@/lib/holdings-store";
import { formatPoolLabel, type PoolLabelSource } from "@/lib/pool-label";
import { formatCurrencyPrice, formatPrice } from "@/lib/price-format";
import { renderStockNameCell } from "@/lib/name-highlight";

import { CASH_ROW_TICKER, type AccountSummary, type HoldingsResponse, type HoldingsRow } from "../assets/assets-helpers";
import { holdingMetricColumns } from "../assets/holding-metric-columns";
import AccountSelect, { type AccountOptionBase } from "../components/AccountSelect";
import { SlackAlarmField } from "../components/SlackAlarmField";
import { readRememberedTradeAccountId, writeRememberedTradeAccountId } from "../components/account-selection";
import { AppAgGrid } from "../components/AppAgGrid";
import { createAppGridTheme } from "../components/app-grid-theme";
import { GridToolbarButton } from "../components/GridToolbarButton";
import { HoldingsGroupButton, HoldingsGroupDialog, HoldingsGroupName, type HoldingsGroupDraft } from "../components/HoldingsGroupControls";
import { AppModal } from "../components/AppModal";
import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";
import { useStockMemoSave } from "../components/useStockMemoSave";
import { UnsavedChangesBadge } from "../components/UnsavedChangesBadge";

type StrategyConfig = {
  // 매매 간격(%) 하나로 진입·추가·매도를 모두 관리한다.
  trigger_pct: number;
  round_size: number;
  rounds: number;
  index_name: string;
};

type IndexStatus = {
  name: string;
  close: number;
  // 최근 지수 저점(라벨은 서버가 정한다) — 회차 지정가가 어느 층에 걸리는지 가늠하는 기준선.
  recent_low: number;
  recent_low_date: string;
  recent_low_label: string;
  recent_high: number;
  recent_high_date: string;
  recent_high_label: string;
};

type Status = {
  held_count: number;
  waiting_first_entry: boolean;
  // 보유 0 일 때 다음 1호 매수 시각(한국시간 "HH:MM").
  first_entry_time: string;
  held_quantity: number;
  // 증권사 계좌 평균단가(이동평균)와 현재가 대비 수익률(%) — 보유가 없으면 null.
  average_price: number | null;
  average_profit_pct: number | null;
  // 다음 회차를 살 계좌 현금이 없어 매수를 멈췄다(한도 소진).
  cash_short: boolean;
  budget_short: boolean;
  next_round: number | null;
  next_ticker: string | null;
  next_name: string | null;
  last_buy_price: number | null;
  // 다음 회차 매수까지 남은 하락률(%, 음수). 도달했으면 0 이상. 다음 회차 없으면 null.
  next_buy_drop_pct: number | null;
  // 가장 가까운 매도 회차와 목표까지 남은 상승률(%, 양수). 0 이하면 도달. 보유 없으면 null.
  next_sell_round: number | null;
  next_sell_rise_pct: number | null;
  // 매도가 먼저 닿아 있어 다음 회차 매수를 그 매도가 체결될 때까지 미루는 중 — 신호는 한 번에 하나다.
  sell_pending: boolean;
};

type TradeSide = "buy" | "sell" | "arbitrary_buy" | "arbitrary_sell";

const FILL_LABELS = {
  buy: "매수",
  sell: "매도",
  arbitrary_buy: "임의 매수",
  arbitrary_sell: "임의 매도",
} as const;

export type RoundRow = {
  /** 사다리 위치(표시용) — 1호 체결가 기준 간격 칸. 건너뛴 칸은 회차 없이 빈 행으로 보인다. */
  position: number;
  round: number;
  fixed_round: number;
  ticker: string;
  name: string;
  held: boolean;
  quantity: number;
  // 실제 체결가·산 날짜·출처 — 사다리 선(buy_index)과 별개다. 안 산 회차는 null.
  fill_price: number | null;
  bought_on: string | null;
  source: "slack" | "manual" | "mixed" | null;
  close: number | null;
  profit_pct: number | null;
  buy_limit: number | null;
  buy_index: number | null;
  sell_limit: number | null;
  sell_index: number | null;
  buy_reached: boolean;
  // 정해진 구간에 사는 1호 — 현재가로 산다.
  close_entry: boolean;
  // 다음 회차를 살 계좌 현금이 없다 — 한도 소진, 이 회차는 사지 않는다.
  cash_short: boolean;
  budget_short: boolean;
  // 이 회차를 팔면 다음 매수선이 매도가 이상이라 바로 다시 사게 된다 — 신호를 '매도 후 재매수'로 낸다.
  rebuy_after_sell: boolean;
  sell_reached: boolean;
  // 오늘 슬랙 신호가 나간 방향(key: 회수할 때 쓰는 발송 기록 키, has_ts: 슬랙 메시지 기록이 있어 회수할 수 있음), 아직 끝나지 않은 주문의 방향.
  slack_signals: SlackSignal[];
  order_sides: Array<"buy" | "sell">;
  // 증권사 앱(MTS)에 걸려 있는 미체결 수동 주문 — 슬랙 주문을 막는다.
  manual_orders: Array<{ side: "buy" | "sell"; price: number; quantity: number }>;
  is_next: boolean;
};

export type SlackSignal = { side: "buy" | "sell"; key: string; has_ts: boolean };

export type StrategyView = {
  strategy_id: string;
  label: string;
  /** 이 전략이 속한 종목풀. */
  pool: string;
  /** 운용 계좌 — 전략마다 고른다. */
  account_id: string;
  account_name: string;
  /** 전략의 국가(kor·us) — 호가 단위·통화·시간표가 이 값으로 정해진다. */
  country: string;
  ticker: string;
  config: StrategyConfig;
  index: IndexStatus;
  status: Status;
  arbitrary_lots: Array<{ id: string; quantity: number; fill_price: number; bought_on: string; sell_limit: number }>;
  /** 사다리 위치별 · 날짜별 매수·매도(체결 순서) — 지금 사이클, 오늘 리셋됐으면 오늘까지 1주일. */
  trades_by_position: Record<string, Record<string, TradeSide[]>>;
  /** 사다리 위치별 가격 — 지금 1호 체결가 × (1 − 간격%)^(위치−1)을 호가로 맞춘 값. 보유가 없으면 비어 있다. */
  position_prices: Record<string, number>;
  /** 보유가 모두 빈 날짜(사이클 종료) — 표의 날짜 칸 오른쪽에 경계선을 긋는다. */
  cycle_end_days: string[];
  // 체결 이력(회차 원장)을 계산한 시각 — 사고팔기 알림 배치가 갱신한다(KST ISO).
  ledger_at: string;
  rounds: RoundRow[];
};

/** 증권사 체결 이력으로 회차를 맞출 수 없는 전략 — 이유만 보여 준다(평균단가로 짐작하지 않는다). */
export type FailedStrategy = {
  strategy_id: string;
  label: string;
  pool: string;
  account_id: string;
  account_name: string;
  ticker: string;
  error: string;
};

/** 저장된 전략 설정 한 줄 — 편집 표의 행. */
type StrategyConfigRow = {
  strategy_id: string;
  ticker: string;
  name: string;
  trigger_pct: number;
  round_size: number;
  rounds: number;
  /** 현재가·일간(%) — 공통 시세 경로 값. 못 받으면 null. */
  price: number | null;
  change_pct: number | null;
};

type TickerOption = { ticker: string; name: string; pools: string[]; kind: "stock" | "etf" };

type View = {
  /** 사고팔기를 운영할 수 있는 계좌 — 증권사 API 가 연동된 운영 계좌. */
  account_options: Array<AccountOptionBase & { country_code: string }>;
  /** 지금 보는 계좌. 운영할 수 있는 계좌가 없으면 null. */
  account_id: string | null;
  /** 지금 보는 계좌의 국가(kor·us) — 통화·금액 단위·가격 표기가 이 값으로 정해진다. */
  country_code: string | null;
  /** 이 계좌의 사고팔기 슬랙 스위치(합성 슬랙과 별개). */
  slack_enabled: boolean;
  groups: HoldingsGroup[];
  pool_options: PoolLabelSource[];
  default_round_size: number;
  /** 종목 통화→원 환율(국장 1) — 회차수량 컬럼용. 못 받았으면 null. */
  krw_rate: number | null;
  ticker_options: TickerOption[];
  /** 저장된 전략 설정 — 이 계좌의 전략 전부. */
  configs: StrategyConfigRow[];
  strategies: Array<StrategyView | FailedStrategy>;
  /** 종목 메모(티커 → 메모) — 메모가 없는 종목은 빠진다. */
  memos: Record<string, string>;
};

// 편집 파라미터는 매매 간격(trigger_pct) 하나 — 백엔드 EDITABLE_PCT_KEYS 와 동일.
const PCT_MIN = 0.1;
const PCT_MAX = 50;

const cardStyle: React.CSSProperties = {
  padding: 10,
  display: "flex",
  flexDirection: "column",
  gap: 6,
};

// 매수·매도 행동 배지 — 이 화면에서 가장 중요한 정보라 테두리로 강조한다.
// 내리면 매수 = 파란색(하락), 오르면 매도 = 빨간색(상승).
function actionBadgeStyle(color: string): React.CSSProperties {
  return {
    fontWeight: 800,
    padding: "1px 8px",
    borderRadius: 6,
    border: `1.5px solid ${color}`,
    color,
  };
}
const BUY_BADGE_COLOR = "var(--down-color, #2f6fd0)";
const SELL_BADGE_COLOR = "var(--up-color, #d64545)";

const thStyle: React.CSSProperties = { padding: "4px 8px", textAlign: "right" };
// 회차 표 고정 열(위치~손익)의 폭 — 날짜 열이 몇 개든 같은 폭이고, 날짜가 적으면 오른쪽이 비어 보인다.
const ROUND_TABLE_FIXED_WIDTHS = [56, 80, 140, 100, 150, 84] as const;
// 날짜 열의 최소 폭 — 그날 매매가 많으면 배지만큼 넓어진다.
const ROUND_TABLE_DAY_MIN_WIDTH = 108;
const WEEKDAY_NAMES = ["일", "월", "화", "수", "목", "금", "토"] as const;
// 오늘 날짜 열 강조색 — 날짜는 한국 시각 기준이다.
const TODAY_COLOR = "var(--tblr-green, #16a34a)";
const tdStyle: React.CSSProperties = { padding: "clamp(3px, calc(1dvh - 6px), 7px) 8px", textAlign: "right" };

/** 계좌 국가별 거래 통화 — 가격 표기(원 0자리·달러 2자리)와 금액 단위를 한 곳에서 정한다. */
export const CURRENCY_BY_COUNTRY: Record<string, string> = { kor: "KRW", us: "USD" };
const TradeCurrencyContext = createContext<string>("KRW");
const useTradeCurrency = () => useContext(TradeCurrencyContext);

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

function StatusBadge({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <span
      style={{
        padding: "1px 5px",
        borderRadius: 4,
        fontSize: "var(--fs-sm)",
        fontWeight: 700,
        color,
        border: `1px solid ${color}`,
      }}
    >
      {children}
    </span>
  );
}

const SIDE_WORDS = { buy: "매수", sell: "매도" } as const;

/** 회차 상태 칸에 보일 것이 있는지 — 설정 표의 상태 요약도 같은 기준을 쓴다. */
function hasRoundStatus(row: RoundRow): boolean {
  return row.buy_reached || row.sell_reached || row.slack_signals.length > 0 || row.order_sides.length > 0 || row.manual_orders.length > 0;
}

/** 상태 칸 — 지정가에 닿았으면 도달(빨강), 슬랙 신호·주문이 나갔으면 방향을 붙여 매도슬랙·매수주문처럼(녹색).
 *  슬랙 뱃지는 슬랙 메시지 기록이 있으면 눌러서 회수할 수 있다(주문이 있으면 서버가 막는다). */
function RoundStatus({ row, onWithdrawSignal }: { row: RoundRow; onWithdrawSignal: (signal: SlackSignal, label: string) => void }) {
  const currency = useTradeCurrency();
  return (
    <span style={{ display: "inline-flex", gap: 4 }}>
      {row.buy_reached || row.sell_reached ? <StatusBadge color="var(--up-color, #d64545)">도달</StatusBadge> : null}
      {row.slack_signals.map((signal) => {
        const label = `${SIDE_WORDS[signal.side]}슬랙`;
        const badge = <StatusBadge color="var(--tblr-green, #2fb344)">{label}</StatusBadge>;
        return signal.has_ts ? (
          <button
            key={`slack-${signal.side}`}
            type="button"
            title="눌러서 이 슬랙 신호 회수"
            onClick={() => onWithdrawSignal(signal, label)}
            style={{ all: "unset", cursor: "pointer" }}
          >
            {badge}
          </button>
        ) : (
          <span key={`slack-${signal.side}`}>{badge}</span>
        );
      })}
      {row.order_sides.map((side) => (
        <StatusBadge key={`order-${side}`} color="var(--tblr-green, #2fb344)">
          {SIDE_WORDS[side]}주문
        </StatusBadge>
      ))}
      {row.manual_orders.map((order, index) => (
        <span key={`manual-${index}`} title={`${formatCurrencyPrice(order.price, currency)} · ${order.quantity.toLocaleString()}주`}>
          <StatusBadge color="var(--tblr-green, #2fb344)">수동{SIDE_WORDS[order.side]}주문</StatusBadge>
        </span>
      ))}
    </span>
  );
}

// 매수는 보라, 매도는 주황 — 빨강·파랑은 상승·하락 손익색이라 헷갈리고 초록은 슬랙·주문 표시가 쓴다.
const FILL_COLORS = { buy: "var(--tblr-purple, #ae3ec9)", sell: "var(--tblr-orange, #f76707)" } as const;

const FILL_SOURCE_WORDS = { slack: "슬랙", manual: "수동" } as const;
const BOUGHT_SOURCE_WORDS = { ...FILL_SOURCE_WORDS, mixed: "혼합" } as const;

function DailyTradeBadges({ sides }: { sides: TradeSide[] }) {
  return (
    <span style={{ display: "inline-flex", gap: 4, whiteSpace: "nowrap" }}>
      {sides.map((side, idx) => (
        <StatusBadge key={idx} color={FILL_COLORS[side.endsWith("buy") ? "buy" : "sell"]}>
          {FILL_LABELS[side]}
        </StatusBadge>
      ))}
    </span>
  );
}

function PriceGapRow({ gap, columnCount }: { gap: { round: number; pct: number }; columnCount: number }) {
  return (
    <tr>
      <td colSpan={columnCount} style={{ padding: "clamp(5px, calc(1dvh - 4px), 9px) 8px", textAlign: "center" }}>
        <span style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
          {gap.pct === 0 ? (
            <>{gap.round}호가 기준선에 있습니다</>
          ) : (
            <>
              {gap.round}호에서 <b style={{ color: signColor(gap.pct) }}>{formatSigned(gap.pct)}</b> 더{" "}
              {gap.pct > 0 ? "올라가야" : "내려가야"} 기준선
            </>
          )}
        </span>
      </td>
    </tr>
  );
}

function RecentPriceRow({ index, columnCount, kind }: { index: IndexStatus; columnCount: number; kind: "high" | "low" }) {
  const currency = useTradeCurrency();
  const [label, price, date, color] = kind === "high"
    ? [index.recent_high_label, index.recent_high, index.recent_high_date, SELL_BADGE_COLOR] as const
    : [index.recent_low_label, index.recent_low, index.recent_low_date, BUY_BADGE_COLOR] as const;
  return (
    <tr>
      <td colSpan={columnCount} style={{ padding: "2px 8px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, color, fontSize: "var(--fs-sm)", fontWeight: 700 }}>
          <span style={{ flex: 1, borderTop: "2px dashed currentColor", opacity: 0.55 }} />
          <span>
            {index.name} {label} {formatPrice(price, currency)} ({date})
          </span>
          <span style={{ flex: 1, borderTop: "2px dashed currentColor", opacity: 0.55 }} />
        </div>
      </td>
    </tr>
  );
}

/** 종목 입력 — 티커를 적고 [확인]으로 그 종목풀의 ETF 인지 확인한다(레버리지 설정과 같은 방식).
 *  확인된 티커만 `onChange` 로 올라가고, 글자를 고치면 다시 확인할 때까지 null 이다. */
/** 설정 표의 한 행 — 전략 하나의 편집 초안. 숫자는 입력 중 상태를 담으려 문자열이다. */
type DraftRow = {
  key: string;
  strategy_id: string | null;
  ticker: string;
  name: string;
  pct: string;
  amount: string;
  rounds: string;
};

let draftKeySeq = 0;
function nextDraftKey(): string {
  draftKeySeq += 1;
  return `row-${draftKeySeq}`;
}

function rowFromConfig(config: StrategyConfigRow): DraftRow {
  return {
    key: config.strategy_id,
    strategy_id: config.strategy_id,
    ticker: config.ticker,
    name: config.name,
    pct: String(config.trigger_pct),
    amount: String(config.round_size),
    rounds: String(config.rounds),
  };
}

/** 변경 비교용 — 입력 중 문자열을 저장 값 모양으로 맞춘다. */
function comparableDrafts(rows: DraftRow[]) {
  return rows.map((row) => [
    row.strategy_id, row.ticker, Number(row.pct), Number(row.amount), Number(row.rounds),
  ]);
}

type TradeListRow = DraftRow & {
  held: boolean;
  heldRounds: string;
  roundQuantity?: number | null;
  is_label?: boolean;
  /** 상태(도달·슬랙·주문)가 있는 회차 — 아래 회차 표의 상태 칸과 같은 값. */
  statusRounds?: RoundRow[];
} & Partial<HoldingsRow>;
const tradeGridTheme = createAppGridTheme();
const { weight: tradeWeightColumn, dailyChange, currentPrice, quantity, returnPct, pnl, valuation } = holdingMetricColumns<TradeListRow>(true);
const TOTAL_ROW_KEY = "__total__";
const TRADE_TOTAL_ROW_KEY = "__trade_total__";
const OTHER_HOLDINGS_TICKER = "__OTHER_HOLDINGS__";
const EMPTY_GROUPS: HoldingsGroup[] = [];

function sumTradeHoldings(rows: HoldingsRow[]) {
  const totals = rows.reduce((sum, row) => {
    sum.valuation_krw += row.valuation_krw;
    sum.buyAmount += row.buy_amount_krw;
    sum.pnl_krw += row.pnl_krw;
    return sum;
  }, { valuation_krw: 0, buyAmount: 0, pnl_krw: 0 });
  return {
    valuation_krw: totals.valuation_krw,
    pnl_krw: totals.pnl_krw,
    return_pct: totals.buyAmount > 0 ? (totals.pnl_krw / totals.buyAmount) * 100 : undefined,
  };
}

function displayWeights(values: number[], total: number): number[] {
  // 소수점 둘째 자리에서 생기는 반올림 차이는 나머지가 큰 행부터 배분한다.
  const parts = values.map((value, index) => {
    const exact = (value / total) * 10_000;
    const floor = Math.floor(exact);
    return { index, floor, remainder: exact - floor };
  });
  const weights = parts.map(({ floor }) => floor);
  const remaining = 10_000 - weights.reduce((sum, weight) => sum + weight, 0);
  parts.sort((a, b) => b.remainder - a.remainder).slice(0, remaining).forEach(({ index }) => { weights[index] += 1; });
  return weights.map((weight) => weight / 100);
}

function tradeListLabel(key: string, name: string): TradeListRow {
  return { key, ticker: "", name, strategy_id: null, pct: "", amount: "", rounds: "", held: false, heldRounds: "", is_label: true };
}

const tradeSettingColumns = [
  { field: "pct", headerName: "간격(%)", headerTooltip: "매매 간격 (%)", width: 78 },
  { field: "rounds", headerName: "회차 수", width: 76 },
  { field: "amount", headerName: "회차금액(만원)", width: 120 },
] as const;

function AccountSettingsCard({
  view,
  accountId,
  onAccountChange,
  onSaved,
  onWithdrawSignal,
}: {
  view: View;
  accountId: string;
  onAccountChange: (accountId: string) => void;
  onSaved: (next: View) => void;
  onWithdrawSignal: (signal: SlackSignal, label: string) => void;
}) {
  const toast = useToast();
  const currency = useTradeCurrency();
  const [drafts, setDrafts] = useState<DraftRow[]>(() => view.configs.map(rowFromConfig));
  const [groups, setGroups] = useState<HoldingsGroup[]>(view.groups ?? EMPTY_GROUPS);
  const [groupDraft, setGroupDraft] = useState<HoldingsGroupDraft | null>(null);
  const [sortActive, setSortActive] = useState(false);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [checkedKeys, setCheckedKeys] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [selection, setSelection] = useState<{ accountId: string; ticker: string } | null>(null);
  const [slack, setSlack] = useState(view.slack_enabled);
  const [slackTesting, setSlackTesting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [accountAssets, setAccountAssets] = useState<{ summary: AccountSummary; holdings: Map<string, HoldingsRow> } | null>(null);
  const [memos, setMemos] = useState(view.memos);
  const saveMemo = useStockMemoSave(
    useCallback((ticker: string, memo: string) => setMemos((prev) => ({ ...prev, [ticker]: memo })), []),
  );

  // 서버가 최신 값을 돌려주면(저장·계좌 전환) 초안도 그 값으로 맞춘다.
  useEffect(() => {
    setDrafts(view.configs.map(rowFromConfig));
    setGroups(view.groups ?? EMPTY_GROUPS);
    setPickerOpen(false);
    setSlack(view.slack_enabled);
    setMemos(view.memos);
  }, [view]);

  useEffect(() => {
    const controller = new AbortController();
    const errorMessage = "계좌 보유 지표를 불러오지 못했습니다.";
    async function loadHoldings() {
      try {
        const response = await fetch(`/api/assets?account=${encodeURIComponent(accountId)}`, {
          cache: "no-store",
          signal: controller.signal,
        });
        const payload = (await response.json()) as HoldingsResponse;
        const summary = payload.account_summaries?.find((row) => row.account_id === accountId);
        if (!response.ok || !payload.rows || !summary) {
          throw new Error(payload.error ?? errorMessage);
        }
        if (controller.signal.aborted) return;
        const rows = payload.rows.filter((row) => row.account_id === accountId);
        setAccountAssets({ summary, holdings: new Map(rows.map((row) => [row.ticker, row])) });
      } catch (error) {
        if (controller.signal.aborted) return;
        setAccountAssets(null);
        toast.error(error instanceof Error ? error.message : errorMessage);
      }
    }
    void loadHoldings();
    return () => controller.abort();
  }, [accountId, view, toast]);

  const isDirty =
    slack !== view.slack_enabled ||
    JSON.stringify(groups) !== JSON.stringify(view.groups ?? EMPTY_GROUPS) ||
    JSON.stringify(comparableDrafts(drafts)) !== JSON.stringify(comparableDrafts(view.configs.map(rowFromConfig)));

  const poolLabel = useCallback(
    (pool: string) => {
      const option = view.pool_options.find((item) => item.ticker_type === pool);
      return option ? formatPoolLabel(option) : pool;
    },
    [view.pool_options],
  );
  const updateRow = useCallback((key: string, patch: Partial<DraftRow>) => {
    setDrafts((prev) => prev.map((row) => (row.key === key ? { ...row, ...patch } : row)));
  }, []);

  const query = search.trim().toLowerCase();
  const usedTickers = new Set(drafts.map((row) => row.ticker));
  const candidates = pickerOpen
    ? view.ticker_options.filter(
        (option) =>
          !usedTickers.has(option.ticker) &&
          (option.ticker.toLowerCase().includes(query) || option.name.toLowerCase().includes(query)),
      )
    : [];
  const selectedRow =
    (selection?.accountId === accountId ? drafts.find((row) => row.ticker === selection.ticker) : undefined) ??
    drafts[0];
  const strategiesById = useMemo(
    () => new Map(view.strategies.map((strategy) => [strategy.strategy_id, strategy])),
    [view.strategies],
  );
  const selectedStrategy = selectedRow?.strategy_id ? strategiesById.get(selectedRow.strategy_id) : undefined;
  const details = selectedStrategy?.ticker === selectedRow?.ticker ? selectedStrategy : undefined;
  const quotesByTicker = useMemo(() => new Map(view.configs.map((config) => [config.ticker, config])), [view.configs]);
  const { gridRows, pinnedRows } = useMemo(() => {
    const stockRows = drafts.map((row): TradeListRow => {
      const strategy = row.strategy_id ? strategiesById.get(row.strategy_id) : undefined;
      const quote = quotesByTicker.get(row.ticker);
      const price = quote?.price ?? null;
      return {
        ...accountAssets?.holdings.get(row.ticker),
        ...row,
        memo: memos[row.ticker] ?? "",
        statusRounds: strategy && strategy.ticker === row.ticker && !("error" in strategy) ? strategy.rounds.filter(hasRoundStatus) : [],
        current_price: price == null ? undefined : String(price),
        daily_change_pct: quote?.change_pct,
        currency,
        // 현재가 기준 회차 수량 — 서버 `round_buy_quantity` 와 같은 식(회차 금액 ÷ (현재가 × 환율), 내림).
        roundQuantity:
          price != null && view.krw_rate != null && Number.isSafeInteger(Number(row.amount))
            ? Math.floor((Number(row.amount) * 10_000) / (price * view.krw_rate))
            : null,
        held: Boolean(strategy && strategy.ticker === row.ticker && !("error" in strategy) && strategy.status.held_quantity > 0),
        heldRounds: !strategy || strategy.ticker !== row.ticker
          ? "미저장"
          : "error" in strategy ? "오류" : `${strategy.status.held_count}/${strategy.config.rounds}`,
      };
    });
    const summary = accountAssets?.summary;
    const stockValues = stockRows.map((row) => row.valuation_krw ?? 0);
    const strategyTickers = new Set(drafts.map((row) => row.ticker));
    const holdings = accountAssets && [...accountAssets.holdings.values()];
    const totalHoldings = holdings ? sumTradeHoldings(holdings) : undefined;
    const otherHoldings = holdings ? sumTradeHoldings(holdings.filter((row) => !strategyTickers.has(row.ticker))) : undefined;
    const tradeHoldings = holdings ? sumTradeHoldings(holdings.filter((row) => strategyTickers.has(row.ticker))) : undefined;
    const otherValue = otherHoldings?.valuation_krw;
    const weights = summary && summary.total_assets_krw > 0 && otherValue != null
      ? displayWeights([summary.cash_balance_krw, otherValue, ...stockValues], summary.total_assets_krw)
      : [];
    const pinnedRows: TradeListRow[] = [
      {
        ...tradeListLabel(TOTAL_ROW_KEY, "합계"),
        pnl_krw: totalHoldings?.pnl_krw,
        return_pct: totalHoldings?.return_pct,
        valuation_krw: summary?.total_assets_krw,
        weight_pct: weights.length ? 100 : undefined,
      },
      { ...tradeListLabel(CASH_ROW_TICKER, "현금"), ticker: CASH_ROW_TICKER, valuation_krw: summary?.cash_balance_krw, weight_pct: weights[0] },
      {
        ...tradeListLabel(OTHER_HOLDINGS_TICKER, "기타 종목"),
        ...otherHoldings,
        weight_pct: weights[1],
      },
      {
        ...tradeListLabel(TRADE_TOTAL_ROW_KEY, "사고팔기 합계"),
        ...tradeHoldings,
        weight_pct: weights.length ? weights.slice(2).reduce((sum, weight) => sum + weight, 0) : undefined,
      },
    ];
    const weightedRows = stockRows.map((row, index) => ({ ...row, weight_pct: weights[index + 2] }));
    if (sortActive || !groups.length) return { gridRows: weightedRows, pinnedRows };

    const tickers = new Set(weightedRows.map((row) => row.ticker));
    const byAnchor = new Map<string | null, TradeListRow[]>();
    for (const group of groups) {
      const anchor = group.before_ticker && tickers.has(group.before_ticker) ? group.before_ticker : null;
      const groupRows = byAnchor.get(anchor) ?? [];
      groupRows.push({ ...tradeListLabel(group.id, group.name), is_group: true });
      byAnchor.set(anchor, groupRows);
    }
    return {
      gridRows: [...weightedRows.flatMap((row) => [...(byAnchor.get(row.ticker) ?? []), row]), ...(byAnchor.get(null) ?? [])],
      pinnedRows,
    };
  }, [accountAssets, drafts, groups, sortActive, strategiesById, quotesByTicker, currency, view.krw_rate, memos]);

  const tradeListColumns = useMemo<ColDef<TradeListRow>[]>(() => [
    gridRowDragColumn<TradeListRow>(({ node }) => !node.rowPinned, 36),
    tickerColumn<TradeListRow>(),
    stockNameColumn<TradeListRow>({
      cellRenderer: ({ data, value }) => data?.is_group ? (
        <HoldingsGroupName
          name={data.name}
          disabled={saving}
          onEdit={() => setGroupDraft({ id: data.key, name: data.name })}
          onDelete={() => setGroups((current) => current.filter((group) => group.id !== data.key))}
        />
      ) : renderStockNameCell(value, { ticker: data?.ticker, isHeld: data?.held }),
    }),
    // 종목 메모 — 순위·전략·자산 관리 화면과 같은 값(종목에 붙는다). 셀을 벗어나면 저장.
    stockMemoColumn<TradeListRow>({
      field: "memo",
      editable: (row) => Boolean(row && !row.is_label),
      onSave: (row, memo) => void saveMemo(row.ticker, memo),
    }),
    {
      colId: "roundStatus",
      headerName: "상태",
      headerTooltip: "아래 회차 표의 상태 — 상태가 있는 회차만 회차 번호와 함께 보인다",
      // 도달·슬랙·주문 뱃지가 한 회차에 같이 붙는 경우까지 들어가는 폭 — 넘치면 옆 칸을 덮지 않고 잘라낸다.
      width: 240,
      sortable: false,
      cellStyle: { overflow: "hidden" },
      cellRenderer: ({ data }: { data?: TradeListRow }) => (
        <span style={{ display: "inline-flex", gap: 6, alignItems: "center", whiteSpace: "nowrap" }}>
          {(data?.statusRounds ?? []).map((row) => (
            <span key={row.round} style={{ display: "inline-flex", gap: 3, alignItems: "center" }}>
              {row.round}호
              <RoundStatus row={row} onWithdrawSignal={onWithdrawSignal} />
            </span>
          ))}
        </span>
      ),
    },
    dailyChange,
    currentPrice,
    tradeWeightColumn,
    { field: "heldRounds", headerName: "보유", headerTooltip: "보유 회차 / 저장된 전체 회차", width: 64, type: "rightAligned" },
    ...tradeSettingColumns.flatMap((column): ColDef<TradeListRow>[] => [
      {
        ...column,
        type: "rightAligned",
        editable: ({ node, data }) => !saving && !node.rowPinned && !data?.is_group,
        cellClass: ({ node, data }) => !saving && !node.rowPinned && !data?.is_group ? "appEditableCell" : undefined,
      },
      // 회차금액 옆 읽기 전용 — 지금 현재가로 그 금액이 몇 주인지.
      ...(column.field === "amount"
        ? [{
            field: "roundQuantity",
            headerName: "회차수량",
            headerTooltip: "회차금액 ÷ (현재가 × 환율), 내림 — 현재가 기준 참고값",
            width: 88,
            type: "rightAligned",
            valueFormatter: ({ value, node }) => (node?.rowPinned || value == null ? "" : `${formatNumber(value)}주`),
          } satisfies ColDef<TradeListRow>]
        : []),
    ]),
    { ...quantity, headerName: "보유수량", width: 88 },
    returnPct, pnl, valuation,
  ], [saving, saveMemo, onWithdrawSignal]);

  const handleRowDragEnd = ({ api }: RowDragEndEvent<TradeListRow>) => {
    // 그리드가 드롭을 반영한 다음 종목 순서와 그룹 위치를 함께 읽는다.
    queueMicrotask(() => {
      if (api.isDestroyed()) return;
      const orderedRows: TradeListRow[] = [];
      api.forEachNodeAfterFilterAndSort(({ data, rowPinned }) => {
        if (data && !rowPinned) orderedRows.push(data);
      });
      setDrafts((rows) => {
        const byKey = new Map(rows.map((row) => [row.key, row]));
        const ordered = orderedRows.flatMap(({ key }) => {
          const row = byKey.get(key);
          return row ? [row] : [];
        });
        return ordered.length === rows.length ? ordered : rows;
      });
      setGroups((current) => {
        const byId = new Map(current.map((group) => [group.id, group]));
        let nextTicker: string | null = null;
        const ordered: HoldingsGroup[] = [];
        for (let index = orderedRows.length - 1; index >= 0; index -= 1) {
          const row = orderedRows[index];
          if (!row.is_group) {
            nextTicker = row.ticker;
            continue;
          }
          const group = byId.get(row.key);
          if (group) ordered.push({ ...group, before_ticker: nextTicker });
        }
        return ordered.length === current.length ? ordered.reverse() : current;
      });
    });
  };

  const selectTicker = (option: TickerOption) => {
    if (!pickerOpen || usedTickers.has(option.ticker)) return;
    const key = nextDraftKey();
    setDrafts((prev) => [
      ...prev,
      { key, strategy_id: null, ticker: option.ticker, name: option.name, pct: "5", amount: String(view.default_round_size), rounds: "4" },
    ]);
    setSelection({ accountId, ticker: option.ticker });
    setPickerOpen(false);
  };

  const deleteChecked = () => {
    const checked = new Set(checkedKeys);
    const deletedTickers = new Set(drafts.filter((row) => checked.has(row.key)).map((row) => row.ticker));
    setGroups((current) => current.map((group) => {
      if (!group.before_ticker || !deletedTickers.has(group.before_ticker)) return group;
      const anchor = drafts.findIndex((row) => row.ticker === group.before_ticker);
      const nextTicker = drafts.slice(anchor + 1).find((row) => !checked.has(row.key))?.ticker ?? null;
      return { ...group, before_ticker: nextTicker };
    }));
    setDrafts((rows) => rows.filter((row) => !checked.has(row.key)));
    setCheckedKeys([]);
  };

  const handleAccountChange = useCallback(
    (next: string) => {
      if (isDirty && !window.confirm("저장하지 않은 변경이 사라집니다. 계좌를 바꿀까요?")) return;
      onAccountChange(next);
    },
    [isDirty, onAccountChange],
  );

  const save = useCallback(async () => {
    const strategies = [];
    for (const row of drafts) {
      const label = row.name || row.ticker || "새 전략";
      const pct = Number(row.pct);
      const amount = Number(row.amount);
      const rounds = Number(row.rounds);
      if (!Number.isFinite(pct) || pct < PCT_MIN || pct > PCT_MAX) {
        return void toast.error(`${label}: 매매 간격은 ${PCT_MIN}~${PCT_MAX} 사이 숫자여야 합니다.`);
      }
      if (!Number.isSafeInteger(amount) || !Number.isSafeInteger(amount * 10_000) || amount < 1) return void toast.error(`${label}: 회차금액(만원)은 1 이상의 정수여야 합니다.`);
      if (!Number.isInteger(rounds) || rounds < 1 || rounds > 12) {
        return void toast.error(`${label}: 회차 수는 1~12 사이 정수여야 합니다.`);
      }
      strategies.push({
        ...(row.strategy_id ? { strategy_id: row.strategy_id } : {}),
        ticker: row.ticker,
        trigger_pct: pct,
        round_size: amount,
        rounds,
      });
    }
    setSaving(true);
    try {
      const resp = await fetch("/api/strategy-trade", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account_id: accountId, slack_enabled: slack, strategies, groups }),
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "설정을 저장하지 못했습니다.");
      onSaved(payload as View);
      toast.success("사고팔기 설정을 저장했습니다.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "설정을 저장하지 못했습니다.");
    } finally {
      setSaving(false);
    }
  }, [accountId, drafts, groups, onSaved, slack, toast]);

  const sendSlackTest = useCallback(async () => {
    setSlackTesting(true);
    try {
      const resp = await fetch("/api/strategy-trade/slack-test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account_id: accountId }),
      });
      const data = (await resp.json()) as { sent?: boolean; count?: number; error?: string };
      if (!resp.ok || data.error || !data.sent) throw new Error(data.error ?? "발송에 실패했습니다.");
      toast.success(`슬랙 발송 완료 (신호 ${data.count ?? 0}건)`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "슬랙 테스트 발송에 실패했습니다.");
    } finally {
      setSlackTesting(false);
    }
  }, [accountId, toast]);

  return (
    <>
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBodyTight">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <AccountSelect
                  accounts={view.account_options}
                  value={accountId}
                  onChange={handleAccountChange}
                  disabled={saving}
                />
                <SlackAlarmField
                  checked={slack}
                  onChange={setSlack}
                  disabled={saving}
                  title="이 계좌 전략의 매수·매도 신호를 슬랙으로 보낸다 (장중 10분 간격 감시). 저장 버튼을 눌러야 반영된다."
                  testing={slackTesting}
                  onTest={() => void sendSlackTest()}
                />
              </div>
              <div className="appMainHeaderRight">
                <UnsavedChangesBadge show={isDirty} />
                <HoldingsGroupButton disabled={saving} onAdd={() => setGroupDraft({ id: null, name: "" })} />
                <GridToolbarButton
                  variant="add"
                  disabled={saving || view.ticker_options.length === 0}
                  onClick={() => {
                    setSearch("");
                    setPickerOpen(true);
                  }}
                />
                <GridToolbarButton variant="save" disabled={saving || !isDirty} onClick={() => void save()}>
                  {saving ? "저장 중…" : "저장"}
                </GridToolbarButton>
                <GridToolbarButton variant="delete" disabled={saving || checkedKeys.length === 0} onClick={deleteChecked} />
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="appSection appPageStack strategyTradeGrid">
        <aside className="card appCard" aria-label="종목 목록">
          <div className="card-body appCardBodyTight">
            <AppAgGrid
              rowData={gridRows}
              columnDefs={tradeListColumns}
              theme={tradeGridTheme}
              height="auto"
              minHeight={0}
              getRowId={({ data }) => data.key}
              getRowClass={({ data }) => [
                data?.is_group || data?.key === TOTAL_ROW_KEY || data?.key === TRADE_TOTAL_ROW_KEY ? "assetsGroupRow" : "",
                stockRowClass({ held: data?.held }),
                data?.key === selectedRow?.key ? "strategyTradeSelectedRow" : "",
              ].filter(Boolean).join(" ")}
              rowClassDeps={[selectedRow?.key]}
              gridOptions={{
                domLayout: "autoHeight",
                rowHeight: 36,
                headerHeight: 34,
                suppressMovableColumns: true,
                suppressCellFocus: false,
                stopEditingWhenCellsLoseFocus: true,
                readOnlyEdit: true,
                pinnedTopRowData: pinnedRows,
                suppressOverlays: ["noRows"],
                rowSelection: {
                  mode: "multiRow", headerCheckbox: true, enableClickSelection: false, hideDisabledCheckboxes: true,
                  isRowSelectable: (node) => Boolean(node.data && !node.rowPinned && !node.data.is_group),
                },
                selectionColumnDef: gridSelectionColumn,
                rowDragManaged: true,
                suppressMoveWhenRowDragging: true,
                suppressRowDrag: saving || sortActive,
                onSortChanged: ({ api }) => setSortActive(api.getColumnState().some((column) => column.sort != null)),
                onRowDragEnd: handleRowDragEnd,
                onSelectionChanged: ({ api }) => setCheckedKeys(api.getSelectedRows().map((row) => row.key)),
                onCellEditRequest: ({ data, colDef, newValue }) => {
                  const field = colDef.field;
                  if (field === "pct" || field === "amount" || field === "rounds") {
                    updateRow(data.key, { [field]: String(newValue ?? "").trim() });
                  }
                },
                onRowClicked: ({ data, node }) => {
                  if (data && !node.rowPinned && !data.is_group) setSelection({ accountId, ticker: data.ticker });
                },
                onCellKeyDown: ({ data, event, node }) => {
                  if (data && !node.rowPinned && !data.is_group && event instanceof KeyboardEvent && event.key === "Enter") {
                    setSelection({ accountId, ticker: data.ticker });
                  }
                },
              }}
            />
          </div>
        </aside>
        <div aria-label="선택 종목 상세">
          {details ? (
            <StrategySection
              key={details.strategy_id}
              view={details}
              poolLabel={poolLabel(details.pool)}
              onWithdrawSignal={onWithdrawSignal}
            />
          ) : (
            <div className="card appCard"><div className="card-body appCardBodyTight tableMuted">
              {selectedRow ? "전략을 저장하면 운용 상세가 표시됩니다." : "헤더에서 종목을 추가하세요."}
            </div></div>
          )}
        </div>
      </section>

      <HoldingsGroupDialog
        draft={groupDraft}
        onChange={setGroupDraft}
        onSave={({ id, name }) => {
          if (id) {
            setGroups((current) => current.map((group) => group.id === id ? { ...group, name } : group));
          } else {
            const group = { id: `group-${Date.now()}`, name, before_ticker: drafts[0]?.ticker ?? null };
            setGroups((current) => [...current, group]);
          }
        }}
      />
      <AppModal open={pickerOpen} title="한국 종목 선택" onClose={() => setPickerOpen(false)}>
        <input
          className="form-control mb-2"
          aria-label="종목명 또는 티커"
          placeholder="종목명 또는 티커 검색"
          value={search}
          autoFocus
          onChange={(event) => setSearch(event.target.value)}
        />
        <div className="list-group" style={{ maxHeight: "50vh", overflowY: "auto" }}>
          {candidates.slice(0, 50).map((option) => (
            <button key={option.ticker} type="button" className="list-group-item list-group-item-action" onClick={() => selectTicker(option)}>
              <strong>{option.ticker}</strong> {option.name}
              <div className="tableMuted">{option.kind === "stock" ? "개별주" : "ETF"} · {option.pools.map(poolLabel).join(" · ")}</div>
            </button>
          ))}
        </div>
        {candidates.length === 0 ? <p className="tableMuted mb-0">선택할 종목이 없습니다.</p> : null}
        {candidates.length > 50 ? <p className="tableMuted mb-0 mt-2">검색어를 입력하면 나머지 종목을 찾을 수 있습니다.</p> : null}
      </AppModal>
    </>
  );
}

/** 선택한 전략의 운용 현황과 회차 표. */
function StrategySection({
  view,
  poolLabel,
  onWithdrawSignal,
}: {
  view: StrategyView | FailedStrategy;
  /** 전략의 가격 출처 표기. */
  poolLabel: string;
  onWithdrawSignal: (signal: SlackSignal, label: string) => void;
}) {
  const currency = useTradeCurrency();
  if ("error" in view) {
    return (
      <div className="card appCard" style={cardStyle}>
        <span style={{ fontWeight: 800, fontSize: "var(--fs-base)" }}>{view.label}</span>
        <span style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
          {poolLabel}
        </span>
        <div style={{ color: "var(--up-color, #d64545)", fontSize: "var(--fs-sm)" }}>
          {view.ticker} 보유 회차를 맞출 수 없습니다: {view.error}
        </div>
      </div>
    );
  }
  const { status, index } = view;
  const tradeDays = [
    ...new Set(Object.values(view.trades_by_position).flatMap((byDay) => Object.keys(byDay))),
  ].sort();
  const columnCount = 6 + tradeDays.length;
  const cycleEnds = new Set(view.cycle_end_days);
  // 사이클이 끝난 날 칸 오른쪽에 경계선 — 그 뒤 체결은 새 사이클이다.
  const cycleEndStyle = (day: string): React.CSSProperties =>
    cycleEnds.has(day) ? { borderRight: "2px solid var(--text-muted)" } : {};
  const today = new Date().toLocaleDateString("sv-SE", { timeZone: "Asia/Seoul" });
  // 사다리 위치마다 한 행 — 회차가 없는 위치(건너뛴 칸)도 그 가격대의 매매와 함께 행으로 둬 간격을 보여 준다.
  const positions = [...view.rounds.map((row) => row.position), ...Object.keys(view.trades_by_position).map(Number)];
  const firstPosition = Math.min(...positions);
  const lastPosition = Math.max(...positions);
  const slots = Array.from({ length: lastPosition - firstPosition + 1 }, (_, idx) => ({
    position: firstPosition + idx,
    row: view.rounds.find((item) => item.position === firstPosition + idx) ?? null,
    trades: view.trades_by_position[String(firstPosition + idx)] ?? {},
  }));
  const firstRoundPrice = view.rounds.find((row) => row.round === 1)?.buy_index;
  const aboveFirstRound = firstRoundPrice != null && firstRoundPrice > 0 && index.recent_high > 0
    ? { round: 1, pct: Number(((index.recent_high / firstRoundPrice - 1) * 100).toFixed(2)) }
    : null;
  // 저점이 회차 범위 아래에 있으면 마지막 회차에서 저점까지의 하락률을 표시한다.
  const lastLeveledRound = [...view.rounds].reverse().find((row) => row.buy_index != null) ?? null;
  const nextRow = view.rounds.find((row) => row.is_next) ?? null;
  const belowLastRound =
    lastLeveledRound?.buy_index != null && lastLeveledRound.buy_index > 0
      ? {
          round: lastLeveledRound.round,
          pct: Number(((index.recent_low / lastLeveledRound.buy_index - 1) * 100).toFixed(2)),
        }
      : null;

  return (
    <div className="card appCard" style={cardStyle}>
      <div style={{ display: "flex", gap: "4px 10px", flexWrap: "wrap", alignItems: "baseline", fontSize: "var(--fs-sm)" }}>
        <span style={{ fontWeight: 800, fontSize: "var(--fs-base)" }}>{view.label}</span>
        <span>
          현재가 <b style={{ fontSize: "var(--fs-lg)", fontWeight: 800 }}>{formatPrice(index.close, currency)}</b>
        </span>
        {status.held_quantity > 0 ? (
          <span>
            평균단가 <b>{formatPrice(status.average_price, currency)}</b>{" "}
            <b style={{ color: signColor(status.average_profit_pct) }}>{formatSigned(status.average_profit_pct)}</b>
          </span>
        ) : null}
        <span style={{ color: "var(--text-muted)" }}>
          체결 이력 {view.ledger_at.slice(5, 16).replace("T", " ")} 기준
        </span>
        {status.waiting_first_entry ? (
          <span style={{ fontWeight: 700 }}>
            1호 진입 대기 — {status.first_entry_time} 매수 예정
          </span>
        ) : status.next_round == null ? (
          <span style={{ fontWeight: 700, color: "var(--text-muted)" }}>회차 소진 — 매도 대기</span>
        ) : null}
        {status.budget_short ? (
          <span style={actionBadgeStyle("var(--text-muted)")}>⏸ {status.next_round}호 회차 금액으로 1주 매수 불가</span>
        ) : status.cash_short ? (
          <span style={actionBadgeStyle("var(--text-muted)")}>⏸ {status.next_round}호 한도 소진 — 현금 부족</span>
        ) : nextRow?.close_entry ? (
          <span style={actionBadgeStyle(BUY_BADGE_COLOR)}>📉 {status.next_round}호 {status.first_entry_time} 매수!</span>
        ) : status.next_buy_drop_pct != null ? (
          <span style={actionBadgeStyle(BUY_BADGE_COLOR)}>
            {status.next_buy_drop_pct >= 0
              ? status.sell_pending
                ? `📉 ${status.next_round}호 매수가 도달 — ${status.next_sell_round}호 매도 체결 뒤 매수`
                : `📉 ${status.next_round}호 매수가 도달!`
              : `📉 ${status.next_round}호 매수까지 ${Math.abs(status.next_buy_drop_pct).toFixed(2)}% 하락`}
          </span>
        ) : null}
        {status.next_sell_rise_pct != null ? (
          <span style={actionBadgeStyle(SELL_BADGE_COLOR)}>
            {status.next_sell_rise_pct <= 0
              ? `📈 ${status.next_sell_round}호 매도가 도달!`
              : `📈 ${status.next_sell_round}호 매도까지 ${status.next_sell_rise_pct.toFixed(2)}% 상승`}
          </span>
        ) : null}
      </div>

      {view.arbitrary_lots.map((lot) => (
        <div key={lot.id} style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", fontSize: "var(--fs-sm)" }}>
          <StatusBadge color={FILL_COLORS.buy}>임의 매수</StatusBadge>
          <span>{lot.bought_on} · {formatCurrencyPrice(lot.fill_price, currency)} · {formatNumber(lot.quantity)}주</span>
          <span style={{ color: SELL_BADGE_COLOR }}>임의 매도 목표 {formatCurrencyPrice(lot.sell_limit, currency)}</span>
        </div>
      ))}

      {/* 회차 표 */}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "auto", borderCollapse: "collapse", fontSize: "var(--fs-sm)", lineHeight: 1.25 }}>
          <colgroup>
            {ROUND_TABLE_FIXED_WIDTHS.map((width, idx) => (
              <col key={idx} style={{ width, minWidth: width }} />
            ))}
            {tradeDays.map((day) => (
              <col key={day} style={{ minWidth: ROUND_TABLE_DAY_MIN_WIDTH }} />
            ))}
          </colgroup>
          <thead>
            <tr style={{ color: "var(--text-muted)" }}>
              <th style={{ ...thStyle, textAlign: "left" }}>위치</th>
              <th style={{ ...thStyle, textAlign: "left" }}>회차</th>
              <th style={{ ...thStyle, textAlign: "left" }}>상태</th>
              <th style={thStyle}>위치 가격</th>
              <th style={thStyle}>체결가</th>
              <th style={thStyle}>손익</th>
              {tradeDays.map((day) => (
                <th
                  key={day}
                  title={cycleEnds.has(day) ? `${day} · 보유가 모두 비었습니다(사이클 종료)` : day}
                  style={{
                    ...thStyle,
                    textAlign: "center",
                    whiteSpace: "nowrap",
                    minWidth: ROUND_TABLE_DAY_MIN_WIDTH,
                    ...(day === today ? { color: TODAY_COLOR, fontWeight: 800 } : null),
                    ...cycleEndStyle(day),
                  }}
                >
                  {Number(day.slice(5, 7))}월 {Number(day.slice(8, 10))}일({WEEKDAY_NAMES[new Date(`${day}T00:00:00Z`).getUTCDay()]})
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <RecentPriceRow index={index} columnCount={columnCount} kind="high" />
            {slots.map(({ position, row, trades }, rowIdx) => {
              const tradeCells = tradeDays.map((day) => (
                <td key={day} style={{ ...tdStyle, textAlign: "center", ...cycleEndStyle(day) }}>
                  {trades[day] ? <DailyTradeBadges sides={trades[day]} /> : null}
                </td>
              ));
              if (row === null) {
                return (
                  <tr key={`empty-${position}`} style={{ borderTop: "1px solid rgba(148,163,184,0.15)" }}>
                    <td style={{ ...tdStyle, textAlign: "left", color: "var(--text-muted)" }}>{position}</td>
                    <td colSpan={2} style={tdStyle} />
                    <td style={{ ...tdStyle, color: "var(--text-muted)" }}>{formatPrice(view.position_prices[String(position)], currency)}</td>
                    <td colSpan={2} style={tdStyle} />
                    {tradeCells}
                  </tr>
                );
              }
              // 최근 저점 라인: 이 행의 지수 환산이 저점 아래로 처음 내려가는 지점 앞에 긋는다.
              // (마지막 행까지 저점 위면 맨 아래에) — 값이 없는 행은 판정에서 건너뛴다.
              const prevLevels = slots
                .slice(0, rowIdx)
                .map((slot) => slot.row?.buy_index)
                .filter((v): v is number => v != null);
              const ownLevel = row.buy_index;
              const lowLineHere =
                ownLevel != null &&
                ownLevel < index.recent_low &&
                (prevLevels.length === 0 || prevLevels[prevLevels.length - 1] >= index.recent_low);
              return (
              <React.Fragment key={row.round}>
              {row.round === 1 && aboveFirstRound ? <PriceGapRow gap={aboveFirstRound} columnCount={columnCount} /> : null}
              {lowLineHere ? <RecentPriceRow index={index} columnCount={columnCount} kind="low" /> : null}
              <tr style={{ borderTop: "1px solid rgba(148,163,184,0.15)" }}>
                <td style={{ ...tdStyle, textAlign: "left", color: "var(--text-muted)" }}>{position}</td>
                <td style={{ ...tdStyle, textAlign: "left", fontWeight: 700 }}>
                  {row.round}호
                  {row.is_next ? (
                    <span style={{ marginLeft: 5, color: "var(--text-muted)", fontSize: "var(--fs-sm)", fontWeight: 400 }}>
                      다음
                    </span>
                  ) : null}
                </td>
                <td style={{ ...tdStyle, textAlign: "left" }}>
                  <RoundStatus row={row} onWithdrawSignal={onWithdrawSignal} />
                </td>
                <td style={{ ...tdStyle, color: "var(--text-muted)" }}>{formatPrice(view.position_prices[String(position)], currency)}</td>
                <td style={tdStyle}>
                  {row.held ? (
                    <>
                      <span style={{ display: "inline-flex", alignItems: "baseline", gap: 6, whiteSpace: "nowrap" }}>
                        <b>{formatPrice(row.fill_price, currency)}</b>
                        {row.bought_on != null && row.source != null ? (
                          <span style={{ color: "var(--text-muted)", fontSize: "var(--fs-xs, 11px)" }}>
                            {row.bought_on.slice(5)} · {BOUGHT_SOURCE_WORDS[row.source]}
                          </span>
                        ) : null}
                      </span>
                      {/* 팔 수 있는 회차(가장 깊은 보유)만 실제 매도 지정가 — 체결가 +간격%. */}
                      {row.round === status.held_count && row.sell_limit != null ? (
                        <div style={{ color: SELL_BADGE_COLOR, fontSize: "var(--fs-xs, 11px)", fontWeight: 700 }}>
                          매도 {formatPrice(row.sell_limit, currency)}
                          {row.rebuy_after_sell ? " 후 재매수" : ""}
                        </div>
                      ) : null}
                    </>
                  ) : row.is_next ? (
                    // 다음 회차의 실제 매수 지정가 — 마지막 체결가 −간격%(위치 가격과 조금 다를 수 있다).
                    // 보유가 없으면 가격이 없고 다음 1호 매수 시각(한국시간)을 적는다.
                    <span style={{ color: BUY_BADGE_COLOR, fontWeight: 700 }}>
                      {row.buy_limit != null
                        ? `다음 매수 ${formatPrice(row.buy_limit, currency)}`
                        : `${status.first_entry_time} 매수 예정`}
                      {row.cash_short ? (
                        <span style={{ marginLeft: 6, color: "var(--text-muted)", fontWeight: 400 }}>한도 소진</span>
                      ) : null}
                    </span>
                  ) : null}
                </td>
                <td style={{ ...tdStyle, color: signColor(row.profit_pct), fontWeight: 700 }}>
                  {formatSigned(row.profit_pct)}
                </td>
                {tradeCells}
              </tr>
              </React.Fragment>
              );
            })}
            {view.rounds.length > 0 &&
            view.rounds.every((r) => r.buy_index == null || r.buy_index >= index.recent_low) ? (
              <>
                {belowLastRound ? <PriceGapRow gap={belowLastRound} columnCount={columnCount} /> : null}
                <RecentPriceRow index={index} columnCount={columnCount} kind="low" />
              </>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function StrategyTradeClient() {
  const toast = useToast();
  const [view, setView] = useState<View | null>(null);
  const [loading, setLoading] = useState(true);
  const [withdrawing, setWithdrawing] = useState<{ key: string; label: string } | null>(null);
  const [withdrawBusy, setWithdrawBusy] = useState(false);

  const load = useCallback(
    async (accountId: string | null, remembered = false) => {
      setLoading(true);
      try {
        const query = accountId ? `?account_id=${encodeURIComponent(accountId)}` : "";
        const resp = await fetch(`/api/strategy-trade${query}`, { cache: "no-store" });
        const payload = await resp.json();
        // 기억해 둔 계좌가 더는 운영 가능한 계좌가 아니면 첫 계좌로 연다(다른 화면과 같은 방식).
        if (!resp.ok && remembered) return await load(null);
        if (!resp.ok) throw new Error(payload?.error ?? "운용 현황을 불러오지 못했습니다.");
        setView(payload as View);
        writeRememberedTradeAccountId((payload as View).account_id);
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "운용 현황을 불러오지 못했습니다.");
      } finally {
        setLoading(false);
      }
    },
    [toast],
  );

  const askWithdrawSignal = useCallback((signal: SlackSignal, label: string) => {
    setWithdrawing({ key: signal.key, label });
  }, []);

  const confirmWithdrawSignal = useCallback(async () => {
    if (!withdrawing) return;
    setWithdrawBusy(true);
    try {
      const resp = await fetch("/api/strategy-trade/withdraw-signal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: withdrawing.key }),
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "슬랙 신호를 회수하지 못했습니다.");
      toast.success("슬랙 신호를 회수했습니다.");
      setWithdrawing(null);
      await load(view?.account_id ?? null);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "슬랙 신호를 회수하지 못했습니다.");
    } finally {
      setWithdrawBusy(false);
    }
  }, [withdrawing, load, toast, view?.account_id]);

  useEffect(() => {
    // 마지막으로 고른 계좌는 브라우저에 기억한다(사고팔기 화면 전용 키).
    void load(readRememberedTradeAccountId(), true);
  }, [load]);

  if (loading && !view) {
    return (
      <PageFrame title="전략 사고팔기" fullWidth>
        <div style={{ color: "var(--text-muted)", padding: 20 }}>불러오는 중…</div>
      </PageFrame>
    );
  }
  if (!view) {
    return (
      <PageFrame title="전략 사고팔기" fullWidth>
        <div style={{ color: "var(--text-muted)", padding: 20 }}>데이터가 없습니다.</div>
      </PageFrame>
    );
  }
  if (view.account_id === null) {
    return (
      <PageFrame title="전략 사고팔기" fullWidth>
        <div style={{ color: "var(--text-muted)", padding: 20 }}>
          사고팔기를 운영할 계좌가 없습니다 — 계좌 설정에서 운영 계좌에 증권사 API(나무·메리츠)를 연동하세요.
        </div>
      </PageFrame>
    );
  }

  const currency = view.country_code ? CURRENCY_BY_COUNTRY[view.country_code] : undefined;
  if (!currency) {
    return (
      <PageFrame title="전략 사고팔기" fullWidth>
        <div style={{ color: "var(--text-muted)", padding: 20 }}>지원하지 않는 계좌 국가입니다: {view.country_code ?? "-"}</div>
      </PageFrame>
    );
  }

  return (
    <PageFrame title="전략 사고팔기" fullWidth>
     <TradeCurrencyContext.Provider value={currency}>
      <div className="appPageStack">
        <AccountSettingsCard
          key={view.account_id}
          view={view}
          accountId={view.account_id}
          onAccountChange={(next) => void load(next)}
          onSaved={setView}
          onWithdrawSignal={askWithdrawSignal}
        />
      </div>

      <AppModal
        open={withdrawing !== null}
        title="슬랙 신호 회수"
        onClose={() => setWithdrawing(null)}
        footer={
          <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, width: "100%" }}>
            <button type="button" className="btn" onClick={() => setWithdrawing(null)} disabled={withdrawBusy}>
              취소
            </button>
            <button type="button" className="btn btn-danger" onClick={() => void confirmWithdrawSignal()} disabled={withdrawBusy}>
              {withdrawBusy ? "회수 중…" : "회수"}
            </button>
          </div>
        }
      >
        <p style={{ margin: 0 }}>
          {withdrawing?.label} 메시지를 회수 한 줄로 바꾸고 그 스레드 답글과 발송 기록을 지웁니다.
          같은 메시지로 나간 다른 신호도 함께 회수되며, 이 신호로 낸 주문이 있으면 회수하지 않습니다.
        </p>
      </AppModal>
     </TradeCurrencyContext.Provider>
    </PageFrame>
  );
}
