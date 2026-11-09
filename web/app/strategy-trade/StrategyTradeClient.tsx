"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import type { ColDef } from "ag-grid-community";

import { gridSelectionColumn, stockNameColumn, stockRowClass, tickerColumn } from "@/lib/grid-cells";
import { formatPoolLabel, type PoolLabelSource } from "@/lib/pool-label";

import type { HoldingsResponse, HoldingsRow } from "../assets/assets-helpers";
import { holdingMetricColumns } from "../assets/holding-metric-columns";
import AccountSelect, { type AccountOptionBase } from "../components/AccountSelect";
import { SlackAlarmField } from "../components/SlackAlarmField";
import { readRememberedTradeAccountId, writeRememberedTradeAccountId } from "../components/account-selection";
import { AppAgGrid } from "../components/AppAgGrid";
import { createAppGridTheme } from "../components/app-grid-theme";
import { GridToolbarButton } from "../components/GridToolbarButton";
import { AppModal } from "../components/AppModal";
import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";
import { UnsavedChangesBadge } from "../components/UnsavedChangesBadge";

type StrategyConfig = {
  // 매매 간격(%) 하나로 진입·추가·매도를 모두 관리한다.
  trigger_pct: number;
  // 회차당 매수 수량(주) — 수량이 고정이어야 보유 수량으로 진입 회차 수를 알 수 있다.
  round_quantity: number;
  rounds: number;
  index_name: string;
};

type IndexStatus = {
  name: string;
  close: number;
  buy_trigger: number;
  // 최근 지수 저점(라벨은 서버가 정한다) — 회차 지정가가 어느 층에 걸리는지 가늠하는 기준선.
  recent_low: number;
  recent_low_date: string;
  recent_low_label: string;
};

type Status = {
  held_count: number;
  waiting_first_entry: boolean;
  // 보유 0 인데 장중 1호 진입선에 못 닿았을 때 1호를 사는 시각(KST, "HH:MM").
  first_entry_close_time: string;
  // 보유 수량이 회차×회차수량과 다르면 마지막 회차는 일부만 보유 중이다 — 경고 표시용.
  quantity_mismatch: boolean;
  held_quantity: number;
  // 다음 회차를 살 계좌 현금이 없어 매수를 멈췄다(한도 소진).
  cash_short: boolean;
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

type TradeSide = "buy" | "sell";

export type RoundRow = {
  /** 사다리 위치(표시용) — 1호 체결가 기준 간격 칸. 건너뛴 칸은 회차 없이 빈 행으로 보인다. */
  position: number;
  round: number;
  fixed_round: number;
  ticker: string;
  name: string;
  held: boolean;
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
  // 진입선이 아니라 장 마감 매수로 닿은 1호 — 현재가로 산다.
  close_entry: boolean;
  // 다음 회차를 살 계좌 현금이 없다 — 한도 소진, 이 회차는 사지 않는다.
  cash_short: boolean;
  sell_reached: boolean;
  // 오늘 슬랙 신호가 나간 방향(key: 지울 때 쓰는 발송 기록 키, has_ts: 슬랙 메시지 기록이 있어 지울 수 있음), 아직 끝나지 않은 주문의 방향.
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
  ticker: string;
  config: StrategyConfig;
  index: IndexStatus;
  status: Status;
  /** 사다리 위치별 · 날짜별 매수·매도(체결 순서) — 체결가에 가장 가까운 위치. 지금 사이클의 것만이다. */
  trades_by_position: Record<string, Record<string, TradeSide[]>>;
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
  round_quantity: number;
  rounds: number;
};

type TickerOption = { ticker: string; name: string; pools: string[]; kind: "stock" | "etf" };

type View = {
  /** 사고팔기를 운영할 수 있는 계좌 — 증권사 API 가 연동된 운영 계좌. */
  account_options: AccountOptionBase[];
  /** 지금 보는 계좌. 운영할 수 있는 계좌가 없으면 null. */
  account_id: string | null;
  /** 이 계좌의 사고팔기 슬랙 스위치(합성 슬랙과 별개). */
  slack_enabled: boolean;
  /** 계좌 원화 예수금(잔고 동기화 값)과 계좌 평가액 대비 비율(%) — 다음 회차를 살 수 없으면 한도 소진. */
  cash: { cash_krw: number; pct: number | null } | null;
  pool_options: PoolLabelSource[];
  ticker_options: TickerOption[];
  /** 저장된 전략 설정 — 이 계좌의 전략 전부. */
  configs: StrategyConfigRow[];
  strategies: Array<StrategyView | FailedStrategy>;
};

// 편집 파라미터는 매매 간격(trigger_pct) 하나 — 백엔드 EDITABLE_PCT_KEYS 와 동일.
const PCT_MIN = 0.1;
const PCT_MAX = 50;

const cardStyle: React.CSSProperties = {
  padding: 14,
  display: "flex",
  flexDirection: "column",
  gap: 10,
};

// 매수·매도 행동 배지 — 이 화면에서 가장 중요한 정보라 테두리로 강조한다.
// 내리면 매수 = 파란색(하락), 오르면 매도 = 빨간색(상승).
function actionBadgeStyle(color: string): React.CSSProperties {
  return {
    fontWeight: 800,
    padding: "2px 10px",
    borderRadius: 6,
    border: `1.5px solid ${color}`,
    color,
  };
}
const BUY_BADGE_COLOR = "var(--down-color, #2f6fd0)";
const SELL_BADGE_COLOR = "var(--up-color, #d64545)";

const thStyle: React.CSSProperties = { padding: "6px 8px", textAlign: "right" };
const tdStyle: React.CSSProperties = { padding: "6px 8px", textAlign: "right" };

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

/** 상태 칸 — 지정가에 닿았으면 도달(빨강), 슬랙 신호·주문이 나갔으면 방향을 붙여 매도슬랙·매수주문처럼(녹색).
 *  슬랙 뱃지는 슬랙 메시지 기록이 있으면 눌러서 지울 수 있다(주문이 있으면 서버가 막는다). */
function RoundStatus({ row, onDeleteSignal }: { row: RoundRow; onDeleteSignal: (signal: SlackSignal, label: string) => void }) {
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
            title="눌러서 이 슬랙 신호 지우기"
            onClick={() => onDeleteSignal(signal, label)}
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
        <span key={`manual-${index}`} title={`${order.price.toLocaleString()}원 · ${order.quantity.toLocaleString()}주`}>
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
        <StatusBadge key={idx} color={FILL_COLORS[side]}>
          {side === "buy" ? "매수" : "매도"}
        </StatusBadge>
      ))}
    </span>
  );
}

/** 마지막 회차와 기준선 사이의 빈 구간 — 저점까지 얼마나 더 내려가야 하는지 적는다.
 *  선만 있으면 "표 밖 어딘가 아래"라는 것만 보이고 거리가 안 보인다. */
function BelowLastGapRow({ gap, columnCount }: { gap: { round: number; pct: number }; columnCount: number }) {
  return (
    <tr>
      <td colSpan={columnCount} style={{ padding: "22px 8px", textAlign: "center" }}>
        <span style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
          {gap.round}호에서 <b style={{ color: "var(--up-color, #d64545)" }}>{formatSigned(gap.pct)}</b> 더 내려가야
          기준선
        </span>
      </td>
    </tr>
  );
}

/** 최근 저점 기준선 행 — 회차 지정가의 지수 환산이 이 아래로 내려가는 경계에 긋는다.
 *
 *  회차 사이에 들어가지 못하면(= 저점이 마지막 회차보다도 아래) 표 맨 아래에 긋고,
 *  마지막 회차에서 얼마나 더 내려가야 저점인지 `belowLast` 로 함께 적는다 —
 *  선만 보고는 "얼마나 남았나"를 알 수 없다.
 */
function RecentLowRow({ index, columnCount }: { index: IndexStatus; columnCount: number }) {
  return (
    <tr>
      <td colSpan={columnCount} style={{ padding: "2px 8px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--up-color, #d64545)", fontSize: "var(--fs-sm)", fontWeight: 700 }}>
          <span style={{ flex: 1, borderTop: "2px dashed var(--up-color, #d64545)", opacity: 0.55 }} />
          <span>
            {index.name} {index.recent_low_label} {formatNumber(index.recent_low)} ({index.recent_low_date})
          </span>
          <span style={{ flex: 1, borderTop: "2px dashed var(--up-color, #d64545)", opacity: 0.55 }} />
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
  qty: string;
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
    qty: String(config.round_quantity),
    rounds: String(config.rounds),
  };
}

/** 변경 비교용 — 입력 중 문자열을 저장 값 모양으로 맞춘다. */
function comparableDrafts(rows: DraftRow[]) {
  return rows.map((row) => [
    row.strategy_id, row.ticker, Number(row.pct), Number(row.qty), Number(row.rounds),
  ]);
}

type TradeListRow = DraftRow & { held: boolean; heldRounds: string } & Partial<HoldingsRow>;
const tradeGridTheme = createAppGridTheme();
const tradeHoldingColumns = Object.values(holdingMetricColumns<TradeListRow>(true));

const tradeSettingColumns = [
  { field: "pct", headerName: "간격(%)", headerTooltip: "매매 간격 (%)", width: 78 },
  { field: "qty", headerName: "수량(주)", headerTooltip: "회차당 수량 (주)", width: 78 },
  { field: "rounds", headerName: "회차 수", width: 76 },
] as const;

function AccountSettingsCard({
  view,
  accountId,
  onAccountChange,
  onSaved,
  onDeleteSignal,
}: {
  view: View;
  accountId: string;
  onAccountChange: (accountId: string) => void;
  onSaved: (next: View) => void;
  onDeleteSignal: (signal: SlackSignal, label: string) => void;
}) {
  const toast = useToast();
  const [drafts, setDrafts] = useState<DraftRow[]>(() => view.configs.map(rowFromConfig));
  const [pickerOpen, setPickerOpen] = useState(false);
  const [checkedKeys, setCheckedKeys] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [selection, setSelection] = useState<{ accountId: string; ticker: string } | null>(null);
  const [slack, setSlack] = useState(view.slack_enabled);
  const [slackTesting, setSlackTesting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [holdings, setHoldings] = useState<Map<string, HoldingsRow>>(() => new Map());

  // 서버가 최신 값을 돌려주면(저장·계좌 전환) 초안도 그 값으로 맞춘다.
  useEffect(() => {
    setDrafts(view.configs.map(rowFromConfig));
    setPickerOpen(false);
    setSlack(view.slack_enabled);
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
        if (!response.ok || !payload.rows || !payload.account_summaries?.some((row) => row.account_id === accountId)) {
          throw new Error(payload.error ?? errorMessage);
        }
        if (controller.signal.aborted) return;
        setHoldings(new Map(payload.rows.map((row) => [row.ticker, row])));
      } catch (error) {
        if (controller.signal.aborted) return;
        setHoldings(new Map());
        toast.error(error instanceof Error ? error.message : errorMessage);
      }
    }
    void loadHoldings();
    return () => controller.abort();
  }, [accountId, view, toast]);

  const isDirty =
    slack !== view.slack_enabled ||
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
  const gridRows = useMemo<TradeListRow[]>(() => drafts.map((row) => {
    const strategy = row.strategy_id ? strategiesById.get(row.strategy_id) : undefined;
    return {
      ...holdings.get(row.ticker),
      ...row,
      held: Boolean(strategy && strategy.ticker === row.ticker && !("error" in strategy) && strategy.status.held_quantity > 0),
      heldRounds: !strategy || strategy.ticker !== row.ticker
        ? "미저장"
        : "error" in strategy ? "오류" : `${strategy.status.held_count}/${strategy.config.rounds}`,
    };
  }), [drafts, holdings, strategiesById]);

  const tradeListColumns = useMemo<ColDef<TradeListRow>[]>(() => [
    tickerColumn<TradeListRow>(),
    stockNameColumn<TradeListRow>({ nameOptions: (row) => ({ isHeld: row?.held }) }),
    { field: "heldRounds", headerName: "보유", headerTooltip: "보유 회차 / 저장된 전체 회차", width: 64, type: "rightAligned" },
    ...tradeSettingColumns.map((column): ColDef<TradeListRow> => ({
      ...column,
      type: "rightAligned",
      editable: !saving,
      cellClass: saving ? undefined : "appEditableCell",
    })),
    ...tradeHoldingColumns,
  ], [saving]);

  const selectTicker = (option: TickerOption) => {
    if (!pickerOpen || usedTickers.has(option.ticker)) return;
    const key = nextDraftKey();
    setDrafts((prev) => [
      ...prev,
      { key, strategy_id: null, ticker: option.ticker, name: option.name, pct: "5", qty: "", rounds: "4" },
    ]);
    setSelection({ accountId, ticker: option.ticker });
    setPickerOpen(false);
  };

  const deleteChecked = () => {
    const checked = new Set(checkedKeys);
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
      const qty = Number(row.qty);
      const rounds = Number(row.rounds);
      if (!Number.isFinite(pct) || pct < PCT_MIN || pct > PCT_MAX) {
        return void toast.error(`${label}: 매매 간격은 ${PCT_MIN}~${PCT_MAX} 사이 숫자여야 합니다.`);
      }
      if (!Number.isInteger(qty) || qty < 1) return void toast.error(`${label}: 회차당 수량은 1 이상의 정수여야 합니다.`);
      if (!Number.isInteger(rounds) || rounds < 1 || rounds > 12) {
        return void toast.error(`${label}: 회차 수는 1~12 사이 정수여야 합니다.`);
      }
      strategies.push({
        ...(row.strategy_id ? { strategy_id: row.strategy_id } : {}),
        ticker: row.ticker,
        trigger_pct: pct,
        round_quantity: qty,
        rounds,
      });
    }
    setSaving(true);
    try {
      const resp = await fetch("/api/strategy-trade", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account_id: accountId, slack_enabled: slack, strategies }),
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
  }, [accountId, drafts, onSaved, slack, toast]);

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
                <div className="appLabeledField" style={{ marginBottom: 0 }}>
                  <span className="appLabeledFieldLabel">현금 비율</span>
                  <span style={{ fontWeight: 700 }}>{view.cash?.pct != null ? `${view.cash.pct.toFixed(1)}%` : "—"}</span>
                </div>
              </div>
              <div className="appMainHeaderRight">
                <UnsavedChangesBadge show={isDirty} />
                <GridToolbarButton
                  variant="add"
                  disabled={saving || view.ticker_options.length === 0}
                  onClick={() => {
                    setSearch("");
                    setPickerOpen(true);
                  }}
                >
                  종목 추가
                </GridToolbarButton>
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
            <div className="d-flex flex-wrap align-items-center gap-2 mb-2">
              <strong>종목</strong>
              <span className="tableMuted">행을 선택해 상세 보기 · 값을 더블 클릭해 수정</span>
            </div>
            {gridRows.length > 0 ? (
              <AppAgGrid
                rowData={gridRows}
                columnDefs={tradeListColumns}
                theme={tradeGridTheme}
                height={Math.min(420, 54 + gridRows.length * 36)}
                minHeight={76}
                getRowId={({ data }) => data.key}
                getRowClass={({ data }) => [
                  stockRowClass({ held: data?.held }),
                  data?.key === selectedRow?.key ? "strategyTradeSelectedRow" : "",
                ].filter(Boolean).join(" ")}
                rowClassDeps={[selectedRow?.key]}
                gridOptions={{
                  rowHeight: 36,
                  headerHeight: 34,
                  suppressMovableColumns: true,
                  suppressCellFocus: false,
                  stopEditingWhenCellsLoseFocus: true,
                  readOnlyEdit: true,
                  rowSelection: { mode: "multiRow", checkboxes: true, headerCheckbox: true, enableClickSelection: false },
                  selectionColumnDef: gridSelectionColumn,
                  onSelectionChanged: ({ api }) => setCheckedKeys(api.getSelectedRows().map((row) => row.key)),
                  onCellEditRequest: ({ data, colDef, newValue }) => {
                    const field = colDef.field;
                    if (field === "pct" || field === "qty" || field === "rounds") {
                      updateRow(data.key, { [field]: String(newValue ?? "").trim() });
                    }
                  },
                  onRowClicked: ({ data }) => {
                    if (data) setSelection({ accountId, ticker: data.ticker });
                  },
                  onCellKeyDown: ({ data, event }) => {
                    if (data && event instanceof KeyboardEvent && event.key === "Enter") {
                      setSelection({ accountId, ticker: data.ticker });
                    }
                  },
                }}
              />
            ) : <div className="tableMuted">등록된 종목이 없습니다.</div>}
          </div>
        </aside>
        <div aria-label="선택 종목 상세">
          {details ? (
            <StrategySection
              key={details.strategy_id}
              view={details}
              poolLabel={poolLabel(details.pool)}
              onDeleteSignal={onDeleteSignal}
            />
          ) : (
            <div className="card appCard"><div className="card-body appCardBodyTight tableMuted">
              {selectedRow ? "전략을 저장하면 운용 상세가 표시됩니다." : "헤더에서 종목을 추가하세요."}
            </div></div>
          )}
        </div>
      </section>

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
  onDeleteSignal,
}: {
  view: StrategyView | FailedStrategy;
  /** 전략의 가격 출처 표기. */
  poolLabel: string;
  onDeleteSignal: (signal: SlackSignal, label: string) => void;
}) {
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
  const columnCount = 7 + tradeDays.length;
  // 사다리 위치마다 한 행 — 회차가 없는 위치(건너뛴 칸)도 그 가격대의 매매와 함께 행으로 둬 간격을 보여 준다.
  const positions = [...view.rounds.map((row) => row.position), ...Object.keys(view.trades_by_position).map(Number)];
  const firstPosition = Math.min(...positions);
  const lastPosition = Math.max(...positions);
  const slots = Array.from({ length: lastPosition - firstPosition + 1 }, (_, idx) => ({
    position: firstPosition + idx,
    row: view.rounds.find((item) => item.position === firstPosition + idx) ?? null,
    trades: view.trades_by_position[String(firstPosition + idx)] ?? {},
  }));
  // 기준선이 회차 사이에 못 들어가는 경우 — 마지막 회차의 지수 환산에서 저점까지 남은 하락률.
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
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "baseline", fontSize: "var(--fs-sm)" }}>
        <span style={{ fontWeight: 800, fontSize: "var(--fs-base)" }}>{view.label}</span>
        <span>
          현재가 <b style={{ fontSize: "var(--fs-xl)", fontWeight: 800 }}>{formatNumber(index.close)}</b>
        </span>
        <span style={{ color: "var(--text-muted)" }}>
          체결 이력 {view.ledger_at.slice(5, 16).replace("T", " ")} 기준
        </span>
      </div>

      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", fontSize: "var(--fs-sm)" }}>
        {status.quantity_mismatch ? (
          <span style={{ color: "var(--up-color, #d64545)", fontWeight: 700 }}>
            ⚠️ 보유 {formatNumber(status.held_quantity)}주가 회차당 수량의 배수가 아님 — 마지막 회차는
            일부만 보유 중입니다
          </span>
        ) : null}
        {status.waiting_first_entry ? (
          <span style={{ fontWeight: 700 }}>
            1호 진입 대기 — {index.name} {formatNumber(index.buy_trigger)} 이하 · 미도달 시{" "}
            {status.first_entry_close_time} 매수
          </span>
        ) : status.next_round == null ? (
          <span style={{ fontWeight: 700, color: "var(--text-muted)" }}>회차 소진 — 매도 대기</span>
        ) : null}
        {status.cash_short ? (
          <span style={actionBadgeStyle("var(--text-muted)")}>⏸ {status.next_round}호 한도 소진 — 현금 부족</span>
        ) : status.next_buy_drop_pct != null ? (
          <span style={actionBadgeStyle(BUY_BADGE_COLOR)}>
            {nextRow?.close_entry
              ? `📉 ${status.next_round}호 장 마감 매수!`
              : status.next_buy_drop_pct >= 0
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

      {/* 회차 표 */}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", minWidth: 600, borderCollapse: "collapse", fontSize: "var(--fs-sm)" }}>
          <thead>
            <tr style={{ color: "var(--text-muted)" }}>
              <th style={{ ...thStyle, textAlign: "left" }}>위치</th>
              <th style={{ ...thStyle, textAlign: "left" }}>회차</th>
              <th style={{ ...thStyle, textAlign: "left" }}>상태</th>
              <th style={thStyle}>체결가</th>
              <th style={thStyle}>손익</th>
              <th style={thStyle}>매수 지정가</th>
              <th style={thStyle}>매도 지정가</th>
              {tradeDays.map((day) => (
                <th key={day} title={day} style={{ ...thStyle, textAlign: "center", whiteSpace: "nowrap" }}>
                  {Number(day.slice(5, 7))}월 {Number(day.slice(8, 10))}일
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {slots.map(({ position, row, trades }, rowIdx) => {
              const tradeCells = tradeDays.map((day) => (
                <td key={day} style={{ ...tdStyle, textAlign: "center" }}>
                  {trades[day] ? <DailyTradeBadges sides={trades[day]} /> : null}
                </td>
              ));
              if (row === null) {
                return (
                  <tr key={`empty-${position}`} style={{ borderTop: "1px solid rgba(148,163,184,0.15)" }}>
                    <td style={{ ...tdStyle, textAlign: "left", color: "var(--text-muted)" }}>{position}</td>
                    <td colSpan={6} style={tdStyle} />
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
              {lowLineHere ? <RecentLowRow index={index} columnCount={columnCount} /> : null}
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
                  <RoundStatus row={row} onDeleteSignal={onDeleteSignal} />
                </td>
                <td style={tdStyle}>
                  {formatNumber(row.fill_price)}
                  {row.bought_on != null && row.source != null ? (
                    <div style={{ color: "var(--text-muted)", fontSize: "var(--fs-xs, 11px)" }}>
                      {row.bought_on.slice(5)} · {BOUGHT_SOURCE_WORDS[row.source]}
                    </div>
                  ) : null}
                </td>
                <td style={{ ...tdStyle, color: signColor(row.profit_pct), fontWeight: 700 }}>
                  {formatSigned(row.profit_pct)}
                </td>
                <td style={{ ...tdStyle, fontWeight: row.is_next ? 700 : undefined }}>
                  {formatNumber(row.buy_limit)}
                  {row.cash_short ? (
                    <span style={{ marginLeft: 6, color: "var(--text-muted)", fontWeight: 400 }}>한도 소진</span>
                  ) : null}
                </td>
                <td style={{ ...tdStyle, fontWeight: row.sell_limit != null ? 700 : undefined }}>
                  {formatNumber(row.sell_limit)}
                </td>
                {tradeCells}
              </tr>
              </React.Fragment>
              );
            })}
            {view.rounds.length > 0 &&
            view.rounds.every((r) => r.buy_index == null || r.buy_index >= index.recent_low) ? (
              <>
                {belowLastRound ? <BelowLastGapRow gap={belowLastRound} columnCount={columnCount} /> : null}
                <RecentLowRow index={index} columnCount={columnCount} />
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
  const [deleting, setDeleting] = useState<{ key: string; label: string } | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);

  const load = useCallback(
    async (accountId: string | null) => {
      setLoading(true);
      try {
        const query = accountId ? `?account_id=${encodeURIComponent(accountId)}` : "";
        const resp = await fetch(`/api/strategy-trade${query}`, { cache: "no-store" });
        const payload = await resp.json();
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

  const askDeleteSignal = useCallback((signal: SlackSignal, label: string) => {
    setDeleting({ key: signal.key, label });
  }, []);

  const confirmDeleteSignal = useCallback(async () => {
    if (!deleting) return;
    setDeleteBusy(true);
    try {
      const resp = await fetch("/api/strategy-trade/delete-signal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: deleting.key }),
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "슬랙 신호를 지우지 못했습니다.");
      toast.success("슬랙 신호를 지웠습니다.");
      setDeleting(null);
      await load(view?.account_id ?? null);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "슬랙 신호를 지우지 못했습니다.");
    } finally {
      setDeleteBusy(false);
    }
  }, [deleting, load, toast, view?.account_id]);

  useEffect(() => {
    // 마지막으로 고른 계좌는 브라우저에 기억한다(사고팔기 화면 전용 키).
    void load(readRememberedTradeAccountId());
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

  return (
    <PageFrame title="전략 사고팔기" fullWidth>
      <div className="appPageStack">
        <AccountSettingsCard
          key={view.account_id}
          view={view}
          accountId={view.account_id}
          onAccountChange={(next) => void load(next)}
          onSaved={setView}
          onDeleteSignal={askDeleteSignal}
        />
      </div>

      <AppModal
        open={deleting !== null}
        title="슬랙 신호 지우기"
        onClose={() => setDeleting(null)}
        footer={
          <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, width: "100%" }}>
            <button type="button" className="btn" onClick={() => setDeleting(null)} disabled={deleteBusy}>
              취소
            </button>
            <button type="button" className="btn btn-danger" onClick={() => void confirmDeleteSignal()} disabled={deleteBusy}>
              {deleteBusy ? "지우는 중…" : "지우기"}
            </button>
          </div>
        }
      >
        <p style={{ margin: 0 }}>
          {deleting?.label} 메시지와 그 스레드(주문·체결 알림)를 슬랙에서 지우고 발송 기록도 지웁니다.
          같은 메시지로 나간 다른 신호도 함께 지워지며, 이 신호로 낸 주문이 있으면 지우지 않습니다.
        </p>
      </AppModal>
    </PageFrame>
  );
}
