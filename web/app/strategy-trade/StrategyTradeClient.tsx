"use client";

import React, { useCallback, useEffect, useState } from "react";

import { formatPoolLabel, type PoolLabelSource } from "@/lib/pool-label";

import { readRememberedTickerType, writeRememberedTickerType } from "../components/account-selection";
import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";
import { UnsavedChangesBadge } from "../components/UnsavedChangesBadge";
import { TradeBacktestSection } from "./TradeBacktestSection";

type StrategyConfig = {
  // 매매 간격(%) 하나로 진입·추가·매도를 모두 관리한다.
  trigger_pct: number;
  // 회차당 매수 수량(주) — 수량이 고정이어야 계좌에서 회차 상태를 역산할 수 있다.
  round_quantity: number;
  rounds: number;
  index_name: string;
};

type IndexStatus = {
  name: string;
  as_of: string;
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
  // 보유 수량이 회차×회차수량과 다르면 역산이 근사가 된다 — 경고 표시용.
  quantity_mismatch: boolean;
  held_quantity: number;
  // 이 전략의 매수 한도(원) · 이미 산 금액(원) · 다음 회차가 한도에 막혔는지.
  budget: number;
  invested: number;
  limit_blocked: boolean;
  next_round: number | null;
  next_ticker: string | null;
  next_name: string | null;
  last_buy_price: number | null;
  // 다음 회차 매수까지 남은 하락률(%, 음수). 도달했으면 0 이상. 다음 회차 없으면 null.
  next_buy_drop_pct: number | null;
  // 가장 가까운 매도 회차와 목표까지 남은 상승률(%, 양수). 0 이하면 도달. 보유 없으면 null.
  next_sell_round: number | null;
  next_sell_rise_pct: number | null;
};

type RoundRow = {
  round: number;
  fixed_round: number;
  ticker: string;
  name: string;
  held: boolean;
  avg_price: number | null;
  close: number | null;
  profit_pct: number | null;
  buy_limit: number | null;
  buy_index: number | null;
  sell_limit: number | null;
  sell_index: number | null;
  buy_reached: boolean;
  // 진입선이 아니라 장 마감 매수로 닿은 1호 — 현재가로 산다.
  close_entry: boolean;
  // 다음 회차가 매수 한도를 넘는다 — 이 회차부터는 사지 않는다.
  over_limit: boolean;
  sell_reached: boolean;
  is_next: boolean;
};

type StrategyView = {
  strategy_id: string;
  label: string;
  /** 이 전략이 속한 종목풀. */
  pool: string;
  ticker: string;
  config: StrategyConfig;
  index: IndexStatus;
  status: Status;
  rounds: RoundRow[];
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

/** 이 종목풀의 사고팔기 합성 슬리브 — 운용 계좌와 매수 한도의 출처. */
type Sleeve = {
  account_id: string;
  account_name: string;
  weight_pct: number;
  limit_krw: number;
  per_strategy_limit_krw?: number;
  /** 슬랙 알람 — 그 계좌의 합성 슬랙 스위치를 따른다(여기서 정하지 않는다). */
  slack_enabled: boolean;
};

type View = {
  /** 합성에서 이 종목풀에 붙인 사고팔기 슬리브. 없으면 null — 운용 현황을 만들 수 없다. */
  sleeve: Sleeve | null;
  /** 저장된 전략 설정 — 슬리브 유무와 무관하게 편집할 수 있다. */
  configs: StrategyConfigRow[];
  /** 사고팔기를 켠 종목풀 — 종목풀 설정의 토글이 단일 소스다. */
  pool_options: PoolLabelSource[];
  /** 백테스트 기간 선택지 — 백엔드가 단일 소스다. */
  month_options: number[];
  /** 지금 보는 종목풀. 켜진 풀이 없으면 null. */
  pool: string | null;
  strategies: StrategyView[];
};

// 편집 파라미터는 매매 간격(trigger_pct) 하나 — 백엔드 EDITABLE_PCT_KEYS 와 동일.
const PCT_MIN = 0.1;
const PCT_MAX = 50;

const pctInputStyle: React.CSSProperties = {
  width: 74,
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "3px 6px",
  fontSize: "var(--fs-sm)",
  textAlign: "right",
};

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

/** 원 금액을 만원 단위로 — 한도·투입금처럼 큰 금액용. */
function formatKrwMan(value: number): string {
  return `${Math.round(value / 10_000).toLocaleString("ko-KR")}만원`;
}

function formatSigned(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

function signColor(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value) || value === 0) return "inherit";
  return value > 0 ? "var(--up-color, #d64545)" : "var(--down-color, #2f6fd0)";
}

function ReachedBadge() {
  return (
    <span
      style={{
        marginLeft: 6,
        padding: "1px 5px",
        borderRadius: 4,
        fontSize: "var(--fs-sm)",
        fontWeight: 700,
        color: "var(--up-color, #d64545)",
        border: "1px solid var(--up-color, #d64545)",
      }}
    >
      도달
    </span>
  );
}

/** 마지막 회차와 기준선 사이의 빈 구간 — 저점까지 얼마나 더 내려가야 하는지 적는다.
 *  선만 있으면 "표 밖 어딘가 아래"라는 것만 보이고 거리가 안 보인다. */
function BelowLastGapRow({ gap }: { gap: { round: number; pct: number } }) {
  return (
    <tr>
      <td colSpan={7} style={{ padding: "22px 8px", textAlign: "center" }}>
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
function RecentLowRow({ index }: { index: IndexStatus }) {
  return (
    <tr>
      <td colSpan={7} style={{ padding: "2px 8px" }}>
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
function EtfTickerInput({
  pool,
  confirmedTicker,
  confirmedName,
  onChange,
  disabled,
}: {
  pool: string;
  confirmedTicker: string;
  confirmedName: string;
  onChange: (ticker: string | null, name: string) => void;
  disabled?: boolean;
}) {
  const toast = useToast();
  const [text, setText] = useState(confirmedTicker);
  const [checking, setChecking] = useState(false);

  // 확인된 값이 바뀌면(저장·서버 갱신) 입력칸을 맞춘다. 글자를 고치는 중(미확인 = 빈 값)에는 건드리지 않는다.
  useEffect(() => {
    if (confirmedTicker) setText(confirmedTicker);
  }, [confirmedTicker]);

  const confirm = useCallback(async () => {
    const raw = text.trim();
    if (!raw) return;
    setChecking(true);
    try {
      const resp = await fetch(`/api/strategy-trade/resolve?pool=${encodeURIComponent(pool)}&ticker=${encodeURIComponent(raw)}`, { cache: "no-store" });
      const data = (await resp.json()) as { ticker?: string; name?: string; error?: string };
      if (!resp.ok || data.error || !data.ticker || !data.name) {
        onChange(null, "");
        toast.error(data.error ?? "존재하지 않는 티커입니다.");
        return;
      }
      setText(data.ticker);
      onChange(data.ticker, data.name);
      toast.success(`${data.name}(${data.ticker}) 확인 완료`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "티커 조회 중 오류가 발생했습니다.");
    } finally {
      setChecking(false);
    }
  }, [onChange, pool, text, toast]);

  const isConfirmed = text.trim().toUpperCase() === confirmedTicker.toUpperCase() && confirmedTicker !== "";
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span className="appLabeledFieldLabel">종목</span>
      <input
        className="form-control form-control-sm"
        style={{ width: 92 }}
        placeholder="티커"
        value={text}
        disabled={disabled || checking}
        onChange={(event) => {
          setText(event.target.value);
          onChange(null, "");
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            void confirm();
          }
        }}
      />
      <button
        type="button"
        className="btn btn-sm btn-outline-secondary"
        disabled={disabled || checking || !text.trim() || isConfirmed}
        onClick={() => void confirm()}
      >
        {checking ? "확인 중…" : "확인"}
      </button>
      <span style={{ color: isConfirmed ? "var(--text-muted)" : "var(--down-color, #2f6fd0)" }}>
        {isConfirmed ? confirmedName : "티커를 확인하세요"}
      </span>
    </span>
  );
}

/** 설정 표의 한 행 — 전략 하나의 편집 초안. 숫자는 입력 중 상태를 담으려 문자열이다. */
type DraftRow = {
  key: string;
  strategy_id: string | null;
  ticker: string | null;
  tickerName: string;
  pct: string;
  qty: string;
  rounds: string;
};

let draftKeySeq = 0;
function nextDraftKey(): string {
  draftKeySeq += 1;
  return `row-${draftKeySeq}`;
}

function rowsFromConfigs(configs: StrategyConfigRow[]): DraftRow[] {
  return configs.map((config) => ({
    key: config.strategy_id,
    strategy_id: config.strategy_id,
    ticker: config.ticker,
    tickerName: config.name,
    pct: String(config.trigger_pct),
    qty: String(config.round_quantity),
    rounds: String(config.rounds),
  }));
}

/** 변경 비교용 — 입력 중 문자열을 저장 값 모양으로 맞춘다. */
function comparableRow(row: DraftRow) {
  return [row.strategy_id, row.ticker, Number(row.pct), Number(row.qty), Number(row.rounds)];
}

/** 상단 설정 카드 — 종목풀 셀렉트, 전략 설정 표, 저장 버튼 하나. */
function PoolSettingsCard({
  view,
  pool,
  onPoolChange,
  onSaved,
  onDirtyChange,
}: {
  view: View;
  pool: string;
  onPoolChange: (pool: string) => void;
  onSaved: (next: View) => void;
  /** 저장 안 한 변경이 생기거나 사라질 때 알린다 — 백테스트는 저장된 설정으로 돌기 때문이다. */
  onDirtyChange: (dirty: boolean) => void;
}) {
  const toast = useToast();
  const [rows, setRows] = useState<DraftRow[]>(() => rowsFromConfigs(view.configs));
  const [saving, setSaving] = useState(false);

  // 서버가 최신 값을 돌려주면(저장·풀 전환) 초안도 그 값으로 맞춘다.
  useEffect(() => {
    setRows(rowsFromConfigs(view.configs));
  }, [view.configs]);

  const isDirty =
    JSON.stringify(rows.map(comparableRow)) !== JSON.stringify(rowsFromConfigs(view.configs).map(comparableRow));
  useEffect(() => {
    onDirtyChange(isDirty);
  }, [isDirty, onDirtyChange]);

  const updateRow = useCallback((key: string, patch: Partial<DraftRow>) => {
    setRows((prev) => prev.map((row) => (row.key === key ? { ...row, ...patch } : row)));
  }, []);

  const addRow = useCallback(() => {
    setRows((prev) => [
      ...prev,
      {
        key: nextDraftKey(),
        strategy_id: null,
        ticker: null,
        tickerName: "",
        pct: "5",
        qty: "",
        rounds: "4",
      },
    ]);
  }, []);

  const handlePoolChange = useCallback(
    (next: string) => {
      if (isDirty && !window.confirm("저장하지 않은 변경이 사라집니다. 종목풀을 바꿀까요?")) return;
      onPoolChange(next);
    },
    [isDirty, onPoolChange],
  );

  const save = useCallback(async () => {
    const strategies = [];
    for (const row of rows) {
      const label = row.tickerName || row.ticker || "새 전략";
      const pct = Number(row.pct);
      const qty = Number(row.qty);
      const rounds = Number(row.rounds);
      if (!row.ticker) return void toast.error(`${label}: 종목 티커를 입력하고 확인하세요.`);
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
        body: JSON.stringify({ pool, strategies }),
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "설정을 저장하지 못했습니다.");
      onSaved(payload as View);
      toast.success("전략 설정을 저장했습니다.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "설정을 저장하지 못했습니다.");
    } finally {
      setSaving(false);
    }
  }, [onSaved, pool, rows, toast]);

  return (
    <section className="appSection">
      <div className="card appCard">
        <div className="card-body appCardBodyTight">
          <div className="appMainHeader">
            <div className="appMainHeaderLeft">
              <label className="appLabeledField">
                <span className="appLabeledFieldLabel">종목풀</span>
                <select
                  className="form-select form-select-sm"
                  value={pool}
                  disabled={saving}
                  onChange={(event) => handlePoolChange(event.target.value)}
                >
                  {view.pool_options.map((option) => (
                    <option key={option.ticker_type} value={option.ticker_type}>
                      {formatPoolLabel(option)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="appMainHeaderRight">
              <UnsavedChangesBadge show={isDirty} />
              <button type="button" className="btn btn-sm btn-outline-secondary" disabled={saving} onClick={addRow}>
                + 전략 추가
              </button>
              <button
                type="button"
                className="btn btn-success btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                disabled={saving || !isDirty}
                onClick={() => void save()}
              >
                <span>{saving ? "저장 중…" : "저장"}</span>
              </button>
            </div>
          </div>

          {view.sleeve ? (
            <div style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", marginTop: 8 }}>
              운용 계좌 {view.sleeve.account_name}({view.sleeve.account_id}) · 합성 슬리브 {view.sleeve.weight_pct}% · 매수
              한도 {formatKrwMan(view.sleeve.limit_krw)}
              {view.sleeve.per_strategy_limit_krw != null
                ? ` (전략당 ${formatKrwMan(view.sleeve.per_strategy_limit_krw)})`
                : ""}
              {` · 슬랙 알람 ${view.sleeve.slack_enabled ? "켜짐" : "꺼짐"}(합성 화면에서 설정)`}
            </div>
          ) : (
            <div className="alert alert-warning" style={{ marginTop: 10, marginBottom: 0 }}>
              이 종목풀의 사고팔기 슬리브가 어느 계좌에도 붙어 있지 않습니다 — 합성 화면에서 「사고팔기」 슬리브를 붙이면
              운용 현황이 나옵니다.
            </div>
          )}

          <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 10 }}>
            {rows.map((row) => (
              <div
                key={row.key}
                style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "center", fontSize: "var(--fs-sm)" }}
              >
                <EtfTickerInput
                  pool={pool}
                  confirmedTicker={row.ticker ?? ""}
                  confirmedName={row.tickerName}
                  onChange={(ticker, name) => updateRow(row.key, { ticker, tickerName: name })}
                  disabled={saving}
                />
                <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  매매 간격
                  <input
                    type="number"
                    step="0.1"
                    min={PCT_MIN}
                    max={PCT_MAX}
                    style={pctInputStyle}
                    value={row.pct}
                    disabled={saving}
                    onChange={(event) => updateRow(row.key, { pct: event.target.value })}
                  />
                  %
                </span>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  회차당 수량
                  <input
                    type="number"
                    step="1"
                    min={1}
                    style={pctInputStyle}
                    value={row.qty}
                    disabled={saving}
                    onChange={(event) => updateRow(row.key, { qty: event.target.value })}
                  />
                  주
                </span>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  회차 수
                  <input
                    type="number"
                    step="1"
                    min={1}
                    max={12}
                    style={pctInputStyle}
                    value={row.rounds}
                    disabled={saving}
                    onChange={(event) => updateRow(row.key, { rounds: event.target.value })}
                  />
                  회
                </span>
                <button
                  type="button"
                  className="btn btn-sm btn-outline-danger"
                  disabled={saving}
                  onClick={() => setRows((prev) => prev.filter((item) => item.key !== row.key))}
                >
                  삭제
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

/** 전략 하나의 운용 현황(현재 판단 + 회차 표) — 설정은 상단 카드에서 편집한다. */
function StrategySection({ view }: { view: StrategyView }) {
  const { config, status, index } = view;
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
        <span style={{ color: "var(--text-muted)" }}>
          {view.ticker} · 간격 {config.trigger_pct}% · 회차당 {formatNumber(config.round_quantity)}주 · {config.rounds}회
          · 한도 {formatKrwMan(status.budget)} 중 {formatKrwMan(status.invested)} 사용
        </span>
      </div>

      {/* 현재 판단 요약 */}
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: "var(--fs-sm)" }}>
        <span>
          {index.name} <b>{formatNumber(index.close)}</b>
          <span style={{ color: "var(--text-muted)" }}> ({index.as_of})</span>
        </span>
        <span>
          {index.recent_low_label} <b>{formatNumber(index.recent_low)}</b>
          <span style={{ color: "var(--text-muted)" }}> ({index.recent_low_date})</span>
        </span>
        <span>
          보유 <b>{status.held_count}</b>/{config.rounds}회차
        </span>
        {status.quantity_mismatch ? (
          <span style={{ color: "var(--up-color, #d64545)", fontWeight: 700 }}>
            ⚠️ 보유 {formatNumber(status.held_quantity)}주가 회차당 수량의 배수가 아님 — 회차
            역산이 근사값입니다
          </span>
        ) : null}
        {status.last_buy_price != null ? (
          <span>
            마지막 매수가 <b>{formatNumber(status.last_buy_price)}</b>
          </span>
        ) : null}
        {status.waiting_first_entry ? (
          <span style={{ fontWeight: 700 }}>
            1호 진입 대기 — {index.name} {formatNumber(index.buy_trigger)} 이하 · 미도달 시{" "}
            {status.first_entry_close_time} 매수
          </span>
        ) : status.next_round != null ? (
          <span style={{ fontWeight: 700 }}>
            다음 진입: {status.next_round}호 {status.next_ticker} {status.next_name}
          </span>
        ) : (
          <span style={{ fontWeight: 700, color: "var(--text-muted)" }}>회차 소진 — 매도 대기</span>
        )}
        {status.limit_blocked ? (
          <span style={actionBadgeStyle("var(--text-muted)")}>⏸ {status.next_round}호 한도 도달 — 추가 매수 중단</span>
        ) : status.next_buy_drop_pct != null ? (
          <span style={actionBadgeStyle(BUY_BADGE_COLOR)}>
            {nextRow?.close_entry
              ? `📉 ${status.next_round}호 장 마감 매수!`
              : status.next_buy_drop_pct >= 0
              ? `📉 ${status.next_round}호 매수가 도달!`
              : `📉 ${index.name} ${Math.abs(status.next_buy_drop_pct).toFixed(2)}% 더 내리면 ${status.next_round}호 매수`}
          </span>
        ) : null}
        {status.next_sell_rise_pct != null ? (
          <span style={actionBadgeStyle(SELL_BADGE_COLOR)}>
            {status.next_sell_rise_pct <= 0
              ? `📈 ${status.next_sell_round}호 매도가 도달!`
              : `📈 ${index.name} ${status.next_sell_rise_pct.toFixed(2)}% 더 오르면 ${status.next_sell_round}호 매도`}
          </span>
        ) : null}
      </div>

      {/* 회차 표 */}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", minWidth: 600, borderCollapse: "collapse", fontSize: "var(--fs-sm)" }}>
          <thead>
            <tr style={{ color: "var(--text-muted)" }}>
              <th style={{ ...thStyle, textAlign: "left" }}>회차</th>
              <th style={{ ...thStyle, textAlign: "left" }}>종목</th>
              <th style={thStyle}>진입가</th>
              <th style={thStyle}>현재가</th>
              <th style={thStyle}>손익</th>
              <th style={thStyle}>매수 지정가</th>
              <th style={thStyle}>매도 지정가</th>
            </tr>
          </thead>
          <tbody>
            {view.rounds.map((row, rowIdx) => {
              // 최근 저점 라인: 이 행의 지수 환산이 저점 아래로 처음 내려가는 지점 앞에 긋는다.
              // (마지막 행까지 저점 위면 맨 아래에) — 값이 없는 행은 판정에서 건너뛴다.
              const prevLevels = view.rounds.slice(0, rowIdx).map((r) => r.buy_index).filter((v): v is number => v != null);
              const ownLevel = row.buy_index;
              const lowLineHere =
                ownLevel != null &&
                ownLevel < index.recent_low &&
                (prevLevels.length === 0 || prevLevels[prevLevels.length - 1] >= index.recent_low);
              return (
              <React.Fragment key={row.round}>
              {lowLineHere ? <RecentLowRow index={index} /> : null}
              <tr style={{ borderTop: "1px solid rgba(148,163,184,0.15)" }}>
                <td style={{ ...tdStyle, textAlign: "left", fontWeight: 700 }}>
                  {row.round}호
                  {row.is_next ? (
                    <span style={{ marginLeft: 5, color: "var(--text-muted)", fontSize: "var(--fs-sm)", fontWeight: 400 }}>
                      다음
                    </span>
                  ) : null}
                </td>
                <td style={{ ...tdStyle, textAlign: "left" }}>
                  {row.ticker} {row.name}
                </td>
                <td style={tdStyle}>{formatNumber(row.avg_price)}</td>
                <td style={tdStyle}>{formatNumber(row.close)}</td>
                <td style={{ ...tdStyle, color: signColor(row.profit_pct), fontWeight: 700 }}>
                  {formatSigned(row.profit_pct)}
                </td>
                <td style={{ ...tdStyle, fontWeight: row.is_next ? 700 : undefined }}>
                  {formatNumber(row.buy_limit)}
                  {row.buy_reached ? <ReachedBadge /> : null}
                  {row.over_limit ? (
                    <span style={{ marginLeft: 6, color: "var(--text-muted)", fontWeight: 400 }}>한도 도달</span>
                  ) : null}
                </td>
                <td style={{ ...tdStyle, fontWeight: row.sell_limit != null ? 700 : undefined }}>
                  {formatNumber(row.sell_limit)}
                  {row.sell_reached ? <ReachedBadge /> : null}
                </td>
              </tr>
              </React.Fragment>
              );
            })}
            {view.rounds.length > 0 &&
            view.rounds.every((r) => r.buy_index == null || r.buy_index >= index.recent_low) ? (
              <>
                {belowLastRound ? <BelowLastGapRow gap={belowLastRound} /> : null}
                <RecentLowRow index={index} />
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
  const [dirty, setDirty] = useState(false);

  const load = useCallback(
    async (pool: string | null) => {
      setLoading(true);
      try {
        const query = pool ? `?pool=${encodeURIComponent(pool)}` : "";
        const resp = await fetch(`/api/strategy-trade${query}`, { cache: "no-store" });
        const payload = await resp.json();
        if (!resp.ok) throw new Error(payload?.error ?? "운용 현황을 불러오지 못했습니다.");
        setView(payload as View);
        writeRememberedTickerType("strategy-trade", (payload as View).pool);
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "운용 현황을 불러오지 못했습니다.");
      } finally {
        setLoading(false);
      }
    },
    [toast],
  );

  useEffect(() => {
    // 마지막으로 고른 풀은 브라우저에 기억한다(다른 전략 화면들과 같은 공용 키 방식).
    void load(readRememberedTickerType("strategy-trade"));
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
  if (view.pool === null) {
    return (
      <PageFrame title="전략 사고팔기" fullWidth>
        <div style={{ color: "var(--text-muted)", padding: 20 }}>
          사고팔기를 켠 종목풀이 없습니다 — 종목풀 설정에서 한국 ETF 종목풀의 「사고팔기」를 켜세요.
        </div>
      </PageFrame>
    );
  }

  return (
    <PageFrame title="전략 사고팔기" fullWidth>
      <div className="appPageStack">
        <PoolSettingsCard
          view={view}
          pool={view.pool}
          onPoolChange={(next) => void load(next)}
          onSaved={setView}
          onDirtyChange={setDirty}
        />

        {/* 운용 현황 — DB 에 저장된 순서, 좌우 2열(좁은 화면은 1열) */}
        <section className="appSection">
          <div className="strategyTradeGrid">
            {view.strategies.map((strategy) => (
              <StrategySection key={strategy.strategy_id} view={strategy} />
            ))}
          </div>
        </section>

        {/* 백테스트 — 설정이나 풀이 바뀌면 결과가 어긋나므로 다시 마운트해 비운다. */}
        <TradeBacktestSection
          key={`${view.pool}:${view.configs.map((c) => `${c.strategy_id}/${c.ticker}/${c.trigger_pct}/${c.round_quantity}/${c.rounds}`).join(",")}:${view.sleeve?.limit_krw ?? ""}`}
          pool={view.pool}
          monthOptions={view.month_options}
          blockedReason={
            dirty
              ? "저장하지 않은 변경이 있습니다 — 저장한 뒤 실행하세요."
              : view.configs.length === 0
                ? "이 종목풀에 전략이 없습니다."
                : view.sleeve === null
                  ? "사고팔기 슬리브가 계좌에 붙어 있지 않습니다."
                  : null
          }
        />
      </div>
    </PageFrame>
  );
}
