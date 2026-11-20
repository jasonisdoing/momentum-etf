"use client";

import { useCallback, useEffect, useState } from "react";

import { formatCurrencyPrice } from "@/lib/price-format";

import { AppModal } from "./AppModal";
import { StatusBadge } from "./StatusBadge";
import { useToast } from "./ToastProvider";

export function SlackSignalBadge({ label, compact, title = "눌러서 주문·정정·취소·회수", onClick }: {
  label: string;
  compact: boolean;
  title?: string;
  onClick?: () => void;
}) {
  const badge = <StatusBadge color="var(--tblr-green, #2fb344)" compact={compact}>{label}</StatusBadge>;
  if (!onClick) return <span>{badge}</span>;

  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      style={{ all: "unset", cursor: "pointer", whiteSpace: "nowrap" }}
    >
      {badge}
    </button>
  );
}

export type SlackSignalPhase = "ready_to_order" | "active_order" | "completed";

export type SlackSignalActionTarget = {
  accountId: string;
  accountLabel: string;
  currency: string;
  title: string;
  text: string;
  sideLabel: string;
  sideColor: string;
  withdrawKey: string | null;
  statusUrl: string;
  actionUrl: string;
  withdrawUrl: string;
  actionPayload: Record<string, unknown>;
  withdrawPayload: Record<string, unknown>;
};

type SignalActionState = {
  phase: SlackSignalPhase;
  status_label: string;
  thread_ts?: string | null;
  reason?: string;
  order_error?: string;
  confirmation_id?: string;
  confirmed?: Record<string, unknown>;
  order_preview?: {
    quantity: number;
    price: number;
    amount: number;
    quote_label: string;
    venue: string;
    country: string;
  };
  order?: {
    id: string;
    status: string;
    price: number;
    quantity: number;
    filled_quantity: number;
    remaining_quantity: number;
    country: string;
  };
  modify_preview?: {
    target_price: number;
    quote_label: string;
    remaining_quantity: number;
  } | null;
  modify_error?: string | null;
  withdraw_key: string | null;
};

const PHASE_COLORS: Record<SlackSignalPhase, string> = {
  ready_to_order: "var(--tblr-green, #2fb344)",
  active_order: "var(--tblr-orange, #f76707)",
  completed: "var(--text-muted)",
};

/**
 * 슬랙 발송 신호 통합 모달(합성 전략·전략 사고팔기 공용) — 주문·호가정정·취소·회수를 처리한다.
 * 열릴 때마다 서버에서 최신 상태를 조회해 슬랙과 동시 작업 시 중복 실행을 막는다.
 */
export function SlackSignalActionModal({ target, onClose, onChanged, onWithdrawn }: {
  target: SlackSignalActionTarget | null;
  onClose: () => void;
  onChanged: () => void;
  onWithdrawn?: (target: SlackSignalActionTarget) => void;
}) {
  const toast = useToast();
  const [state, setState] = useState<SignalActionState | null>(null);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [withdrawBusy, setWithdrawBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadState = useCallback(async (current: SlackSignalActionTarget) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(current.statusUrl, { cache: "no-store" });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error ?? "슬랙 발송 주문 상태를 조회하지 못했습니다.");
      setState(result as SignalActionState);
    } catch (err) {
      setError(err instanceof Error ? err.message : "슬랙 발송 주문 상태를 조회하지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!target) {
      setState(null);
      setError(null);
      return;
    }
    void loadState(target);
  }, [target, loadState]);

  if (!target) return null;

  const busy = loading || submitting || withdrawBusy;
  const phaseColor = state ? PHASE_COLORS[state.phase] : "var(--text-muted)";
  const effectiveWithdrawKey = state ? state.withdraw_key : target.withdrawKey;

  async function runAction(kind: "order" | "modify" | "cancel") {
    if (!target || !state || busy) return;
    setSubmitting(true);
    setError(null);
    try {
      const extra =
        kind === "order"
          ? {
              thread_ts: state.thread_ts,
              ...(state.confirmation_id ? { confirmation_id: state.confirmation_id } : null),
              ...(state.confirmed ? { confirmed: state.confirmed } : null),
            }
          : {
              thread_ts: state.thread_ts,
              order_id: state.order?.id,
            };
      const response = await fetch(target.actionUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind, ...target.actionPayload, ...extra }),
      });
      const result = await response.json();
      if (!response.ok) {
        await loadState(target);
        throw new Error(result.error ?? "요청을 처리하지 못했습니다.");
      }
      setState(result as SignalActionState);
      onChanged();
      toast.success(
        kind === "order"
          ? `${target.sideLabel} 주문을 접수했습니다.`
          : kind === "modify"
            ? "호가 정정을 요청했습니다."
            : "주문 취소를 요청했습니다.",
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "요청을 처리하지 못했습니다.");
    } finally {
      setSubmitting(false);
    }
  }

  async function runWithdraw(key: string) {
    if (!target || busy) return;
    setWithdrawBusy(true);
    setError(null);
    try {
      const response = await fetch(target.withdrawUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...target.withdrawPayload, key }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error ?? "슬랙 신호를 회수하지 못했습니다.");
      toast.success("슬랙 신호를 회수했습니다.");
      onWithdrawn?.(target);
      onChanged();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "슬랙 신호를 회수하지 못했습니다.");
    } finally {
      setWithdrawBusy(false);
    }
  }

  const dlStyle: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: "92px 1fr",
    gap: "6px 12px",
    margin: 0,
    fontSize: "var(--fs-sm)",
  };
  const boxStyle: React.CSSProperties = {
    marginTop: 12,
    padding: "10px 12px",
    borderRadius: 8,
    border: "1px solid var(--border-color, #dee2e6)",
    background: "var(--bg-subtle, rgba(0, 0, 0, 0.02))",
  };

  const footer = (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, width: "100%" }}>
      <div>
        {state?.phase === "ready_to_order" && effectiveWithdrawKey ? (
          <button
            type="button"
            className="btn btn-outline-danger"
            disabled={busy}
            onClick={() => void runWithdraw(effectiveWithdrawKey)}
          >
            {withdrawBusy ? "회수 중…" : "신호 회수"}
          </button>
        ) : null}
        {state?.phase === "active_order" && state.order ? (
          <button
            type="button"
            className="btn btn-outline-danger"
            disabled={busy}
            onClick={() => void runAction("cancel")}
          >
            {submitting ? "처리 중…" : "주문 취소"}
          </button>
        ) : null}
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <button type="button" className="btn" disabled={busy} onClick={onClose}>
          닫기
        </button>
        {state?.phase === "ready_to_order" && state.order_preview ? (
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy}
            onClick={() => void runAction("order")}
          >
            {submitting ? "주문 중…" : `${target.sideLabel} 주문하기`}
          </button>
        ) : null}
        {state?.phase === "active_order" && state.order ? (
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy || !state.modify_preview}
            onClick={() => void runAction("modify")}
          >
            {submitting ? "정정 중…" : "호가 정정"}
          </button>
        ) : null}
        {state?.phase === "completed" ? (
          <button
            type="button"
            className="btn btn-outline-secondary"
            disabled={busy}
            onClick={() => void loadState(target)}
          >
            상태 새로고침
          </button>
        ) : null}
      </div>
    </div>
  );

  return (
    <AppModal
      open={target !== null}
      title={`슬랙 발송 주문 관리 — ${target.title}`}
      onClose={() => { if (!busy) onClose(); }}
      footer={footer}
    >
      <dl style={dlStyle}>
        <dt>계좌</dt>
        <dd style={{ margin: 0 }}>{target.accountLabel}</dd>
        <dt>대상</dt>
        <dd style={{ margin: 0 }}>
          <strong style={{ color: target.sideColor }}>{target.title}</strong> — {target.text}
        </dd>
        <dt>현재 상태</dt>
        <dd style={{ margin: 0 }}>
          <StatusBadge compact color={phaseColor}>
            {loading ? "확인 중…" : state?.status_label ?? "확인 실패"}
          </StatusBadge>
        </dd>
      </dl>

      {state?.phase === "ready_to_order" ? (
        <div style={boxStyle}>
          {state.order_preview ? (
            <>
              <div style={{ fontWeight: 600, marginBottom: 8, fontSize: "var(--fs-sm)" }}>
                주문 확인 (슬랙 주문하기와 동일 기준)
              </div>
              <dl style={dlStyle}>
                <dt>주문 수량</dt>
                <dd style={{ margin: 0 }}>{state.order_preview.quantity.toLocaleString()}주</dd>
                <dt>지정가(호가)</dt>
                <dd style={{ margin: 0 }}>
                  {formatCurrencyPrice(state.order_preview.price, target.currency)} ({state.order_preview.quote_label})
                </dd>
                <dt>예상 금액</dt>
                <dd style={{ margin: 0 }}>{formatCurrencyPrice(state.order_preview.amount, target.currency)}</dd>
              </dl>
              <p style={{ margin: "8px 0 0", color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
                실행 시 슬랙과 같은 락·중복 검사 후 주문을 접수하고 슬랙 스레드 버튼도 함께 갱신합니다.
              </p>
            </>
          ) : (
            <p style={{ margin: 0, fontSize: "var(--fs-sm)" }} className="text-danger">
              {state.order_error ?? "주문 가능 정보를 확인할 수 없습니다."}
            </p>
          )}
        </div>
      ) : null}

      {state?.phase === "active_order" && state.order ? (
        <div style={boxStyle}>
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: "var(--fs-sm)" }}>
            미체결 주문 현황 (정정·취소)
          </div>
          <dl style={dlStyle}>
            <dt>접수 주문가</dt>
            <dd style={{ margin: 0 }}>{formatCurrencyPrice(state.order.price, target.currency)}</dd>
            <dt>체결 / 잔여</dt>
            <dd style={{ margin: 0 }}>
              {state.order.filled_quantity.toLocaleString()}주 체결 · {state.order.remaining_quantity.toLocaleString()}주 미체결
            </dd>
            <dt>정정 예정가</dt>
            <dd style={{ margin: 0 }}>
              {state.modify_preview
                ? `${formatCurrencyPrice(state.modify_preview.target_price, target.currency)} (${state.modify_preview.quote_label})`
                : state.modify_error ?? "정정 불가"}
            </dd>
          </dl>
          <p style={{ margin: "8px 0 0", color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
            정정 또는 취소 전 증권사 최신 미체결 잔량을 다시 확인한 뒤 반영합니다.
          </p>
        </div>
      ) : null}

      {state?.phase === "completed" ? (
        <div style={boxStyle}>
          <div style={{ fontWeight: 600, marginBottom: 6, fontSize: "var(--fs-sm)" }}>
            {state.status_label}
          </div>
          <p style={{ margin: 0, color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
            {state.reason ?? "이미 처리된 신호입니다."}
          </p>
          {state.order ? (
            <dl style={{ ...dlStyle, marginTop: 8 }}>
              <dt>주문 내역</dt>
              <dd style={{ margin: 0 }}>
                {state.order.filled_quantity.toLocaleString()} / {state.order.quantity.toLocaleString()}주 ×{" "}
                {formatCurrencyPrice(state.order.price, target.currency)} ({state.order.status})
              </dd>
            </dl>
          ) : null}
        </div>
      ) : null}

      {error ? (
        <p role="alert" className="text-danger mb-0" style={{ marginTop: 10, fontSize: "var(--fs-sm)" }}>
          {error}
        </p>
      ) : null}
    </AppModal>
  );
}
