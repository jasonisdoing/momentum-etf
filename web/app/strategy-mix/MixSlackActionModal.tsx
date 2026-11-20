"use client";

import { useState } from "react";

import { formatCurrencyPrice } from "@/lib/price-format";

import { AppModal } from "../components/AppModal";
import { StatusBadge } from "../components/StatusBadge";

export type MixSlackPhase = "ready_to_order" | "active_order" | "completed";

export type MixSlackActionTarget = {
  accountId: string;
  accountLabel: string;
  currency: string;
  ticker: string;
  title: string;
  text: string;
  side: "buy" | "sell";
  quantity: number;
  withdrawKey: string | null;
};

type Props = {
  target: MixSlackActionTarget | null;
  busy: boolean;
  onClose: () => void;
  onWithdraw: (target: MixSlackActionTarget) => void;
};

const PHASE_LABELS: Record<MixSlackPhase, { label: string; color: string }> = {
  ready_to_order: { label: "주문 대기", color: "var(--tblr-green, #2fb344)" },
  active_order: { label: "미체결 진행 중", color: "var(--tblr-orange, #f76707)" },
  completed: { label: "처리 완료", color: "var(--text-muted)" },
};

const PHASE_TABS: Array<{ key: MixSlackPhase; label: string }> = [
  { key: "ready_to_order", label: "1. 주문 전" },
  { key: "active_order", label: "2. 미체결 (정정·취소)" },
  { key: "completed", label: "3. 이미 처리됨" },
];

/**
 * 슬랙 발송됨 통합 액션 모달 — 슬랙에서 가능한 주문·호가정정·취소·회수를 웹에서도 처리한다.
 * 1단계 UI 확인용으로 상태별 화면 전환 미리보기를 포함한다.
 */
export function MixSlackActionModal({ target, busy, onClose, onWithdraw }: Props) {
  const [phase, setPhase] = useState<MixSlackPhase>("ready_to_order");
  const [notice, setNotice] = useState<string | null>(null);

  if (!target) return null;

  const sideLabel = target.side === "sell" ? "매도" : "매수";
  const phaseMeta = PHASE_LABELS[phase];
  const samplePrice = target.currency === "USD" ? 102.4 : 11660;
  const sampleModifyPrice = target.currency === "USD" ? 102.6 : 11670;
  const sampleAmount = samplePrice * target.quantity;

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
        {phase === "ready_to_order" && target.withdrawKey ? (
          <button
            type="button"
            className="btn btn-outline-danger"
            disabled={busy}
            onClick={() => onWithdraw(target)}
          >
            {busy ? "회수 중…" : "신호 회수"}
          </button>
        ) : null}
        {phase === "active_order" ? (
          <button
            type="button"
            className="btn btn-outline-danger"
            disabled={busy}
            onClick={() => setNotice("주문 취소 요청은 백엔드 연동 단계에서 실제 증권사 취소로 연결됩니다.")}
          >
            주문 취소
          </button>
        ) : null}
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <button type="button" className="btn" disabled={busy} onClick={onClose}>
          닫기
        </button>
        {phase === "ready_to_order" ? (
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy}
            onClick={() => setNotice("주문 실행은 백엔드 연동 단계에서 슬랙과 동일한 주문 경로로 연결됩니다.")}
          >
            {sideLabel} 주문하기
          </button>
        ) : null}
        {phase === "active_order" ? (
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy}
            onClick={() => setNotice("호가 정정 요청은 백엔드 연동 단계에서 실제 정정 경로로 연결됩니다.")}
          >
            호가 정정
          </button>
        ) : null}
        {phase === "completed" ? (
          <button
            type="button"
            className="btn btn-outline-secondary"
            disabled={busy}
            onClick={() => setNotice("상태 새로고침은 백엔드 상태 조회 API 연결 후 동작합니다.")}
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
      {/* 1단계 UI 확인용 상태 전환 바 — 2단계 백엔드 연동 시 실제 조회 상태로 대체한다. */}
      <div
        role="tablist"
        aria-label="상태 미리보기"
        style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 12 }}
      >
        {PHASE_TABS.map((tab) => (
          <button
            key={tab.key}
            type="button"
            role="tab"
            aria-selected={phase === tab.key}
            className={`btn btn-sm ${phase === tab.key ? "btn-primary" : "btn-outline-secondary"}`}
            onClick={() => {
              setPhase(tab.key);
              setNotice(null);
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <dl style={dlStyle}>
        <dt>계좌</dt>
        <dd style={{ margin: 0 }}>{target.accountLabel}</dd>
        <dt>액션</dt>
        <dd style={{ margin: 0 }}>
          <strong style={{ color: target.side === "sell" ? "var(--down-color, #2f6fd0)" : "var(--up-color, #d64545)" }}>
            {target.title}
          </strong>{" "}
          — {target.text}
        </dd>
        <dt>현재 상태</dt>
        <dd style={{ margin: 0 }}>
          <StatusBadge compact color={phaseMeta.color}>{phaseMeta.label}</StatusBadge>
        </dd>
      </dl>

      {phase === "ready_to_order" ? (
        <div style={boxStyle}>
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: "var(--fs-sm)" }}>
            주문 확인 (슬랙 주문하기 모달과 동일 기준)
          </div>
          <dl style={dlStyle}>
            <dt>주문 수량</dt>
            <dd style={{ margin: 0 }}>{target.quantity.toLocaleString()}주</dd>
            <dt>지정가(호가)</dt>
            <dd style={{ margin: 0 }}>{formatCurrencyPrice(samplePrice, target.currency)} (최우선 호가)</dd>
            <dt>예상 금액</dt>
            <dd style={{ margin: 0 }}>{formatCurrencyPrice(sampleAmount, target.currency)}</dd>
            <dt>신호 대비</dt>
            <dd style={{ margin: 0 }}>+0.00%</dd>
          </dl>
          <p style={{ margin: "8px 0 0", color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
            실행 시 슬랙과 같은 락·중복 검사 후 주문을 접수하고 슬랙 스레드 버튼도 함께 갱신합니다.
          </p>
        </div>
      ) : null}

      {phase === "active_order" ? (
        <div style={boxStyle}>
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: "var(--fs-sm)" }}>
            미체결 주문 현황 (정정·취소)
          </div>
          <dl style={dlStyle}>
            <dt>접수 주문가</dt>
            <dd style={{ margin: 0 }}>{formatCurrencyPrice(samplePrice, target.currency)}</dd>
            <dt>체결 / 잔여</dt>
            <dd style={{ margin: 0 }}>0주 체결 · {target.quantity.toLocaleString()}주 미체결</dd>
            <dt>정정 예정가</dt>
            <dd style={{ margin: 0 }}>{formatCurrencyPrice(sampleModifyPrice, target.currency)} (현재 호가 기준)</dd>
          </dl>
          <p style={{ margin: "8px 0 0", color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
            정정 또는 취소 전 증권사 최신 미체결 잔량을 다시 확인한 뒤 반영합니다.
          </p>
        </div>
      ) : null}

      {phase === "completed" ? (
        <div style={boxStyle}>
          <div style={{ fontWeight: 600, marginBottom: 6, fontSize: "var(--fs-sm)" }}>
            이미 처리된 신호입니다
          </div>
          <p style={{ margin: 0, color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
            슬랙 또는 웹에서 이미 체결 완료·취소·만료·회수된 건은 중복 주문을 막기 위해 실행 버튼이 비활성화됩니다.
          </p>
        </div>
      ) : null}

      {notice ? (
        <p role="status" className="text-info mb-0" style={{ marginTop: 10, fontSize: "var(--fs-sm)" }}>
          {notice}
        </p>
      ) : null}
    </AppModal>
  );
}
