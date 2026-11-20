import type { ReactNode } from "react";

import { AppModal } from "./AppModal";
import { StatusBadge } from "./StatusBadge";

export function SlackSignalBadge({ label, compact, showWithdraw, title = "눌러서 이 슬랙 신호 회수", onWithdraw }: {
  label: string;
  compact: boolean;
  showWithdraw: boolean;
  title?: string;
  onWithdraw?: () => void;
}) {
  const badge = <StatusBadge color="var(--tblr-green, #2fb344)" compact={compact}>{label}</StatusBadge>;
  if (!onWithdraw) return <span>{badge}</span>;

  return (
    <button type="button" title={title} onClick={onWithdraw}
      style={{ all: "unset", cursor: "pointer", whiteSpace: "nowrap" }}>
      {badge}
      {showWithdraw ? <span style={{ marginLeft: 4, textDecoration: "underline" }}>회수</span> : null}
    </button>
  );
}

export function SlackWithdrawModal({ label, busy, onClose, onConfirm, children }: {
  label: string | null;
  busy: boolean;
  onClose: () => void;
  onConfirm: () => void;
  children?: ReactNode;
}) {
  return (
    <AppModal
      open={label !== null}
      title="슬랙 신호 회수"
      onClose={() => { if (!busy) onClose(); }}
      footer={
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, width: "100%" }}>
          <button type="button" className="btn" onClick={onClose} disabled={busy}>취소</button>
          <button type="button" className="btn btn-danger" onClick={onConfirm} disabled={busy}>
            {busy ? "회수 중…" : "회수"}
          </button>
        </div>
      }
    >
      <p style={{ margin: 0 }}>
        {label} 메시지를 회수 한 줄로 바꾸고 그 스레드 답글과 발송 기록을 지웁니다.
        이 신호로 낸 주문이 있으면 회수하지 않습니다.
        {children}
      </p>
    </AppModal>
  );
}
