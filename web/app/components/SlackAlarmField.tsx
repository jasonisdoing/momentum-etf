"use client";

import type { CSSProperties } from "react";

const hintStyle: CSSProperties = { color: "var(--text-muted)", fontSize: "var(--fs-sm)" };

/** 슬랙 알람 스위치 + 지금 발송(테스트) 버튼 — 합성·사고팔기 화면 공용. 스위치는 화면의 저장 버튼으로 반영된다. */
export function SlackAlarmField({
  label = "슬랙 알람",
  checked,
  onChange,
  disabled,
  title,
  testing,
  onTest,
}: {
  label?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled: boolean;
  /** 스위치 설명(마우스 오버). */
  title: string;
  testing: boolean;
  onTest: () => void;
}) {
  return (
    <label className="appLabeledField" style={{ marginBottom: 0 }}>
      <span className="appLabeledFieldLabel">{label}</span>
      <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
        <div className="form-check form-switch" style={{ marginBottom: 0 }}>
          <input
            className="form-check-input"
            type="checkbox"
            role="switch"
            checked={checked}
            disabled={disabled}
            onChange={(e) => onChange(e.target.checked)}
            title={title}
          />
        </div>
        <span style={hintStyle}>{checked ? "켜짐" : "꺼짐"}</span>
        <button type="button" className="btn btn-sm btn-outline-secondary" disabled={testing} onClick={onTest}>
          {testing ? "발송 중…" : "지금 발송(테스트)"}
        </button>
      </span>
    </label>
  );
}
