"use client";

import type { CSSProperties, ReactNode } from "react";

const hintStyle: CSSProperties = { color: "var(--text-muted)", fontSize: "var(--fs-sm)" };

/** 라벨 + 스위치 + 켜짐/꺼짐 표시 — 헤더의 켜고 끄는 설정 공용(슬랙 알람·권장 간격 사용). ``children`` 은 스위치 오른쪽에 붙는다. */
export function LabeledSwitchField({
  label,
  checked,
  onChange,
  disabled,
  title,
  children,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled: boolean;
  /** 스위치 설명(마우스 오버). */
  title: string;
  children?: ReactNode;
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
        {children}
      </span>
    </label>
  );
}
