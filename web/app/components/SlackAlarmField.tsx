"use client";

import { LabeledSwitchField } from "./LabeledSwitchField";

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
    <LabeledSwitchField label={label} checked={checked} onChange={onChange} disabled={disabled} title={title}>
      <button type="button" className="btn btn-sm btn-outline-secondary" disabled={testing} onClick={onTest}>
        {testing ? "발송 중…" : "지금 발송(테스트)"}
      </button>
    </LabeledSwitchField>
  );
}
