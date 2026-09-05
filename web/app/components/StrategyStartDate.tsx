"use client";

/** 전략·종목풀별 시작일 입력 — 세 전략 화면이 같은 형태를 사용한다. */
export function StrategyStartDate({ value, disabled, onChange }: {
  value: string | null | undefined;
  disabled: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <label className="appLabeledField">
      <span className="appLabeledFieldLabel">전략 시작일</span>
      <input
        type="date"
        required
        value={value ?? ""}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        title="시작일을 선택하고 저장하면 이 날짜부터 전략을 계산합니다."
        style={{ width: 160, height: 30, padding: "4px 8px", borderRadius: 4,
          border: "1px solid var(--app-border)", background: "var(--app-bg)",
          color: "var(--text-primary)", fontSize: "var(--fs-sm)" }}
      />
    </label>
  );
}
