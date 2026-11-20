"use client";

/** 그리드 셀 체크박스 — AG Grid 편집 사이클을 거치지 않고 초안 값을 그대로 그린다.
 *
 *  `cellDataType: "boolean"` + `editable` 로 두면 클릭이 편집 시작과 값 토글을 겸하는데,
 *  `stopEditingWhenCellsLoseFocus` 와 맞물려 켤 때 편집 중 상태가 남아 체크 표시가 보이지
 *  않았다(값은 바뀌어 저장은 됐다). 초안 값을 그대로 그리고 onChange 로만 바꾸면 화면과 값이 항상 같다.
 *  셀은 `GRID_CHECKBOX_CELL_STYLE` 로 가운데 정렬한다. */
export function GridCheckbox({
  checked,
  disabled,
  label,
  onChange,
}: {
  checked: boolean;
  disabled: boolean;
  label: string;
  onChange: (checked: boolean) => void;
}) {
  return (
    <input
      type="checkbox"
      className="form-check-input"
      style={{ margin: 0, cursor: disabled ? "not-allowed" : "pointer" }}
      aria-label={label}
      checked={checked}
      disabled={disabled}
      onChange={(event) => onChange(event.target.checked)}
    />
  );
}

export const GRID_CHECKBOX_CELL_STYLE = { display: "flex", alignItems: "center", justifyContent: "center" } as const;
