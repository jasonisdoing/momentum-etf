"use client";

/**
 * 이평선 일수 셀렉트 — 시스템 공용. 종목풀 설정·순위·종목풀 백테스트·모멘텀·신고가·알림이
 * 전부 이 컴포넌트를 쓴다. 선택지는 백엔드(`utils/ma_options.py`)가 응답으로 내려준 목록만
 * 렌더하고, 폭·표기("20일")는 여기(`appMaDaysSelect`)서만 정한다 — 화면은 값과 핸들러만 준다.
 */

export type MaOptionsPayload = {
  short_ma_options: number[];
  long_ma_options: number[];
};

export function MaDaysSelect({
  value,
  options,
  onChange,
  disabled,
  title,
}: {
  value: number | null | undefined;
  options: number[] | undefined;
  onChange: (days: number) => void;
  disabled?: boolean;
  title?: string;
}) {
  const list = options ?? [];
  // 선택지가 바뀌어 저장값이 목록 밖이면 숨기지 않고 그대로 보여줘 사용자가 바꾸게 한다.
  const outside = value != null && !list.includes(value);
  return (
    <select
      className="form-select form-select-sm appMaDaysSelect"
      value={value == null ? "" : String(value)}
      disabled={disabled || list.length === 0}
      title={title}
      onChange={(event) => onChange(Number(event.target.value))}
    >
      {list.map((days) => (
        <option key={days} value={days}>
          {days}일
        </option>
      ))}
      {outside ? <option value={String(value)}>{value}일 (선택지 밖)</option> : null}
    </select>
  );
}
