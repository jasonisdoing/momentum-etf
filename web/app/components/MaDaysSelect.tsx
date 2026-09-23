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

/** 이평선 표기 — 종류가 있으면 "SMA 20일", 없으면 "20일". 표기 규칙은 여기 한 곳이다. */
export function formatMaLabel(days: number | null | undefined, maType?: string | null): string {
  if (days == null) return "-";
  const type = String(maType ?? "").trim().toUpperCase();
  return type ? `${type} ${days}일` : `${days}일`;
}

/**
 * 이평선 **종류** 셀렉트 — 종목풀별 설정(SMA/EMA). 일수와 같은 자리에서 고른다.
 * 시장지수 추세·레버리지는 종목풀에 속하지 않아 `config.MOVING_AVERAGE_TYPE` 공통값을 쓴다.
 */
export function MaTypeSelect({
  value,
  options,
  onChange,
  disabled,
  title,
}: {
  value: string | null | undefined;
  options: string[] | undefined;
  onChange: (maType: string) => void;
  disabled?: boolean;
  title?: string;
}) {
  const list = options ?? [];
  const current = String(value ?? "").trim().toUpperCase();
  const outside = current !== "" && !list.includes(current);
  return (
    <select
      className="form-select form-select-sm appMaDaysSelect"
      value={current}
      disabled={disabled || list.length === 0}
      title={title}
      onChange={(event) => onChange(event.target.value)}
    >
      {current === "" ? <option value="" disabled>종류 선택</option> : null}
      {list.map((type) => (
        <option key={type} value={type}>
          {type}
        </option>
      ))}
      {outside ? <option value={current}>{current} (선택지 밖)</option> : null}
    </select>
  );
}

export function MaDaysSelect({
  value,
  options,
  maType,
  onChange,
  disabled,
  title,
}: {
  value: number | null | undefined;
  options: number[] | undefined;
  /** 그 종목풀의 이평 종류 — 주면 "SMA 20일" 로 표기한다. 없으면 "20일". */
  maType?: string | null;
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
          {formatMaLabel(days, maType)}
        </option>
      ))}
      {outside ? <option value={String(value)}>{formatMaLabel(value, maType)} (선택지 밖)</option> : null}
    </select>
  );
}
