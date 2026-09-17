"use client";

/** 시총 상위 N 셀렉트 — 마켓/인덱스 세그먼트 토글 옆에 붙는 작은 셀렉트(미국·한국 개별주 공용).
 *
 * 응답이 시총 내림차순이라 상위 N 은 이미 받아둔 행을 자르기만 한다(재조회 없음).
 * 마지막 선택은 localStorage 에 기억해 다음 방문에도 같은 범위로 연다.
 * 키 형식은 시스템 공통(`momentum-etf:<화면>:<항목>`)을 따른다.
 */

/** 저장된 상위 N. `"all"`(전체)이거나 값이 없으면 null. */
export function readRememberedTopCount(storageKey: string): number | null {
  if (typeof window === "undefined") {
    return null;
  }
  const raw = window.localStorage.getItem(storageKey);
  if (!raw || raw === "all") {
    return null;
  }
  const parsed = Number(raw);
  // 못 읽는 값은 전체로 둔다 — 임의의 숫자로 잘라 보여주면 무엇이 적용됐는지 알 수 없다.
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

export function writeRememberedTopCount(storageKey: string, value: number | null): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(storageKey, value === null ? "all" : String(value));
}

/**
 * 지금 고른 값이 목록에 없으면 함께 노출한다 — 빼면 셀렉트가 빈칸이 되어
 * 무엇이 적용 중인지 알 수 없다.
 */
export function withCurrentTopOption(
  options: (number | null)[],
  current: number | null,
): (number | null)[] {
  if (current !== null && !options.includes(current)) {
    options.push(current);
    options.sort((a, b) => (a ?? -1) - (b ?? -1));
  }
  return options;
}

/**
 * `step` 단위로 실제 종목 수 직전까지 채운 선택지(전체 포함).
 * 전체 개수 이상은 `전체` 와 같은 결과라 만들지 않는다.
 */
export function stepTopOptions(
  step: number,
  rowCount: number,
  current: number | null,
): (number | null)[] {
  const options: (number | null)[] = [
    null,
    ...Array.from(
      { length: Math.max(0, Math.ceil(rowCount / step) - 1) },
      (_, i) => (i + 1) * step,
    ),
  ];
  return withCurrentTopOption(options, current);
}

export function TopCountSelect({
  value,
  options,
  onChange,
}: {
  value: number | null;
  options: (number | null)[];
  onChange: (next: number | null) => void;
}) {
  return (
    <select
      value={value === null ? "all" : String(value)}
      onChange={(event) => onChange(event.target.value === "all" ? null : Number(event.target.value))}
      style={{
        border: "1px solid rgba(148,163,184,0.4)",
        borderRadius: 6,
        padding: "3px 6px",
        fontSize: "var(--fs-sm)",
        marginLeft: 6,
      }}
    >
      {options.map((count) => (
        <option key={count ?? "all"} value={count === null ? "all" : String(count)}>
          {count === null ? "전체" : `상위 ${count}`}
        </option>
      ))}
    </select>
  );
}
