/** 시스템 공용 시각 포맷 (KST). 저장/갱신 시각 등에 일관 사용한다. */

/**
 * UTC ISO(또는 tz 식별자 없는 UTC 문자열) → `2026. 6. 27(토) 오후 11:33` (KST).
 *
 * - tz 표기가 없고 `T` 가 포함된 ISO 면 UTC 로 해석하도록 `Z` 를 붙인다.
 * - 항상 `Asia/Seoul` 로 변환하고 요일을 괄호로 함께 표시한다.
 */
const WEEKDAYS = ["일", "월", "화", "수", "목", "금", "토"];

/** `2026-08-12` → `2026-08-12 (수)`.
 *  UTC 로 파싱하면 하루 밀릴 수 있어 로컬 자정으로 읽는다. */
export function formatDateWithWeekday(date: string): string {
  const parsed = new Date(`${date}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return date;
  return `${date} (${WEEKDAYS[parsed.getDay()]})`;
}

export function formatKstDateTime(input?: string | null): string {
  if (!input) return "-";
  let s = input;
  if (s.includes("T") && !s.endsWith("Z") && !/[+-]\d\d:?\d\d$/.test(s)) {
    s = `${s}Z`;
  }
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return input;

  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("ko-KR", {
      timeZone: "Asia/Seoul",
      year: "numeric",
      month: "numeric",
      day: "numeric",
      weekday: "short",
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    })
      .formatToParts(d)
      .map((p) => [p.type, p.value]),
  ) as Record<string, string>;

  return `${parts.year}. ${parts.month}. ${parts.day}(${parts.weekday}) ${parts.dayPeriod} ${parts.hour}:${parts.minute}`;
}
