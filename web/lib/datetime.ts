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

/** `formatKstDateTime` 과 같은 입력을 Date 로 — tz 표기 없는 ISO 는 UTC 로 읽는다. */
function parseKst(input?: string | null): Date | null {
  if (!input) return null;
  let s = input;
  if (s.includes("T") && !s.endsWith("Z") && !/[+-]\d\d:?\d\d$/.test(s)) {
    s = `${s}Z`;
  }
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
}

/**
 * 지난 시간을 `2년, 3개월, 5일, 4시간, 12분전` 으로. 값이 0인 단위는 빼고 큰 단위부터 잇는다.
 *
 * 절대 시각만으로는 "얼마나 오래된 설정인지"가 한눈에 안 들어와서 옆에 붙여 쓴다.
 * 월은 30일, 년은 365일로 계산한다(달력 길이를 따지지 않는 대략치라는 뜻).
 */
export function formatElapsedKorean(input?: string | null, now: Date = new Date()): string | null {
  const d = parseKst(input);
  if (!d) return null;

  let minutes = Math.floor((now.getTime() - d.getTime()) / 60000);
  if (minutes < 0) return null; // 미래 시각이면 표시하지 않는다
  if (minutes < 1) return "방금";

  const units: [number, string][] = [
    [365 * 24 * 60, "년"],
    [30 * 24 * 60, "개월"],
    [24 * 60, "일"],
    [60, "시간"],
    [1, "분"],
  ];
  const parts: string[] = [];
  for (const [size, label] of units) {
    const value = Math.floor(minutes / size);
    if (value > 0) {
      parts.push(`${value}${label}`);
      minutes -= value * size;
    }
  }
  return `${parts.join(", ")}전`;
}
