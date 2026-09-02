/** 종목명 하이라이트 표준.
 *
 * 레버리지 등 위험·특성 키워드를 색+굵게 강조하고 뒤에 이모지를 붙인다.
 * 여러 화면(종목풀 순위·자산 헬퍼 등)이 같은 규칙을 쓰도록 여기서만 정의한다.
 * 신규상장(백테스트 기간 미달) 종목은 호출부가 isNew 를 넘기면 맨 뒤에 🆕 를 붙인다.
 */

import type { ReactNode } from "react";

const NAME_HIGHLIGHT_KEYWORDS: Record<string, { color: string; emoji: string }> = {
  레버리지: { color: "#d63939", emoji: "💣" },
  Geared: { color: "#d63939", emoji: "💣" },
  "3X": { color: "#d63939", emoji: "💣" },
  Ultra: { color: "#d63939", emoji: "💣" },
};

const NAME_HIGHLIGHT_RE = new RegExp(`(${Object.keys(NAME_HIGHLIGHT_KEYWORDS).join("|")})`, "i");

const NEW_LISTING_BADGE = "🆕";

/** 추세가 꺾인 종목 배지. 보유종목 알람의 이동선 이탈 표시와 같은 기호다. */
const TREND_BROKEN_BADGE = "❗";

/** 장기·단기 이격 중 하나라도 음수면 보유 대상이 될 수 없다고 본다.
 *
 * 값이 없으면 판단 자체가 불가하므로 같이 이탈로 둔다(상장 직후 등 이평선 계산 불가).
 * 백엔드 `utils/holdings_alarm_service._ma_status` 의 이동선 이탈 판정과 같은 기준이다.
 */
export function isTrendBroken(
  shortDisparity: number | null | undefined,
  longDisparity: number | null | undefined,
): boolean {
  if (shortDisparity == null || longDisparity == null) return true;
  return shortDisparity < 0 || longDisparity < 0;
}

function getNameHighlight(part: string): { color: string; emoji: string } | undefined {
  const lower = part.toLowerCase();
  for (const [keyword, style] of Object.entries(NAME_HIGHLIGHT_KEYWORDS)) {
    if (keyword.toLowerCase() === lower) {
      return style;
    }
  }
  return undefined;
}

export type StockNameOptions = {
  /** 상장 기간이 백테스트 기준 창보다 짧은 종목 */
  isNew?: boolean;
  /** 단기·장기 이평선 중 하나 이상 이탈 — `isTrendBroken()` 의 결과를 넘긴다 */
  trendBroken?: boolean;
  /** 티커·종목명 검색어 — 일치하는 글자만 굵게 표시한다. */
  searchQuery?: string;
};

export function renderTextWithSearchHighlight(text: string, query: string | null | undefined): ReactNode {
  const needle = String(query ?? "").trim().toLocaleLowerCase();
  if (!needle) return text;

  const normalizedText = text.toLocaleLowerCase();
  const parts: ReactNode[] = [];
  let cursor = 0;
  let matchIndex = normalizedText.indexOf(needle, cursor);
  while (matchIndex >= 0) {
    if (matchIndex > cursor) parts.push(text.slice(cursor, matchIndex));
    const matchEnd = matchIndex + needle.length;
    parts.push(
      <strong key={`${matchIndex}:${matchEnd}`} style={{ fontWeight: 800 }}>
        {text.slice(matchIndex, matchEnd)}
      </strong>,
    );
    cursor = matchEnd;
    matchIndex = normalizedText.indexOf(needle, cursor);
  }
  if (cursor === 0) return text;
  if (cursor < text.length) parts.push(text.slice(cursor));
  return <>{parts}</>;
}

export function renderNameWithLeverageHighlight(
  name: string,
  options?: StockNameOptions,
): ReactNode {
  const newBadge = options?.isNew ? (
    <span title="신규상장 — 백테스트 기준 기간(12개월)보다 상장 기간이 짧습니다"> {NEW_LISTING_BADGE}</span>
  ) : null;
  const brokenBadge = options?.trendBroken ? (
    <span title="단기·장기 이평선 중 하나 이상 이탈"> {TREND_BROKEN_BADGE}</span>
  ) : null;

  const parts = name.split(NAME_HIGHLIGHT_RE);
  if (parts.length === 1) {
    const highlightedName = renderTextWithSearchHighlight(name, options?.searchQuery);
    return newBadge || brokenBadge ? (
      <>
        {highlightedName}
        {newBadge}
        {brokenBadge}
      </>
    ) : (
      highlightedName
    );
  }
  const emojis: string[] = [];
  const rendered = parts.map((part, index) => {
    const style = index % 2 === 1 ? getNameHighlight(part) : undefined;
    if (!style) {
      return <span key={index}>{renderTextWithSearchHighlight(part, options?.searchQuery)}</span>;
    }
    if (!emojis.includes(style.emoji)) {
      emojis.push(style.emoji);
    }
    return (
      <span key={index} style={{ color: style.color, fontWeight: 700 }}>
        {renderTextWithSearchHighlight(part, options?.searchQuery)}
      </span>
    );
  });
  return (
    <>
      {rendered}
      {emojis.length > 0 && <span> {emojis.join("")}</span>}
      {newBadge}
      {brokenBadge}
    </>
  );
}

/** 그리드 종목명 셀 표준. 말줄임 스타일·툴팁·배지를 한곳에서 정한다.
 *
 * 종목명을 보여주는 화면은 전부 이걸 쓴다 — 화면마다 배지가 붙었다 말았다 하지 않도록.
 */
export function renderStockNameCell(
  name: string | null | undefined,
  options?: StockNameOptions & {
    /** 보유종목 알람 배지 등 화면 고유의 꼬리표. 종목명 맨 뒤에 붙는다. */
    badge?: string;
  },
): ReactNode {
  const value = String(name ?? "-") || "-";
  const title = options?.trendBroken ? `${value} (추세 이탈)` : value;
  return (
    <span className="appNameCellText" title={title}>
      {renderNameWithLeverageHighlight(value, options)}
      {options?.badge ? <span> {options.badge}</span> : null}
    </span>
  );
}
