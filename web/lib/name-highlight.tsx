/** 종목명 하이라이트 표준.
 *
 * 레버리지 등 위험·특성 키워드를 색+굵게 강조하고 뒤에 이모지를 붙인다.
 * 여러 화면(종목풀 순위·자산 헬퍼 등)이 같은 규칙을 쓰도록 여기서만 정의한다.
 */

import type { ReactNode } from "react";

const NAME_HIGHLIGHT_KEYWORDS: Record<string, { color: string; emoji: string }> = {
  레버리지: { color: "#d63939", emoji: "💣" },
  Geared: { color: "#d63939", emoji: "💣" },
  "3X": { color: "#d63939", emoji: "💣" },
  Ultra: { color: "#d63939", emoji: "💣" },
};

const NAME_HIGHLIGHT_RE = new RegExp(`(${Object.keys(NAME_HIGHLIGHT_KEYWORDS).join("|")})`, "i");

function getNameHighlight(part: string): { color: string; emoji: string } | undefined {
  const lower = part.toLowerCase();
  for (const [keyword, style] of Object.entries(NAME_HIGHLIGHT_KEYWORDS)) {
    if (keyword.toLowerCase() === lower) {
      return style;
    }
  }
  return undefined;
}

export function renderNameWithLeverageHighlight(name: string): ReactNode {
  const parts = name.split(NAME_HIGHLIGHT_RE);
  if (parts.length === 1) {
    return name;
  }
  const emojis: string[] = [];
  const rendered = parts.map((part, index) => {
    const style = index % 2 === 1 ? getNameHighlight(part) : undefined;
    if (!style) {
      return <span key={index}>{part}</span>;
    }
    if (!emojis.includes(style.emoji)) {
      emojis.push(style.emoji);
    }
    return (
      <span key={index} style={{ color: style.color, fontWeight: 700 }}>
        {part}
      </span>
    );
  });
  return (
    <>
      {rendered}
      {emojis.length > 0 && <span> {emojis.join("")}</span>}
    </>
  );
}
