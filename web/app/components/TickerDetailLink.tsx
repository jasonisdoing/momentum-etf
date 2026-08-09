"use client";

import { IconExternalLink } from "@tabler/icons-react";

type TickerDetailLinkProps = {
  ticker: string | null | undefined;
  displayTicker?: string | null;
  className?: string;
};

function normalizeTickerForDetailRoute(ticker: string | null | undefined): string {
  const upper = String(ticker || "").trim().toUpperCase();
  // 호주 시장 접두사(ASX:)만 보존. 미국 티커는 접두사 없이 사용.
  if (upper.startsWith("ASX:")) {
    return upper;
  }
  if (upper.endsWith(".AX")) {
    return `ASX:${upper.slice(0, -3)}`;
  }
  if (upper.endsWith(".KS") || upper.endsWith(".KQ")) {
    return upper.split(".")[0];
  }
  return upper;
}

/** 외부 조회용 — `ASX:` 접두사를 벗긴 티커. 화면 표시에는 쓰지 않는다. */
export function stripAsxPrefix(ticker: string | null | undefined): string {
  const text = String(ticker ?? "").trim();
  return text.startsWith("ASX:") ? text.slice(4) : text;
}

export function TickerDetailLink({ ticker, displayTicker, className }: TickerDetailLinkProps) {
  const routeTicker = normalizeTickerForDetailRoute(ticker);
  // 호주 종목은 `ASX:` 를 붙인 채로 보여준다 — 미국에 같은 티커가 있어 구분이 필요하다.
  // (docs/developer_guide.md "호주 티커(ASX) 식별 규칙")
  const text = String(displayTicker ?? ticker ?? "-").trim() || "-";
  // IS(International Shares)는 실제 상장 종목이 아니라 /assets 에서 수동 입력하는 고정자산이라
  // 상세 화면이 없다. 접두사가 붙은 `ASX:IS` 로 와도 똑같이 링크를 걸지 않는다.
  const disabled =
    !routeTicker ||
    routeTicker === "-" ||
    routeTicker === "IS" ||
    routeTicker === "ASX:IS" ||
    routeTicker === "__CASH__";
  const href = `/ticker?ticker=${encodeURIComponent(routeTicker)}`;

  if (disabled) {
    return (
      <span className={className ? `appCodeText ${className}` : "appCodeText"}>
        {text === "__CASH__" ? "-" : text}
      </span>
    );
  }

  return (
    <span className={className ? `tickerDetailLink ${className}` : "tickerDetailLink"}>
      <span className="appCodeText tickerDetailLinkText">{text}</span>
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="tickerDetailLinkButton"
        aria-label={`${text} 상세 보기`}
        title="상세 보기"
        onMouseDown={(event) => event.stopPropagation()}
        onClick={(event) => event.stopPropagation()}
      >
        <IconExternalLink size={12} stroke={2.2} />
      </a>
    </span>
  );
}
