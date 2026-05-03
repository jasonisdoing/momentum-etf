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

export function TickerDetailLink({ ticker, displayTicker, className }: TickerDetailLinkProps) {
  const routeTicker = normalizeTickerForDetailRoute(ticker);
  const text = String(displayTicker ?? ticker ?? "-").trim() || "-";
  const disabled = !routeTicker || routeTicker === "-" || routeTicker === "IS" || routeTicker === "__CASH__";
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
