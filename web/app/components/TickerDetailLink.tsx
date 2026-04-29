"use client";

import { IconExternalLink } from "@tabler/icons-react";
import { useRouter } from "next/navigation";

type TickerDetailLinkProps = {
  ticker: string | null | undefined;
  displayTicker?: string | null;
  className?: string;
};

function normalizeTickerForDetailRoute(ticker: string | null | undefined): string {
  const upper = String(ticker || "").trim().toUpperCase().replace(/^ASX:/, "");
  if (upper.endsWith(".KS") || upper.endsWith(".KQ") || upper.endsWith(".AX")) {
    return upper.split(".")[0];
  }
  return upper;
}

export function TickerDetailLink({ ticker, displayTicker, className }: TickerDetailLinkProps) {
  const router = useRouter();
  const routeTicker = normalizeTickerForDetailRoute(ticker);
  const text = String(displayTicker ?? ticker ?? "-").trim() || "-";
  const disabled = !routeTicker || routeTicker === "-" || routeTicker === "IS" || routeTicker === "__CASH__";

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
      <button
        type="button"
        className="tickerDetailLinkButton"
        aria-label={`${text} 상세 보기`}
        title="상세 보기"
        onMouseDown={(event) => event.stopPropagation()}
        onClick={(event) => {
          event.stopPropagation();
          router.push(`/ticker?ticker=${encodeURIComponent(routeTicker)}`);
        }}
      >
        <IconExternalLink size={12} stroke={2.2} />
      </button>
    </span>
  );
}
