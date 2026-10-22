"use client";

/** 실시간 시세를 못 받은 종목 경고 — 순위·전략 화면 공용.
 *
 * 시세를 못 받으면 서버가 일간(%)을 비우고 현재가는 마지막 종가로 둔다(`effective_prices.realtime_missing`).
 * 빈 칸만으로는 장애인지 알기 어려워 화면 맨 위에 같은 문구로 알린다. 목록이 비면 그리지 않는다.
 */
const PREVIEW_COUNT = 8;

export function RealtimeMissingNotice({ tickers }: { tickers: string[] }) {
  if (tickers.length === 0) return null;
  const preview = tickers.slice(0, PREVIEW_COUNT).join(", ");
  const rest = tickers.length > PREVIEW_COUNT ? ` 외 ${tickers.length - PREVIEW_COUNT}개` : "";
  return (
    <div className="appBannerStack">
      <div className="alert alert-warning mb-0">
        실시간 시세를 받지 못한 종목 {tickers.length}개 — 현재가는 마지막 종가이고 일간(%)은 비워 둡니다. ({preview}
        {rest})
      </div>
    </div>
  );
}
