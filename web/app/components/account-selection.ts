const MOMENTUM_ETF_ACCOUNT_KEY = "momentum-etf:selected-account-id";
const MOMENTUM_ETF_TICKER_TYPE_KEY = "momentum-etf:selected-ticker-type";

export function readRememberedMomentumEtfAccountId(): string | null {
  if (typeof window === "undefined") {
    return null;
  }

  const value = window.localStorage.getItem(MOMENTUM_ETF_ACCOUNT_KEY);
  return value && value.trim() ? value : null;
}

export function writeRememberedMomentumEtfAccountId(accountId: string | null | undefined) {
  if (typeof window === "undefined") {
    return;
  }

  if (!accountId || !accountId.trim()) {
    window.localStorage.removeItem(MOMENTUM_ETF_ACCOUNT_KEY);
    return;
  }

  window.localStorage.setItem(MOMENTUM_ETF_ACCOUNT_KEY, accountId.trim());
}

/** 화면 이름 — 종목풀 선택은 **화면마다 따로** 기억한다.
 *
 *  예전에는 한 키를 모든 화면이 같이 썼다. 순위에서 고른 풀이 전략 화면에도 따라와 편했지만,
 *  모멘텀은 A 풀 · 신고가는 B 풀로 작업할 때 화면을 오갈 때마다 다시 고르게 됐다.
 *  게다가 전략마다 쓰는 풀이 달라진 뒤로는, 그 전략이 안 쓰는 풀이 따라와 화면이 열리지도
 *  않았다("지원하지 않는 종목풀입니다").
 *
 *  일부 화면만 공통으로 두면 「이 화면은 왜 따라오지?」를 매번 따져야 해서 전부 나눈다.
 */
export type PoolScreen =
  | "rank"
  | "pools-backtest"
  | "market"
  | "market-stock-kor"
  | "market-stock-us"
  | "market-stock-aus"
  | "strategy-momentum"
  | "strategy-new-high"
  | "strategy-portfolio";

function tickerTypeKey(screen: PoolScreen): string {
  return `${MOMENTUM_ETF_TICKER_TYPE_KEY}:${screen}`;
}

export function readRememberedTickerType(screen: PoolScreen): string | null {
  if (typeof window === "undefined") {
    return null;
  }

  const value = window.localStorage.getItem(tickerTypeKey(screen));
  return value && value.trim() ? value : null;
}

export function writeRememberedTickerType(screen: PoolScreen, tickerType: string | null | undefined) {
  if (typeof window === "undefined") {
    return;
  }

  if (!tickerType || !tickerType.trim()) {
    window.localStorage.removeItem(tickerTypeKey(screen));
    return;
  }

  window.localStorage.setItem(tickerTypeKey(screen), tickerType.trim());
}
