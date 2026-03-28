const MOMENTUM_ETF_ACCOUNT_KEY = "momentum-etf:selected-account-id";

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
