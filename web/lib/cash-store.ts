import { fetchFastApiJson } from "./internal-api";

type CashAccountItem = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
  country_code: string;
  currency: string;
  total_principal: number;
  cash_balance_krw: number;
  cash_balance_native: number | null;
  cash_currency: string;
  intl_shares_value: number | null;
  intl_shares_change: number | null;
  updated_at: string | null;
};

type CashAccountUpdate = {
  account_id: string;
  total_principal: number;
  cash_balance_krw: number;
  cash_balance_native: number | null;
  cash_currency: string;
  intl_shares_value: number | null;
  intl_shares_change: number | null;
};

function normalizeNumber(value: unknown): number {
  return Number(value ?? 0);
}

function normalizeNullableNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  return Number(value);
}

export async function loadCashAccounts(): Promise<CashAccountItem[]> {
  const payload = await fetchFastApiJson<{ accounts: CashAccountItem[] }>("/internal/cash");
  return payload.accounts ?? [];
}

export async function saveCashAccounts(updates: CashAccountUpdate[]): Promise<void> {
  await fetchFastApiJson("/internal/cash", {
    method: "POST",
    body: JSON.stringify({ accounts: updates }),
  });
}

export type { CashAccountItem, CashAccountUpdate };
