import { type WithId } from "mongodb";

import { loadAccountConfigs, type AccountConfig } from "./accounts";
import { getMongoDb } from "./mongo";

type PortfolioAccountDoc = {
  account_id: string;
  total_principal?: number;
  cash_balance?: number;
  cash_balance_native?: number | null;
  cash_currency?: string | null;
  intl_shares_value?: number;
  intl_shares_change?: number;
  holdings?: Array<Record<string, unknown>>;
  updated_at?: Date | string;
};

type PortfolioMasterDoc = {
  master_id: string;
  accounts?: PortfolioAccountDoc[];
};

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

function toUpdatedAtText(value: unknown): string | null {
  if (!value) {
    return null;
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  return String(value);
}

async function loadPortfolioMaster(): Promise<WithId<PortfolioMasterDoc> | null> {
  const db = await getMongoDb();
  return db.collection<PortfolioMasterDoc>("portfolio_master").findOne({ master_id: "GLOBAL" });
}

export async function loadCashAccounts(): Promise<CashAccountItem[]> {
  const [configs, doc] = await Promise.all([loadAccountConfigs(), loadPortfolioMaster()]);
  const accountDocs = new Map((doc?.accounts ?? []).map((account) => [account.account_id, account]));

  return configs.map((config: AccountConfig) => {
    const accountDoc = accountDocs.get(config.account_id);
    const cashCurrency = String(accountDoc?.cash_currency ?? "").trim().toUpperCase() || config.currency;

    return {
      account_id: config.account_id,
      order: config.order,
      name: config.name,
      icon: config.icon,
      country_code: config.country_code,
      currency: config.currency,
      total_principal: normalizeNumber(accountDoc?.total_principal),
      cash_balance_krw: normalizeNumber(accountDoc?.cash_balance),
      cash_balance_native: normalizeNullableNumber(accountDoc?.cash_balance_native),
      cash_currency: cashCurrency,
      intl_shares_value: config.account_id === "aus_account" ? normalizeNullableNumber(accountDoc?.intl_shares_value) : null,
      intl_shares_change:
        config.account_id === "aus_account" ? normalizeNullableNumber(accountDoc?.intl_shares_change) : null,
      updated_at: toUpdatedAtText(accountDoc?.updated_at),
    };
  });
}

export async function saveCashAccounts(updates: CashAccountUpdate[]): Promise<void> {
  const db = await getMongoDb();
  const collection = db.collection<PortfolioMasterDoc>("portfolio_master");
  const doc = (await collection.findOne({ master_id: "GLOBAL" })) ?? { master_id: "GLOBAL", accounts: [] };
  const accounts = Array.isArray(doc.accounts) ? [...doc.accounts] : [];
  const now = new Date();

  for (const update of updates) {
    const index = accounts.findIndex((item) => item.account_id === update.account_id);
    if (index >= 0) {
      const current = accounts[index];
      accounts[index] = {
        ...current,
        total_principal: update.total_principal,
        cash_balance: update.cash_balance_krw,
        cash_balance_native: update.cash_balance_native,
        cash_currency: update.cash_currency,
        intl_shares_value: update.intl_shares_value ?? undefined,
        intl_shares_change: update.intl_shares_change ?? undefined,
        holdings: Array.isArray(current.holdings) ? current.holdings : [],
        updated_at: now,
      };
      continue;
    }

    accounts.push({
      account_id: update.account_id,
      total_principal: update.total_principal,
      cash_balance: update.cash_balance_krw,
      cash_balance_native: update.cash_balance_native,
      cash_currency: update.cash_currency,
      intl_shares_value: update.intl_shares_value ?? undefined,
      intl_shares_change: update.intl_shares_change ?? undefined,
      holdings: [],
      updated_at: now,
    });
  }

  await collection.updateOne({ master_id: "GLOBAL" }, { $set: { accounts } }, { upsert: true });
}

export type { CashAccountItem, CashAccountUpdate };
