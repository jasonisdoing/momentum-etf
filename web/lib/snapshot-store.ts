import { type ObjectId, type WithId } from "mongodb";

import { loadAccountConfigs } from "./accounts";
import { getMongoDb } from "./mongo";

type SnapshotAccountDoc = {
  account_id: string;
  total_assets?: number;
  total_principal?: number;
  cash_balance?: number;
  valuation_krw?: number;
};

type DailySnapshotDoc = {
  _id: ObjectId;
  snapshot_date: string;
  total_assets?: number;
  total_principal?: number;
  cash_balance?: number;
  valuation_krw?: number;
  accounts?: SnapshotAccountDoc[];
};

type SnapshotAccountItem = {
  account_id: string;
  account_name: string;
  order: number;
  total_assets: number;
  total_principal: number;
  cash_balance: number;
  valuation_krw: number;
};

type SnapshotListItem = {
  id: string;
  snapshot_date: string;
  total_assets: number;
  total_principal: number;
  cash_balance: number;
  valuation_krw: number;
  account_count: number;
  accounts: SnapshotAccountItem[];
};

function normalizeNumber(value: unknown): number {
  return Number(value ?? 0);
}

async function loadDailySnapshots(): Promise<Array<WithId<DailySnapshotDoc>>> {
  const db = await getMongoDb();
  return db.collection<DailySnapshotDoc>("daily_snapshots").find().sort("snapshot_date", -1).toArray();
}

export async function loadSnapshotList(): Promise<SnapshotListItem[]> {
  const [configs, docs] = await Promise.all([loadAccountConfigs(), loadDailySnapshots()]);
  const accountMap = new Map(
    configs.map((config) => [config.account_id, { name: config.name, order: config.order }]),
  );

  return docs.map((doc) => {
    const accounts = (doc.accounts ?? [])
      .map((account) => {
        const config = accountMap.get(account.account_id);
        return {
          account_id: account.account_id,
          account_name: config?.name ?? account.account_id,
          order: config?.order ?? 999,
          total_assets: normalizeNumber(account.total_assets),
          total_principal: normalizeNumber(account.total_principal),
          cash_balance: normalizeNumber(account.cash_balance),
          valuation_krw: normalizeNumber(account.valuation_krw),
        };
      })
      .sort((left, right) => left.order - right.order);

    return {
      id: String(doc._id),
      snapshot_date: String(doc.snapshot_date ?? ""),
      total_assets: normalizeNumber(doc.total_assets),
      total_principal: normalizeNumber(doc.total_principal),
      cash_balance: normalizeNumber(doc.cash_balance),
      valuation_krw: normalizeNumber(doc.valuation_krw),
      account_count: accounts.length,
      accounts,
    };
  });
}

export type { SnapshotAccountItem, SnapshotListItem };
