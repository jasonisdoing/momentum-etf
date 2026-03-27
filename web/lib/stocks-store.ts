import { loadAccountConfigs } from "./accounts";
import { getMongoDb } from "./mongo";

type StockMetaDoc = {
  account_id?: string;
  ticker?: string;
  name?: string;
  bucket?: number;
  added_date?: string;
  listing_date?: string;
  deleted_reason?: string;
  deleted_at?: Date | string;
  is_deleted?: boolean;
  updated_at?: Date | string;
  ["1_week_avg_volume"]?: number;
  ["1_week_earn_rate"]?: number;
  ["2_week_earn_rate"]?: number;
  ["1_month_earn_rate"]?: number;
  ["3_month_earn_rate"]?: number;
  ["6_month_earn_rate"]?: number;
  ["12_month_earn_rate"]?: number;
};

type StocksAccountItem = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
};

type StocksRowItem = {
  ticker: string;
  name: string;
  bucket_id: number;
  bucket_name: string;
  added_date: string;
  listing_date: string;
  week_volume: number | null;
  return_1w: number | null;
  return_2w: number | null;
  return_1m: number | null;
  return_3m: number | null;
  return_6m: number | null;
  return_12m: number | null;
};

type StocksTableData = {
  accounts: StocksAccountItem[];
  rows: StocksRowItem[];
  account_id: string;
};

const BUCKETS: Record<number, string> = {
  1: "1. 모멘텀",
  2: "2. 혁신기술",
  3: "3. 시장지수",
  4: "4. 배당방어",
  5: "5. 대체헷지",
};

function normalizeNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeText(value: unknown, fallback = "-"): string {
  const text = String(value ?? "").trim();
  return text || fallback;
}

export function getBucketOptions(): Array<{ id: number; name: string }> {
  return Object.entries(BUCKETS).map(([id, name]) => ({ id: Number(id), name }));
}

export async function loadStocksTable(accountId?: string): Promise<StocksTableData> {
  const configs = await loadAccountConfigs();
  if (configs.length === 0) {
    throw new Error("계좌 설정이 없습니다.");
  }

  const accounts: StocksAccountItem[] = configs.map((config) => ({
    account_id: config.account_id,
    order: config.order,
    name: config.name,
    icon: config.icon,
  }));

  const targetAccountId = String(accountId ?? accounts[0]?.account_id ?? "").trim().toLowerCase();
  if (!targetAccountId) {
    throw new Error("계좌를 찾을 수 없습니다.");
  }

  const db = await getMongoDb();
  const docs = await db
    .collection<StockMetaDoc>("stock_meta")
    .find({
      account_id: targetAccountId,
      is_deleted: { $ne: true },
    })
    .project({
      ticker: 1,
      name: 1,
      bucket: 1,
      added_date: 1,
      listing_date: 1,
      "1_week_avg_volume": 1,
      "1_week_earn_rate": 1,
      "2_week_earn_rate": 1,
      "1_month_earn_rate": 1,
      "3_month_earn_rate": 1,
      "6_month_earn_rate": 1,
      "12_month_earn_rate": 1,
    })
    .toArray();

  const rows = docs
    .map((doc) => {
      const bucketId = Number(doc.bucket ?? 1);
      return {
        ticker: normalizeText(doc.ticker, ""),
        name: normalizeText(doc.name, ""),
        bucket_id: bucketId,
        bucket_name: BUCKETS[bucketId] ?? BUCKETS[1],
        added_date: normalizeText(doc.added_date),
        listing_date: normalizeText(doc.listing_date),
        week_volume: normalizeNumber(doc["1_week_avg_volume"]),
        return_1w: normalizeNumber(doc["1_week_earn_rate"]),
        return_2w: normalizeNumber(doc["2_week_earn_rate"]),
        return_1m: normalizeNumber(doc["1_month_earn_rate"]),
        return_3m: normalizeNumber(doc["3_month_earn_rate"]),
        return_6m: normalizeNumber(doc["6_month_earn_rate"]),
        return_12m: normalizeNumber(doc["12_month_earn_rate"]),
      };
    })
    .sort((left, right) => {
      const bucketDiff = left.bucket_id - right.bucket_id;
      if (bucketDiff !== 0) {
        return bucketDiff;
      }
      return (right.return_1w ?? Number.NEGATIVE_INFINITY) - (left.return_1w ?? Number.NEGATIVE_INFINITY);
    });

  return {
    accounts,
    rows,
    account_id: targetAccountId,
  };
}

export async function updateStockBucket(accountId: string, ticker: string, bucketId: number): Promise<void> {
  const db = await getMongoDb();
  const result = await db.collection<StockMetaDoc>("stock_meta").updateOne(
    {
      account_id: String(accountId ?? "").trim().toLowerCase(),
      ticker: String(ticker ?? "").trim().toUpperCase(),
      is_deleted: { $ne: true },
    },
    {
      $set: {
        bucket: Number(bucketId),
        updated_at: new Date(),
      },
    },
  );

  if (result.matchedCount === 0) {
    throw new Error("수정할 종목을 찾을 수 없습니다.");
  }
}

export async function softDeleteStock(accountId: string, ticker: string, reason?: string): Promise<void> {
  const db = await getMongoDb();
  const result = await db.collection<StockMetaDoc>("stock_meta").updateOne(
    {
      account_id: String(accountId ?? "").trim().toLowerCase(),
      ticker: String(ticker ?? "").trim().toUpperCase(),
      is_deleted: { $ne: true },
    },
    {
      $set: {
        is_deleted: true,
        deleted_reason: String(reason ?? "").trim(),
        deleted_at: new Date(),
        updated_at: new Date(),
      },
    },
  );

  if (result.matchedCount === 0) {
    throw new Error("삭제할 종목을 찾을 수 없습니다.");
  }
}

export type { StocksAccountItem, StocksRowItem, StocksTableData };
