import { loadAccountConfigs } from "./accounts";
import { getMongoDb } from "./mongo";

type StockMetaDoc = {
  account_id?: string;
  ticker?: string;
  name?: string;
  bucket?: number;
  added_date?: string;
  listing_date?: string;
  deleted_reason?: string | null;
  deleted_at?: Date | string | null;
  is_deleted?: boolean;
  updated_at?: Date | string | null;
  ["1_week_avg_volume"]?: number;
  ["1_week_earn_rate"]?: number;
  ["2_week_earn_rate"]?: number;
  ["1_month_earn_rate"]?: number;
  ["3_month_earn_rate"]?: number;
  ["6_month_earn_rate"]?: number;
  ["12_month_earn_rate"]?: number;
};

type DeletedStocksAccountItem = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
};

type DeletedStocksRowItem = {
  ticker: string;
  name: string;
  bucket_id: number;
  bucket_name: string;
  listing_date: string;
  week_volume: number | null;
  return_1w: number | null;
  return_2w: number | null;
  return_1m: number | null;
  return_3m: number | null;
  return_6m: number | null;
  return_12m: number | null;
  deleted_date: string;
  deleted_reason: string;
};

type DeletedStocksTableData = {
  accounts: DeletedStocksAccountItem[];
  rows: DeletedStocksRowItem[];
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

function formatDeletedDate(value: Date | string | undefined): string {
  if (!value) {
    return "-";
  }

  if (value instanceof Date) {
    return value.toISOString().slice(0, 10);
  }

  const text = String(value).trim();
  if (!text) {
    return "-";
  }
  return text.slice(0, 10);
}

export async function loadDeletedStocksTable(accountId?: string): Promise<DeletedStocksTableData> {
  const configs = await loadAccountConfigs();
  if (configs.length === 0) {
    throw new Error("계좌 설정이 없습니다.");
  }

  const accounts: DeletedStocksAccountItem[] = configs.map((config) => ({
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
      is_deleted: true,
    })
    .project({
      ticker: 1,
      name: 1,
      bucket: 1,
      listing_date: 1,
      "1_week_avg_volume": 1,
      "1_week_earn_rate": 1,
      "2_week_earn_rate": 1,
      "1_month_earn_rate": 1,
      "3_month_earn_rate": 1,
      "6_month_earn_rate": 1,
      "12_month_earn_rate": 1,
      deleted_at: 1,
      deleted_reason: 1,
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
        listing_date: normalizeText(doc.listing_date),
        week_volume: normalizeNumber(doc["1_week_avg_volume"]),
        return_1w: normalizeNumber(doc["1_week_earn_rate"]),
        return_2w: normalizeNumber(doc["2_week_earn_rate"]),
        return_1m: normalizeNumber(doc["1_month_earn_rate"]),
        return_3m: normalizeNumber(doc["3_month_earn_rate"]),
        return_6m: normalizeNumber(doc["6_month_earn_rate"]),
        return_12m: normalizeNumber(doc["12_month_earn_rate"]),
        deleted_date: formatDeletedDate(doc.deleted_at),
        deleted_reason: normalizeText(doc.deleted_reason),
      };
    })
    .sort((left, right) => {
      const bucketDiff = left.bucket_id - right.bucket_id;
      if (bucketDiff !== 0) {
        return bucketDiff;
      }
      return right.deleted_date.localeCompare(left.deleted_date);
    });

  return {
    accounts,
    rows,
    account_id: targetAccountId,
  };
}

export async function restoreDeletedStocks(accountId: string, tickers: string[]): Promise<number> {
  const accountNorm = String(accountId ?? "").trim().toLowerCase();
  const tickerList = tickers
    .map((ticker) => String(ticker ?? "").trim().toUpperCase())
    .filter(Boolean);

  if (!accountNorm || tickerList.length === 0) {
    throw new Error("복구할 종목을 선택하세요.");
  }

  const db = await getMongoDb();
  const now = new Date();
  const result = await db.collection<StockMetaDoc>("stock_meta").updateMany(
    {
      account_id: accountNorm,
      ticker: { $in: tickerList },
      is_deleted: true,
    },
    {
      $set: {
        is_deleted: false,
        deleted_at: null,
        deleted_reason: null,
        added_date: now.toISOString().slice(0, 10),
        updated_at: now,
      },
    },
  );

  return result.modifiedCount;
}

export async function hardDeleteStocks(accountId: string, tickers: string[]): Promise<number> {
  const accountNorm = String(accountId ?? "").trim().toLowerCase();
  const tickerList = tickers
    .map((ticker) => String(ticker ?? "").trim().toUpperCase())
    .filter(Boolean);

  if (!accountNorm || tickerList.length === 0) {
    throw new Error("삭제할 종목을 선택하세요.");
  }

  const db = await getMongoDb();
  const result = await db.collection<StockMetaDoc>("stock_meta").deleteMany({
    account_id: accountNorm,
    ticker: { $in: tickerList },
    is_deleted: true,
  });

  return result.deletedCount;
}

export type { DeletedStocksAccountItem, DeletedStocksRowItem, DeletedStocksTableData };
