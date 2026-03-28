import { fetchFastApiJson } from "./internal-api";

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

type StockValidationResult = {
  ticker: string;
  name: string;
  listing_date: string;
  status: "active" | "deleted" | "new";
  is_deleted: boolean;
  deleted_reason: string;
  bucket_id: number;
  account_id: string;
  country_code: string;
};

type StockCreateResult = {
  ticker: string;
  name: string;
  listing_date: string;
  bucket_id: number;
  bucket_name: string;
  status: "active" | "deleted" | "new";
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
  return fetchFastApiJson<StocksTableData>(
    `/internal/stocks${accountId ? `?account_id=${encodeURIComponent(accountId)}` : ""}`,
  );
}

export async function updateStockBucket(accountId: string, ticker: string, bucketId: number): Promise<void> {
  await fetchFastApiJson("/internal/stocks", {
    method: "PATCH",
    body: JSON.stringify({
      account_id: accountId,
      ticker,
      bucket_id: bucketId,
    }),
  });
}

export async function softDeleteStock(accountId: string, ticker: string, reason?: string): Promise<void> {
  await fetchFastApiJson("/internal/stocks", {
    method: "DELETE",
    body: JSON.stringify({
      account_id: accountId,
      ticker,
      reason,
    }),
  });
}

export async function validateStockCandidate(accountId: string, ticker: string): Promise<StockValidationResult> {
  return fetchFastApiJson<StockValidationResult>("/internal/stocks/validate", {
    method: "POST",
    body: JSON.stringify({
      account_id: accountId,
      ticker,
    }),
  });
}

export async function addStockCandidate(accountId: string, ticker: string, bucketId: number): Promise<StockCreateResult> {
  return fetchFastApiJson<StockCreateResult>("/internal/stocks", {
    method: "POST",
    body: JSON.stringify({
      account_id: accountId,
      ticker,
      bucket_id: bucketId,
    }),
  });
}

export type { StockCreateResult, StockValidationResult, StocksAccountItem, StocksRowItem, StocksTableData };
