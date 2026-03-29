import { fetchFastApiJson } from "./internal-api";

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
  return fetchFastApiJson<DeletedStocksTableData>(
    `/internal/stocks/deleted${accountId ? `?account_id=${encodeURIComponent(accountId)}` : ""}`,
  );
}

export async function restoreDeletedStocks(accountId: string, tickers: string[]): Promise<number> {
  const payload = await fetchFastApiJson<{ restored_count: number }>("/internal/stocks/deleted", {
    method: "PATCH",
    body: JSON.stringify({
      account_id: accountId,
      tickers,
    }),
  });
  return Number(payload.restored_count ?? 0);
}

export async function hardDeleteStocks(accountId: string, tickers: string[]): Promise<number> {
  const payload = await fetchFastApiJson<{ deleted_count: number }>("/internal/stocks/deleted", {
    method: "DELETE",
    body: JSON.stringify({
      account_id: accountId,
      tickers,
    }),
  });
  return Number(payload.deleted_count ?? 0);
}

export type { DeletedStocksAccountItem, DeletedStocksRowItem, DeletedStocksTableData };
