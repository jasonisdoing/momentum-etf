import { type WithId } from "mongodb";

import { loadAccountConfigs, type AccountConfig } from "./accounts";
import { getMongoDb } from "./mongo";

type PortfolioHolding = {
  ticker?: string;
  name?: string;
  quantity?: number;
  average_buy_price?: number;
  currency?: string;
  bucket?: number;
  first_buy_date?: string;
};

type PortfolioAccount = {
  account_id: string;
  total_principal?: number;
  cash_balance?: number;
  cash_balance_native?: number;
  cash_currency?: string;
  intl_shares_value?: number;
  intl_shares_change?: number;
  holdings?: PortfolioHolding[];
  updated_at?: Date | string;
};

type PortfolioMasterDoc = {
  master_id: string;
  accounts?: PortfolioAccount[];
};

type StockMetaDoc = {
  account_id?: string;
  ticker?: string;
  name?: string;
  is_deleted?: boolean;
};

type ParsedImportRow = {
  account_name: string;
  account_id: string;
  currency: string;
  bucket_text: string;
  bucket: number;
  ticker: string;
  name: string;
  quantity: number;
  average_buy_price: number;
};

type ImportPreviewData = {
  rows: ParsedImportRow[];
  account_count: number;
  row_count: number;
};

const BUCKET_NAME_TO_ID = new Map<string, number>([
  ["1. 모멘텀", 1],
  ["2. 혁신기술", 2],
  ["3. 시장지수", 3],
  ["4. 배당방어", 4],
  ["5. 대체헷지", 5],
]);

function normalizeTicker(value: string): string {
  const trimmed = String(value ?? "").trim();
  if (!trimmed.includes(":")) {
    return trimmed.toUpperCase();
  }
  return trimmed.split(":").at(-1)?.trim().toUpperCase() ?? "";
}

function normalizeNumericText(value: string): number {
  const normalized = String(value ?? "").replaceAll(",", "").replace(/[^\d.-]/g, "");
  const parsed = Number(normalized);
  if (!Number.isFinite(parsed)) {
    return 0;
  }
  return parsed;
}

function getLastBusinessDayText(): string {
  const formatter = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  const current = new Date();

  while (true) {
    const weekday = new Intl.DateTimeFormat("en-US", {
      timeZone: "Asia/Seoul",
      weekday: "short",
    }).format(current);
    if (weekday !== "Sat" && weekday !== "Sun") {
      return formatter.format(current);
    }
    current.setUTCDate(current.getUTCDate() - 1);
  }
}

function getAccountNameMap(configs: AccountConfig[]): Map<string, AccountConfig> {
  const entries: Array<[string, AccountConfig]> = [];

  for (const config of configs) {
    const plainName = config.name.trim();
    const orderedName = `${config.order}. ${plainName}`;
    entries.push([plainName, config]);
    entries.push([orderedName, config]);
  }

  return new Map(entries);
}

function buildAccountOrderMap(configs: AccountConfig[]): Map<string, number> {
  return new Map(configs.map((config) => [config.account_id, config.order]));
}

function parseTsvRows(rawText: string): string[][] {
  return rawText
    .split(/\r?\n/)
    .map((line) => line.trimEnd())
    .filter((line) => line.trim().length > 0)
    .map((line) => line.split("\t"));
}

export async function parseBulkImportText(rawText: string): Promise<ImportPreviewData> {
  const text = String(rawText ?? "");
  if (!text.trim()) {
    throw new Error("붙여넣은 데이터가 비어 있습니다.");
  }

  const configs = await loadAccountConfigs();
  const accountNameMap = getAccountNameMap(configs);
  const accountOrderMap = buildAccountOrderMap(configs);
  const parsedLines = parseTsvRows(text);

  if (parsedLines.length === 0) {
    throw new Error("붙여넣은 데이터가 비어 있습니다.");
  }

  const errors: string[] = [];
  const rows: ParsedImportRow[] = [];

  parsedLines.forEach((columns, index) => {
    const lineNumber = index + 1;
    if (columns.length < 7) {
      errors.push(`${lineNumber}행: TSV 7컬럼이 필요합니다.`);
      return;
    }

    const [accountNameRaw, currencyRaw, bucketTextRaw, tickerRaw, stockNameRaw, quantityRaw, priceRaw] = columns;
    const accountName = String(accountNameRaw ?? "").trim();
    const currency = String(currencyRaw ?? "").trim().toUpperCase();
    const bucketText = String(bucketTextRaw ?? "").trim();
    const ticker = normalizeTicker(tickerRaw);
    const stockName = String(stockNameRaw ?? "").trim();
    const quantity = normalizeNumericText(quantityRaw);
    const averageBuyPrice = normalizeNumericText(priceRaw);

    const account = accountNameMap.get(accountName);
    if (!account) {
      errors.push(`${lineNumber}행: 계좌 '${accountName}'을(를) 찾을 수 없습니다.`);
      return;
    }

    const bucket = BUCKET_NAME_TO_ID.get(bucketText);
    if (!bucket) {
      errors.push(`${lineNumber}행: 버킷 '${bucketText}'을(를) 찾을 수 없습니다.`);
      return;
    }

    if (!ticker) {
      errors.push(`${lineNumber}행: 티커가 비어 있습니다.`);
      return;
    }

    rows.push({
      account_name: accountName,
      account_id: account.account_id,
      currency,
      bucket_text: bucketText,
      bucket,
      ticker,
      name: stockName,
      quantity,
      average_buy_price: averageBuyPrice,
    });
  });

  if (errors.length > 0) {
    throw new Error(errors.join("\n"));
  }

  rows.sort((left, right) => {
    const orderDiff = (accountOrderMap.get(left.account_id) ?? 999) - (accountOrderMap.get(right.account_id) ?? 999);
    if (orderDiff !== 0) {
      return orderDiff;
    }
    return left.ticker.localeCompare(right.ticker, "ko");
  });

  return {
    rows,
    row_count: rows.length,
    account_count: new Set(rows.map((row) => row.account_id)).size,
  };
}

async function loadReferenceNameMap(accountId: string, tickers: string[]): Promise<Map<string, string>> {
  const db = await getMongoDb();
  const docs = await db
    .collection<WithId<StockMetaDoc>>("stock_meta")
    .find({
      account_id: accountId,
      ticker: { $in: tickers },
      is_deleted: { $ne: true },
    })
    .project({ ticker: 1, name: 1 })
    .toArray();

  const byAccount = new Map<string, string>();
  for (const doc of docs) {
    const ticker = String(doc.ticker ?? "").trim().toUpperCase();
    const name = String(doc.name ?? "").trim();
    if (ticker && name) {
      byAccount.set(ticker, name);
    }
  }

  const missingTickers = tickers.filter((ticker) => !byAccount.has(ticker));
  if (missingTickers.length === 0) {
    return byAccount;
  }

  const fallbackDocs = await db
    .collection<WithId<StockMetaDoc>>("stock_meta")
    .find({
      ticker: { $in: missingTickers },
      is_deleted: { $ne: true },
    })
    .project({ ticker: 1, name: 1 })
    .toArray();

  for (const doc of fallbackDocs) {
    const ticker = String(doc.ticker ?? "").trim().toUpperCase();
    const name = String(doc.name ?? "").trim();
    if (ticker && name && !byAccount.has(ticker)) {
      byAccount.set(ticker, name);
    }
  }

  return byAccount;
}

function toIsoText(value: Date | string | undefined): string | null {
  if (!value) {
    return null;
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  return String(value);
}

export async function saveBulkImportRows(rows: ParsedImportRow[]): Promise<{ updated_accounts: number }> {
  if (rows.length === 0) {
    throw new Error("반영할 데이터가 없습니다.");
  }

  const db = await getMongoDb();
  const doc =
    (await db.collection<PortfolioMasterDoc>("portfolio_master").findOne({ master_id: "GLOBAL" })) ??
    ({ master_id: "GLOBAL", accounts: [] } as PortfolioMasterDoc);

  const existingAccounts = Array.isArray(doc.accounts) ? [...doc.accounts] : [];
  const groupedRows = new Map<string, ParsedImportRow[]>();

  for (const row of rows) {
    const current = groupedRows.get(row.account_id) ?? [];
    current.push(row);
    groupedRows.set(row.account_id, current);
  }

  const now = new Date();
  const defaultFirstBuyDate = getLastBusinessDayText();
  let updatedAccounts = 0;

  for (const [accountId, accountRows] of groupedRows.entries()) {
    const existingAccount = existingAccounts.find((account) => account.account_id === accountId);
    const existingHoldings = Array.isArray(existingAccount?.holdings) ? existingAccount.holdings : [];
    const existingDateMap = new Map<string, string>();
    const existingNameMap = new Map<string, string>();

    for (const holding of existingHoldings) {
      const ticker = String(holding.ticker ?? "").trim().toUpperCase();
      if (!ticker) {
        continue;
      }
      const firstBuyDate = String(holding.first_buy_date ?? "").trim();
      const name = String(holding.name ?? "").trim();
      if (firstBuyDate) {
        existingDateMap.set(ticker, firstBuyDate);
      }
      if (name) {
        existingNameMap.set(ticker, name);
      }
    }

    const tickers = accountRows.map((row) => row.ticker);
    const referenceNameMap = await loadReferenceNameMap(accountId, tickers);
    const nextHoldings: PortfolioHolding[] = accountRows.map((row) => ({
      ticker: row.ticker,
      name: existingNameMap.get(row.ticker) ?? referenceNameMap.get(row.ticker) ?? row.name ?? row.ticker,
      quantity: Math.floor(Number(row.quantity ?? 0)),
      average_buy_price: Number(row.average_buy_price ?? 0),
      currency: row.currency,
      bucket: row.bucket,
      first_buy_date: existingDateMap.get(row.ticker) ?? defaultFirstBuyDate,
    }));

    if (existingAccount) {
      existingAccount.holdings = nextHoldings;
      existingAccount.updated_at = now;
    } else {
      existingAccounts.push({
        account_id: accountId,
        total_principal: 0,
        cash_balance: 0,
        holdings: nextHoldings,
        updated_at: now,
      });
    }

    updatedAccounts += 1;
  }

  await db.collection<PortfolioMasterDoc>("portfolio_master").updateOne(
    { master_id: "GLOBAL" },
    {
      $set: {
        master_id: "GLOBAL",
        accounts: existingAccounts,
      },
    },
    { upsert: true },
  );

  return { updated_accounts: updatedAccounts };
}

export type { ImportPreviewData, ParsedImportRow };
