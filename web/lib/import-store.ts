import { fetchFastApiJson } from "./internal-api";

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

export async function parseBulkImportText(rawText: string): Promise<ImportPreviewData> {
  return fetchFastApiJson<ImportPreviewData>("/internal/import/preview", {
    method: "POST",
    body: JSON.stringify({ text: String(rawText ?? "") }),
  });
}

export async function saveBulkImportRows(rows: ParsedImportRow[]): Promise<{ updated_accounts: number }> {
  return fetchFastApiJson<{ updated_accounts: number }>("/internal/import/save", {
    method: "POST",
    body: JSON.stringify({ rows }),
  });
}

export type { ImportPreviewData, ParsedImportRow };
