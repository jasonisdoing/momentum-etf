import { fetchFastApiJson } from "./internal-api";

type DailyEditableField = {
  key: string;
  label: string;
  type: "int" | "float" | "text";
};

type DailyRow = {
  date: string;
  date_display: string;
  withdrawal_personal: number;
  withdrawal_mom: number;
  nh_principal_interest: number;
  total_expense: number;
  deposit_withdrawal: number;
  total_principal: number;
  total_assets: number;
  purchase_amount: number;
  valuation_amount: number;
  profit_loss: number;
  cumulative_profit: number;
  cumulative_return_pct: number;
  memo: string;
  exchange_rate: number;
  exchange_rate_change_pct: number;
  bucket_pct_momentum: number;
  bucket_pct_market: number;
  bucket_pct_dividend: number;
  bucket_pct_alternative: number;
  bucket_pct_cash: number;
  total_stocks: number;
  profit_count: number;
  loss_count: number;
  updated_at: string | null;
};

type DailyTableData = {
  latest_date: string;
  rows: DailyRow[];
  editable_fields: DailyEditableField[];
  read_only_keys: string[];
  core_hidden_keys: string[];
};

export async function loadDailyTableData(): Promise<DailyTableData> {
  return fetchFastApiJson<DailyTableData>("/internal/daily");
}

export async function updateDailyRow(
  date: string,
  payload: Record<string, unknown>,
): Promise<{ date: string }> {
  return fetchFastApiJson<{ date: string }>("/internal/daily", {
    method: "PATCH",
    body: JSON.stringify({
      ...payload,
      date,
    }),
  });
}

export type { DailyEditableField, DailyRow, DailyTableData };
