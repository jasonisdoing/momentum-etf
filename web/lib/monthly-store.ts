import { fetchFastApiJson } from "./internal-api";

type MonthlyEditableField = {
  key: string;
  label: string;
  type: "int" | "float" | "text";
};

type MonthlyRow = {
  month_date: string;
  month_date_display: string;
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
  monthly_profit: number;
  monthly_return_pct: number;
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

type MonthlyTableData = {
  active_month_date: string;
  rows: MonthlyRow[];
  editable_fields: MonthlyEditableField[];
  core_hidden_keys: string[];
};

export async function loadMonthlyTableData(): Promise<MonthlyTableData> {
  return fetchFastApiJson<MonthlyTableData>("/internal/monthly");
}

export async function updateMonthlyRow(
  monthDate: string,
  payload: Record<string, unknown>,
): Promise<{ month_date: string }> {
  return fetchFastApiJson<{ month_date: string }>("/internal/monthly", {
    method: "PATCH",
    body: JSON.stringify({
      ...payload,
      month_date: monthDate,
    }),
  });
}

export type { MonthlyEditableField, MonthlyRow, MonthlyTableData };
