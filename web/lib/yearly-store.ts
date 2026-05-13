import { fetchFastApiJson } from "./internal-api";

type YearlyEditableField = {
  key: string;
  label: string;
  type: "int" | "float" | "text";
};

type YearlyRow = {
  year_date: string;
  year_date_display: string;
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
  yearly_profit: number;
  yearly_return_pct: number;
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

type YearlyTableData = {
  active_year_date: string;
  rows: YearlyRow[];
  editable_fields: YearlyEditableField[];
  core_hidden_keys: string[];
};

export async function loadYearlyTableData(): Promise<YearlyTableData> {
  return fetchFastApiJson<YearlyTableData>("/internal/yearly");
}

export async function updateYearlyRow(
  yearDate: string,
  payload: Record<string, unknown>,
): Promise<{ year_date: string }> {
  return fetchFastApiJson<{ year_date: string }>("/internal/yearly", {
    method: "PATCH",
    body: JSON.stringify({
      ...payload,
      year_date: yearDate,
    }),
  });
}

export type { YearlyEditableField, YearlyRow, YearlyTableData };
