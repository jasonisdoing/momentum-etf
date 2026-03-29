import { fetchFastApiJson } from "./internal-api";

type BacktestTickerItem = {
  ticker: string;
  name: string;
  listing_date: string;
};

type BacktestGroupItem = {
  group_id?: string;
  name: string;
  weight: number;
  tickers: BacktestTickerItem[];
};

type BacktestConfigListItem = {
  config_id: string;
  name: string;
  period_months: number;
  slippage_pct: number;
  benchmark?: BacktestTickerItem | null;
  saved_at: string;
};

type BacktestConfigDetail = {
  config_id: string;
  name: string;
  period_months: number;
  slippage_pct: number;
  benchmark?: BacktestTickerItem | null;
  groups: BacktestGroupItem[];
  saved_at: string;
};

type BacktestRunResult = {
  initial_buy_date: string;
  latest_trading_day: string;
  cumulative_return_pct: number;
  cagr_pct: number;
  mdd_pct: number;
  benchmark?: {
    ticker: string;
    name: string;
    cumulative_return_pct: number;
    cagr_pct: number;
    mdd_pct: number;
  } | null;
  equity_curve: Array<{
    date: string;
    equity: number;
  }>;
};

type BacktestTickerValidation = {
  ticker: string;
  name: string;
  listing_date: string;
  status: string;
  is_deleted: boolean;
  deleted_reason: string;
};

export async function listBacktestConfigs() {
  return fetchFastApiJson<{ items: BacktestConfigListItem[] }>("/internal/backtest");
}

export async function loadBacktestConfig(configId: string) {
  return fetchFastApiJson<BacktestConfigDetail>(`/internal/backtest?config_id=${encodeURIComponent(configId)}`);
}

export async function saveBacktestConfig(
  name: string,
  periodMonths: number,
  slippagePct: number,
  benchmark: BacktestTickerItem | null,
  groups: BacktestGroupItem[],
) {
  return fetchFastApiJson<{ config_id: string; name: string; saved_at: string; duplicated?: boolean }>(
    "/internal/backtest",
    {
    method: "POST",
    body: JSON.stringify({
      name,
      period_months: periodMonths,
      slippage_pct: slippagePct,
      benchmark,
      groups,
    }),
  });
}

export async function validateBacktestTicker(ticker: string) {
  return fetchFastApiJson<BacktestTickerValidation>("/internal/backtest/validate", {
    method: "POST",
    body: JSON.stringify({ ticker }),
  });
}

export async function deleteBacktestConfig(configId: string) {
  return fetchFastApiJson<{ config_id: string }>("/internal/backtest/delete", {
    method: "POST",
    body: JSON.stringify({ config_id: configId }),
  });
}

export async function runBacktest(
  periodMonths: number,
  slippagePct: number,
  benchmark: BacktestTickerItem | null,
  groups: BacktestGroupItem[],
) {
  return fetchFastApiJson<BacktestRunResult>("/internal/backtest/run", {
    method: "POST",
    body: JSON.stringify({
      period_months: periodMonths,
      slippage_pct: slippagePct,
      benchmark,
      groups,
    }),
  });
}

export type {
  BacktestConfigDetail,
  BacktestConfigListItem,
  BacktestGroupItem,
  BacktestRunResult,
  BacktestTickerItem,
  BacktestTickerValidation,
};
