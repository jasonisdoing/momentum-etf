import { fetchFastApiJson } from "./internal-api";

export type TickerDetailRow = {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
  change_pct: number | null;
};

export type TickerDetailData = {
  ticker: string;
  rows: TickerDetailRow[];
  error?: string;
};

export async function loadTickerDetailData(params: {
  ticker: string;
  ticker_type: string;
  country_code?: string;
  months?: number;
}): Promise<TickerDetailData> {
  const search = new URLSearchParams();
  search.set("ticker", params.ticker);
  search.set("ticker_type", params.ticker_type);
  if (params.country_code) {
    search.set("country_code", params.country_code);
  }
  if (params.months) {
    search.set("months", String(params.months));
  }
  return fetchFastApiJson<TickerDetailData>(`/internal/ticker-detail?${search.toString()}`);
}
