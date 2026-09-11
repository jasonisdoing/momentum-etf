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

export type TickerHoldingRow = {
  ticker: string;
  name: string;
  contracts: number | null;
  amount: number | null;
  raw_code?: string | null;
  raw_name?: string | null;
  reuters_code?: string | null;
  yahoo_symbol?: string | null;
  current_price?: number | null;
  previous_close?: number | null;
  change_pct?: number | null;
  price_currency?: string | null;
  weight: number | null;
};

export type TickerDetailData = {
  ticker: string;
  rows: TickerDetailRow[];
  holdings: TickerHoldingRow[];
  holdings_as_of_date?: string | null;
  holdings_price_as_of_date?: string | null;
  holdings_error?: string | null;
  error?: string;
};

type TickerMetaItem = {
  ticker: string;
  name: string;
  ticker_type: string;
  country_code: string;
  is_etf?: boolean;
  has_holdings?: boolean;
};

async function resolveTickerMeta(ticker: string): Promise<TickerMetaItem> {
  return fetchFastApiJson<TickerMetaItem>(`/internal/ticker-detail/resolve?ticker=${encodeURIComponent(ticker)}`);
}

export async function loadTickerDetailData(params: {
  ticker: string;
  ticker_type?: string;
  country_code?: string;
}): Promise<TickerDetailData> {
  const resolvedMeta = params.ticker_type ? null : await resolveTickerMeta(params.ticker);

  const search = new URLSearchParams();
  search.set("ticker", params.ticker);
  search.set("ticker_type", params.ticker_type ?? resolvedMeta!.ticker_type);

  const countryCode = params.country_code ?? resolvedMeta?.country_code;
  if (countryCode) {
    search.set("country_code", countryCode);
  }
  return fetchFastApiJson<TickerDetailData>(`/internal/ticker-detail?${search.toString()}`);
}

// 비교(여러 ETF 일괄) 호출은 SSE 스트림으로 이전했다 — `/api/ticker-detail-compare` 가
// `stream-proxy` 로 FastAPI `/internal/ticker-detail/compare` 를 그대로 통과시킨다.
// JSON 일괄 응답은 8종목 × 구성종목 조회에서 프록시 타임아웃(90초)에 걸려 폐기했다.
