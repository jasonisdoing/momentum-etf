import { fetchFastApiJson } from "./internal-api";

type RankTickerType = {
  ticker_type: string;
  order: number;
  name: string;
  icon: string;
  country_code: string;
  top_n_hold?: number;
  type_source?: string;
  currency?: string;
  include?: string[];
};

type RankMaRule = {
  short_ma_days: number;
  main_ma_days: number;
  score_column: string;
};

type RankRow = {
  [key: string]: string | number | null | undefined;
  순번: string;
  순위: number | null;
  이전순위: number | null;
  "1주순위": number | null;
  버킷: string;
  bucket: number;
  티커: string;
  종목명: string;
  상장일: string;
  분류: string;
  "전체 분류": string;
  현재가: number | null;
  "괴리율": number | null;
  "일간(%)": number | null;
  "1주(%)": number | null;
  "2주(%)": number | null;
  "3주(%)": number | null;
  "4주(%)": number | null;
  "1달(%)": number | null;
  "2달(%)": number | null;
  "3달(%)": number | null;
  "4달(%)": number | null;
  "5달(%)": number | null;
  "6달(%)": number | null;
  "7달(%)": number | null;
  "8달(%)": number | null;
  "9달(%)": number | null;
  "10달(%)": number | null;
  "11달(%)": number | null;
  "12달(%)": number | null;
  고점: number | null;
  RSI: number | null;
  추세: number | null;
  배열?: string | null;
};

type RankData = {
  ticker_types: RankTickerType[];
  ticker_type: string;
  ma_rules: RankMaRule[];
  as_of_date: string | null;
  monthly_return_labels: string[];
  rows: RankRow[];
  cache_blocked: boolean;
  latest_trading_day: string | null;
  cache_updated_at: string | null;
  ranking_computed_at: string | null;
  realtime_fetched_at: string | null;
  previous_trading_day: string | null;
  missing_tickers: string[];
  missing_ticker_labels: string[];
  stale_tickers: string[];
};

type RankToolbarData = {
  ticker_types: RankTickerType[];
  ticker_type: string;
  ma_rules: RankMaRule[];
};

export async function loadRankToolbarData(params?: {
  ticker_type?: string;
}, signal?: AbortSignal): Promise<RankToolbarData> {
  const search = new URLSearchParams();
  if (params?.ticker_type) {
    search.set("ticker_type", params.ticker_type);
  }

  const query = search.size > 0 ? `?${search.toString()}` : "";
  return fetchFastApiJson<RankToolbarData>(`/internal/rank/toolbar${query}`, { signal });
}

export async function loadRankData(params?: {
  ticker_type?: string;
  ma_rule_override?: RankMaRule;
  as_of_date?: string;
}, signal?: AbortSignal): Promise<RankData> {
  const search = new URLSearchParams();
  if (params?.ticker_type) {
    search.set("ticker_type", params.ticker_type);
  }
  if (params?.as_of_date) {
    search.set("as_of_date", params.as_of_date);
  }
  if (params?.ma_rule_override) {
    search.set("short_ma_days", String(params.ma_rule_override.short_ma_days));
    search.set("main_ma_days", String(params.ma_rule_override.main_ma_days));
  }

  const query = search.size > 0 ? `?${search.toString()}` : "";
  return fetchFastApiJson<RankData>(`/internal/rank${query}`, { signal });
}

export type { RankTickerType, RankMaRule, RankData, RankRow };
