import { fetchFastApiJson } from "./internal-api";

type RankAccount = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
  country_code: string;
};

type RankRow = {
  보유여부: string;
  버킷: string;
  bucket: number;
  티커: string;
  종목명: string;
  상장일: string;
  추세: number | null;
  지속: number | null;
  보유: string;
  현재가: number | null;
  "괴리율": number | null;
  "일간(%)": number | null;
  "1주(%)": number | null;
  "2주(%)": number | null;
  "1달(%)": number | null;
  "3달(%)": number | null;
  "6달(%)": number | null;
  "12달(%)": number | null;
  고점: number | null;
  RSI: number | null;
};

type RankData = {
  accounts: RankAccount[];
  account_id: string;
  ma_type: string;
  ma_months: number;
  ma_type_options: string[];
  ma_months_max: number;
  rows: RankRow[];
  cache_blocked: boolean;
  latest_trading_day: string | null;
  cache_updated_at: string | null;
  ranking_computed_at: string | null;
  realtime_fetched_at: string | null;
  missing_tickers: string[];
  missing_ticker_labels: string[];
  stale_tickers: string[];
};

export async function loadRankData(params?: {
  account_id?: string;
  ma_type?: string;
  ma_months?: number;
}): Promise<RankData> {
  const search = new URLSearchParams();
  if (params?.account_id) {
    search.set("account_id", params.account_id);
  }
  if (params?.ma_type) {
    search.set("ma_type", params.ma_type);
  }
  if (params?.ma_months) {
    search.set("ma_months", String(params.ma_months));
  }

  const query = search.size > 0 ? `?${search.toString()}` : "";
  return fetchFastApiJson<RankData>(`/internal/rank${query}`);
}

export type { RankAccount, RankData, RankRow };
