import { fetchFastApiJson } from "./internal-api";

type RankTickerType = {
  ticker_type: string;
  order: number;
  name: string;
  icon: string;
  country_code: string;
  top_n_hold?: number;
  /** 업종 상한 — 종목풀 저장값. null/미설정 = 제한 없음. */
  max_per_industry?: number | null;
  currency?: string;
  include?: string[];
};

type RankMaRule = {
  short_ma_days: number;
  long_ma_days: number;
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
  /** 종목 메모 — 자산 관리 화면과 같은 값(종목에 붙는다). */
  메모?: string;
  상장일: string;
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
  /** 이평선 일수 선택지 — 백엔드 상수(utils/ma_options)가 단일 소스. */
  short_ma_options: number[];
  long_ma_options: number[];
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
  short_ma_options: number[];
  long_ma_options: number[];
  /** 종목 수·업종 상한 선택지 — 백엔드 `config` 가 단일 소스(-1 = 제한 없음). */
  top_n_options: number[];
  max_per_industry_options: number[];
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

// 화면에서 임시로 바꿔 보는 이평선 값. 넘긴 항목만 저장 규칙을 대신하고, 빠진 항목은 저장값을 쓴다.
type RankMaRuleOverride = {
  short_ma_days?: number;
  long_ma_days?: number;
};

export async function loadRankData(params?: {
  ticker_type?: string;
  ma_rule_override?: RankMaRuleOverride;
  as_of_date?: string;
  /** 종목 수 — 화면 상단에서 바꿔 보는 값. 생략하면 종목풀 저장값. */
  top_n?: number | null;
  /** 업종 상한 — 화면 상단에서 바꿔 보는 값. 생략하면 종목풀 저장값, -1 은 '제한 없음'. */
  max_per_industry?: number | null;
}, signal?: AbortSignal): Promise<RankData> {
  const search = new URLSearchParams();
  if (params?.ticker_type) {
    search.set("ticker_type", params.ticker_type);
  }
  if (params?.as_of_date) {
    search.set("as_of_date", params.as_of_date);
  }
  const override = params?.ma_rule_override;
  if (override) {
    for (const key of ["short_ma_days", "long_ma_days"] as const) {
      const value = override[key];
      if (value != null) {
        search.set(key, String(value));
      }
    }
  }

  if (params?.top_n != null) {
    search.set("top_n", String(params.top_n));
  }
  if (params?.max_per_industry != null) {
    search.set("max_per_industry", String(params.max_per_industry));
  }

  const query = search.size > 0 ? `?${search.toString()}` : "";
  return fetchFastApiJson<RankData>(`/internal/rank${query}`, { signal });
}

export type { RankTickerType, RankMaRule, RankMaRuleOverride, RankData, RankRow };
