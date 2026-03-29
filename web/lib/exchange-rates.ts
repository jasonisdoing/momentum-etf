type YahooChartResponse = {
  chart?: {
    result?: Array<{
      timestamp?: number[];
      indicators?: {
        quote?: Array<{
          close?: Array<number | null>;
        }>;
      };
    }>;
  };
};

type ExchangeRateItem = {
  rate: number;
  change_pct: number;
};

type ExchangeRateSummary = {
  USD: ExchangeRateItem;
  AUD: ExchangeRateItem;
  updated_at: string | null;
};

function getDefaultSummary(): ExchangeRateSummary {
  return {
    USD: { rate: 0, change_pct: 0 },
    AUD: { rate: 0, change_pct: 0 },
    updated_at: null,
  };
}

async function fetchRateSummary(symbol: string): Promise<{ rate: number; changePct: number; updatedAt: string | null }> {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?range=5d&interval=1d`;
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`환율 조회 실패: ${symbol}`);
  }

  const payload = (await response.json()) as YahooChartResponse;
  const timestamps = payload.chart?.result?.[0]?.timestamp ?? [];
  const closes = payload.chart?.result?.[0]?.indicators?.quote?.[0]?.close ?? [];
  const normalizedPoints = closes
    .map((value, index) => ({
      close: typeof value === "number" && Number.isFinite(value) ? value : null,
      timestamp: typeof timestamps[index] === "number" ? timestamps[index] : null,
    }))
    .filter((item): item is { close: number; timestamp: number | null } => item.close !== null);

  const lastPoint = normalizedPoints.at(-1);
  if (!lastPoint) {
    throw new Error(`환율 종가가 없습니다: ${symbol}`);
  }

  const previousPoint = normalizedPoints.length > 1 ? normalizedPoints.at(-2) : null;
  const changePct =
    previousPoint && previousPoint.close > 0 ? ((lastPoint.close / previousPoint.close) - 1) * 100 : 0;

  return {
    rate: lastPoint.close,
    changePct,
    updatedAt: lastPoint.timestamp ? new Date(lastPoint.timestamp * 1000).toISOString() : null,
  };
}

export async function loadExchangeRateSummary(): Promise<ExchangeRateSummary> {
  const summary = getDefaultSummary();

  try {
    const usd = await fetchRateSummary("KRW=X");
    summary.USD = { rate: usd.rate, change_pct: usd.changePct };
    summary.updated_at = usd.updatedAt;
  } catch {
    summary.USD = { rate: 0, change_pct: 0 };
  }

  try {
    const aud = await fetchRateSummary("AUDKRW=X");
    summary.AUD = { rate: aud.rate, change_pct: aud.changePct };
    summary.updated_at = summary.updated_at ?? aud.updatedAt;
  } catch {
    summary.AUD = { rate: 0, change_pct: 0 };
  }

  return summary;
}

export async function loadExchangeRates(): Promise<Record<string, number>> {
  const summary = await loadExchangeRateSummary();
  return {
    USD: summary.USD.rate,
    AUD: summary.AUD.rate,
  };
}

export type { ExchangeRateSummary };
