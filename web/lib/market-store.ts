import { type ObjectId } from "mongodb";

import { getMongoDb } from "./mongo";

type EtfMarketMasterRow = {
  티커?: string;
  종목명?: string;
  상장일?: string;
  전일거래량?: number;
  시가총액?: number;
};

type EtfMarketMasterDoc = {
  _id: ObjectId;
  master_id: string;
  updated_at?: Date | string;
  rows?: EtfMarketMasterRow[];
};

type MarketRowItem = {
  ticker: string;
  name: string;
  listed_at: string;
  daily_change_pct: number | null;
  current_price: number | null;
  nav: number | null;
  deviation: number | null;
  return_3m_pct: number | null;
  prev_volume: number;
  market_cap: number;
};

type MarketTableData = {
  updated_at: string | null;
  rows: MarketRowItem[];
};

function normalizeNumber(value: unknown): number {
  return Number(value ?? 0);
}

function normalizeNullableNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  const parsed = Number(String(value).replaceAll(",", ""));
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeText(value: unknown): string {
  return String(value ?? "").trim();
}

function toUpdatedAtText(value: unknown): string | null {
  if (!value) {
    return null;
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  return String(value);
}

async function loadKorEtfRealtimeSnapshot(
  tickers: string[],
): Promise<Record<string, { changeRate?: number; nowVal?: number; nav?: number; deviation?: number; threeMonthEarnRate?: number }>> {
  if (tickers.length === 0) {
    return {};
  }

  const response = await fetch("https://finance.naver.com/api/sise/etfItemList.nhn", {
    headers: {
      "User-Agent":
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
      Referer: "https://finance.naver.com/sise/etfList.nhn",
      Accept: "application/json, text/plain, */*",
    },
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`네이버 ETF 실시간 스냅샷 조회 실패 (${response.status})`);
  }

  const payload = (await response.json()) as {
    result?: {
      etfItemList?: Array<Record<string, unknown>>;
    };
  };
  const items = payload.result?.etfItemList;
  if (!Array.isArray(items)) {
    throw new Error("네이버 ETF 실시간 스냅샷 응답 형식이 올바르지 않습니다.");
  }

  const tickerSet = new Set(tickers);
  const snapshot: Record<string, { changeRate?: number; nowVal?: number; nav?: number; deviation?: number; threeMonthEarnRate?: number }> = {};

  for (const item of items) {
    const code = String(item.itemcode ?? "").trim().toUpperCase();
    if (!code || !tickerSet.has(code)) {
      continue;
    }

    const nowVal = normalizeNullableNumber(item.nowVal);
    const nav = normalizeNullableNumber(item.nav);
    const changeRate = normalizeNullableNumber(item.changeRate);
    const threeMonthEarnRate = normalizeNullableNumber(item.threeMonthEarnRate);
    const deviation = nowVal !== null && nav !== null && nav > 0 ? ((nowVal / nav) - 1) * 100 : null;

    snapshot[code] = {
      changeRate: changeRate ?? undefined,
      nowVal: nowVal ?? undefined,
      nav: nav ?? undefined,
      deviation: deviation ?? undefined,
      threeMonthEarnRate: threeMonthEarnRate ?? undefined,
    };
  }

  return snapshot;
}

export async function loadEtfMarketTable(): Promise<MarketTableData> {
  const db = await getMongoDb();
  const doc = await db
    .collection<EtfMarketMasterDoc>("etf_market_master")
    .findOne({ master_id: "kor_etf_market" });

  if (!doc) {
    throw new Error("ETF 마켓 캐시가 없습니다. stock_meta_updater를 먼저 실행하세요.");
  }

  const rows = Array.isArray(doc.rows)
    ? doc.rows.map((row) => ({
        ticker: normalizeText(row.티커),
        name: normalizeText(row.종목명),
        listed_at: normalizeText(row.상장일),
        prev_volume: normalizeNumber(row.전일거래량),
        market_cap: normalizeNumber(row.시가총액),
      }))
    : [];

  if (rows.length === 0) {
    throw new Error("ETF 마켓 캐시가 비어 있습니다. stock_meta_updater를 다시 실행하세요.");
  }

  const tickers = rows.map((row) => row.ticker).filter(Boolean);
  const snapshot = await loadKorEtfRealtimeSnapshot(tickers);

  return {
    updated_at: toUpdatedAtText(doc.updated_at),
    rows: rows.map((row) => {
      const realtime = snapshot[row.ticker] ?? {};
      return {
        ...row,
        daily_change_pct: realtime.changeRate ?? null,
        current_price: realtime.nowVal ?? null,
        nav: realtime.nav ?? null,
        deviation: realtime.deviation ?? null,
        return_3m_pct: realtime.threeMonthEarnRate ?? null,
      };
    }),
  };
}

export type { MarketRowItem, MarketTableData };
