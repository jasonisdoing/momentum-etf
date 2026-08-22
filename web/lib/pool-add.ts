/**
 * 시장 화면(한국·미국·호주)에서 고른 종목을 종목풀에 담는 공용 로직.
 *
 * 세 화면이 같은 반복문을 각자 들고 있었다. 이미 그 풀에 있는 종목까지 서버로 보내면
 * 종목당 왕복이 한 번씩 생기고, 서버는 "이미 등록됨"을 알아내려고 조회를 한 번 더 한다.
 * 그래서 **보내기 전에 화면에서 걸러낸다** — 표가 이미 풀 목록(`ticker_pool_types`)을
 * 들고 있으므로 추가 조회 없이 판별된다.
 */

import { addStockCandidate } from "@/lib/stocks-store";

/** 시장 표의 한 행 — 여기서 쓰는 필드만 좁게 요구한다. */
export type PoolMembershipRow = {
  ticker: string;
  /** 이 종목이 이미 들어 있는 종목풀 id 목록. 서버가 내려준다. */
  ticker_pool_types?: string[];
};

export type PoolAddResult = {
  /** 새로 추가된 종목 수. */
  added: number;
  /** 이미 그 풀에 있어 보내지 않은 종목 수. */
  skipped: number;
  /** 추가에 실패한 티커. */
  failed: string[];
};

/** 추가 전 안내 문구 — 몇 개 중 몇 개가 이미 있어 빠지는지 미리 알려준다. */
export function describePoolAddPlan(tickers: string[], rows: PoolMembershipRow[], tickerType: string): string {
  const total = tickers.length;
  if (total === 0) return "추가할 종목을 선택하세요.";
  if (!tickerType) return `선택한 종목 ${total}개를 추가합니다.`;
  const { fresh, already } = splitByPoolMembership(tickers, rows, tickerType);
  if (already.length === 0) return `선택한 종목 ${total}개를 추가합니다.`;
  if (fresh.length === 0) return `선택한 ${total}개가 모두 이미 이 종목풀에 있습니다.`;
  return `선택한 ${total}개 중 이미 있는 ${already.length}개를 빼고 ${fresh.length}개를 추가합니다.`;
}

/** 이미 그 풀에 들어 있는 티커. 표가 들고 있는 풀 목록으로만 판단한다(서버 조회 없음). */
export function splitByPoolMembership(
  tickers: string[],
  rows: PoolMembershipRow[],
  tickerType: string,
): { fresh: string[]; already: string[] } {
  const pool = String(tickerType || "").trim().toLowerCase();
  const poolsByTicker = new Map(
    rows.map((row) => [
      String(row.ticker || "").trim().toUpperCase(),
      (row.ticker_pool_types ?? []).map((value) => String(value).trim().toLowerCase()),
    ] as const),
  );
  const fresh: string[] = [];
  const already: string[] = [];
  for (const ticker of tickers) {
    const pools = poolsByTicker.get(String(ticker || "").trim().toUpperCase());
    // 표에 없는 티커는 판단 근거가 없으므로 그냥 보낸다 — 서버가 최종 판정한다.
    if (pools && pools.includes(pool)) {
      already.push(ticker);
    } else {
      fresh.push(ticker);
    }
  }
  return { fresh, already };
}

/** 고른 종목을 종목풀에 담는다. 이미 있는 종목은 요청조차 보내지 않는다. */
export async function addTickersToPool(
  tickers: string[],
  rows: PoolMembershipRow[],
  tickerType: string,
  bucketId: number,
): Promise<PoolAddResult> {
  const { fresh, already } = splitByPoolMembership(tickers, rows, tickerType);
  let added = 0;
  let skipped = already.length;
  const failed: string[] = [];

  for (const ticker of fresh) {
    try {
      await addStockCandidate(tickerType, ticker, bucketId);
      added += 1;
    } catch (error) {
      const message = error instanceof Error ? error.message : "종목 추가 처리에 실패했습니다.";
      // 표가 최신이 아니면 여기서 걸린다 — 실패가 아니라 건너뛴 것으로 센다.
      if (message.includes("이미 등록된 종목입니다.")) {
        skipped += 1;
        continue;
      }
      failed.push(ticker);
    }
  }

  return { added, skipped, failed };
}
