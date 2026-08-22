/**
 * 시장 화면(한국·미국·호주)에서 고른 종목을 종목풀에 담는 공용 로직.
 *
 * 세 화면이 같은 반복문을 각자 들고 있었다. 이미 그 풀에 있는 종목까지 서버로 보내면
 * 종목당 왕복이 한 번씩 생기고, 서버는 "이미 등록됨"을 알아내려고 조회를 한 번 더 한다.
 * 그래서 **보내기 전에 화면에서 걸러낸다** — 표가 이미 풀 목록(`ticker_pool_types`)을
 * 들고 있으므로 추가 조회 없이 판별된다.
 *
 * 건너뛰는 이유가 둘이라 화면이 색을 나눠 알린다.
 *   - 보내기 전 제외(표 기준)  → 시작 전 노란 안내. 정상 동작이다.
 *   - 보낸 뒤 서버가 거절     → 종료 후 빨간 안내. 표가 최신이 아니었다는 뜻이다.
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
  /** 보냈는데 서버가 "이미 등록됨"으로 거절한 수 — 표가 최신이 아니었던 경우. */
  skipped: number;
  /** 추가에 실패한 티커. */
  failed: string[];
};

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

/** 시작 전 노란 안내 문구. 건너뛸 게 없으면 null — 그때는 알리지 않는다. */
export function buildPoolAddSkipNotice(total: number, alreadyCount: number): string | null {
  if (alreadyCount <= 0) return null;
  if (alreadyCount >= total) return `선택한 ${total}개가 모두 이미 이 종목풀에 있어 건너뜁니다.`;
  return `선택한 ${total}개 중 이미 있는 ${alreadyCount}개는 건너뛰고 ${total - alreadyCount}개를 추가합니다.`;
}

/** 걸러낸 신규 종목만 실제로 담는다. 호출부가 `splitByPoolMembership` 으로 미리 나눈다. */
export async function addTickersToPool(
  tickers: string[],
  tickerType: string,
  bucketId: number,
): Promise<PoolAddResult> {
  let added = 0;
  let skipped = 0;
  const failed: string[] = [];

  for (const ticker of tickers) {
    try {
      await addStockCandidate(tickerType, ticker, bucketId);
      added += 1;
    } catch (error) {
      const message = error instanceof Error ? error.message : "종목 추가 처리에 실패했습니다.";
      if (message.includes("이미 등록된 종목입니다.")) {
        skipped += 1;
        continue;
      }
      failed.push(ticker);
    }
  }

  return { added, skipped, failed };
}
