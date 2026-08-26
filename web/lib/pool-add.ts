/**
 * 시장 화면(한국·미국·호주)에서 고른 종목을 종목풀에 담는 공용 로직.
 *
 * 세 화면이 같은 반복문을 각자 들고 있었다. 이미 등록된 종목까지 서버로 보내면
 * 종목당 왕복이 한 번씩 생기고, 서버는 "이미 등록됨"을 알아내려고 조회를 한 번 더 한다.
 * 그래서 **보내기 전에 화면에서 걸러낸다** — 표가 이미 풀 목록(`ticker_pool_types`)을
 * 들고 있으므로 추가 조회 없이 판별된다.
 *
 * 한 티커는 **한 종목풀에만** 들어갈 수 있다(서버 `add_active_stock` 이 최종 판정).
 * 계좌 보유 종목의 소속 풀이 유일해야 그 풀의 이평선으로 이탈을 판정할 수 있기 때문이다.
 * 그래서 "이 풀에 이미 있음"뿐 아니라 "다른 풀에 이미 있음"도 미리 걸러낸다.
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
  /** 보냈는데 서버가 "다른 종목풀에 있음"으로 거절한 수 — 표가 최신이 아니었던 경우. */
  blocked: number;
  /** 추가에 실패한 티커. */
  failed: string[];
};

export type PoolMembershipSplit = {
  /** 실제로 서버에 보낼 티커. */
  fresh: string[];
  /** 이미 이 종목풀에 있어 건너뛸 티커. */
  already: string[];
  /** 다른 종목풀에 있어 건너뛸 티커 — 한 티커는 한 풀에만 들어갈 수 있다. */
  otherPool: string[];
};

/** 이미 어딘가의 풀에 들어 있는 티커. 표가 들고 있는 풀 목록으로만 판단한다(서버 조회 없음). */
export function splitByPoolMembership(
  tickers: string[],
  rows: PoolMembershipRow[],
  tickerType: string,
): PoolMembershipSplit {
  const pool = String(tickerType || "").trim().toLowerCase();
  const poolsByTicker = new Map(
    rows.map((row) => [
      String(row.ticker || "").trim().toUpperCase(),
      (row.ticker_pool_types ?? []).map((value) => String(value).trim().toLowerCase()),
    ] as const),
  );
  const fresh: string[] = [];
  const already: string[] = [];
  const otherPool: string[] = [];
  for (const ticker of tickers) {
    const pools = poolsByTicker.get(String(ticker || "").trim().toUpperCase());
    // 표에 없는 티커는 판단 근거가 없으므로 그냥 보낸다 — 서버가 최종 판정한다.
    if (!pools || pools.length === 0) {
      fresh.push(ticker);
    } else if (pools.includes(pool)) {
      already.push(ticker);
    } else {
      otherPool.push(ticker);
    }
  }
  return { fresh, already, otherPool };
}

/** 시작 전 노란 안내 문구. 건너뛸 게 없으면 null — 그때는 알리지 않는다. */
export function buildPoolAddSkipNotice(total: number, split: PoolMembershipSplit): string | null {
  const reasons: string[] = [];
  if (split.already.length > 0) reasons.push(`이미 이 종목풀에 있는 ${split.already.length}개`);
  if (split.otherPool.length > 0) reasons.push(`다른 종목풀에 있는 ${split.otherPool.length}개`);
  if (reasons.length === 0) return null;

  const joined = reasons.join(", ");
  if (split.fresh.length === 0) {
    return `선택한 ${total}개(${joined})가 모두 이미 등록돼 있어 건너뜁니다.`;
  }
  return `선택한 ${total}개 중 ${joined}는 건너뛰고 ${split.fresh.length}개를 추가합니다.`;
}

/** 한 종목을 처리할 때마다 불린다. 서버가 종목당 시세·메타 캐시까지 채워 수 초씩 걸려서,
 *  진행도를 알리지 않으면 화면이 멈춘 것처럼 보인다. */
export type PoolAddProgress = { done: number; total: number; ticker: string; name?: string };

/** 걸러낸 신규 종목만 실제로 담는다. 호출부가 `splitByPoolMembership` 으로 미리 나눈다.
 *
 * `nameByTicker` 는 진행도 표시에만 쓴다 — 티커만 보이면 무엇을 담는 중인지 알기 어렵다.
 * 표가 이미 종목명을 들고 있으므로 추가 조회 없이 넘긴다. */
export async function addTickersToPool(
  tickers: string[],
  tickerType: string,
  bucketId: number,
  onProgress?: (progress: PoolAddProgress) => void,
  nameByTicker?: Map<string, string>,
): Promise<PoolAddResult> {
  let added = 0;
  let skipped = 0;
  let blocked = 0;
  const failed: string[] = [];

  for (const [index, ticker] of tickers.entries()) {
    const name = nameByTicker?.get(String(ticker || "").trim().toUpperCase());
    onProgress?.({ done: index, total: tickers.length, ticker, name });
    try {
      await addStockCandidate(tickerType, ticker, bucketId);
      added += 1;
    } catch (error) {
      const message = error instanceof Error ? error.message : "종목 추가 처리에 실패했습니다.";
      if (message.includes("이미 등록된 종목입니다.")) {
        skipped += 1;
        continue;
      }
      // 서버 문구: "이미 다른 종목풀에 있습니다: 🇰🇷 국내상장 국내 ETF (kor_kr)"
      if (message.includes("이미 다른 종목풀에 있습니다")) {
        blocked += 1;
        continue;
      }
      failed.push(ticker);
    } finally {
      onProgress?.({ done: index + 1, total: tickers.length, ticker, name });
    }
  }

  return { added, skipped, blocked, failed };
}
