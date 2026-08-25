/** 계좌 알람 배지(종목명 옆 이모지) 클라이언트 store.
 *
 * 계좌 설정(`/account-settings`)의 알람 On/Off·판정(이동선 이탈·손절)을 그대로 재사용해,
 * 자산 관리·자산 헬퍼 종목명에 붙일 티커→아이콘 맵을 가져온다.
 * 화면마다 fetch 를 복붙하지 말고 이 함수를 공유한다.
 */

import { readSessionTtlCache, writeSessionTtlCache } from "./session-ttl-cache";

export type AlertBadges = Record<string, string>;

/** 배지 맵 + 이동선 이탈 종목 목록. 이탈 종목은 행 전체를 회색 처리하는 데 쓴다. */
export type AlertBadgeInfo = {
  badgeByTicker: AlertBadges;
  /** 이동선 이탈로 배지가 붙은 티커(정규화된 형태). */
  maTickers: string[];
};

const EMPTY_BADGE_INFO: AlertBadgeInfo = { badgeByTicker: {}, maTickers: [] };

// 서버 배지 계산이 수 초 걸려, 화면 재방문 시 세션 캐시로 즉시 표시한다(서버에도 별도 TTL 캐시 있음).
const BADGES_SESSION_CACHE_TTL_MS = 60_000;
const BADGES_SESSION_CACHE_PREFIX = "alert-badges:";

/** 티커 정규화 — 'ASX:XXX' 접두사 제거 + 대문자(배지 맵 키와 동일 규칙). */
export function normalizeBadgeTicker(ticker: string): string {
  const raw = String(ticker ?? "").trim().toUpperCase();
  const parts = raw.split(":");
  return parts[parts.length - 1] ?? "";
}

/** 계좌의 배지 정보 조회(세션 TTL 캐시 우선). 보조 정보라 실패 시 빈 값(화면은 그대로). */
export async function fetchAlertBadges(accountId: string): Promise<AlertBadgeInfo> {
  const cacheKey = `${BADGES_SESSION_CACHE_PREFIX}${accountId}`;
  const cached = readSessionTtlCache<AlertBadgeInfo>(cacheKey, BADGES_SESSION_CACHE_TTL_MS);
  // 예전 버전이 남긴 캐시(티커→아이콘 맵)는 형태가 달라 그대로 쓰면 화면이 깨진다. 모양을 확인한다.
  if (cached !== null && cached.badgeByTicker) return cached;
  try {
    const resp = await fetch(`/api/alarms/badges?account=${encodeURIComponent(accountId)}`, { cache: "no-store" });
    const payload = (await resp.json()) as {
      badge_by_ticker?: Record<string, string>;
      ma_tickers?: string[];
      error?: string;
    };
    if (!resp.ok || payload.error) return EMPTY_BADGE_INFO;
    const info: AlertBadgeInfo = {
      badgeByTicker: payload.badge_by_ticker ?? {},
      maTickers: payload.ma_tickers ?? [],
    };
    writeSessionTtlCache(cacheKey, info);
    return info;
  } catch {
    return EMPTY_BADGE_INFO;
  }
}
