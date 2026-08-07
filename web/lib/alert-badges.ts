/** 계좌 알람 배지(종목명 옆 이모지) 클라이언트 store.
 *
 * /alarms 의 계좌별 알람 설정·판정(이동선 이탈·손절)을 그대로 재사용해,
 * 자산 관리·자산 헬퍼 종목명에 붙일 티커→아이콘 맵을 가져온다.
 * 화면마다 fetch 를 복붙하지 말고 이 함수를 공유한다.
 */

import { readSessionTtlCache, writeSessionTtlCache } from "./session-ttl-cache";

export type AlertBadges = Record<string, string>;

// 서버 배지 계산이 수 초 걸려, 화면 재방문 시 세션 캐시로 즉시 표시한다(서버에도 별도 TTL 캐시 있음).
const BADGES_SESSION_CACHE_TTL_MS = 60_000;
const BADGES_SESSION_CACHE_PREFIX = "alert-badges:";

/** 티커 정규화 — 'ASX:XXX' 접두사 제거 + 대문자(배지 맵 키와 동일 규칙). */
export function normalizeBadgeTicker(ticker: string): string {
  const raw = String(ticker ?? "").trim().toUpperCase();
  const parts = raw.split(":");
  return parts[parts.length - 1] ?? "";
}

/** 계좌의 배지 맵 조회(세션 TTL 캐시 우선). 배지는 보조 정보라 실패 시 빈 맵(화면은 그대로). */
export async function fetchAlertBadges(accountId: string): Promise<AlertBadges> {
  const cacheKey = `${BADGES_SESSION_CACHE_PREFIX}${accountId}`;
  const cached = readSessionTtlCache<AlertBadges>(cacheKey, BADGES_SESSION_CACHE_TTL_MS);
  if (cached !== null) return cached;
  try {
    const resp = await fetch(`/api/alarms/badges?account=${encodeURIComponent(accountId)}`, { cache: "no-store" });
    const payload = (await resp.json()) as { badge_by_ticker?: Record<string, string>; error?: string };
    if (!resp.ok || payload.error) return {};
    const badges = payload.badge_by_ticker ?? {};
    writeSessionTtlCache(cacheKey, badges);
    return badges;
  } catch {
    return {};
  }
}
