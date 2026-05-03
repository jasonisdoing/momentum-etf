type CacheEnvelope<T> = {
  savedAt: number;
  value: T;
};

function isBrowser(): boolean {
  return typeof window !== "undefined";
}

export function readSessionTtlCache<T>(key: string, ttlMs: number): T | null {
  if (!isBrowser()) {
    return null;
  }

  const raw = window.sessionStorage.getItem(key);
  if (!raw) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw) as CacheEnvelope<T>;
    if (typeof parsed?.savedAt !== "number") {
      window.sessionStorage.removeItem(key);
      return null;
    }
    if (Date.now() - parsed.savedAt > ttlMs) {
      window.sessionStorage.removeItem(key);
      return null;
    }
    return parsed.value ?? null;
  } catch {
    window.sessionStorage.removeItem(key);
    return null;
  }
}

function isQuotaError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  // Safari, Chrome, Firefox 모두 다른 코드/이름을 사용
  return (
    err.name === "QuotaExceededError" ||
    err.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
    /quota/i.test(err.message)
  );
}

function evictPrefixEntries(prefix: string, exceptKey: string): number {
  let removed = 0;
  try {
    for (let i = window.sessionStorage.length - 1; i >= 0; i--) {
      const k = window.sessionStorage.key(i);
      if (k && k !== exceptKey && k.startsWith(prefix)) {
        window.sessionStorage.removeItem(k);
        removed += 1;
      }
    }
  } catch {
    // 무시
  }
  return removed;
}

export function writeSessionTtlCache<T>(key: string, value: T): void {
  if (!isBrowser()) {
    return;
  }

  const payload: CacheEnvelope<T> = {
    savedAt: Date.now(),
    value,
  };
  const serialized = JSON.stringify(payload);
  try {
    window.sessionStorage.setItem(key, serialized);
    return;
  } catch (err) {
    if (!isQuotaError(err)) {
      return;
    }
    // 동일 prefix 의 오래된 엔트리들을 제거하고 1회 재시도
    const prefix = key.includes(":") ? key.slice(0, key.indexOf(":") + 1) : "";
    if (prefix) {
      evictPrefixEntries(prefix, key);
      try {
        window.sessionStorage.setItem(key, serialized);
        return;
      } catch {
        // 여전히 실패 — 캐시 저장 포기 (페이지 동작은 유지)
      }
    }
  }
}

