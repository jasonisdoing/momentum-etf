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

export function writeSessionTtlCache<T>(key: string, value: T): void {
  if (!isBrowser()) {
    return;
  }

  const payload: CacheEnvelope<T> = {
    savedAt: Date.now(),
    value,
  };
  window.sessionStorage.setItem(key, JSON.stringify(payload));
}

