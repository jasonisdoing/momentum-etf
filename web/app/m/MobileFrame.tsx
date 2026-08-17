"use client";

import Link from "next/link";
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

import styles from "./mobile.module.css";

/** 금액 가림 — `/m` 화면들이 공유하는 상태. 저장값이 없으면 **가림**으로 시작한다.
 *  데스크톱 전역 설정(`jason-invest-hide-money`)과는 키를 나눈다 — 폰은 남이 볼 수 있어
 *  기본값이 반대이고, 한쪽을 바꿨다고 다른 쪽이 따라 바뀌면 안 된다. */
const MASK_STORAGE_KEY = "momentum-etf:m:hide-money";

const MobileMaskContext = createContext<{ hidden: boolean; toggle: () => void }>({
  hidden: true,
  toggle: () => {},
});

export function MobileMaskProvider({ children }: { children: ReactNode }) {
  // 서버 렌더와 첫 그림을 맞추려고 가림으로 시작하고, 저장값은 마운트 후 읽는다.
  const [hidden, setHidden] = useState(true);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(MASK_STORAGE_KEY);
      if (stored === "0") setHidden(false);
    } catch {
      // 저장소를 못 읽으면 기본값(가림)을 쓴다.
    }
  }, []);

  const toggle = useCallback(() => {
    setHidden((current) => {
      const next = !current;
      try {
        window.localStorage.setItem(MASK_STORAGE_KEY, next ? "1" : "0");
      } catch {
        // 저장 실패해도 이번 세션 동작은 유지한다.
      }
      return next;
    });
  }, []);

  return <MobileMaskContext.Provider value={{ hidden, toggle }}>{children}</MobileMaskContext.Provider>;
}

type Props = {
  title: string;
  /** 홈이 아니면 뒤로 갈 주소 — 주면 왼쪽에 `←` 가 나온다. */
  backHref?: string;
  /** 기준 시각 — 이 화면 데이터를 받아온 시점. */
  loadedAt?: Date | null;
  onRefresh?: () => void;
  refreshing?: boolean;
  children: ReactNode;
};

function formatTime(value: Date): string {
  return value.toLocaleTimeString("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

/** 모바일 화면 공용 틀 — 헤더(제목·금액 숨김) + 본문 + 푸터(기준 시각·새로고침). */
export function MobileFrame({
  title,
  backHref,
  loadedAt,
  onRefresh,
  refreshing,
  children,
}: Props) {
  const { hidden, toggle } = useContext(MobileMaskContext);

  return (
    <div className={styles.frame}>
      <header className={styles.header}>
        <span className={styles.headerLeft}>
          {backHref ? (
            <Link href={backHref} className={styles.backLink} aria-label="뒤로">
              ←
            </Link>
          ) : null}
          <span className={styles.headerTitle}>{title}</span>
        </span>
        <button
          type="button"
          className={styles.iconButton}
          onClick={toggle}
          aria-label={hidden ? "금액 표시" : "금액 숨김"}
        >
          {hidden ? "🙈" : "👁"}
        </button>
      </header>

      <main className={styles.page}>{children}</main>

      <footer className={styles.footer}>
        <span>{loadedAt ? `${formatTime(loadedAt)} 기준` : "-"}</span>
        {onRefresh ? (
          <button
            type="button"
            className={styles.iconButton}
            onClick={onRefresh}
            disabled={refreshing}
            aria-label="새로고침"
          >
            {refreshing ? "…" : "↻"}
          </button>
        ) : null}
      </footer>
    </div>
  );
}

/** 금액 가림이 켜져 있으면 `••••` 로 바꾼다 — 화면마다 같은 규칙을 쓰려고 여기 둔다. */
export function useMaskedAmount(): (text: string) => string {
  const { hidden } = useContext(MobileMaskContext);
  return (text: string) => (hidden ? "••••" : text);
}
