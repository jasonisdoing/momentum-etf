"use client";

import Link from "next/link";
import { createContext, useCallback, useContext, useState, type ReactNode } from "react";

import styles from "./mobile.module.css";

/** 금액 가림 — `/m` 전용 상태다. 기본은 **가림**이고 눈 아이콘으로 잠깐 연다.
 *  데스크톱의 전역 설정(HideMoneyProvider)과 분리한다 — 폰은 남이 볼 수 있어 기본값이 반대다.
 *  저장하지 않으므로 앱을 다시 열면 다시 가려진다. */
const MobileMaskContext = createContext<{ hidden: boolean; toggle: () => void }>({
  hidden: true,
  toggle: () => {},
});

export function MobileMaskProvider({ children }: { children: ReactNode }) {
  const [hidden, setHidden] = useState(true);
  const toggle = useCallback(() => setHidden((current) => !current), []);
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
