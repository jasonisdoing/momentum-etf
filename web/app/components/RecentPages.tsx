"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { getNavLabel } from "@/lib/nav-menu";
import { useFitOneLine } from "@/lib/use-fit-one-line";

// 최근 방문한 페이지(메뉴)를 상단 헤더 아래에 칩으로 나열한다.
// - 홈(/)과 메뉴 정의(nav-menu)에 없는 경로는 제외
// - 최신순, localStorage 유지, 현재 페이지는 활성 강조
// - 화면 폭에 "한 줄로 들어가는 최대 개수"만 표시(넘치는 칩은 숨김), 나머지는 보관만
const STORAGE_KEY = "momentum-recent-pages";
const MAX_STORED = 20;

type RecentPage = { href: string; label: string };

function loadRecent(): RecentPage[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as RecentPage[];
    return Array.isArray(parsed)
      ? parsed.filter((item) => item && typeof item.href === "string" && typeof item.label === "string")
      : [];
  } catch {
    return [];
  }
}

export function RecentPages() {
  const pathname = usePathname();
  const [recent, setRecent] = useState<RecentPage[]>([]);
  const barRef = useRef<HTMLDivElement>(null);

  // 최초 마운트 시 localStorage 에서 복원.
  useEffect(() => {
    setRecent(loadRecent());
  }, []);

  // 경로 변경 시 현재 페이지를 최근 목록에 반영(최신순·중복 제거·최대 보관 개수).
  useEffect(() => {
    if (!pathname || pathname === "/") return;
    const label = getNavLabel(pathname);
    if (!label) return;
    setRecent((prev) => {
      const withoutCurrent = prev.filter((item) => item.href !== pathname);
      const next = [{ href: pathname, label }, ...withoutCurrent].slice(0, MAX_STORED);
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        /* 저장 실패는 무시 */
      }
      return next;
    });
  }, [pathname]);

  // 화면 폭에 한 줄로 들어가는 칩만 보이게 한다(넘치는 칩은 숨김). 최근목록/리사이즈 변경 시 재계산.
  useFitOneLine(barRef, [recent]);

  const remove = (href: string) => {
    setRecent((prev) => {
      const next = prev.filter((item) => item.href !== href);
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        /* 저장 실패는 무시 */
      }
      return next;
    });
  };

  if (recent.length === 0) return null;

  return (
    <nav ref={barRef} className="recentPagesBar" aria-label="최근 방문 페이지">
      {recent.map((item) => {
        const active = item.href === pathname;
        return (
          <span
            key={item.href}
            data-fit-hideable
            className={active ? "recentPageChip recentPageChip--active" : "recentPageChip"}
          >
            <Link href={item.href} className="recentPageChipLink">
              {item.label}
            </Link>
            <button
              type="button"
              className="recentPageChipClose"
              aria-label={`${item.label} 최근목록에서 제거`}
              onClick={() => remove(item.href)}
            >
              ×
            </button>
          </span>
        );
      })}
      <style jsx>{`
        .recentPagesBar {
          display: flex;
          align-items: center;
          gap: 8px;
          flex-wrap: nowrap;
          overflow: hidden;
          padding: 8px 0 4px;
          margin-bottom: 4px;
          /* 부모(.appContent)가 세로 flex 라 높이가 눌리면 칩이 찌그러진다 — 자체적으로 축소를 막는다. */
          flex: 0 0 auto;
          min-height: 40px;
        }
        .recentPageChip {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          height: 28px;
          padding: 0 6px 0 12px;
          border: 1px solid rgba(148, 163, 184, 0.45);
          border-radius: 999px;
          background: var(--surface, #ffffff);
          font-size: 0.82rem;
          line-height: 1;
          white-space: nowrap;
          flex: 0 0 auto;
        }
        .recentPageChip--active {
          border-color: #2563eb;
          background: rgba(37, 99, 235, 0.1);
        }
        .recentPageChipLink {
          color: var(--text-strong, #1e293b);
          text-decoration: none;
          font-weight: 600;
        }
        .recentPageChip--active .recentPageChipLink {
          color: #1d4ed8;
        }
        .recentPageChipClose {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 18px;
          height: 18px;
          padding: 0;
          border: none;
          border-radius: 999px;
          background: transparent;
          color: var(--text-muted, #64748b);
          font-size: 1rem;
          line-height: 1;
          cursor: pointer;
        }
        .recentPageChipClose:hover {
          background: rgba(148, 163, 184, 0.25);
          color: var(--text-strong, #1e293b);
        }
      `}</style>
    </nav>
  );
}
