"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { getNavLabel } from "@/lib/nav-menu";
import { useFitOneLine } from "@/lib/use-fit-one-line";

// 자주 방문한 페이지(메뉴)를 상단 헤더 아래에 칩으로 나열한다.
// - 홈(/)과 메뉴 정의(nav-menu)에 없는 경로는 제외
// - 방문할 때마다 카운트를 올리고 **많이 간 순**으로 정렬. 지금 보고 있는 화면만 맨 왼쪽 고정
// - 카운트가 같으면 최근 방문한 쪽이 앞 (동점 순서가 매번 뒤바뀌면 칩 위치를 못 외운다)
// - localStorage 유지. 화면 폭에 한 줄로 들어가는 개수만 표시(넘치는 칩은 숨김)
// - 지금은 정렬을 눈으로 확인하려고 칩에 `메뉴명(카운트)` 로 횟수를 같이 적는다(임시).
const STORAGE_KEY = "momentum-page-visits";
// 최신순으로만 쌓던 시절의 키 — 카운트가 없어 이어 쓸 수 없다. 남겨두면 용량만 먹으므로 지운다.
const LEGACY_STORAGE_KEY = "momentum-recent-pages";

type PageVisit = { href: string; label: string; count: number; lastVisitedAt: number };

function loadVisits(): PageVisit[] {
  if (typeof window === "undefined") return [];
  try {
    window.localStorage.removeItem(LEGACY_STORAGE_KEY);
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as PageVisit[];
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item) =>
        item &&
        typeof item.href === "string" &&
        typeof item.label === "string" &&
        typeof item.count === "number" &&
        typeof item.lastVisitedAt === "number",
    );
  } catch {
    return [];
  }
}

function save(visits: PageVisit[]): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(visits));
  } catch {
    /* 저장 실패는 무시 — 표시용 편의 기능이라 화면을 막지 않는다 */
  }
}

/** 많이 간 순 → 같으면 최근 순. */
function byFrequency(a: PageVisit, b: PageVisit): number {
  return b.count - a.count || b.lastVisitedAt - a.lastVisitedAt;
}

export function RecentPages() {
  const pathname = usePathname();
  const [visits, setVisits] = useState<PageVisit[]>([]);
  const barRef = useRef<HTMLDivElement>(null);
  // 같은 경로에서 이펙트가 두 번 돌아도(개발 모드 StrictMode) 카운트를 두 번 올리지 않는다.
  const countedPathRef = useRef<string | null>(null);

  // 최초 마운트 시 localStorage 에서 복원.
  useEffect(() => {
    setVisits(loadVisits());
  }, []);

  // 경로가 바뀔 때마다 그 페이지의 방문 횟수를 1 올린다.
  useEffect(() => {
    if (!pathname || pathname === "/") return;
    const label = getNavLabel(pathname);
    if (!label) return;
    if (countedPathRef.current === pathname) return;
    countedPathRef.current = pathname;

    setVisits((prev) => {
      const current = prev.find((item) => item.href === pathname);
      const updated: PageVisit = {
        href: pathname,
        label,
        count: (current?.count ?? 0) + 1,
        lastVisitedAt: Date.now(),
      };
      const next = [...prev.filter((item) => item.href !== pathname), updated].sort(byFrequency);
      save(next);
      return next;
    });
  }, [pathname]);

  // 지금 보고 있는 화면은 카운트와 무관하게 맨 왼쪽 — 활성 칩 위치가 매번 달라지지 않게 한다.
  const ordered = [...visits].sort(byFrequency);
  const currentIndex = ordered.findIndex((item) => item.href === pathname);
  if (currentIndex > 0) ordered.unshift(...ordered.splice(currentIndex, 1));

  // 화면 폭에 한 줄로 들어가는 칩만 보이게 한다(넘치는 칩은 숨김). 목록/리사이즈 변경 시 재계산.
  useFitOneLine(barRef, [ordered.map((item) => item.href).join("|")]);

  if (ordered.length === 0) return null;

  return (
    <nav ref={barRef} className="recentPagesBar" aria-label="자주 방문한 페이지">
      {ordered.map((item) => {
        const active = item.href === pathname;
        return (
          <span
            key={item.href}
            data-fit-hideable
            className={active ? "recentPageChip recentPageChip--active" : "recentPageChip"}
          >
            <Link href={item.href} className="recentPageChipLink">
              {item.label}({item.count})
            </Link>
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
          height: 28px;
          padding: 0 12px;
          border: 1px solid rgba(148, 163, 184, 0.45);
          border-radius: 999px;
          background: var(--surface, #ffffff);
          font-size: var(--fs-sm);
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
      `}</style>
    </nav>
  );
}
