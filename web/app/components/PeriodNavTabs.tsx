"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { href: "/daily", label: "일별" },
  { href: "/weekly", label: "주별" },
  { href: "/monthly", label: "월별" },
  { href: "/yearly", label: "년별" },
];

/** 기간별(일/주/월/년) 페이지 공용 탭 — 사이드바 단일 메뉴('기간별') 하위 이동용. */
export function PeriodNavTabs() {
  const pathname = usePathname();
  return (
    <div className="appSegmentedToggle" role="group" aria-label="기간별 내역" style={{ marginBottom: 10, alignSelf: "flex-start" }}>
      {TABS.map((tab) => (
        <Link
          key={tab.href}
          href={tab.href}
          className={`btn appSegmentedToggleButton ${pathname === tab.href ? "is-active" : ""}`.trim()}
        >
          {tab.label}
        </Link>
      ))}
    </div>
  );
}
