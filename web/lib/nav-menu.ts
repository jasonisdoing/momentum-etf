/** 앱 메뉴 정의 표준(단일 소스).
 *
 * 사이드바(AppShell)와 홈 허브 타일(HubMenu)이 같은 메뉴 정의를 공유한다.
 * 메뉴 추가/변경은 여기서만 한다 — 화면별로 목록을 복붙하지 않는다.
 */

import type { ComponentType } from "react";
import {
  IconActivity,
  IconCash,
  IconChartHistogram,
  IconChartInfographic,
  IconChartPie,
  IconChartLine,
  IconDatabase,
  IconHome,
  IconLayoutDashboard,
  IconList,
  IconListDetails,
  IconMedal2,
  IconNotes,
  IconReceipt2,
  IconSettings,
  IconTable,
  IconTrendingUp,
} from "@tabler/icons-react";

export type NavIconComponent = ComponentType<{ size?: number; stroke?: number }>;
export type NavIcon = NavIconComponent | string;

export type NavItem = {
  href: string;
  label: string;
  icon: NavIcon;
};

// 루트 메뉴는 항상 컴포넌트 아이콘(문자열 이모지 없음) — 렌더 단순화를 위해 타입을 좁힌다.
export type RootNavItem = {
  href: string;
  label: string;
  icon: NavIconComponent;
};

export type NavGroup = {
  id: string;
  title: string;
  icon: NavIconComponent;
  items: readonly NavItem[];
};

// 루트(그룹 밖) 메뉴
export const HOME_ITEM: RootNavItem = { href: "/", label: "홈", icon: IconHome };
export const ROOT_ITEMS: readonly RootNavItem[] = [
  { href: "/dashboard", label: "자산 요약", icon: IconLayoutDashboard },
  { href: "/holdings", label: "보유종목", icon: IconActivity },
  { href: "/holdings_details", label: "보유종목 상세", icon: IconListDetails },
];

export const NAV_GROUPS: readonly NavGroup[] = [
  {
    id: "assets",
    title: "계좌",
    icon: IconCash,
    items: [
      { href: "/assets", label: "자산 관리", icon: IconList },
      { href: "/asset-helper", label: "자산 헬퍼", icon: IconListDetails },
      { href: "/asset-status", label: "자산 현황", icon: IconTrendingUp },
      { href: "/daily", label: "기간별", icon: IconReceipt2 },
      { href: "/snapshots", label: "스냅샷", icon: IconReceipt2 },
      { href: "/account-settings", label: "설정", icon: IconSettings },
    ],
  },
  {
    id: "info",
    title: "정보",
    icon: IconTrendingUp,
    items: [
      { href: "/market-trend", label: "시장지수 추세", icon: IconChartLine },
      { href: "/compare", label: "ETF 비교", icon: IconListDetails },
      { href: "/kor-market-stock", label: "한국 개별주", icon: "🇰🇷" },
      { href: "/us-market-stock", label: "미국 개별주", icon: "🇺🇸" },
      { href: "/aus-market-stock", label: "호주 개별주", icon: "🇦🇺" },
      { href: "/kor-market-etf", label: "한국 ETF", icon: "🇰🇷" },
      { href: "/live-24h", label: "24H 시세", icon: "⏰" },
      { href: "/kor-dividend", label: "한국 배당주", icon: "🇰🇷" },
    ],
  },
  {
    id: "pools",
    title: "종목풀",
    icon: IconTrendingUp,
    items: [
      { href: "/pools-rank", label: "순위", icon: IconMedal2 },
      { href: "/pools-settings", label: "설정", icon: IconSettings },
      { href: "/pools-backtest", label: "백테스트", icon: IconChartHistogram },
    ],
  },
  {
    id: "leverage",
    title: "전략",
    icon: IconChartLine,
    items: [
      { href: "/leverage-settings", label: "레버리지", icon: IconSettings },
      { href: "/strategy-mix", label: "합성 전략", icon: IconChartInfographic },
      { href: "/strategy-momentum", label: "모멘텀 전략", icon: IconTrendingUp },
      { href: "/strategy-new-high", label: "신고가 돌파 전략", icon: IconTrendingUp },
      { href: "/strategy-portfolio", label: "포트폴리오 전략", icon: IconChartPie },
    ],
  },
  {
    id: "system",
    title: "시스템",
    icon: IconSettings,
    items: [
      { href: "/memos", label: "메모", icon: IconNotes },
      { href: "/data_source", label: "데이터 소스", icon: IconDatabase },
      { href: "/data-tables", label: "테이블", icon: IconTable },
      { href: "/batch", label: "배치", icon: IconListDetails },
    ],
  },
];

export function isNavItemActive(itemHref: string, currentPathname: string | null): boolean {
  if (!currentPathname) return false;
  if (itemHref === currentPathname) return true;
  // 기간별 통합 메뉴(/daily)는 주별/월별/년별에서도 활성 표시
  if (itemHref === "/daily" && ["/weekly", "/monthly", "/yearly"].includes(currentPathname)) return true;
  // /ticker → /ticker/XXX 같은 동적 라우트 매칭
  if (itemHref !== "/" && currentPathname.startsWith(itemHref + "/")) return true;
  return false;
}

// 전체폭(사이드바 자동 숨김) 라우트 — 큰 그리드 중심 화면. 화면별로 점진 조정한다.
export const FULL_WIDTH_ROUTES: readonly string[] = [
  "/assets",
  "/asset-helper",
  "/pools-rank",
  "/pools-backtest",
  "/kor-market-stock",
  "/us-market-stock",
  "/aus-market-stock",
  "/kor-market-etf",
  "/kor-dividend",
  "/holdings_details",
];

export function isFullWidthRoute(pathname: string | null): boolean {
  if (!pathname) return false;
  return FULL_WIDTH_ROUTES.some((route) => pathname === route || pathname.startsWith(route + "/"));
}

// 메뉴 정의(단일 소스)에서 경로에 해당하는 메뉴 이름을 찾는다. 최근 방문 페이지 표시 등에서 사용.
// 그룹 항목은 "그룹-라벨"(예: 계좌-자산 관리)로, 루트/홈은 라벨만. 정확 일치 우선, 없으면 동적 하위경로 접두어 매칭(가장 긴 것).
export function getNavLabel(pathname: string | null): string | null {
  if (!pathname) return null;

  // (href, 표시라벨) 목록 — 그룹 항목은 그룹명 접두어를 붙인다.
  const entries: { href: string; label: string }[] = [
    { href: HOME_ITEM.href, label: HOME_ITEM.label },
    ...ROOT_ITEMS.map((item) => ({ href: item.href, label: item.label })),
    ...NAV_GROUPS.flatMap((group) => group.items.map((item) => ({ href: item.href, label: `${group.title}-${item.label}` }))),
  ];

  const exact = entries.find((entry) => entry.href === pathname);
  if (exact) return exact.label;

  const prefixMatch = entries
    .filter((entry) => entry.href !== "/" && pathname.startsWith(entry.href + "/"))
    .sort((a, b) => b.href.length - a.href.length)[0];
  return prefixMatch ? prefixMatch.label : null;
}
