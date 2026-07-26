/** 앱 메뉴 정의 표준(단일 소스).
 *
 * 사이드바(AppShell)와 홈 허브 타일(HubMenu)이 같은 메뉴 정의를 공유한다.
 * 메뉴 추가/변경은 여기서만 한다 — 화면별로 목록을 복붙하지 않는다.
 */

import type { ComponentType } from "react";
import {
  IconActivity,
  IconBell,
  IconCash,
  IconChartHistogram,
  IconChartLine,
  IconDatabase,
  IconHome,
  IconLayoutDashboard,
  IconList,
  IconListDetails,
  IconMedal2,
  IconReceipt2,
  IconSettings,
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
    title: "레버리지",
    icon: IconChartLine,
    items: [
      { href: "/leverage-scalp", label: "단타", icon: IconChartLine },
      { href: "/leverage-settings", label: "설정", icon: IconSettings },
    ],
  },
  {
    id: "system",
    title: "시스템",
    icon: IconSettings,
    items: [
      { href: "/alarms", label: "알람", icon: IconBell },
      { href: "/data_source", label: "데이터 소스", icon: IconDatabase },
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
  "/holdings_details",
];

export function isFullWidthRoute(pathname: string | null): boolean {
  if (!pathname) return false;
  return FULL_WIDTH_ROUTES.some((route) => pathname === route || pathname.startsWith(route + "/"));
}
