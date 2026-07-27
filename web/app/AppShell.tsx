"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import type { ReactNode } from "react";
import {
  IconChevronDown,
  IconHome,
  IconMoodSmile,
  IconMenu2,
  IconX,
} from "@tabler/icons-react";

import { useHideMoney } from "@/lib/hide-money-context";
import { HOME_ITEM, NAV_GROUPS, ROOT_ITEMS, isNavItemActive } from "@/lib/nav-menu";
import { HubMenu } from "./components/HubMenu";
import { GlobalTickerSearch } from "./components/GlobalTickerSearch";
import { RecentPages } from "./components/RecentPages";

// 메뉴 정의는 nav-menu.ts 가 단일 소스 — 홈 허브 타일(HubMenu)과 공유한다.
const homeItem = HOME_ITEM;
const navGroups = NAV_GROUPS;

type AppShellProps = {
  children: ReactNode;
};

type FxSummary = {
  USD?: { rate: number; change_pct: number };
  AUD?: { rate: number; change_pct: number };
  updated_at?: string | null;
};

type FearGreedSummary = {
  score: number | null;
  label: string | null;
  previous_close_score: number | null;
  updated_at?: string | null;
};

type DashboardMetricItem = {
  label: string;
  value: number;
  kind: string;
  sub_value?: number;
  sub_kind?: string;
};

type DashboardSummary = {
  metrics_row1?: DashboardMetricItem[];
  metrics_row2?: DashboardMetricItem[];
  period_profits?: {
    daily?: { profit: number; return_pct: number };
    weekly?: { profit: number; return_pct: number };
    monthly?: { profit: number; return_pct: number };
    yearly?: { profit: number; return_pct: number };
  };
  is_deploying?: boolean;
};

type NqFutureSummary = {
  price?: number | null;
  change_pct?: number | null;
  error?: string;
};

type ProfitMetric = {
  amount: number;
  pct: number | null;
};

function pickProfitMetric(items: DashboardMetricItem[] | undefined, label: string): ProfitMetric | null {
  if (!items) return null;
  const found = items.find((item) => item.label === label);
  if (!found) return null;
  const pct = found.sub_kind === "percent" && typeof found.sub_value === "number" ? found.sub_value : null;
  return { amount: found.value, pct };
}

function formatProfitAmount(value: number): string {
  if (!Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  const eok = Math.floor(abs / 100_000_000);
  const man = Math.round((abs % 100_000_000) / 10_000);
  const sign = value < 0 ? "-" : "";
  if (eok > 0 && man > 0) return `${sign}${eok}억 ${man.toLocaleString("ko-KR")}만원`;
  if (eok > 0) return `${sign}${eok}억원`;
  return `${sign}${man.toLocaleString("ko-KR")}만원`;
}

function formatProfitPct(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "";
  const sign = value > 0 ? "+" : "";
  return `(${sign}${value.toFixed(2)}%)`;
}

function formatRate(value: number | undefined): string {
  if (!value || Number.isNaN(value)) {
    return "-";
  }
  return `${new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 2 }).format(value)}원`;
}

function formatChangePct(value: number | undefined): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "";
  }
  return `(${value.toFixed(2)}%)`;
}

function getFxChangeClass(value: number | undefined): string | undefined {
  if (value === undefined || value === null || Number.isNaN(value) || value === 0) {
    return undefined;
  }
  return value < 0 ? "metricNegative" : "metricPositive";
}

function formatFearGreedScore(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(1);
}

function formatFearGreedLabel(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }

  const normalized = value.trim();
  if (!normalized) {
    return "-";
  }

  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function formatFearGreedDelta(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "";
  }
  const sign = value > 0 ? "+" : "";
  return `(${sign}${value.toFixed(1)})`;
}

async function loadFearGreedSummary(): Promise<FearGreedSummary | null> {
  const apiResponse = await fetch("/api/fear-greed", { cache: "no-store" });
  if (apiResponse.ok) {
    return (await apiResponse.json()) as FearGreedSummary;
  }
  return null;
}

function getDefaultOpenGroups(pathname: string | null) {
  return Object.fromEntries(
    navGroups.map((group) => [group.id, true]),
  );
}

export function AppShell({ children }: AppShellProps) {
  const pathname = usePathname();
  const router = useRouter();
  const { hideMoney } = useHideMoney();
  const [fx, setFx] = useState<FxSummary | null>(null);
  const [isFxLoading, setIsFxLoading] = useState(true);
  const [fearGreed, setFearGreed] = useState<FearGreedSummary | null>(null);
  const [isFearGreedLoading, setIsFearGreedLoading] = useState(true);
  const [dashboardSummary, setDashboardSummary] = useState<DashboardSummary | null>(null);
  const [isDashboardSummaryLoading, setIsDashboardSummaryLoading] = useState(true);
  const [nqFuture, setNqFuture] = useState<NqFutureSummary | null>(null);
  const [isNqLoading, setIsNqLoading] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // 데스크톱 좌측 사이드바는 제거됨. 메뉴는 홈의 우측 레일(HubMenu)로 통일하고,
  // 다른 화면은 사이드바 없이 전체폭 + 상단 홈 버튼으로 복귀한다. (모바일은 햄버거 메뉴 유지)
  const isHome = pathname === "/";
  // 레일 기간손익은 기본 숨김 — "수익률 보기" 클릭 시 펼친다.
  const [showRailPeriod, setShowRailPeriod] = useState(false);
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>(() => getDefaultOpenGroups(pathname));
  const [isDbError, setIsDbError] = useState(false);
  const isLoginPage = pathname === "/login";

  const loadTopBarData = useCallback(async (initial = false) => {
    if (isLoginPage) {
      return;
    }

    try {
      if (initial) {
        setIsFxLoading(true);
        setIsFearGreedLoading(true);
        setIsDashboardSummaryLoading(true);
        setIsNqLoading(true);
      }

      const [fxResponse, fearGreedSummary, dashboardResponse, nqResponse] = await Promise.all([
        fetch("/api/fx", { cache: "no-store" }),
        loadFearGreedSummary().catch(() => null),
        fetch("/api/dashboard", { cache: "no-store" }).catch(() => null),
        fetch("/api/nq-future", { cache: "no-store" }).catch(() => null),
      ]);

      const payload = fxResponse.ok ? ((await fxResponse.json()) as FxSummary) : null;
      const dashboardPayload = dashboardResponse?.ok
        ? ((await dashboardResponse.json()) as DashboardSummary)
        : null;
      const nqPayload = nqResponse?.ok ? ((await nqResponse.json()) as NqFutureSummary) : null;

      setFx(payload);
      setFearGreed(fearGreedSummary);
      setDashboardSummary(dashboardPayload);
      setNqFuture(nqPayload);

      if (initial) {
        setIsFxLoading(false);
        setIsFearGreedLoading(false);
        setIsDashboardSummaryLoading(false);
        setIsNqLoading(false);
      }
    } catch {
      setFx(null);
      setFearGreed(null);
      setDashboardSummary(null);
      setNqFuture(null);
      if (initial) {
        setIsFxLoading(false);
        setIsFearGreedLoading(false);
        setIsDashboardSummaryLoading(false);
        setIsNqLoading(false);
      }
    }
  }, [isLoginPage]);

  useEffect(() => {
    void loadTopBarData(true);

    // 10초마다 상단 헤더 데이터(나스닥 선물, 환율 등)를 자동으로 갱신
    const timerId = setInterval(() => {
      void loadTopBarData(false);
    }, 10000);

    function handlePageShow() {
      void loadTopBarData(false);
    }

    window.addEventListener("pageshow", handlePageShow);
    return () => {
      clearInterval(timerId);
      window.removeEventListener("pageshow", handlePageShow);
    };
  }, [loadTopBarData]);

  useEffect(() => {
    if (isLoginPage) return;

    const errorHandler = () => {
      setIsDbError(true);
    };

    window.addEventListener("db_error_occurred", errorHandler);

    return () => {
      window.removeEventListener("db_error_occurred", errorHandler);
    };
  }, [isLoginPage]);

  useEffect(() => {
    if (isLoginPage) return;

    async function checkHealth() {
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        // 401(로그인 필요)은 DB 이슈가 아님 — 무시
        if (res.status === 401) {
          setIsDbError(false);
          return;
        }
        setIsDbError(!res.ok);
      } catch {
        setIsDbError(true);
      }
    }

    void checkHealth();
  }, [isLoginPage]);

  useEffect(() => {
    setSidebarOpen(false);
  }, [pathname]);

  useEffect(() => {
    setOpenGroups((current) => {
      const next = { ...current };
      for (const group of navGroups) {
        if (group.items.some((item) => isNavItemActive(item.href, pathname))) {
          next[group.id] = true;
        }
      }
      return next;
    });
  }, [pathname]);

  async function handleLogout() {
    await fetch("/api/auth/logout", { method: "POST" });
    router.replace("/login");
    router.refresh();
  }

  function toggleGroup(groupId: string) {
    setOpenGroups((current) => ({ ...current, [groupId]: !current[groupId] }));
  }

  if (isLoginPage) {
    return <div className="appContent loginAppContent">{children}</div>;
  }

  const fearGreedDelta =
    fearGreed?.score !== null &&
      fearGreed?.score !== undefined &&
      fearGreed.previous_close_score !== null &&
      fearGreed.previous_close_score !== undefined
      ? fearGreed.score - fearGreed.previous_close_score
      : null;

  const periodProfits = dashboardSummary?.period_profits;

  const sentimentWidget = (
    <div className="appSidebarSentiment">
      <a
        href="https://fear-and-greed.jason.ai.kr"
        target="_blank"
        rel="noopener noreferrer"
        className="appSidebarSentimentItem"
      >
        <span className="appSidebarSentimentLabel">
          <IconMoodSmile size={14} stroke={1.9} />
          <span>CNN</span>
        </span>
        {isFearGreedLoading ? (
          <span className="topbarSpinner" aria-label="CNN 로딩 중" />
        ) : fearGreed?.score != null ? (
          <span className="appSidebarSentimentValue">
            <strong>{formatFearGreedScore(fearGreed?.score)}</strong>
            <span className="appSidebarSentimentSub">({formatFearGreedLabel(fearGreed?.label)})</span>
            {fearGreedDelta !== null ? (
              <span className={getFxChangeClass(fearGreedDelta)}>
                {formatFearGreedDelta(fearGreedDelta)}
              </span>
            ) : null}
          </span>
        ) : (
          <strong>-</strong>
        )}
      </a>
    </div>
  );

  const navList = (
    <div className="appSidebarNavGroups">
      <div className="navbar-nav appSidebarNav appSidebarNavRoot">
        {[homeItem, ...ROOT_ITEMS].map((item) => {
          const ItemIcon = item.icon;
          return (
            <div key={item.href} className="nav-item appSidebarItem">
              <Link href={item.href} className={pathname === item.href ? "nav-link active" : "nav-link"}>
                <span className="appSidebarIcon" aria-hidden="true">
                  <ItemIcon size={18} stroke={1.9} />
                </span>
                <span className="nav-link-title">{item.label}</span>
              </Link>
            </div>
          );
        })}
      </div>
      {navGroups.map((group) => {
        const isExpanded = openGroups[group.id] ?? false;
        const hasActiveChild = group.items.some((item) => isNavItemActive(item.href, pathname));
        const GroupIcon = group.icon;
        return (
          <div key={group.id} className="appSidebarGroup">
            <button
              type="button"
              className={`appSidebarGroupToggle ${hasActiveChild ? "is-active" : ""}`.trim()}
              aria-expanded={isExpanded}
              onClick={() => toggleGroup(group.id)}
            >
              <span className="appSidebarGroupLead">
                <span className="appSidebarIcon" aria-hidden="true">
                  <GroupIcon size={18} stroke={1.9} />
                </span>
                <span className="appSidebarGroupLabel">{group.title}</span>
              </span>
              <span className={`appSidebarChevron ${isExpanded ? "is-open" : ""}`.trim()} aria-hidden="true">
                <IconChevronDown size={16} stroke={1.9} />
              </span>
            </button>
            <div className={`navbar-nav appSidebarNav appSidebarSubnav ${isExpanded ? "is-open" : ""}`.trim()}>
              {group.items.map((item) => {
                const isActive = isNavItemActive(item.href, pathname);
                const Icon = item.icon;
                return (
                  <div key={item.href} className="nav-item appSidebarItem">
                    <Link href={item.href} className={isActive ? "nav-link active" : "nav-link"}>
                      <span className="appSidebarIcon" aria-hidden="true">
                        {typeof Icon === "string" ? (
                          <span className="appSidebarEmojiIcon">{Icon}</span>
                        ) : (
                          <Icon size={18} stroke={1.9} />
                        )}
                      </span>
                      <span className="nav-link-title">{item.label}</span>
                    </Link>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );

  // 레일용 기간 수익률 행(금일/금주/금월/금년) — 이전엔 헤더에 있던 것을 CNN 아래로 옮김.
  const renderRailPeriodRow = (label: string, pct: number | undefined) => (
    <div className="appRailPeriodRow" key={label}>
      <span className="appRailPeriodLabel">{label}</span>
      {isDashboardSummaryLoading ? (
        <span className="topbarSpinner" aria-label={`${label} 로딩 중`} />
      ) : pct !== undefined && pct !== null && !Number.isNaN(pct) ? (
        <strong className={getFxChangeClass(pct)}>{`${pct > 0 ? "+" : ""}${pct.toFixed(2)}%`}</strong>
      ) : (
        <strong>-</strong>
      )}
    </div>
  );

  // 홈 우측 메뉴 레일 — 좌측 사이드바를 대체(메뉴 + CNN + 기간손익 + 로그아웃). 홈에서만 표시.
  const homeRail = (
    <aside className="appHomeRail">
      <HubMenu />
      <div className="appHomeRailFooter">
        {sentimentWidget}
        {showRailPeriod ? (
          <div className="appRailPeriod">
            {renderRailPeriodRow("금일", periodProfits?.daily?.return_pct)}
            {renderRailPeriodRow("금주", periodProfits?.weekly?.return_pct)}
            {renderRailPeriodRow("금월", periodProfits?.monthly?.return_pct)}
            {renderRailPeriodRow("금년", periodProfits?.yearly?.return_pct)}
          </div>
        ) : (
          <button
            className="btn btn-outline-secondary btn-sm w-100"
            type="button"
            onClick={() => setShowRailPeriod(true)}
          >
            수익률 보기
          </button>
        )}
        <button className="btn btn-outline-secondary btn-sm w-100" type="button" onClick={handleLogout}>
          로그아웃
        </button>
      </div>
    </aside>
  );

  return (
    <div className="appLayout appLayoutNoSidebar">
      <div className="appMain">
        <header className="navbar d-print-none appTopHeader">
          <div className="container-fluid appTopHeaderInner">
            <div className="appHeaderLeft">
              {/* 사이드바 없는 화면에서 홈(허브)으로 복귀 */}
              <Link
                href="/"
                className="btn btn-icon btn-sm appSidebarDesktopToggle"
                aria-label="홈(허브)으로 이동"
                title="홈(허브)"
              >
                <IconHome size={18} stroke={1.9} />
              </Link>
              {dashboardSummary?.is_deploying ? (
                <span
                  title="서버 배포 진행 중 — DB가 일시적으로 느려질 수 있습니다"
                  style={{
                    marginLeft: "0.5rem",
                    padding: "3px 10px",
                    borderRadius: 6,
                    background: "#fef3c7",
                    color: "#92400e",
                    fontWeight: 700,
                    fontSize: "0.82rem",
                    whiteSpace: "nowrap",
                  }}
                >
                  🚧 배포 진행 중
                </span>
              ) : null}
            </div>
            <div className="appMobileHeader">
              <button
                className="btn btn-icon btn-sm appSidebarToggle"
                type="button"
                aria-label={sidebarOpen ? "메뉴 닫기" : "메뉴 열기"}
                onClick={() => setSidebarOpen((current) => !current)}
              >
                {sidebarOpen ? <IconX size={18} stroke={1.9} /> : <IconMenu2 size={18} stroke={1.9} />}
              </button>
              <Link href="/" className="appMobileBrand">
                Jason 투자
              </Link>
            </div>
            <div className="topbarFx">
              {isDbError && (
                <span className="topbarFxItem" style={{ color: "#e03131", fontWeight: 600, background: "#ffe3e3", padding: "2px 8px", borderRadius: "4px" }}>
                  ⚠️ 몽고디비 이슈
                </span>
              )}
              <span className="topbarFxItem topbarTickerSearchItem">
                <GlobalTickerSearch />
              </span>
              <span className="topbarFxItem">
                나스닥선물:{" "}
                {isNqLoading ? (
                  <span className="topbarFxLoading" aria-label="나스닥선물 로딩 중">
                    <span className="topbarSpinner" />
                  </span>
                ) : nqFuture?.price != null ? (
                  <>
                    <strong>{new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(nqFuture.price)}</strong>
                    <span className={getFxChangeClass(nqFuture.change_pct ?? undefined)}>
                      {formatChangePct(nqFuture.change_pct ?? undefined)}
                    </span>
                  </>
                ) : (
                  <strong>-</strong>
                )}
              </span>
              <span className="topbarFxItem">
                USD/KRW:{" "}
                {isFxLoading ? (
                  <span className="topbarFxLoading" aria-label="환율 로딩 중">
                    <span className="topbarSpinner" />
                  </span>
                ) : (
                  <>
                    <strong>{formatRate(fx?.USD?.rate)}</strong>
                    <span className={getFxChangeClass(fx?.USD?.change_pct)}>
                      {formatChangePct(fx?.USD?.change_pct)}
                    </span>
                  </>
                )}
              </span>
              <span className="topbarFxItem">
                AUD/KRW:{" "}
                {isFxLoading ? (
                  <span className="topbarFxLoading" aria-label="환율 로딩 중">
                    <span className="topbarSpinner" />
                  </span>
                ) : (
                  <>
                    <strong>{formatRate(fx?.AUD?.rate)}</strong>
                    <span className={getFxChangeClass(fx?.AUD?.change_pct)}>
                      {formatChangePct(fx?.AUD?.change_pct)}
                    </span>
                  </>
                )}
              </span>
            </div>
          </div>
        </header>
        <div className={`appMobileMenu ${sidebarOpen ? "is-open" : ""}`.trim()}>
          <div className="container-fluid appMobileMenuInner">
            {navList}
            <div className="appSidebarFooter appMobileMenuFooter">
              {sentimentWidget}
              <button className="btn btn-outline-secondary btn-sm w-100" type="button" onClick={handleLogout}>
                로그아웃
              </button>
            </div>
          </div>
        </div>
        <div className="appBody">
          {isHome ? (
            <div className="appContent appContentHome">
              <div className="appHomeMainCol">{children}</div>
              {homeRail}
            </div>
          ) : (
            <div className="container-fluid appContent">
              <RecentPages />
              {children}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
