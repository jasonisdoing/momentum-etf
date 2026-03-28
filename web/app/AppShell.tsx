"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import type { ReactNode } from "react";
import {
  IconCash,
  IconChevronDown,
  IconFileImport,
  IconMoodSmile,
  IconHome,
  IconListDetails,
  IconMenu2,
  IconNotebook,
  IconReceipt2,
  IconSettings,
  IconSparkles,
  IconTrendingUp,
  IconX,
} from "@tabler/icons-react";

import { parseFearGreedSummary } from "@/lib/fear-greed";

const homeItem = { href: "/", label: "Home", icon: IconHome };

const navGroups = [
  {
    id: "assets",
    title: "자산",
    icon: IconCash,
    items: [
      { href: "/cash", label: "자산 관리", icon: IconCash },
      { href: "/import", label: "벌크 입력", icon: IconFileImport },
      { href: "/snapshots", label: "스냅샷", icon: IconReceipt2 },
      { href: "/weekly", label: "주별", icon: IconReceipt2 },
    ],
  },
  {
    id: "momentum-etf",
    title: "Momentum ETF",
    icon: IconListDetails,
    items: [
      { href: "/stocks", label: "종목 관리", icon: IconListDetails },
      { href: "/market", label: "ETF 마켓", icon: IconTrendingUp },
      { href: "/note", label: "계좌 메모", icon: IconNotebook },
      { href: "/summary", label: "AI용 요약", icon: IconSparkles },
    ],
  },
  {
    id: "system",
    title: "시스템",
    icon: IconSettings,
    items: [{ href: "/system", label: "정보", icon: IconSettings }],
  },
] as const;

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
  const sign = value > 0 ? "+" : "";
  return `(${sign}${value.toFixed(2)}%)`;
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

  const browserResponse = await fetch("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", {
    cache: "no-store",
  });

  if (!browserResponse.ok) {
    return null;
  }

  const payload = (await browserResponse.json()) as unknown;
  const summary = parseFearGreedSummary(payload);

  if (summary.score === null || !summary.label) {
    return null;
  }

  return summary;
}

function getDefaultOpenGroups(pathname: string | null) {
  return Object.fromEntries(
    navGroups.map((group) => [group.id, true]),
  );
}

export function AppShell({ children }: AppShellProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [fx, setFx] = useState<FxSummary | null>(null);
  const [isFxLoading, setIsFxLoading] = useState(true);
  const [fearGreed, setFearGreed] = useState<FearGreedSummary | null>(null);
  const [isFearGreedLoading, setIsFearGreedLoading] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>(() => getDefaultOpenGroups(pathname));
  const isLoginPage = pathname === "/login";

  useEffect(() => {
    if (isLoginPage) {
      return;
    }

    let alive = true;

    async function loadFx() {
      try {
        if (alive) {
          setIsFxLoading(true);
        }
        if (alive) {
          setIsFearGreedLoading(true);
        }
        const [fxResponse, fearGreedSummary] = await Promise.all([
          fetch("/api/fx", { cache: "no-store" }),
          loadFearGreedSummary().catch(() => null),
        ]);

        const payload = fxResponse.ok ? ((await fxResponse.json()) as FxSummary) : null;
        if (alive) {
          setFx(payload);
          setIsFxLoading(false);
          setFearGreed(fearGreedSummary);
          setIsFearGreedLoading(false);
        }
      } catch {
        if (alive) {
          setFx(null);
          setIsFxLoading(false);
          setFearGreed(null);
          setIsFearGreedLoading(false);
        }
      }
    }

    loadFx();
    return () => {
      alive = false;
    };
  }, [isLoginPage]);

  useEffect(() => {
    setSidebarOpen(false);
  }, [pathname]);

  useEffect(() => {
    setOpenGroups((current) => {
      const next = { ...current };
      for (const group of navGroups) {
        if (group.items.some((item) => item.href === pathname)) {
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

  const HomeIcon = homeItem.icon;
  const fearGreedDelta =
    fearGreed?.score !== null &&
    fearGreed?.score !== undefined &&
    fearGreed.previous_close_score !== null &&
    fearGreed.previous_close_score !== undefined
      ? fearGreed.score - fearGreed.previous_close_score
      : null;

  const navList = (
    <div className="appSidebarNavGroups">
      <div className="navbar-nav appSidebarNav appSidebarNavRoot">
        <div className="nav-item appSidebarItem">
          <Link href={homeItem.href} className={pathname === homeItem.href ? "nav-link active" : "nav-link"}>
            <span className="appSidebarIcon" aria-hidden="true">
              <HomeIcon size={18} stroke={1.9} />
            </span>
            <span className="nav-link-title">{homeItem.label}</span>
          </Link>
        </div>
      </div>
      {navGroups.map((group) => {
        const isExpanded = openGroups[group.id] ?? false;
        const hasActiveChild = group.items.some((item) => item.href === pathname);
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
                const isActive = pathname === item.href;
                const Icon = item.icon;
                return (
                  <div key={item.href} className="nav-item appSidebarItem">
                    <Link href={item.href} className={isActive ? "nav-link active" : "nav-link"}>
                      <span className="appSidebarIcon" aria-hidden="true">
                        <Icon size={18} stroke={1.9} />
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

  return (
    <div className="appLayout">
      <aside className="navbar navbar-vertical navbar-expand-lg appSidebar appSidebarDesktop">
        <div className="container-fluid appSidebarInner">
          <Link href="/" className="navbar-brand navbar-brand-autodark appSidebarBrand">
            Jason 투자
          </Link>
          {navList}
          <div className="appSidebarFooter">
            <button className="btn btn-outline-light btn-sm w-100" type="button" onClick={handleLogout}>
              로그아웃
            </button>
          </div>
        </div>
      </aside>
      <div className="appMain">
        <header className="navbar d-print-none appTopHeader">
          <div className="container-fluid appTopHeaderInner">
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
              <span className="topbarFxItem topbarSentimentItem">
                <span className="topbarSentimentLabel">
                  <IconMoodSmile size={15} stroke={1.9} />
                  CNN:
                </span>
                {isFearGreedLoading ? (
                  <span className="topbarFxLoading" aria-label="CNN 공포탐욕지수 로딩 중">
                    <span className="topbarSpinner" />
                  </span>
                ) : fearGreed?.score !== null ? (
                  <span className="topbarSentimentValue">
                    <strong>{formatFearGreedScore(fearGreed.score)}</strong>
                    <span className="topbarSentimentText">({formatFearGreedLabel(fearGreed.label)})</span>
                    {fearGreedDelta !== null ? (
                      <span className={getFxChangeClass(fearGreedDelta)}>
                        {formatFearGreedDelta(fearGreedDelta)}
                      </span>
                    ) : null}
                  </span>
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
              <button className="btn btn-outline-light btn-sm w-100" type="button" onClick={handleLogout}>
                로그아웃
              </button>
            </div>
          </div>
        </div>
        <div className="appBody">
          <div className="container-fluid appContent">{children}</div>
        </div>
      </div>
    </div>
  );
}
