"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import type { ReactNode } from "react";
import {
  IconCash,
  IconFileImport,
  IconHome,
  IconListDetails,
  IconMenu2,
  IconReceipt2,
  IconTrendingUp,
  IconX,
} from "@tabler/icons-react";

const navItems = [
  { href: "/", label: "Home", icon: IconHome },
  { href: "/import", label: "벌크 입력", icon: IconFileImport },
  { href: "/stocks", label: "종목 관리", icon: IconListDetails },
  { href: "/cash", label: "자산관리", icon: IconCash },
  { href: "/snapshots", label: "스냅샷", icon: IconReceipt2 },
  { href: "/market", label: "ETF 마켓", icon: IconTrendingUp },
];

type AppShellProps = {
  children: ReactNode;
};

type FxSummary = {
  USD?: { rate: number; change_pct: number };
  AUD?: { rate: number; change_pct: number };
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

export function AppShell({ children }: AppShellProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [fx, setFx] = useState<FxSummary | null>(null);
  const [isFxLoading, setIsFxLoading] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
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
        const response = await fetch("/api/fx", { cache: "no-store" });
        if (!response.ok) {
          if (alive) {
            setFx(null);
            setIsFxLoading(false);
          }
          return;
        }
        const payload = (await response.json()) as FxSummary;
        if (alive) {
          setFx(payload);
          setIsFxLoading(false);
        }
      } catch {
        if (alive) {
          setFx(null);
          setIsFxLoading(false);
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

  async function handleLogout() {
    await fetch("/api/auth/logout", { method: "POST" });
    router.replace("/login");
    router.refresh();
  }

  if (isLoginPage) {
    return <div className="appContent loginAppContent">{children}</div>;
  }

  const navList = (
    <div className="navbar-nav appSidebarNav">
      {navItems.map((item) => {
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
  );

  return (
    <div className="appLayout">
      <aside className="navbar navbar-vertical navbar-expand-lg appSidebar appSidebarDesktop">
        <div className="container-fluid appSidebarInner">
          <Link href="/" className="navbar-brand navbar-brand-autodark appSidebarBrand">
            Momentum ETF
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
                Momentum ETF
              </Link>
            </div>
            <div className="topbarFx">
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
