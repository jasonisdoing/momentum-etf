"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import type { ReactNode } from "react";

const navItems = [
  { href: "/", label: "앱 메뉴" },
  { href: "/dashboard", label: "대시보드" },
  { href: "/import", label: "벌크 입력" },
  { href: "/stocks", label: "종목 관리" },
  { href: "/cash", label: "자산관리" },
  { href: "/snapshots", label: "스냅샷" },
  { href: "/market", label: "ETF 마켓" },
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

  async function handleLogout() {
    await fetch("/api/auth/logout", { method: "POST" });
    router.replace("/login");
    router.refresh();
  }

  if (isLoginPage) {
    return <div className="appContent loginAppContent">{children}</div>;
  }

  return (
    <div className="appFrame">
      <header className="appTopbar">
        <Link href="/" className="appBrand appBrandTop">
          <strong>Momentum ETF</strong>
        </Link>
        <nav className="appNav appNavTop">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={isActive ? "appNavLink appNavLinkActive" : "appNavLink"}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
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
          <button className="topbarLogout" type="button" onClick={handleLogout}>
            로그아웃
          </button>
        </div>
      </header>
      <div className="appContent">{children}</div>
    </div>
  );
}
