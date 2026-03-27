"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

const navItems = [
  { href: "/", label: "앱 메뉴" },
  { href: "/dashboard", label: "대시보드" },
  { href: "/cash", label: "자산관리" },
  { href: "/snapshots", label: "스냅샷" },
  { href: "/market", label: "ETF 마켓" },
  { href: "/py/rank", label: "Python 순위" },
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
  const [fx, setFx] = useState<FxSummary | null>(null);

  useEffect(() => {
    let alive = true;

    async function loadFx() {
      try {
        const response = await fetch("/api/fx", { cache: "no-store" });
        if (!response.ok) {
          return;
        }
        const payload = (await response.json()) as FxSummary;
        if (alive) {
          setFx(payload);
        }
      } catch {
        if (alive) {
          setFx(null);
        }
      }
    }

    loadFx();
    return () => {
      alive = false;
    };
  }, []);

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
            USD/KRW: <strong>{formatRate(fx?.USD?.rate)}</strong>
            <span className={getFxChangeClass(fx?.USD?.change_pct)}>
              {formatChangePct(fx?.USD?.change_pct)}
            </span>
          </span>
          <span className="topbarFxItem">
            AUD/KRW: <strong>{formatRate(fx?.AUD?.rate)}</strong>
            <span className={getFxChangeClass(fx?.AUD?.change_pct)}>
              {formatChangePct(fx?.AUD?.change_pct)}
            </span>
          </span>
        </div>
      </header>
      <div className="appContent">{children}</div>
    </div>
  );
}
