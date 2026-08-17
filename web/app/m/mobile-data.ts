"use client";

/** 모바일 화면(`/m`) 공용 데이터·표기.
 *
 * 데이터는 데스크톱 자산 화면과 같은 `/api/assets`(보유) + `/api/dashboard`(수익률) 를 쓴다.
 * 새 API 를 만들지 않는다 — 같은 숫자를 두 곳에서 따로 계산하면 화면끼리 값이 갈린다.
 */

import type { AccountSummary, HoldingsRow } from "../assets/assets-helpers";

/** 계좌 한 줄 — 보유 요약(assets)과 수익률(dashboard)을 합친 것. */
export type MobileAccount = {
  account_id: string;
  name: string;
  icon: string;
  order: number;
  total_assets_krw: number;
  cash_balance_krw: number;
  daily_return_pct: number;
  daily_profit: number;
  net_profit: number;
  net_profit_pct: number;
};

export type MobileTotals = {
  total_assets: number;
  daily_profit: number;
  daily_return_pct: number;
  net_profit: number;
  net_profit_pct: number;
};

export type MobileSnapshot = {
  accounts: MobileAccount[];
  totals: MobileTotals;
  rows: HoldingsRow[];
};

type DashboardAccount = {
  account_id: string;
  daily_profit?: number;
  daily_return_pct?: number;
  net_profit?: number;
  net_profit_pct?: number;
};

/** 보유·수익률을 한 번에 받아 화면이 쓰는 형태로 합친다. */
export async function loadMobileSnapshot(): Promise<MobileSnapshot> {
  const [assetsResponse, dashboardResponse] = await Promise.all([
    fetch("/api/assets", { cache: "no-store" }),
    fetch("/api/dashboard", { cache: "no-store" }),
  ]);

  const assets = (await assetsResponse.json()) as {
    rows?: HoldingsRow[];
    account_summaries?: AccountSummary[];
    error?: string;
  };
  if (!assetsResponse.ok || assets.error) {
    throw new Error(assets.error ?? "자산 정보를 불러오지 못했습니다.");
  }

  const dashboard = (await dashboardResponse.json()) as {
    accounts?: DashboardAccount[];
    totals?: Partial<MobileTotals> & { total_assets?: number; total_principal?: number };
    error?: string;
  };
  if (!dashboardResponse.ok || dashboard.error) {
    throw new Error(dashboard.error ?? "수익률 정보를 불러오지 못했습니다.");
  }

  const byAccount = new Map((dashboard.accounts ?? []).map((row) => [row.account_id, row]));
  const accounts: MobileAccount[] = (assets.account_summaries ?? []).map((summary) => {
    const metrics = byAccount.get(summary.account_id);
    return {
      account_id: summary.account_id,
      name: summary.name,
      icon: summary.icon,
      order: summary.order,
      total_assets_krw: summary.total_assets_krw,
      cash_balance_krw: summary.cash_balance_krw,
      daily_return_pct: metrics?.daily_return_pct ?? 0,
      daily_profit: metrics?.daily_profit ?? 0,
      net_profit: metrics?.net_profit ?? 0,
      net_profit_pct: metrics?.net_profit_pct ?? 0,
    };
  });
  accounts.sort((a, b) => a.order - b.order);

  const totalAssets = dashboard.totals?.total_assets ?? accounts.reduce((sum, a) => sum + a.total_assets_krw, 0);
  const netProfit = accounts.reduce((sum, account) => sum + account.net_profit, 0);
  const principal = totalAssets - netProfit;
  return {
    accounts,
    rows: assets.rows ?? [],
    totals: {
      total_assets: totalAssets,
      daily_profit: dashboard.totals?.daily_profit ?? 0,
      daily_return_pct: dashboard.totals?.daily_return_pct ?? 0,
      net_profit: netProfit,
      net_profit_pct: principal > 0 ? (netProfit / principal) * 100 : 0,
    },
  };
}

export function formatKrw(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR").format(Math.round(value));
}

/** 큰 금액은 폰 폭에 맞게 억·만 단위로 줄여 쓴다. */
export function formatCompactKrw(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "-";
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 100_000_000) return `${sign}${(abs / 100_000_000).toFixed(2)}억`;
  if (abs >= 10_000) return `${sign}${Math.round(abs / 10_000).toLocaleString("ko-KR")}만`;
  return `${sign}${Math.round(abs).toLocaleString("ko-KR")}`;
}

export function formatPct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "-";
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

/** 등락 색 — 전략 화면과 같은 기준(양수 초록·음수 빨강). */
export function signColorOf(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value) || value === 0) return "var(--text-muted)";
  return value > 0 ? "#2f9e44" : "#d62828";
}
