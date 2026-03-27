"use client";

import { useEffect, useState } from "react";

type DashboardMetricItem = {
  label: string;
  value: number;
  kind: "money" | "percent";
};

type DashboardAccountSummaryItem = {
  account_id: string;
  account_name: string;
  order: number;
  total_assets: number;
  total_principal: number;
  valuation_krw: number;
  cash_balance: number;
  cash_ratio: number;
  net_profit: number;
  net_profit_pct: number;
};

type DashboardBucketItem = {
  label: string;
  weight_pct: number;
};

type DashboardData = {
  metrics?: DashboardMetricItem[];
  accounts?: DashboardAccountSummaryItem[];
  buckets?: DashboardBucketItem[];
  stats?: Array<{ label: string; value: number; kind: "money" | "percent" | "count" }>;
  latest_snapshot_date?: string | null;
  weekly_date?: string | null;
  updated_at?: string | null;
  error?: string;
};

function formatMoney(value: number): string {
  return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
}

function formatPercent(value: number): string {
  return `${value.toFixed(2)}%`;
}

function formatMetricValue(value: number, kind: "money" | "percent" | "count"): string {
  if (kind === "money") {
    return formatMoney(value);
  }
  if (kind === "percent") {
    return formatPercent(value);
  }
  return new Intl.NumberFormat("ko-KR").format(value);
}

function formatUpdatedAt(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("ko-KR", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function getSignedClass(value: number): string | undefined {
  if (value > 0) {
    return "metricPositive";
  }
  if (value < 0) {
    return "metricNegative";
  }
  return undefined;
}

function shouldHighlightMetric(label: string): boolean {
  return label.includes("손익") || label.includes("수익률");
}

export function DashboardManager() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hideMoney, setHideMoney] = useState(false);

  function formatMoneyDisplay(value: number): string {
    if (hideMoney) {
      return "••••••";
    }
    return formatMoney(value);
  }

  function formatMetricDisplay(value: number, kind: "money" | "percent" | "count"): string {
    if (kind === "money") {
      return formatMoneyDisplay(value);
    }
    return formatMetricValue(value, kind);
  }

  useEffect(() => {
    let alive = true;

    async function load() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch("/api/dashboard", { cache: "no-store" });
        const payload = (await response.json()) as DashboardData;
        if (!response.ok) {
          throw new Error(payload.error ?? "대시보드 데이터를 불러오지 못했습니다.");
        }
        if (alive) {
          setData(payload);
        }
      } catch (loadError) {
        if (alive) {
          setError(loadError instanceof Error ? loadError.message : "대시보드 데이터를 불러오지 못했습니다.");
        }
      } finally {
        if (alive) {
          setLoading(false);
        }
      }
    }

    load();
    return () => {
      alive = false;
    };
  }, []);

  if (loading) {
    return <section className="section"><p>대시보드 데이터를 불러오는 중...</p></section>;
  }

  return (
    <>
      <section className="pageHeaderCompact">
        <div><h1>대시보드</h1></div>
        <button
          type="button"
          className={`toggleButton ${hideMoney ? "toggleButtonActive" : ""}`.trim()}
          onClick={() => setHideMoney((current) => !current)}
        >
          금액 가리기 {hideMoney ? "ON" : "OFF"}
        </button>
      </section>

      <section className="section">
        {error ? <div className="bannerError">{error}</div> : null}
        <div className="dashboardMetricGrid">
          {(data?.metrics ?? []).map((metric) => (
            <div key={metric.label} className="dashboardMetricCard">
              <div className="dashboardMetricLabel">{metric.label}</div>
              <div className={shouldHighlightMetric(metric.label) ? getSignedClass(metric.value) : undefined}>
                {formatMetricDisplay(metric.value, metric.kind)}
              </div>
            </div>
          ))}
        </div>
        <div className="tableFooterMeta">
          스냅샷 기준일: {data?.latest_snapshot_date ?? "-"} | 주별 기준일: {data?.weekly_date ?? "-"} | 갱신:{" "}
          {formatUpdatedAt(data?.updated_at)}
        </div>
      </section>

      <section className="section">
        <div className="sectionHeaderCompact">
          <h2>포트폴리오 구성 비중</h2>
        </div>
        <div className="dashboardInfoRibbon">
          {(data?.buckets ?? []).map((bucket) => (
            <article key={bucket.label} className="dashboardInfoCard">
              <div className="dashboardInfoCardLabel">{bucket.label}</div>
              <div className="dashboardInfoCardValue">{formatPercent(bucket.weight_pct)}</div>
            </article>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="sectionHeaderCompact">
          <h2>자산/주간 통계</h2>
        </div>
        <div className="dashboardInfoRibbon">
          {(data?.stats ?? []).map((stat) => (
            <article key={stat.label} className="dashboardInfoCard">
              <div className="dashboardInfoCardLabel">{stat.label}</div>
              <div className={`dashboardInfoCardValue ${shouldHighlightMetric(stat.label) ? getSignedClass(stat.value) ?? "" : ""}`.trim()}>
                {formatMetricDisplay(stat.value, stat.kind)}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="sectionHeaderCompact">
          <h2>계좌별 요약</h2>
        </div>
        <div className="tableWrap">
          <table className="erpTable dashboardAccountTable">
            <colgroup>
              <col className="dashboardAccountColName" />
              <col className="dashboardAccountColValue" />
              <col className="dashboardAccountColValue" />
              <col className="dashboardAccountColValue" />
              <col className="dashboardAccountColValue" />
              <col className="dashboardAccountColRatio" />
              <col className="dashboardAccountColValue" />
              <col className="dashboardAccountColRatio" />
            </colgroup>
            <thead>
              <tr>
                <th>계좌</th>
                <th className="tableAlignRight">총 자산</th>
                <th className="tableAlignRight">총 원금</th>
                <th className="tableAlignRight">평가 금액</th>
                <th className="tableAlignRight">현금</th>
                <th className="tableAlignRight">현금 비중</th>
                <th className="tableAlignRight">계좌 손익</th>
                <th className="tableAlignRight">계좌 수익률</th>
              </tr>
            </thead>
            <tbody>
              {(data?.accounts ?? []).map((account) => (
                <tr key={account.account_id}>
                  <td>{account.account_name}</td>
                  <td className="tableAlignRight">{formatMoneyDisplay(account.total_assets)}</td>
                  <td className="tableAlignRight">{formatMoneyDisplay(account.total_principal)}</td>
                  <td className="tableAlignRight">{formatMoneyDisplay(account.valuation_krw)}</td>
                  <td className="tableAlignRight">{formatMoneyDisplay(account.cash_balance)}</td>
                  <td className="tableAlignRight">{formatPercent(account.cash_ratio)}</td>
                  <td className={`tableAlignRight ${getSignedClass(account.net_profit) ?? ""}`.trim()}>
                    {formatMoneyDisplay(account.net_profit)}
                  </td>
                  <td className={`tableAlignRight ${getSignedClass(account.net_profit_pct) ?? ""}`.trim()}>
                    {formatPercent(account.net_profit_pct)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

    </>
  );
}
