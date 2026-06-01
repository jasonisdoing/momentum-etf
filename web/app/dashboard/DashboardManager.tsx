"use client";

import { useEffect, useRef, useState } from "react";
import { Area, AreaChart, Cell, Pie, PieChart, Tooltip, XAxis, YAxis } from "recharts";

import { BUCKET_COLORS } from "@/lib/bucket-theme";
import { useHideMoney } from "@/lib/hide-money-context";
import { AppLoadingState } from "../components/AppLoadingState";

type DashboardMetricItem = {
  label: string;
  value: number;
  kind: "money" | "percent";
  sub_value?: number;
  sub_kind?: "money" | "percent";
};

type DashboardAccountSummaryItem = {
  account_id: string;
  account_name: string;
  account_url?: string | null;
  order: number;
  total_assets: number;
  total_principal: number;
  valuation_krw: number;
  cash_balance: number;
  cash_ratio: number;
  net_profit: number;
  net_profit_pct: number;
  daily_profit: number;
  weekly_profit: number;
};

type DashboardBucketItem = {
  label: string;
  weight_pct: number;
};

type DashboardData = {
  metrics_row1?: DashboardMetricItem[];
  metrics_row2?: DashboardMetricItem[];
  period_profits?: {
    daily: { profit: number; return_pct: number };
    weekly: { profit: number; return_pct: number };
    monthly: { profit: number; return_pct: number };
    yearly: { profit: number; return_pct: number };
  };
  accounts?: DashboardAccountSummaryItem[];
  buckets?: DashboardBucketItem[];
  account_buckets?: Record<string, DashboardBucketItem[]>;
  sparklines?: Record<string, Array<{ date: string; value: number }>>;
  latest_snapshot_date?: string | null;
  weekly_date?: string | null;
  updated_at?: string | null;
  error?: string;
};

function formatMoney(value: number): string {
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1_0000_0000) {
    const eok = Math.floor(abs / 1_0000_0000);
    const man = Math.round((abs % 1_0000_0000) / 1_0000);
    return man > 0 ? `${sign}${eok}억 ${new Intl.NumberFormat("ko-KR").format(man)}만원` : `${sign}${eok}억원`;
  }
  if (abs >= 1_0000) {
    const man = Math.round(abs / 1_0000);
    return `${sign}${new Intl.NumberFormat("ko-KR").format(man)}만원`;
  }
  return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
}

function formatPercent(value: number): string {
  return `${value.toFixed(2)}%`;
}

function formatMetricValue(value: number, kind: "money" | "percent" | "count"): string {
  if (kind === "money") return formatMoney(value);
  if (kind === "percent") return formatPercent(value);
  return new Intl.NumberFormat("ko-KR").format(value);
}

function formatUpdatedAt(value: string | null | undefined): string {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function getSignedClass(value: number): string {
  if (value > 0) return "metricPositive";
  if (value < 0) return "metricNegative";
  return "";
}

function shouldHighlight(label: string): boolean {
  return label.includes("손익") || label.includes("수익률");
}

function renderAccountNameCell(account: DashboardAccountSummaryItem) {
  if (!account.account_url) {
    return <span className="fw-medium">{account.account_name}</span>;
  }

  return (
    <a
      href={account.account_url}
      target="_blank"
      rel="noreferrer"
      className="fw-medium"
      style={{ textDecoration: "underline" }}
    >
      {account.account_name}
    </a>
  );
}

type SparklinePoint = { date: string; value: number };

function SparklineTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: SparklinePoint }> }) {
  if (!active || !payload?.[0]) return null;
  const point = payload[0].payload;
  return (
    <div
      style={{
        background: "#1e293b",
        color: "#fff",
        padding: "4px 8px",
        borderRadius: 4,
        fontSize: "0.72rem",
        lineHeight: 1.4,
        whiteSpace: "nowrap",
      }}
    >
      <div style={{ fontWeight: 600 }}>{point.date}</div>
      <div>{new Intl.NumberFormat("ko-KR").format(Math.round(point.value))}</div>
    </div>
  );
}

let sparklineCounter = 0;

function formatYAxisTick(value: number, kind: "money" | "percent" | "count" = "money"): string {
  if (!Number.isFinite(value)) return "";
  if (kind === "percent") return `${value.toFixed(0)}%`;
  // money / count → 억/만원 단위로 축약
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 100_000_000) {
    const eok = value / 100_000_000;
    return `${sign}${Math.abs(eok).toFixed(eok % 1 === 0 ? 0 : 1).replace(/^-/, "")}억`;
  }
  if (abs >= 10_000) {
    const man = Math.round(value / 10_000);
    return `${man.toLocaleString("ko-KR")}만`;
  }
  return value.toLocaleString("ko-KR");
}

function Sparkline({
  data,
  color = "#206bc4",
  kind = "money",
  hideYAxisValues = false,
}: {
  data: SparklinePoint[];
  color?: string;
  kind?: "money" | "percent" | "count";
  hideYAxisValues?: boolean;
}) {
  const [id] = useState(() => `spark-${++sparklineCounter}`);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      const nextWidth = Math.floor(rect?.width ?? 0);
      const nextHeight = Math.floor(rect?.height ?? 0);
      if (nextWidth > 0 && nextHeight > 0) {
        setSize({ width: nextWidth, height: nextHeight });
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  if (data.length < 2) return null;
  const tickStyle = { fontSize: 12, fill: "#4a5568", fontWeight: 500 };
  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%", minHeight: 120, minWidth: 0 }}>
      {size.width > 0 && size.height > 0 ? (
        <AreaChart
          data={data}
          width={size.width}
          height={size.height}
          margin={{ top: 4, right: 8, bottom: 4, left: 4 }}
        >
          <defs>
            <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.2} />
              <stop offset="100%" stopColor={color} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="date"
            tick={tickStyle}
            tickFormatter={(value: string) => formatSparkDateLabel(value)}
            tickLine={false}
            axisLine={{ stroke: "#e6e8ec" }}
            minTickGap={32}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={hideYAxisValues ? false : tickStyle}
            tickFormatter={(value: number) => formatYAxisTick(value, kind)}
            tickLine={false}
            axisLine={{ stroke: "#e6e8ec" }}
            width={hideYAxisValues ? 12 : 52}
            tickCount={4}
            domain={["auto", "auto"]}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={1.5}
            fill={`url(#${id})`}
            dot={false}
            isAnimationActive={false}
          />
          <Tooltip content={<SparklineTooltip />} cursor={{ stroke: color, strokeWidth: 1, strokeDasharray: "3 3" }} />
        </AreaChart>
      ) : null}
    </div>
  );
}

function DashboardDonutChart({ buckets }: { buckets: DashboardBucketItem[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      const nextWidth = Math.floor(rect?.width ?? 0);
      const nextHeight = Math.floor(rect?.height ?? 0);
      if (nextWidth > 0 && nextHeight > 0) {
        setSize({ width: nextWidth, height: nextHeight });
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const chartSize = size.width > 0 && size.height > 0 ? Math.min(size.width, size.height, 300) : 0;

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%", minWidth: 0, minHeight: 120 }}>
      {chartSize > 0 ? (
        <PieChart width={chartSize} height={chartSize} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
          <Pie
            data={buckets}
            dataKey="weight_pct"
            nameKey="label"
            cx="50%"
            cy="50%"
            innerRadius={chartSize * 0.26}
            outerRadius={chartSize * 0.46}
            paddingAngle={2}
            strokeWidth={0}
          >
            {buckets.map((_, i) => (
              <Cell key={i} fill={BUCKET_COLORS[i % BUCKET_COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            formatter={(value) => `${Number(value).toFixed(2)}%`}
            contentStyle={{ fontSize: "0.82rem", borderRadius: 6 }}
          />
        </PieChart>
      ) : null}
    </div>
  );
}

function DashboardBucketLegend({ buckets }: { buckets: DashboardBucketItem[] }) {
  return (
    <div className="row g-1">
      {buckets.map((bucket, index) => (
        <div key={bucket.label} className="col-6">
          <div className="d-flex align-items-center gap-1" style={{ minWidth: 0 }}>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                backgroundColor: BUCKET_COLORS[index % BUCKET_COLORS.length],
                flex: "0 0 auto",
              }}
            />
            <span
              className="text-secondary"
              style={{
                fontSize: "0.72rem",
                minWidth: 0,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {bucket.label.replace(/^\d+\.\s*/, "")} {bucket.weight_pct.toFixed(1)}%
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

const PERIOD_OPTIONS = [
  { label: "1달", months: 1 },
  { label: "3달", months: 3 },
  { label: "6달", months: 6 },
  { label: "1년", months: 12 },
  { label: "2년", months: 24 },
  { label: "3년", months: 36 },
];

function formatSparkDateLabel(value: string): string {
  if (!value) return "";
  const date = new Date(value.length === 10 ? `${value}T00:00:00` : value);
  if (Number.isNaN(date.getTime())) return value;
  const yy = String(date.getFullYear()).slice(2);
  const mm = String(date.getMonth() + 1);
  return `${yy}.${mm}`;
}

function calcChange(
  sparkData: SparklinePoint[] | undefined,
  months: number,
): { pct: number; delta: number } | null {
  if (!sparkData || sparkData.length < 2) return null;
  const weeksBack = Math.round(months * 4.33);
  const baseIndex = Math.max(0, sparkData.length - 1 - weeksBack);
  const baseValue = sparkData[baseIndex].value;
  const currentValue = sparkData[sparkData.length - 1].value;
  if (baseValue === 0) return null;
  const delta = currentValue - baseValue;
  const pct = (delta / Math.abs(baseValue)) * 100;
  return { pct, delta };
}

type DashboardRenderableMetricItem = {
  label: string;
  value: number;
  kind: "money" | "percent" | "count";
  sub_value?: number;
  sub_kind?: "money" | "percent" | "count";
};

const DASHBOARD_ROW1_LABELS = ["총 자산", "투자 원금", "금일 손익", "금주 손익"] as const;
const DASHBOARD_ROW2_LABELS = ["누적 손익", "현금 잔고", "금일 손익", "금주 손익"] as const;
const DASHBOARD_LEFT_LABELS = ["총 자산", "투자 원금", "누적 손익", "평가 손익 (인출분 합산)"] as const;

const DASHBOARD_ACCOUNT_WEIGHTS: { account_id: string; icon: string; label: string }[] = [
  { account_id: "kor_account", icon: "🇰🇷", label: "국내 계좌" },
  { account_id: "isa_account", icon: "🇰🇷", label: "ISA 계좌" },
  { account_id: "pension_account", icon: "🇰🇷", label: "연금저축 계좌" },
  { account_id: "core_account", icon: "💼", label: "장기보유 계좌" },
  { account_id: "aus_account", icon: "🇦🇺", label: "호주 계좌" },
];

function orderMetricItems(
  items: DashboardRenderableMetricItem[],
  labels: readonly string[],
): DashboardRenderableMetricItem[] {
  const itemMap = new Map(items.map((item) => [item.label, item]));
  return labels
    .map((label) => itemMap.get(label))
    .filter((item): item is DashboardRenderableMetricItem => item !== undefined);
}

function DashboardMetricCard({
  item,
  hideMoney,
  periodMonths,
  sparklines,
}: {
  item: DashboardRenderableMetricItem;
  hideMoney: boolean;
  periodMonths: number;
  sparklines: Record<string, SparklinePoint[]>;
}) {
  const highlighted = shouldHighlight(item.label);
  const signClass = highlighted ? getSignedClass(item.value) : "";
  const fullSparkData = sparklines[item.label];
  const sparkData = (() => {
    if (!fullSparkData || fullSparkData.length === 0) return fullSparkData;
    const weeksBack = Math.round(periodMonths * 4.33);
    const startIndex = Math.max(0, fullSparkData.length - 1 - weeksBack);
    return fullSparkData.slice(startIndex);
  })();
  const sparkColor = highlighted ? (item.value >= 0 ? "#2fb344" : "#d63939") : "#206bc4";
  const change = calcChange(fullSparkData, periodMonths);
  const isMoneyKind = item.kind === "money";

  function mask(value: number, kind: "money" | "percent" | "count" = "money"): string {
    if (hideMoney && kind === "money") return "••••••";
    return formatMetricValue(value, kind);
  }

  return (
    <div className="card card-sm dashboardMetricCard">
      <div
        className={`card-body ${sparkData ? "dashboardMetricCardBody" : "dashboardMetricCardBody dashboardMetricCardBodyTextOnly"}`}
        style={{ overflow: "hidden", paddingTop: "0.85rem", paddingBottom: "0.85rem" }}
      >
        <div className="d-flex align-items-start justify-content-between">
          <div
            className="subheader"
            style={{ fontSize: "1rem", fontWeight: 700, color: "#1f2d3d", letterSpacing: 0 }}
          >
            {item.label}
          </div>
          {change !== null ? (() => {
            const pctRounded = Number(change.pct.toFixed(1));
            const isZero = pctRounded === 0;
            const colorClass = isZero ? "" : change.pct >= 0 ? "metricPositive" : "metricNegative";
            const arrow = isZero ? "" : change.pct >= 0 ? " \u2197" : " \u2198";
            const deltaText = isMoneyKind ? (hideMoney ? "•••••• " : `${formatMoney(change.delta)} `) : "";
            const periodOption = PERIOD_OPTIONS.find((opt) => opt.months === periodMonths);
            const periodLabel = periodOption ? `${periodOption.label} \uc804 \ub300\ube44` : "";
            return (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 2 }}>
                <span
                  className="subheader"
                  style={{ fontSize: "0.78rem", fontWeight: 600, color: "#8a94a6" }}
                >
                  {periodLabel}
                </span>
                <span
                  className={colorClass}
                  style={{ fontSize: "0.94rem", fontWeight: 600, whiteSpace: "nowrap" }}
                >
                  {deltaText}
                  {pctRounded.toFixed(1)}%{arrow}
                </span>
              </div>
            );
          })() : null}
        </div>
        {item.sub_value !== undefined && item.sub_kind === "count" ? (
          <div className="d-flex align-items-baseline gap-1" style={{ whiteSpace: "nowrap" }}>
            <span className="h1 mb-0 metricPositive" style={{ fontSize: "1.1rem" }}>
              {formatMetricValue(item.value, item.kind)}
            </span>
            <span className="text-secondary" style={{ fontSize: "1rem", fontWeight: 600 }}>
              /
            </span>
            <span className="h1 mb-0 metricNegative" style={{ fontSize: "1.1rem" }}>
              {formatMetricValue(item.sub_value, item.sub_kind)}
            </span>
          </div>
        ) : (
          <div className="d-flex align-items-baseline gap-2">
            <div className={`h1 mb-0 ${signClass}`.trim()} style={{ fontSize: "1.32rem" }}>
              {mask(item.value, item.kind)}
            </div>
            {item.sub_value !== undefined && item.sub_kind ? (
              <span
                className={getSignedClass(item.sub_value)}
                style={{ fontSize: "0.94rem", fontWeight: 600, whiteSpace: "nowrap" }}
              >
                {formatMetricValue(item.sub_value, item.sub_kind)}
              </span>
            ) : null}
          </div>
        )}
        {sparkData ? (
          <div className="dashboardMetricCardSparkline">
            <Sparkline
              data={sparkData}
              color={sparkColor}
              kind={item.kind}
              hideYAxisValues={hideMoney && item.kind === "money"}
            />
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function DashboardManager() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { hideMoney, toggleHideMoney } = useHideMoney();
  const [periodMonths, setPeriodMonths] = useState(12);

  function mask(value: number, kind: "money" | "percent" | "count" = "money"): string {
    if (hideMoney && kind === "money") return "••••••";
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
        if (alive) setData(payload);
      } catch (loadError) {
        if (alive) setError(loadError instanceof Error ? loadError.message : "대시보드 데이터를 불러오지 못했습니다.");
      } finally {
        if (alive) setLoading(false);
      }
    }

    load();
    return () => { alive = false; };
  }, []);

  if (loading) {
    return (
      <div className="appPageStack">
        <div className="appPageLoading">
          <AppLoadingState label="대시보드 데이터를 불러오는 중..." />
        </div>
      </div>
    );
  }

  const row1 = data?.metrics_row1 ?? [];
  const row2 = data?.metrics_row2 ?? [];
  const dashboardMetricItems = [...row1, ...row2].map((item) => ({
    ...item,
    kind: item.kind as "money" | "percent" | "count",
    sub_kind: item.sub_kind as "money" | "percent" | "count" | undefined,
  }));
  const leftMetricItems = orderMetricItems(dashboardMetricItems, DASHBOARD_LEFT_LABELS);
  const holdingsStatusMetric = dashboardMetricItems.find((item) => item.label === "수익/손실 종목 수");
  const sparklines = data?.sparklines ?? {};

  return (
    <div className="appPageStack">
      {error ? (
        <div className="appBannerStack">
          <div className="alert alert-danger mb-0">{error}</div>
        </div>
      ) : null}

      {/* 헤더 + 기준일 */}
      <div className="d-flex align-items-center justify-content-between">
        <div className="text-secondary" style={{ fontSize: "0.82rem" }}>
          스냅샷 {data?.latest_snapshot_date ?? "-"} · 주별 {data?.weekly_date ?? "-"} · 갱신 {formatUpdatedAt(data?.updated_at)}
        </div>
        <div className="d-flex align-items-center gap-2">
          <div className="btn-group btn-group-sm" role="group" aria-label="기간 선택">
            {PERIOD_OPTIONS.map((opt) => (
              <button
                key={opt.months}
                type="button"
                className={`btn ${periodMonths === opt.months ? "btn-primary" : "btn-outline-secondary"}`}
                onClick={() => setPeriodMonths(opt.months)}
              >
                지난 {opt.label}
              </button>
            ))}
          </div>
          {holdingsStatusMetric && holdingsStatusMetric.sub_value !== undefined ? (
            <div
              className="text-secondary"
              style={{ fontSize: "0.85rem", fontWeight: 600, whiteSpace: "nowrap" }}
            >
              수익/손실 종목 수{" "}
              <span className="metricPositive">{formatMetricValue(holdingsStatusMetric.value, holdingsStatusMetric.kind)}</span>
              <span className="text-secondary">/</span>
              <span className="metricNegative">
                {formatMetricValue(holdingsStatusMetric.sub_value, holdingsStatusMetric.sub_kind ?? "count")}
              </span>
            </div>
          ) : null}
          <button
            type="button"
            className={`btn btn-sm ${hideMoney ? "btn-primary" : "btn-outline-secondary"}`}
            onClick={toggleHideMoney}
          >
            {hideMoney ? "금액 보이기" : "금액 가리기"}
          </button>
        </div>
      </div>

      <div className="dashboardOverviewGrid">
        <div className="dashboardOverviewLeft">
          {leftMetricItems.map((item) => (
            <div key={`left-${item.label}`} className="dashboardOverviewMetric">
              <DashboardMetricCard
                item={item}
                hideMoney={hideMoney}
                periodMonths={periodMonths}
                sparklines={sparklines}
              />
            </div>
          ))}
        </div>
      </div>

      {data?.period_profits ? (
        <div className="dashboardPeriodProfitGrid">
          {(
            [
              { key: "daily", label: "금일" },
              { key: "weekly", label: "금주" },
              { key: "monthly", label: "금월" },
              { key: "yearly", label: "금년" },
            ] as const
          ).map((entry) => {
            const item = data.period_profits?.[entry.key];
            const profit = item?.profit ?? 0;
            const pct = item?.return_pct ?? 0;
            const signClass = getSignedClass(profit);
            return (
              <div key={entry.key} className="card card-sm dashboardPeriodProfitCard">
                <div className="card-body dashboardPeriodProfitBody">
                  <div className="dashboardPeriodProfitLabel">{entry.label}</div>
                  <div className="dashboardPeriodProfitValueRow">
                    <span className={`dashboardPeriodProfitAmount ${signClass}`.trim()}>
                      {mask(profit, "money")}
                    </span>
                    <span className={`dashboardPeriodProfitPct ${signClass}`.trim()}>
                      {`${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
