import { type ObjectId, type WithId } from "mongodb";

import { loadAccountConfigs } from "./accounts";
import { getMongoDb } from "./mongo";

type PortfolioAccountDoc = {
  account_id: string;
  total_principal?: number;
  cash_balance?: number;
  updated_at?: Date | string;
};

type PortfolioMasterDoc = {
  master_id: string;
  accounts?: PortfolioAccountDoc[];
};

type SnapshotAccountDoc = {
  account_id: string;
  total_assets?: number;
  total_principal?: number;
  cash_balance?: number;
  valuation_krw?: number;
};

type DailySnapshotDoc = {
  _id: ObjectId;
  snapshot_date: string;
  total_assets?: number;
  total_principal?: number;
  cash_balance?: number;
  valuation_krw?: number;
  accounts?: SnapshotAccountDoc[];
};

type WeeklyFundDoc = {
  week_date: string;
  withdrawal_personal?: number;
  withdrawal_mom?: number;
  nh_principal_interest?: number;
  deposit_withdrawal?: number;
  total_assets?: number;
  purchase_amount?: number;
  valuation_amount?: number;
  exchange_rate?: number;
  bucket_pct_momentum?: number;
  bucket_pct_innovation?: number;
  bucket_pct_market?: number;
  bucket_pct_dividend?: number;
  bucket_pct_alternative?: number;
  bucket_pct_cash?: number;
  profit_count?: number;
  loss_count?: number;
  total_expense?: number;
  total_stocks?: number;
  total_principal?: number;
  cumulative_profit?: number;
  weekly_profit?: number;
  weekly_return_pct?: number;
  cumulative_return_pct?: number;
  updated_at?: Date | string;
};

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
  metrics: DashboardMetricItem[];
  accounts: DashboardAccountSummaryItem[];
  buckets: DashboardBucketItem[];
  stats: Array<{ label: string; value: number; kind: "money" | "percent" | "count" }>;
  latest_snapshot_date: string | null;
  weekly_date: string | null;
  updated_at: string | null;
};

const INITIAL_TOTAL_PRINCIPAL_DATE = "2024-01-31";
const INITIAL_TOTAL_PRINCIPAL_VALUE = 56_000_000;

function normalizeNumber(value: unknown): number {
  return Number(value ?? 0);
}

function toIsoString(value: unknown): string | null {
  if (!value) {
    return null;
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  return String(value);
}

function calculateWeeklyDocs(docs: WeeklyFundDoc[]): WeeklyFundDoc[] {
  const docsByDate = new Map(
    docs.map((doc) => [
      doc.week_date,
      {
        ...doc,
        total_expense:
          normalizeNumber(doc.withdrawal_personal) +
          normalizeNumber(doc.withdrawal_mom) +
          normalizeNumber(doc.nh_principal_interest),
        total_stocks: normalizeNumber(doc.profit_count) + normalizeNumber(doc.loss_count),
      },
    ]),
  );

  let runningTotal = INITIAL_TOTAL_PRINCIPAL_VALUE;
  let runningExpense = 0;
  let previousCumulativeProfit = 0;

  for (const weekDate of [...docsByDate.keys()].sort()) {
    const doc = docsByDate.get(weekDate)! as WeeklyFundDoc & {
      total_expense: number;
      total_stocks: number;
      total_principal?: number;
      cumulative_profit?: number;
      weekly_profit?: number;
      weekly_return_pct?: number;
      cumulative_return_pct?: number;
    };

    if (weekDate <= INITIAL_TOTAL_PRINCIPAL_DATE) {
      doc.total_principal = INITIAL_TOTAL_PRINCIPAL_VALUE;
    } else {
      runningTotal += normalizeNumber(doc.deposit_withdrawal);
      doc.total_principal = runningTotal;
    }

    runningExpense += normalizeNumber(doc.total_expense);
    doc.cumulative_profit = normalizeNumber(doc.total_assets) - normalizeNumber(doc.total_principal) - runningExpense;
    doc.weekly_profit = normalizeNumber(doc.cumulative_profit) - previousCumulativeProfit;

    if (normalizeNumber(doc.total_principal) > 0) {
      doc.weekly_return_pct = (normalizeNumber(doc.weekly_profit) / normalizeNumber(doc.total_principal)) * 100;
      doc.cumulative_return_pct = (normalizeNumber(doc.cumulative_profit) / normalizeNumber(doc.total_principal)) * 100;
    } else {
      doc.weekly_return_pct = 0;
      doc.cumulative_return_pct = 0;
    }

    previousCumulativeProfit = normalizeNumber(doc.cumulative_profit);
  }

  return docs
    .map((doc) => docsByDate.get(doc.week_date)!)
    .sort((left, right) => String(right.week_date).localeCompare(String(left.week_date)));
}

export async function loadDashboardData(): Promise<DashboardData> {
  const db = await getMongoDb();
  const [configs, portfolioDoc, snapshotDocs, weeklyDocs] = await Promise.all([
    loadAccountConfigs(),
    db.collection<PortfolioMasterDoc>("portfolio_master").findOne({ master_id: "GLOBAL" }),
    db.collection<DailySnapshotDoc>("daily_snapshots").find().sort("snapshot_date", -1).limit(2).toArray(),
    db.collection<WithId<WeeklyFundDoc>>("weekly_fund_data").find().sort("week_date", -1).toArray(),
  ]);

  const latestSnapshot = snapshotDocs[0] ?? null;
  const previousSnapshot = snapshotDocs[1] ?? null;
  const latestWeekly = calculateWeeklyDocs(weeklyDocs)[0] ?? null;
  const portfolioAccounts = new Map((portfolioDoc?.accounts ?? []).map((account) => [account.account_id, account]));
  const snapshotAccounts = new Map((latestSnapshot?.accounts ?? []).map((account) => [account.account_id, account]));

  const accounts = configs.map((config) => {
    const portfolioAccount = portfolioAccounts.get(config.account_id);
    const snapshotAccount = snapshotAccounts.get(config.account_id);
    const totalPrincipal = normalizeNumber(portfolioAccount?.total_principal ?? snapshotAccount?.total_principal);
    const cashBalance = normalizeNumber(portfolioAccount?.cash_balance ?? snapshotAccount?.cash_balance);
    const valuationKrw = normalizeNumber(snapshotAccount?.valuation_krw);
    const totalAssets = valuationKrw + cashBalance;
    const netProfit = totalAssets - totalPrincipal;
    const netProfitPct = totalPrincipal > 0 ? (netProfit / totalPrincipal) * 100 : 0;
    const cashRatio = totalAssets > 0 ? (cashBalance / totalAssets) * 100 : 0;

    return {
      account_id: config.account_id,
      account_name: config.name,
      order: config.order,
      total_assets: totalAssets,
      total_principal: totalPrincipal,
      valuation_krw: valuationKrw,
      cash_balance: cashBalance,
      cash_ratio: cashRatio,
      net_profit: netProfit,
      net_profit_pct: netProfitPct,
    };
  });

  const totalAssets = accounts.reduce((sum, account) => sum + account.total_assets, 0);
  const totalPrincipal = accounts.reduce((sum, account) => sum + account.total_principal, 0);
  const totalCash = accounts.reduce((sum, account) => sum + account.cash_balance, 0);
  const valuationAmount = accounts.reduce((sum, account) => sum + account.valuation_krw, 0);
  const dailyProfit = latestSnapshot && previousSnapshot ? totalAssets - normalizeNumber(previousSnapshot.total_assets) : 0;
  const dailyReturnPct =
    previousSnapshot && normalizeNumber(previousSnapshot.total_assets) > 0
      ? (dailyProfit / normalizeNumber(previousSnapshot.total_assets)) * 100
      : 0;

  const metrics: DashboardMetricItem[] = [
    { label: "총 자산", value: totalAssets, kind: "money" },
    { label: "투자 원금", value: totalPrincipal, kind: "money" },
    { label: "현금 잔고", value: totalCash, kind: "money" },
    { label: "금일 손익", value: dailyProfit, kind: "money" },
    { label: "금주 손익", value: normalizeNumber(latestWeekly?.weekly_profit), kind: "money" },
    { label: "누적 손익", value: normalizeNumber(latestWeekly?.cumulative_profit), kind: "money" },
  ];

  const buckets: DashboardBucketItem[] = [
    { label: "1. 모멘텀", weight_pct: normalizeNumber(latestWeekly?.bucket_pct_momentum) },
    { label: "2. 혁신기술", weight_pct: normalizeNumber(latestWeekly?.bucket_pct_innovation) },
    { label: "3. 시장지수", weight_pct: normalizeNumber(latestWeekly?.bucket_pct_market) },
    { label: "4. 배당방어", weight_pct: normalizeNumber(latestWeekly?.bucket_pct_dividend) },
    { label: "5. 대체헷지", weight_pct: normalizeNumber(latestWeekly?.bucket_pct_alternative) },
    { label: "6. 현금", weight_pct: normalizeNumber(latestWeekly?.bucket_pct_cash) },
  ];

  const stats = [
    { label: "매입 금액", value: normalizeNumber(latestWeekly?.purchase_amount), kind: "money" as const },
    { label: "평가 금액", value: valuationAmount || normalizeNumber(latestWeekly?.valuation_amount), kind: "money" as const },
    { label: "현금 비중", value: totalAssets > 0 ? (totalCash / totalAssets) * 100 : 0, kind: "percent" as const },
    { label: "일간 수익률", value: dailyReturnPct, kind: "percent" as const },
    { label: "주 수익률", value: normalizeNumber(latestWeekly?.weekly_return_pct), kind: "percent" as const },
    { label: "누적 수익률", value: normalizeNumber(latestWeekly?.cumulative_return_pct), kind: "percent" as const },
    { label: "수익 종목 수", value: normalizeNumber(latestWeekly?.profit_count), kind: "count" as const },
    { label: "손실 종목 수", value: normalizeNumber(latestWeekly?.loss_count), kind: "count" as const },
  ];

  const updatedAtCandidates = [
    toIsoString(latestSnapshot?.snapshot_date ? `${latestSnapshot.snapshot_date}T00:00:00+09:00` : null),
    toIsoString(latestWeekly?.updated_at),
    ...Array.from(portfolioAccounts.values()).map((account) => toIsoString(account.updated_at)),
  ].filter((value): value is string => Boolean(value));

  return {
    metrics,
    accounts,
    buckets,
    stats,
    latest_snapshot_date: latestSnapshot?.snapshot_date ?? null,
    weekly_date: latestWeekly?.week_date ?? null,
    updated_at: updatedAtCandidates.sort().at(-1) ?? null,
  };
}

export type { DashboardData };
