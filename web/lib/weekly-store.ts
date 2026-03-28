import { loadAccountConfigs } from "./accounts";
import { loadExchangeRateSummary } from "./exchange-rates";
import { getMongoDb } from "./mongo";

const WEEKLY_COLLECTION = "weekly_fund_data";
const INITIAL_TOTAL_PRINCIPAL_DATE = "2024-01-31";
const INITIAL_TOTAL_PRINCIPAL_VALUE = 56_000_000;

const CORE_VIEW_HIDDEN_KEYS = ["withdrawal_personal", "withdrawal_mom", "nh_principal_interest", "deposit_withdrawal"] as const;

const READ_ONLY_FIELDS = new Set([
  "total_expense",
  "total_principal",
  "total_assets",
  "purchase_amount",
  "valuation_amount",
  "profit_loss",
  "cumulative_profit",
  "weekly_profit",
  "weekly_return_pct",
  "cumulative_return_pct",
  "exchange_rate",
  "total_stocks",
]);

const FIELD_DEFS = [
  { key: "withdrawal_personal", label: "개인 인출", type: "int" },
  { key: "withdrawal_mom", label: "엄마", type: "int" },
  { key: "nh_principal_interest", label: "농협원리금", type: "int" },
  { key: "total_expense", label: "지출 합계", type: "int" },
  { key: "deposit_withdrawal", label: "입출금", type: "int" },
  { key: "total_principal", label: "총 원금", type: "int" },
  { key: "total_assets", label: "총 자산", type: "int" },
  { key: "purchase_amount", label: "매입 금액", type: "int" },
  { key: "valuation_amount", label: "평가 금액", type: "int" },
  { key: "profit_loss", label: "평가 손익", type: "int" },
  { key: "cumulative_profit", label: "누적 손익", type: "int" },
  { key: "weekly_profit", label: "금주 손익", type: "int" },
  { key: "weekly_return_pct", label: "주수익률 (%)", type: "float" },
  { key: "cumulative_return_pct", label: "누적 수익률 (%)", type: "float" },
  { key: "memo", label: "비고", type: "text" },
  { key: "exchange_rate", label: "환율", type: "float" },
  { key: "bucket_pct_momentum", label: "1. 모멘텀 (%)", type: "float" },
  { key: "bucket_pct_innovation", label: "2. 혁신기술 (%)", type: "float" },
  { key: "bucket_pct_market", label: "3. 시장지수 (%)", type: "float" },
  { key: "bucket_pct_dividend", label: "4. 배당방어 (%)", type: "float" },
  { key: "bucket_pct_alternative", label: "5. 대체헷지 (%)", type: "float" },
  { key: "bucket_pct_cash", label: "6. 현금 (%)", type: "float" },
  { key: "total_stocks", label: "총 종목 수", type: "int" },
  { key: "profit_count", label: "수익 종목 수", type: "int" },
  { key: "loss_count", label: "손실 종목 수", type: "int" },
] as const;

type WeeklyFieldType = (typeof FIELD_DEFS)[number]["type"];

type WeeklyFundDoc = {
  week_date: string;
  withdrawal_personal?: number;
  withdrawal_mom?: number;
  nh_principal_interest?: number;
  deposit_withdrawal?: number;
  total_assets?: number;
  purchase_amount?: number;
  valuation_amount?: number;
  memo?: string;
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
  total_principal?: number;
  profit_loss?: number;
  total_stocks?: number;
  cumulative_profit?: number;
  weekly_profit?: number;
  weekly_return_pct?: number;
  cumulative_return_pct?: number;
  created_at?: Date | string;
  updated_at?: Date | string;
};

type PortfolioHolding = {
  quantity?: number;
  average_buy_price?: number;
  currency?: string;
  bucket?: number;
};

type PortfolioAccountDoc = {
  account_id: string;
  cash_balance?: number;
  holdings?: PortfolioHolding[];
};

type PortfolioMasterDoc = {
  master_id: string;
  accounts?: PortfolioAccountDoc[];
};

type DailySnapshotAccountDoc = {
  account_id: string;
  cash_balance?: number;
  valuation_krw?: number;
  purchase_amount?: number;
};

type DailySnapshotDoc = {
  snapshot_date: string;
  total_assets?: number;
  purchase_amount?: number;
  valuation_krw?: number;
  cash_balance?: number;
  accounts?: DailySnapshotAccountDoc[];
};

type WeeklyEditableField = {
  key: string;
  label: string;
  type: WeeklyFieldType;
};

type WeeklyRow = {
  week_date: string;
  week_date_display: string;
  withdrawal_personal: number;
  withdrawal_mom: number;
  nh_principal_interest: number;
  total_expense: number;
  deposit_withdrawal: number;
  total_principal: number;
  total_assets: number;
  purchase_amount: number;
  valuation_amount: number;
  profit_loss: number;
  cumulative_profit: number;
  weekly_profit: number;
  weekly_return_pct: number;
  cumulative_return_pct: number;
  memo: string;
  exchange_rate: number;
  exchange_rate_change_pct: number;
  bucket_pct_momentum: number;
  bucket_pct_innovation: number;
  bucket_pct_market: number;
  bucket_pct_dividend: number;
  bucket_pct_alternative: number;
  bucket_pct_cash: number;
  total_stocks: number;
  profit_count: number;
  loss_count: number;
  updated_at: string | null;
};

type WeeklyTableData = {
  active_week_date: string;
  rows: WeeklyRow[];
  editable_fields: WeeklyEditableField[];
  core_hidden_keys: string[];
};

function getNowKst(): Date {
  const now = new Date();
  return new Date(now.toLocaleString("en-US", { timeZone: "Asia/Seoul" }));
}

function normalizeNumber(value: unknown): number {
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function normalizeText(value: unknown): string {
  return String(value ?? "").trim();
}

function toIsoString(value: Date | string | undefined): string | null {
  if (!value) {
    return null;
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  return String(value);
}

function roundNumber(value: number, digits = 4): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function formatWeekDateDisplay(dateText: string): string {
  const date = new Date(`${dateText}T12:00:00+09:00`);
  const parts = new Intl.DateTimeFormat("ko-KR", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "numeric",
    day: "numeric",
    weekday: "short",
  }).formatToParts(date);

  const year = parts.find((part) => part.type === "year")?.value ?? "";
  const month = parts.find((part) => part.type === "month")?.value ?? "";
  const day = parts.find((part) => part.type === "day")?.value ?? "";
  const weekday = (parts.find((part) => part.type === "weekday")?.value ?? "").replace("요일", "");

  return `${year}. ${month}. ${day} (${weekday})`;
}

function getActiveWeekDate(now = getNowKst()): string {
  const weekDay = now.getDay();
  const normalizedWeekDay = weekDay === 0 ? 6 : weekDay - 1;
  const monday = new Date(now);
  monday.setDate(now.getDate() - normalizedWeekDay);
  monday.setHours(0, 0, 0, 0);
  const friday = new Date(monday);
  friday.setDate(monday.getDate() + 4);

  if (normalizedWeekDay === 0 && now.getHours() < 9) {
    friday.setDate(friday.getDate() - 7);
  }

  const year = friday.getFullYear();
  const month = String(friday.getMonth() + 1).padStart(2, "0");
  const day = String(friday.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function buildEmptyWeeklyDoc(weekDate: string): WeeklyFundDoc {
  const now = getNowKst();
  return {
    week_date: weekDate,
    withdrawal_personal: 0,
    withdrawal_mom: 0,
    nh_principal_interest: 0,
    deposit_withdrawal: 0,
    total_assets: 0,
    purchase_amount: 0,
    valuation_amount: 0,
    memo: "",
    exchange_rate: 0,
    bucket_pct_momentum: 0,
    bucket_pct_innovation: 0,
    bucket_pct_market: 0,
    bucket_pct_dividend: 0,
    bucket_pct_alternative: 0,
    bucket_pct_cash: 0,
    profit_count: 0,
    loss_count: 0,
    created_at: now,
    updated_at: now,
  };
}

function applyDerivedFields(doc: WeeklyFundDoc): WeeklyFundDoc {
  const totalExpense =
    normalizeNumber(doc.withdrawal_personal) +
    normalizeNumber(doc.withdrawal_mom) +
    normalizeNumber(doc.nh_principal_interest);
  const profitLoss = normalizeNumber(doc.valuation_amount) - normalizeNumber(doc.purchase_amount);
  const totalStocks = normalizeNumber(doc.profit_count) + normalizeNumber(doc.loss_count);

  return {
    ...doc,
    total_expense: totalExpense,
    profit_loss: profitLoss,
    total_stocks: totalStocks,
  };
}

function applyRunningCalculatedFields(docs: WeeklyFundDoc[]): WeeklyFundDoc[] {
  const docsByDate = new Map(docs.map((doc) => [doc.week_date, applyDerivedFields(doc)]));
  let runningTotalPrincipal = INITIAL_TOTAL_PRINCIPAL_VALUE;
  let runningTotalExpense = 0;
  let previousCumulativeProfit = 0;

  for (const weekDate of [...docsByDate.keys()].sort()) {
    const current = docsByDate.get(weekDate)!;
    if (weekDate <= INITIAL_TOTAL_PRINCIPAL_DATE) {
      current.total_principal = INITIAL_TOTAL_PRINCIPAL_VALUE;
    } else {
      runningTotalPrincipal += normalizeNumber(current.deposit_withdrawal);
      current.total_principal = runningTotalPrincipal;
    }

    runningTotalExpense += normalizeNumber(current.total_expense);
    current.cumulative_profit =
      normalizeNumber(current.total_assets) - normalizeNumber(current.total_principal) - runningTotalExpense;
    current.weekly_profit = normalizeNumber(current.cumulative_profit) - previousCumulativeProfit;

    if (normalizeNumber(current.total_principal) === 0) {
      current.weekly_return_pct = 0;
      current.cumulative_return_pct = 0;
    } else {
      current.weekly_return_pct = roundNumber(
        (normalizeNumber(current.weekly_profit) / normalizeNumber(current.total_principal)) * 100,
        4,
      );
      current.cumulative_return_pct = roundNumber(
        (normalizeNumber(current.cumulative_profit) / normalizeNumber(current.total_principal)) * 100,
        4,
      );
    }

    previousCumulativeProfit = normalizeNumber(current.cumulative_profit);
  }

  return docs
    .map((doc) => docsByDate.get(doc.week_date)!)
    .sort((left, right) => String(right.week_date).localeCompare(String(left.week_date)));
}

function buildWeeklyRows(docs: WeeklyFundDoc[]): WeeklyRow[] {
  return docs.map((doc, index) => {
    const olderRate = index + 1 < docs.length ? normalizeNumber(docs[index + 1]?.exchange_rate) : 0;
    const currentRate = normalizeNumber(doc.exchange_rate);
    const exchangeRateChangePct = olderRate > 0 ? roundNumber(((currentRate / olderRate) - 1) * 100, 4) : 0;

    return {
      week_date: doc.week_date,
      week_date_display: formatWeekDateDisplay(doc.week_date),
      withdrawal_personal: normalizeNumber(doc.withdrawal_personal),
      withdrawal_mom: normalizeNumber(doc.withdrawal_mom),
      nh_principal_interest: normalizeNumber(doc.nh_principal_interest),
      total_expense: normalizeNumber(doc.total_expense),
      deposit_withdrawal: normalizeNumber(doc.deposit_withdrawal),
      total_principal: normalizeNumber(doc.total_principal),
      total_assets: normalizeNumber(doc.total_assets),
      purchase_amount: normalizeNumber(doc.purchase_amount),
      valuation_amount: normalizeNumber(doc.valuation_amount),
      profit_loss: normalizeNumber(doc.profit_loss),
      cumulative_profit: normalizeNumber(doc.cumulative_profit),
      weekly_profit: normalizeNumber(doc.weekly_profit),
      weekly_return_pct: normalizeNumber(doc.weekly_return_pct),
      cumulative_return_pct: normalizeNumber(doc.cumulative_return_pct),
      memo: normalizeText(doc.memo),
      exchange_rate: currentRate,
      exchange_rate_change_pct: exchangeRateChangePct,
      bucket_pct_momentum: normalizeNumber(doc.bucket_pct_momentum),
      bucket_pct_innovation: normalizeNumber(doc.bucket_pct_innovation),
      bucket_pct_market: normalizeNumber(doc.bucket_pct_market),
      bucket_pct_dividend: normalizeNumber(doc.bucket_pct_dividend),
      bucket_pct_alternative: normalizeNumber(doc.bucket_pct_alternative),
      bucket_pct_cash: normalizeNumber(doc.bucket_pct_cash),
      total_stocks: normalizeNumber(doc.total_stocks),
      profit_count: normalizeNumber(doc.profit_count),
      loss_count: normalizeNumber(doc.loss_count),
      updated_at: toIsoString(doc.updated_at),
    };
  });
}

async function ensureActiveWeekRow() {
  const db = await getMongoDb();
  const activeWeekDate = getActiveWeekDate();
  const existing = await db.collection<WeeklyFundDoc>(WEEKLY_COLLECTION).findOne({ week_date: activeWeekDate });
  if (!existing) {
    await db.collection<WeeklyFundDoc>(WEEKLY_COLLECTION).insertOne(buildEmptyWeeklyDoc(activeWeekDate));
  }
  return activeWeekDate;
}

function normalizeCurrency(value: unknown): string {
  return String(value ?? "").trim().toUpperCase() || "KRW";
}

function estimateBucketRatiosFromHoldings(
  accounts: PortfolioAccountDoc[],
  snapshotByAccount: Map<string, DailySnapshotAccountDoc>,
  exchangeRates: Record<string, number>,
) {
  const purchaseByBucket = new Map<number, number>();

  for (const account of accounts) {
    const snapshot = snapshotByAccount.get(account.account_id);
    const holdings = Array.isArray(account.holdings) ? account.holdings : [];
    if (!snapshot || holdings.length === 0) {
      continue;
    }

    const purchaseTotal = holdings.reduce((sum, holding) => {
      const quantity = Math.floor(normalizeNumber(holding.quantity));
      const averageBuyPrice = normalizeNumber(holding.average_buy_price);
      const currency = normalizeCurrency(holding.currency);
      const exchangeRate = currency === "KRW" ? 1 : normalizeNumber(exchangeRates[currency]);
      return sum + quantity * averageBuyPrice * (exchangeRate || 1);
    }, 0);

    const valuationKrw = normalizeNumber(snapshot.valuation_krw);
    const multiplier = purchaseTotal > 0 && valuationKrw > 0 ? valuationKrw / purchaseTotal : 1;

    for (const holding of holdings) {
      const bucketId = Number(holding.bucket ?? 0);
      if (!Number.isFinite(bucketId) || bucketId <= 0) {
        continue;
      }
      const quantity = Math.floor(normalizeNumber(holding.quantity));
      const averageBuyPrice = normalizeNumber(holding.average_buy_price);
      const currency = normalizeCurrency(holding.currency);
      const exchangeRate = currency === "KRW" ? 1 : normalizeNumber(exchangeRates[currency]);
      const purchaseKrw = quantity * averageBuyPrice * (exchangeRate || 1);
      const estimatedValuation = purchaseKrw * multiplier;
      purchaseByBucket.set(bucketId, normalizeNumber(purchaseByBucket.get(bucketId)) + estimatedValuation);
    }
  }

  return {
    momentum: normalizeNumber(purchaseByBucket.get(1)),
    innovation: normalizeNumber(purchaseByBucket.get(2)),
    market: normalizeNumber(purchaseByBucket.get(3)),
    dividend: normalizeNumber(purchaseByBucket.get(4)),
    alternative: normalizeNumber(purchaseByBucket.get(5)),
  };
}

function redistributeBucketPercentages(
  activeDoc: WeeklyFundDoc | null,
  estimatedBucketAmounts: ReturnType<typeof estimateBucketRatiosFromHoldings>,
  cashPct: number,
  totalAssets: number,
) {
  const stockTargetPct = Math.max(0, 100 - cashPct);

  const existingStockTotal =
    normalizeNumber(activeDoc?.bucket_pct_momentum) +
    normalizeNumber(activeDoc?.bucket_pct_innovation) +
    normalizeNumber(activeDoc?.bucket_pct_market) +
    normalizeNumber(activeDoc?.bucket_pct_dividend) +
    normalizeNumber(activeDoc?.bucket_pct_alternative);

  if (existingStockTotal > 0) {
    return {
      bucket_pct_momentum: roundNumber((normalizeNumber(activeDoc?.bucket_pct_momentum) / existingStockTotal) * stockTargetPct, 4),
      bucket_pct_innovation: roundNumber((normalizeNumber(activeDoc?.bucket_pct_innovation) / existingStockTotal) * stockTargetPct, 4),
      bucket_pct_market: roundNumber((normalizeNumber(activeDoc?.bucket_pct_market) / existingStockTotal) * stockTargetPct, 4),
      bucket_pct_dividend: roundNumber((normalizeNumber(activeDoc?.bucket_pct_dividend) / existingStockTotal) * stockTargetPct, 4),
      bucket_pct_alternative: roundNumber((normalizeNumber(activeDoc?.bucket_pct_alternative) / existingStockTotal) * stockTargetPct, 4),
      bucket_pct_cash: roundNumber(cashPct, 4),
    };
  }

  const estimatedTotal =
    estimatedBucketAmounts.momentum +
    estimatedBucketAmounts.innovation +
    estimatedBucketAmounts.market +
    estimatedBucketAmounts.dividend +
    estimatedBucketAmounts.alternative;

  if (estimatedTotal > 0 && totalAssets > 0) {
    return {
      bucket_pct_momentum: roundNumber((estimatedBucketAmounts.momentum / totalAssets) * 100, 4),
      bucket_pct_innovation: roundNumber((estimatedBucketAmounts.innovation / totalAssets) * 100, 4),
      bucket_pct_market: roundNumber((estimatedBucketAmounts.market / totalAssets) * 100, 4),
      bucket_pct_dividend: roundNumber((estimatedBucketAmounts.dividend / totalAssets) * 100, 4),
      bucket_pct_alternative: roundNumber((estimatedBucketAmounts.alternative / totalAssets) * 100, 4),
      bucket_pct_cash: roundNumber(cashPct, 4),
    };
  }

  return {
    bucket_pct_momentum: 0,
    bucket_pct_innovation: 0,
    bucket_pct_market: 0,
    bucket_pct_dividend: 0,
    bucket_pct_alternative: 0,
    bucket_pct_cash: roundNumber(cashPct, 4),
  };
}

export async function loadWeeklyTableData(): Promise<WeeklyTableData> {
  const activeWeekDate = await ensureActiveWeekRow();
  const db = await getMongoDb();
  const docs = await db.collection<WeeklyFundDoc>(WEEKLY_COLLECTION).find().sort("week_date", -1).toArray();
  const weeklyDocs = applyRunningCalculatedFields(docs);

  return {
    active_week_date: activeWeekDate,
    rows: buildWeeklyRows(weeklyDocs),
    editable_fields: FIELD_DEFS.filter((field) => !READ_ONLY_FIELDS.has(field.key)).map((field) => ({ ...field })),
    core_hidden_keys: [...CORE_VIEW_HIDDEN_KEYS],
  };
}

export async function aggregateActiveWeekData(): Promise<{ week_date: string }> {
  const activeWeekDate = await ensureActiveWeekRow();
  const db = await getMongoDb();
  const [snapshot, portfolioDoc, exchangeRateSummary, configs] = await Promise.all([
    db.collection<DailySnapshotDoc>("daily_snapshots").find().sort("snapshot_date", -1).limit(1).next(),
    db.collection<PortfolioMasterDoc>("portfolio_master").findOne({ master_id: "GLOBAL" }),
    loadExchangeRateSummary(),
    loadAccountConfigs(),
  ]);

  if (!snapshot) {
    throw new Error("최신 스냅샷이 없어 주별 데이터를 집계할 수 없습니다.");
  }

  const activeDoc =
    (await db.collection<WeeklyFundDoc>(WEEKLY_COLLECTION).findOne({ week_date: activeWeekDate })) ?? buildEmptyWeeklyDoc(activeWeekDate);

  const showAccountIds = new Set(configs.map((config) => config.account_id));
  const snapshotAccounts = new Map(
    (snapshot.accounts ?? [])
      .filter((account) => showAccountIds.has(account.account_id))
      .map((account) => [account.account_id, account]),
  );

  const portfolioAccounts = (portfolioDoc?.accounts ?? []).filter((account) => showAccountIds.has(account.account_id));
  const totalAssets = normalizeNumber(snapshot.total_assets);
  const totalPurchase = normalizeNumber(snapshot.purchase_amount);
  const totalValuation = normalizeNumber(snapshot.valuation_krw);
  const totalCash = normalizeNumber(snapshot.cash_balance);
  const cashPct = totalAssets > 0 ? (totalCash / totalAssets) * 100 : 0;

  const estimatedBucketAmounts = estimateBucketRatiosFromHoldings(portfolioAccounts, snapshotAccounts, {
    USD: normalizeNumber(exchangeRateSummary.USD.rate),
    AUD: normalizeNumber(exchangeRateSummary.AUD.rate),
  });

  const bucketPercents = redistributeBucketPercentages(activeDoc, estimatedBucketAmounts, cashPct, totalAssets);
  const nextProfitCount = normalizeNumber(activeDoc.profit_count) > 0 || normalizeNumber(activeDoc.loss_count) > 0
    ? normalizeNumber(activeDoc.profit_count)
    : 0;
  const nextLossCount = normalizeNumber(activeDoc.profit_count) > 0 || normalizeNumber(activeDoc.loss_count) > 0
    ? normalizeNumber(activeDoc.loss_count)
    : 0;

  await db.collection<WeeklyFundDoc>(WEEKLY_COLLECTION).updateOne(
    { week_date: activeWeekDate },
    {
      $set: {
        total_assets: Math.round(totalAssets),
        purchase_amount: Math.round(totalPurchase),
        valuation_amount: Math.round(totalValuation),
        exchange_rate: roundNumber(normalizeNumber(exchangeRateSummary.USD.rate), 2),
        ...bucketPercents,
        profit_count: nextProfitCount,
        loss_count: nextLossCount,
        updated_at: getNowKst(),
      },
    },
  );

  return { week_date: activeWeekDate };
}

export async function updateWeeklyRow(
  weekDate: string,
  payload: Record<string, unknown>,
): Promise<{ week_date: string }> {
  const targetWeekDate = normalizeText(weekDate);
  if (!targetWeekDate) {
    throw new Error("수정할 주차를 찾을 수 없습니다.");
  }

  const updateDoc: Record<string, unknown> = {};
  for (const field of FIELD_DEFS) {
    if (READ_ONLY_FIELDS.has(field.key) || !(field.key in payload)) {
      continue;
    }

    const rawValue = payload[field.key];
    if (field.type === "text") {
      updateDoc[field.key] = normalizeText(rawValue);
      continue;
    }

    const numericValue = Number(rawValue ?? 0);
    if (!Number.isFinite(numericValue)) {
      throw new Error(`${field.label} 값이 올바르지 않습니다.`);
    }
    updateDoc[field.key] = field.type === "int" ? Math.trunc(numericValue) : roundNumber(numericValue, 4);
  }

  updateDoc.updated_at = getNowKst();

  const db = await getMongoDb();
  const result = await db.collection<WeeklyFundDoc>(WEEKLY_COLLECTION).updateOne(
    { week_date: targetWeekDate },
    { $set: updateDoc },
  );

  if (result.matchedCount === 0) {
    throw new Error("수정할 주별 데이터를 찾지 못했습니다.");
  }

  return { week_date: targetWeekDate };
}

export type { WeeklyEditableField, WeeklyRow, WeeklyTableData };
