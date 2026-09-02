/**
 * /assets 화면 공용 타입·순수 헬퍼 — AssetsManager 에서 분리(이동만, 로직 불변).
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import type { ColDef, ColumnState, GridApi, GridOptions, RowClassParams } from "ag-grid-community";
import { IconLoader2 } from "@tabler/icons-react";
import { AppAgGrid } from "../components/AppAgGrid";
import { GridToolbarButton } from "../components/GridToolbarButton";
import { StableInlineInput } from "../components/StableInlineInput";
import { AppLoadingState } from "../components/AppLoadingState";
import { AppModal } from "../components/AppModal";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";
import { reorderHoldings } from "@/lib/holdings-store";
import { fetchAlertBadges, normalizeBadgeTicker, type AlertBadges } from "@/lib/alert-badges";

export type HoldingsRow = {
  account_id: string;
  account_name: string;
  currency: string;
  bucket: string;
  bucket_id: number;
  ticker: string;
  name: string;
  quantity: number;
  average_buy_price: string | number;
  current_price: string;
  current_price_num?: number;
  fx_rate_krw?: number;
  pnl_krw: number;
  return_pct: number;
  weight_pct: number;
  daily_change_pct?: number | null;
  buy_amount_krw: number;
  valuation_krw: number;
  target_ratio?: number | null;
  target_quantity?: number | null;
  memo?: string | null;
  sort_order?: number | null;
  original_quantity?: number;
  original_average_buy_price?: number;
  original_buy_amount_krw?: number;
  original_valuation_krw?: number;
};

export type GridRow = HoldingsRow & { id: string };

/** 현금 저장 응답(`PUT /api/assets`) — 화면이 전체 리로드 없이 즉시 반영하는 값. */
export type SavedCashAccount = {
  account_id: string;
  cash: Record<string, number>;
  cash_balance_krw: number;
  cash_balance_native: number | null;
  cash_target_ratio: number;
  total_principal: number;
  updated_at: string | null;
  updated_by: string;
};

export type AccountSummary = {
  account_id: string;
  order: number;
  name: string;
  account_url?: string | null;
  icon: string;
  currency: string;
  total_principal: number;
  cash_balance_krw: number;
  cash_balance_native: number | null;
  cash_currency: string;
  cash?: Record<string, number>;
  cash_currencies?: string[];
  cash_display_native?: number;
  cash_display_currency?: string;
  cash_target_ratio: number;
  // 자산 헬퍼에서 저장한 현금 목표 비중(%) — 미저장이면 null ('-' 표시, 파생·기본값 없음)
  helper_cash_weight_pct?: number | null;
  intl_shares_value: number | null;
  intl_shares_change: number | null;
  updated_at: string | null;
  /** 마지막 변경 주체 — "user"(수기) 또는 커넥터 id(예: "NAMU_PLUG"). */
  updated_by?: string | null;
  valuation_krw: number;
  total_assets_krw: number;
  holdings_count: number;
  target_ratio_total: number;
  cash_ratio: number;
  net_profit: number;
  net_profit_pct: number;
  daily_profit: number;
  daily_return_pct: number;
  weekly_profit: number;
  weekly_return_pct: number;
};

export type ParentGridRow =
  | (AccountSummary & {
    id: string;
    rowType: "main";
    /** 표시용 현금(원화 환산). 이 그리드에서는 편집하지 않는다. */
    cash_krw: number;
  })
  | {
    id: string;
    rowType: "total";
    name: string;
    total_assets_krw: number;
    valuation_krw: number;
    total_principal: number;
    /** 표시용 현금(원화 환산). 이 그리드에서는 편집하지 않는다. */
    cash_krw: number;
    target_ratio_total: number | null;
    holdings_count: number;
    cash_ratio: number;
    net_profit: number;
    net_profit_pct: number;
    daily_profit: number;
    daily_return_pct: number;
    weekly_profit: number;
    weekly_return_pct: number;
  }
  | {
    id: string;
    rowType: "detail";
    parentId: string;
    summary: AccountSummary;
    rows: HoldingsRow[];
  };

export type HoldingsResponse = {
  rows?: HoldingsRow[];
  account_summaries?: AccountSummary[];
  error?: string;
};

export type AssetsHeaderSummary = {
  totalAssets: number;
  totalValuation: number;
  totalCash: number;
  accountCount: number;
};

export type AddingRowState = {
  ticker: string;
  quantity: string;
  average_buy_price: string;
  isValidatingTicker?: boolean;
  name?: string;
  bucketId?: number;
  isValidated?: boolean;
};

export const CASH_ROW_TICKER = "__CASH__";

export type HoldingEditableSnapshot = {
  quantity: number;
  average_buy_price: number;
  memo: string;
};

export const assetsGridTheme = createAppGridTheme();

export function buildGridRowId(row: Pick<HoldingsRow, "ticker" | "account_id">): string {
  return `${row.account_id}-${row.ticker}`;
}

export function buildDirtyCellKey(rowId: string, field: string): string {
  return `${rowId}::${field}`;
}

export function formatKrw(value: number): string {
  return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
}

export function formatPrice(value: number | null | undefined, currency: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const normalizedCurrency = String(currency || "KRW").trim().toUpperCase();
  if (normalizedCurrency === "AUD") {
    return `A$${new Intl.NumberFormat("en-AU", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 4,
    }).format(value)}`;
  }
  if (normalizedCurrency === "USD") {
    return `$${new Intl.NumberFormat("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 4,
    }).format(value)}`;
  }
  return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
}

export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 2 }).format(value);
}

export function getSignedClass(value: number): string {
  if (value === 0 || Number.isNaN(value)) return "";
  return value > 0 ? "metricPositive" : "metricNegative";
}

export function getSignedNullableClass(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) return "";
  return value > 0 ? "metricPositive" : "metricNegative";
}


export function getBucketCellClass(bucketId: number): string {
  if (!bucketId) return "appBucketCell";
  return `appBucketCell appBucketCell${bucketId}`;
}

export function parseRawPrice(formatted: unknown): string {
  if (formatted === null || formatted === undefined) return "0";
  return String(formatted).replace(/A\$|\$|₩|원|,|\s/g, "");
}

/**
 * 같은 거절 메시지가 잇달아 뜨는 것을 막는다.
 *
 * AG Grid 는 편집 확정 한 번에 `valueParser` 를 여러 번 부른다. 그대로 두면 토스트가
 * 같은 문구로 두세 개씩 쌓인다.
 */
const REJECT_DEDUPE_MS = 800;
let lastRejection: { message: string; at: number } | null = null;

/**
 * 숫자 입력 칸의 공용 파서.
 *
 * **빈 값은 0 으로 본다.** 값을 다 지우고 확정하는 건 "0 으로 만들겠다"는 뜻이고,
 * 세 칸(수량·매입 단가·총 원금) 모두 0 이 유효한 값이다.
 *
 * 숫자가 아니거나 음수면 원래 값으로 되돌리고 **그 이유를 알린다**. 예전에는 조용히
 * 되돌렸는데, 그러면 그리드가 `newValue === oldValue` 로 보고 저장 요청조차 하지 않아
 * 화면에는 "수정이 안 된다"로만 보였다.
 */
export function parseNumericCell(
  newValue: unknown,
  oldValue: unknown,
  options: { label: string; integer?: boolean; onReject: (message: string) => void },
): unknown {
  const reject = (message: string) => {
    const now = Date.now();
    if (lastRejection && lastRejection.message === message && now - lastRejection.at < REJECT_DEDUPE_MS) {
      return oldValue;
    }
    lastRejection = { message, at: now };
    options.onReject(message);
    return oldValue;
  };

  const raw = parseRawPrice(newValue).trim();
  if (raw === "") {
    return 0;
  }
  const parsed = options.integer ? parseInt(raw, 10) : parseFloat(raw);
  if (Number.isNaN(parsed)) {
    return reject(`${options.label}에 숫자를 입력하세요 (받은 값: ${String(newValue)}).`);
  }
  if (parsed < 0) {
    return reject(`${options.label}은(는) 0 보다 작을 수 없습니다 (받은 값: ${parsed}).`);
  }
  return parsed;
}

export function safeParseFloat(value: unknown): number {
  const parsed = parseFloat(parseRawPrice(value));
  return Number.isNaN(parsed) ? 0 : parsed;
}

export function parseEditableQuantity(value: unknown): number {
  const parsed = parseInt(parseRawPrice(value), 10);
  return Number.isNaN(parsed) ? 0 : parsed;
}

export function buildHoldingEditableSnapshot(
  row: Pick<HoldingsRow, "quantity" | "average_buy_price" | "memo">,
): HoldingEditableSnapshot {
  return {
    quantity: parseEditableQuantity(row.quantity),
    average_buy_price: safeParseFloat(row.average_buy_price),
    memo: String(row.memo ?? "").trim(),
  };
}

export function formatHiddenAmount(showAmounts: boolean, value: string): string {
  return showAmounts ? value : "••••";
}

// (제거됨) 기간 수익률은 백엔드 dashboard_service 의 입출금 제거 daily_return_pct/weekly_return_pct 사용.
// 상세는 docs/developer_guide.md (자산 수익률 계산 정책) 참고.

export function buildAutoSaveToastMessage(row: Pick<HoldingsRow, "name" | "currency">, before: HoldingEditableSnapshot, after: HoldingEditableSnapshot): string | null {
  const changes: string[] = [];
  if (before.quantity !== after.quantity) {
    changes.push(`수량 ${new Intl.NumberFormat("ko-KR").format(before.quantity)}→${new Intl.NumberFormat("ko-KR").format(after.quantity)}`);
  }
  if (before.average_buy_price !== after.average_buy_price) {
    changes.push(`매입단가 ${formatPrice(before.average_buy_price, row.currency)}→${formatPrice(after.average_buy_price, row.currency)}`);
  }
  if (before.memo !== after.memo) {
    changes.push(`메모 ${before.memo || "(없음)"}→${after.memo || "(없음)"}`);
  }
  if (changes.length === 0) {
    return null;
  }
  return `${row.name} 저장: ${changes.join(", ")}`;
}

export function getCurrentPriceNumber(row: GridRow): number {
  const currentPriceNum = Number(row.current_price_num ?? NaN);
  if (!Number.isNaN(currentPriceNum) && currentPriceNum > 0) {
    return currentPriceNum;
  }
  return safeParseFloat(row.current_price);
}

export function getPreviewQuantity(row: GridRow): number {
  return parseEditableQuantity(row.quantity);
}

export function getPreviewAverageBuyPrice(row: GridRow): number {
  return safeParseFloat(row.average_buy_price);
}

export function getOriginalQuantity(row: GridRow): number {
  return Number(row.original_quantity ?? row.quantity ?? 0);
}

export function getOriginalAverageBuyPrice(row: GridRow): number {
  return Number(row.original_average_buy_price ?? safeParseFloat(row.average_buy_price));
}

export function getOriginalBuyAmountKrw(row: GridRow): number {
  return Number(row.original_buy_amount_krw ?? row.buy_amount_krw ?? 0);
}

export function getOriginalValuationKrw(row: GridRow): number {
  return Number(row.original_valuation_krw ?? row.valuation_krw ?? 0);
}

export function getPreviewValuationKrw(row: GridRow): number {
  if (String(row.ticker || "").trim().toUpperCase() === CASH_ROW_TICKER) {
    return Number(row.valuation_krw ?? 0);
  }
  const quantity = getPreviewQuantity(row);
  if (quantity <= 0) {
    return 0;
  }
  if (row.currency === "KRW") {
    return getCurrentPriceNumber(row) * quantity;
  }
  const currentPrice = getCurrentPriceNumber(row);
  // 외화는 종목 통화의 원화 환율 배수를 직접 사용한다(수량 0 신규 종목도 즉시 계산됨).
  const fxRate = Number(row.fx_rate_krw ?? NaN);
  if (currentPrice > 0 && !Number.isNaN(fxRate) && fxRate > 0) {
    return currentPrice * quantity * fxRate;
  }
  // 폴백: 환율이 없으면 원래 값에서 역산(구버전 응답 호환).
  const originalQuantity = getOriginalQuantity(row);
  const originalValuationKrw = getOriginalValuationKrw(row);
  if (currentPrice > 0 && originalQuantity > 0 && originalValuationKrw > 0) {
    const fixedFxFactor = originalValuationKrw / (originalQuantity * currentPrice);
    return currentPrice * quantity * fixedFxFactor;
  }
  return 0;
}

export function getPreviewBuyAmountKrw(row: GridRow): number {
  if (String(row.ticker || "").trim().toUpperCase() === CASH_ROW_TICKER) {
    return Number(row.buy_amount_krw ?? 0);
  }
  const quantity = getPreviewQuantity(row);
  if (quantity <= 0) {
    return 0;
  }
  const averageBuyPrice = getPreviewAverageBuyPrice(row);
  if (row.currency === "KRW") {
    return averageBuyPrice * quantity;
  }
  // 외화는 종목 통화의 원화 환율 배수를 직접 사용한다(수량 0 신규 종목도 즉시 계산됨).
  const fxRate = Number(row.fx_rate_krw ?? NaN);
  if (!Number.isNaN(fxRate) && fxRate > 0) {
    return averageBuyPrice * quantity * fxRate;
  }
  // 폴백: 환율이 없으면 원래 값에서 역산(구버전 응답 호환).
  const originalQuantity = getOriginalQuantity(row);
  const originalAverageBuyPrice = getOriginalAverageBuyPrice(row);
  const originalBuyAmountKrw = getOriginalBuyAmountKrw(row);
  if (originalQuantity > 0 && originalAverageBuyPrice > 0 && originalBuyAmountKrw > 0) {
    const fixedFxFactor = originalBuyAmountKrw / (originalQuantity * originalAverageBuyPrice);
    return averageBuyPrice * quantity * fixedFxFactor;
  }
  return 0;
}

export function getPreviewWeightPct(row: GridRow, rows: HoldingsRow[], summary: AccountSummary): number {
  const normalizedTicker = String(row.ticker || "").trim().toUpperCase();
  // IS(International Shares 고정자산)도 총자산에 포함해 비중을 계산한다 — 백엔드 weight_pct 와 동일 기준.
  const previewTotalValuation = rows.reduce((sum, currentRow) => {
    return sum + getPreviewValuationKrw({ ...currentRow, id: buildGridRowId(currentRow) });
  }, 0);
  const denominator = Number(summary.cash_balance_krw ?? 0) + previewTotalValuation;
  if (denominator <= 0) {
    return 0;
  }
  if (normalizedTicker === CASH_ROW_TICKER) {
    return (Number(summary.cash_balance_krw ?? 0) / denominator) * 100;
  }
  const rowId = buildGridRowId(row);
  const targetRow = rows.find((currentRow) => buildGridRowId(currentRow) === rowId);
  if (!targetRow) {
    return 0;
  }
  const rowValuation = getPreviewValuationKrw({ ...targetRow, id: rowId });
  return (rowValuation / denominator) * 100;
}

export function buildSyncedHoldingRows(rows: HoldingsRow[], summary: AccountSummary): HoldingsRow[] {
  return rows.map((row) => {
    const previewRow = { ...row, id: buildGridRowId(row) };
    const quantity = getPreviewQuantity(previewRow);
    const averageBuyPrice = getPreviewAverageBuyPrice(previewRow);
    const valuationKrw = Math.round(getPreviewValuationKrw(previewRow));
    const buyAmountKrw = Math.round(getPreviewBuyAmountKrw(previewRow));
    const pnlKrw = valuationKrw - buyAmountKrw;
    const returnPct = buyAmountKrw > 0 ? Number(((pnlKrw / buyAmountKrw) * 100).toFixed(2)) : 0;
    const weightPct = Number(getPreviewWeightPct(previewRow, rows, summary).toFixed(2));

    return {
      ...row,
      quantity,
      average_buy_price: averageBuyPrice,
      valuation_krw: valuationKrw,
      buy_amount_krw: buyAmountKrw,
      pnl_krw: pnlKrw,
      pnl_krw_num: pnlKrw,
      return_pct: returnPct,
      weight_pct: weightPct,
    };
  });
}

export function isDetailRow(row: ParentGridRow | undefined): row is Extract<ParentGridRow, { rowType: "detail" }> {
  return row?.rowType === "detail";
}

export function isTotalRow(row: ParentGridRow | undefined): row is Extract<ParentGridRow, { rowType: "total" }> {
  return row?.rowType === "total";
}

export function formatAccountCash(summary: AccountSummary): string {
  // 현금 컬럼은 계좌 주 통화 환산 값으로 표시(다통화 현금은 원화로 환산해 합산됨).
  const displayCurrency = String(summary.cash_display_currency || summary.currency || "KRW").trim().toUpperCase();
  if (displayCurrency === "KRW") {
    return formatKrw(summary.cash_display_native ?? summary.cash_balance_krw);
  }
  return formatPrice(summary.cash_display_native ?? summary.cash_balance_native, displayCurrency);
}

export function buildCashGridRow(summary: AccountSummary): GridRow {
  const cashValue = Number(summary.cash_balance_krw ?? 0);
  return {
    id: `${summary.account_id}-${CASH_ROW_TICKER}`,
    account_id: summary.account_id,
    account_name: summary.name,
    currency: "KRW",
    bucket: "",
    bucket_id: 0,
    ticker: CASH_ROW_TICKER,
    name: "현금",
    quantity: 0,
    average_buy_price: 0,
    current_price: "-",
    current_price_num: 0,
    pnl_krw: 0,
    return_pct: 0,
    weight_pct: 0,
    daily_change_pct: null,
    buy_amount_krw: cashValue,
    valuation_krw: cashValue,
    target_ratio: Number(summary.cash_target_ratio ?? 0),
    sort_order: -1,
    original_quantity: 0,
    original_average_buy_price: 0,
  };
}


export function reorderRowsByTickers(rows: HoldingsRow[], orderedTickers: string[]): HoldingsRow[] {
  const normalizedTickers = orderedTickers.map((ticker) => String(ticker || "").trim().toUpperCase());
  const rowMap = new Map(rows.map((row) => [String(row.ticker || "").trim().toUpperCase(), row] as const));
  const orderedRows: HoldingsRow[] = [];
  const seen = new Set<string>();

  for (const ticker of normalizedTickers) {
    const row = rowMap.get(ticker);
    if (!row || seen.has(ticker)) {
      continue;
    }
    orderedRows.push({ ...row });
    seen.add(ticker);
  }

  const remainingRows = rows.filter((row) => !seen.has(String(row.ticker || "").trim().toUpperCase()));

  return [...orderedRows, ...remainingRows.map((row) => ({ ...row }))].map((row, index) => ({
    ...row,
    sort_order: index,
  }));
}

export const ASSETS_WEIGHT_TEXT_COLOR = "#7952b3";

export function stopActionButtonMouseDown(event: ReactMouseEvent<HTMLButtonElement>) {
  event.preventDefault();
  event.stopPropagation();
}

export function stopActionButtonClick(event: ReactMouseEvent<HTMLButtonElement>) {
  event.preventDefault();
  event.stopPropagation();
}


/** 변경 주체 표시명 — 커넥터 id 를 사람이 읽는 이름으로. */
export function formatUpdatedBy(updatedBy: string | null | undefined): string {
  if (!updatedBy || updatedBy === "user") return "사용자";
  if (updatedBy === "NAMU_PLUG") return "나무증권";
  return updatedBy;
}
