"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { iconSetQuartzBold, themeQuartz } from "ag-grid-community";
import type { ColDef, GridOptions, RowClassParams } from "ag-grid-community";
import { IconCheck, IconLoader2, IconPlus, IconTrash } from "@tabler/icons-react";

import { AppAgGrid } from "../components/AppAgGrid";
import { AppLoadingState } from "../components/AppLoadingState";
import { useToast } from "../components/ToastProvider";

type HoldingsRow = {
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
  days_held: string;
  pnl_krw: number;
  return_pct: number;
  weight_pct: number;
  daily_change_pct?: number | null;
  buy_amount_krw: number;
  valuation_krw: number;
  target_ratio?: number | null;
  target_quantity?: number | null;
  target_amount?: number | null;
  sort_order?: number | null;
  original_quantity?: number;
  original_average_buy_price?: number;
};

type GridRow = HoldingsRow & { id: string };

type AccountSummary = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
  currency: string;
  total_principal: number;
  cash_balance_krw: number;
  cash_balance_native: number | null;
  cash_currency: string;
  intl_shares_value: number | null;
  intl_shares_change: number | null;
  updated_at: string | null;
  valuation_krw: number;
  total_assets_krw: number;
  holdings_count: number;
  target_ratio_total: number;
};

type ParentGridRow =
  | (AccountSummary & {
      id: string;
      rowType: "main";
      cash_edit_value: number;
    })
  | {
      id: string;
      rowType: "detail";
      parentId: string;
      summary: AccountSummary;
      rows: HoldingsRow[];
    };

type HoldingsResponse = {
  rows?: HoldingsRow[];
  account_summaries?: AccountSummary[];
  error?: string;
};

type AssetsHeaderSummary = {
  totalAssets: number;
  totalValuation: number;
  totalCash: number;
  accountCount: number;
};

type AddingRowState = {
  ticker: string;
  quantity: string;
  average_buy_price: string;
  target_ratio: string;
  isValidatingTicker?: boolean;
  name?: string;
  bucketId?: number;
  isValidated?: boolean;
};

const assetsGridTheme = themeQuartz.withPart(iconSetQuartzBold).withParams({
  accentColor: "#206bc4",
  backgroundColor: "#ffffff",
  foregroundColor: "#182433",
  headerBackgroundColor: "#f8fafc",
  headerTextColor: "#5b6778",
  spacing: 8,
  fontSize: 14,
  wrapperBorderRadius: 10,
  rowHeight: 38,
  headerHeight: 38,
  cellHorizontalPadding: 12,
  headerColumnBorder: true,
  headerColumnBorderHeight: "70%",
  columnBorder: true,
  oddRowBackgroundColor: "#fbfdff",
  headerCellHoverBackgroundColor: "#eef4fb",
  headerCellMovingBackgroundColor: "#e8f0fb",
  iconButtonHoverBackgroundColor: "#eef4fb",
  iconButtonHoverColor: "#206bc4",
  iconSize: 18,
});

function buildGridRowId(row: Pick<HoldingsRow, "ticker" | "account_id">): string {
  return `${row.account_id}-${row.ticker}`;
}

function buildDirtyCellKey(rowId: string, field: string): string {
  return `${rowId}::${field}`;
}

function formatKrw(value: number): string {
  return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
}

function formatPrice(value: number | null | undefined, currency: string): string {
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

function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 2 }).format(value);
}

function formatTargetAmount(value: number | null | undefined, currency: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (String(currency || "KRW").toUpperCase() === "AUD") {
    return `A$${new Intl.NumberFormat("en-AU", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)}`;
  }
  return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
}

function formatTargetQuantity(value: number | null | undefined, currency: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (String(currency || "KRW").toUpperCase() === "AUD") {
    return new Intl.NumberFormat("en-AU", {
      minimumFractionDigits: 0,
      maximumFractionDigits: 4,
    }).format(value);
  }
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 0 }).format(value);
}

function formatExpectedChangeQuantity(
  targetQuantity: number | null | undefined,
  currentQuantity: number | null | undefined,
  currency: string,
): string {
  if (
    targetQuantity === null ||
    targetQuantity === undefined ||
    Number.isNaN(targetQuantity) ||
    currentQuantity === null ||
    currentQuantity === undefined ||
    Number.isNaN(currentQuantity)
  ) {
    return "-";
  }

  const changeQuantity = targetQuantity - currentQuantity;
  const sign = changeQuantity > 0 ? "+" : "";
  if (String(currency || "KRW").toUpperCase() === "AUD") {
    return `${sign}${new Intl.NumberFormat("en-AU", {
      minimumFractionDigits: 0,
      maximumFractionDigits: 4,
    }).format(changeQuantity)}`;
  }
  return `${sign}${new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 0 }).format(changeQuantity)}`;
}

function getSignedClass(value: number): string {
  if (value === 0 || Number.isNaN(value)) return "";
  return value > 0 ? "metricPositive" : "metricNegative";
}

function getSignedNullableClass(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) return "";
  return value > 0 ? "metricPositive" : "metricNegative";
}

function getBucketCellClass(bucketId: number): string {
  if (!bucketId) return "appBucketCell";
  return `appBucketCell appBucketCell${bucketId}`;
}

function parseRawPrice(formatted: unknown): string {
  if (formatted === null || formatted === undefined) return "0";
  return String(formatted).replace(/A\$|\$|₩|원|,|\s/g, "");
}

function safeParseFloat(value: unknown): number {
  const parsed = parseFloat(parseRawPrice(value));
  return Number.isNaN(parsed) ? 0 : parsed;
}

function parseEditableQuantity(value: unknown): number {
  const parsed = parseInt(parseRawPrice(value), 10);
  return Number.isNaN(parsed) ? 0 : parsed;
}

function roundTargetQuantity(quantity: number, currency: string): number {
  if (String(currency || "KRW").toUpperCase() === "AUD") {
    return Math.round(quantity * 10000) / 10000;
  }
  return Math.max(Math.floor(quantity), 0);
}

function getCurrentPriceNumber(row: GridRow): number {
  const currentPriceNum = Number(row.current_price_num ?? NaN);
  if (!Number.isNaN(currentPriceNum) && currentPriceNum > 0) {
    return currentPriceNum;
  }
  return safeParseFloat(row.current_price);
}

function getPreviewQuantity(row: GridRow): number {
  return parseEditableQuantity(row.quantity);
}

function getPreviewAverageBuyPrice(row: GridRow): number {
  return safeParseFloat(row.average_buy_price);
}

function getPreviewValuationKrw(row: GridRow): number {
  const quantity = getPreviewQuantity(row);
  if (quantity <= 0) {
    return 0;
  }
  const currentQuantity = Number(row.original_quantity ?? row.quantity ?? 0);
  if (currentQuantity > 0) {
    return (Number(row.valuation_krw ?? 0) / currentQuantity) * quantity;
  }
  if (row.currency === "KRW") {
    return Number(row.current_price_num ?? 0) * quantity;
  }
  return Number(row.valuation_krw ?? 0);
}

function getPreviewBuyAmountKrw(row: GridRow): number {
  const quantity = getPreviewQuantity(row);
  if (quantity <= 0) {
    return 0;
  }
  const averageBuyPrice = getPreviewAverageBuyPrice(row);
  if (row.currency === "KRW") {
    return averageBuyPrice * quantity;
  }
  const currentQuantity = Number(row.original_quantity ?? row.quantity ?? 0);
  const currentAverageBuyPrice = Number(row.original_average_buy_price ?? safeParseFloat(row.average_buy_price));
  if (currentQuantity > 0 && currentAverageBuyPrice > 0) {
    const fxFactor = Number(row.buy_amount_krw ?? 0) / (currentQuantity * currentAverageBuyPrice);
    return averageBuyPrice * quantity * fxFactor;
  }
  return Number(row.buy_amount_krw ?? 0);
}

function getPreviewTargetRatio(row: GridRow): number | null {
  const parsed = Number(row.target_ratio ?? null);
  if (Number.isNaN(parsed) || parsed < 0) {
    return 0;
  }
  return parsed;
}

function getPreviewWeightPct(row: GridRow, rows: HoldingsRow[], summary: AccountSummary): number {
  if (String(row.ticker || "").trim().toUpperCase() === "IS") {
    return 0;
  }
  const rowId = buildGridRowId(row);
  const previewTotalValuation = rows.reduce((sum, currentRow) => {
    return sum + getPreviewValuationKrw({ ...currentRow, id: buildGridRowId(currentRow) });
  }, 0);
  const previewIsValuation = rows.reduce((sum, currentRow) => {
    if (String(currentRow.ticker || "").trim().toUpperCase() !== "IS") {
      return sum;
    }
    return sum + getPreviewValuationKrw({ ...currentRow, id: buildGridRowId(currentRow) });
  }, 0);
  const currency = String(summary.currency || "KRW").trim().toUpperCase();
  const denominator =
    currency === "AUD"
      ? Number(summary.cash_balance_krw ?? 0) + previewTotalValuation - previewIsValuation
      : Number(summary.cash_balance_krw ?? 0) + previewTotalValuation;
  if (denominator <= 0) {
    return 0;
  }
  const targetRow = rows.find((currentRow) => buildGridRowId(currentRow) === rowId);
  if (!targetRow) {
    return 0;
  }
  const rowValuation = getPreviewValuationKrw({ ...targetRow, id: rowId });
  return (rowValuation / denominator) * 100;
}

function computeAccountTotalAssetsNative(summary: AccountSummary, rows: HoldingsRow[]): number {
  const currency = String(summary.currency || "KRW").trim().toUpperCase();
  if (currency === "AUD") {
    const holdingsNative = rows.reduce((sum, row) => {
      return sum + getCurrentPriceNumber(row as GridRow) * Number(row.quantity ?? 0);
    }, 0);
    return holdingsNative + Number(summary.cash_balance_native ?? 0);
  }
  return rows.reduce((sum, row) => sum + Number(row.valuation_krw ?? 0), 0) + Number(summary.cash_balance_krw ?? 0);
}

function getPreviewTargetAmount(row: GridRow, summary: AccountSummary, rows: HoldingsRow[]): number | null {
  const targetRatio = getPreviewTargetRatio(row);
  if (targetRatio === null) {
    return null;
  }
  const totalAssetsNative = computeAccountTotalAssetsNative(summary, rows);
  return Math.round(totalAssetsNative * (targetRatio / 100) * 100) / 100;
}

function getPreviewTargetQuantity(row: GridRow, summary: AccountSummary, rows: HoldingsRow[]): number | null {
  const targetAmount = getPreviewTargetAmount(row, summary, rows);
  if (targetAmount === null) {
    return null;
  }
  const currentPrice = getCurrentPriceNumber(row);
  if (currentPrice <= 0) {
    return null;
  }
  return roundTargetQuantity(targetAmount / currentPrice, row.currency);
}

function isDetailRow(row: ParentGridRow | undefined): row is Extract<ParentGridRow, { rowType: "detail" }> {
  return row?.rowType === "detail";
}

function formatAccountCash(summary: AccountSummary): string {
  const currency = String(summary.currency || "KRW").trim().toUpperCase();
  if (currency === "AUD") {
    return formatPrice(summary.cash_balance_native, "AUD");
  }
  return formatKrw(summary.cash_balance_krw);
}


function reorderRowsByTickers(rows: HoldingsRow[], orderedTickers: string[]): HoldingsRow[] {
  const normalizedTickers = orderedTickers.map((ticker) => String(ticker || "").trim().toUpperCase());
  const rowMap = new Map(
    rows
      .filter((row) => String(row.ticker || "").trim().toUpperCase() !== "IS")
      .map((row) => [String(row.ticker || "").trim().toUpperCase(), row] as const),
  );
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

  const remainingRows = rows.filter((row) => {
    const ticker = String(row.ticker || "").trim().toUpperCase();
    return ticker === "IS" || !seen.has(ticker);
  });

  return [...orderedRows, ...remainingRows.map((row) => ({ ...row }))].map((row, index) => ({
    ...row,
    sort_order: index,
  }));
}

const ASSETS_WEIGHT_TEXT_COLOR = "#7952b3";

function StableInlineInput({
  initialValue,
  onSave,
  onCancel,
  onChange,
  className,
  style,
  placeholder,
  autoFocus = false,
  disabled = false,
}: {
  initialValue: string;
  onSave?: (val: string) => void;
  onCancel?: () => void;
  onChange?: (val: string) => void;
  className?: string;
  style?: CSSProperties;
  placeholder?: string;
  autoFocus?: boolean;
  disabled?: boolean;
}) {
  const [localValue, setLocalValue] = useState(initialValue);

  useEffect(() => {
    setLocalValue(initialValue);
  }, [initialValue]);

  return (
    <input
      type="text"
      className={className}
      style={style}
      placeholder={placeholder}
      value={localValue}
      autoFocus={autoFocus}
      disabled={disabled}
      onMouseDown={(event) => {
        event.stopPropagation();
      }}
      onClick={(event) => {
        event.stopPropagation();
      }}
      onDoubleClick={(event) => {
        event.stopPropagation();
      }}
      onChange={(event) => {
        event.stopPropagation();
        setLocalValue(event.target.value);
        onChange?.(event.target.value);
      }}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.nativeEvent.isComposing) return;
        if (event.key === "Enter") {
          onSave?.(localValue);
        } else if (event.key === "Escape") {
          setLocalValue(initialValue);
          onCancel?.();
        }
      }}
      onBlur={() => {
        if (localValue !== initialValue) {
          onSave?.(localValue);
        }
      }}
    />
  );
}

function AccountHoldingsDetailPanel({
  summary,
  initialRows,
  onReload,
  onPreviewTargetRatioTotalChange,
}: {
  summary: AccountSummary;
  initialRows: HoldingsRow[];
  onReload: () => Promise<void>;
  onPreviewTargetRatioTotalChange: (accountId: string, total: number) => void;
}) {
  const toast = useToast();
  const hydrateRows = useCallback(
    (sourceRows: HoldingsRow[]) =>
      sourceRows.map((row) => ({
        ...row,
        original_quantity: Number(row.original_quantity ?? row.quantity ?? 0),
        original_average_buy_price: Number(row.original_average_buy_price ?? safeParseFloat(row.average_buy_price)),
      })),
    [],
  );
  const [rows, setRows] = useState<HoldingsRow[]>(() => hydrateRows(initialRows));
  const [addingRow, setAddingRow] = useState<AddingRowState | null>(null);
  const [editingRowId, setEditingRowId] = useState<string | null>(null);
  const [dirtyRowIds, setDirtyRowIds] = useState<string[]>([]);
  const [dirtyCellKeys, setDirtyCellKeys] = useState<string[]>([]);
  const [selectedRowIds, setSelectedRowIds] = useState<string[]>([]);
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [isReorderDirty, setIsReorderDirty] = useState(false);
  const qtyRef = useRef<HTMLInputElement>(null);
  const priceRef = useRef<HTMLInputElement>(null);
  const targetRatioRef = useRef<HTMLInputElement>(null);
  const rowsRef = useRef<HoldingsRow[]>(initialRows);
  const childSaveTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const childSavingRowIdsRef = useRef<Set<string>>(new Set());
  const childQueuedRowIdsRef = useRef<Set<string>>(new Set());
  const reorderSaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reorderSavingRef = useRef(false);
  const reorderQueuedRef = useRef(false);
  useEffect(() => {
    const nextRows = hydrateRows(initialRows);
    setRows(nextRows);
    rowsRef.current = nextRows;
    setDirtyRowIds([]);
    setDirtyCellKeys([]);
    setSelectedRowIds([]);
    setEditingRowId(null);
    setAddingRow(null);
    setIsReorderDirty(false);
  }, [hydrateRows, initialRows]);

  useEffect(() => {
    rowsRef.current = rows;
  }, [rows]);

  useEffect(() => {
    const previewTargetRatioTotal = rows.reduce((sum, row) => {
      return sum + Number(row.target_ratio ?? 0);
    }, 0);
    onPreviewTargetRatioTotalChange(summary.account_id, parseFloat(previewTargetRatioTotal.toFixed(1)));
  }, [onPreviewTargetRatioTotalChange, rows, summary.account_id]);

  useEffect(() => {
    return () => {
      childSaveTimersRef.current.forEach((timer) => clearTimeout(timer));
      childSaveTimersRef.current.clear();
      if (reorderSaveTimerRef.current) {
        clearTimeout(reorderSaveTimerRef.current);
      }
    };
  }, []);

  const isEditableHoldingRow = useCallback(
    (row: GridRow | undefined | null) => Boolean(row && row.id !== "__adding__" && row.ticker !== "IS"),
    [],
  );

  const gridRows = useMemo<GridRow[]>(() => {
    const baseRows = rows
      .map((row, index) => ({
        ...row,
        id: buildGridRowId(row),
        quantity: typeof row.quantity === "number" ? row.quantity : parseInt(String(row.quantity), 10) || 0,
        average_buy_price: safeParseFloat(row.average_buy_price),
        target_ratio: row.target_ratio ?? 0,
      }));

    if (!addingRow) {
      return baseRows;
    }

    return [
      {
        id: "__adding__",
        account_id: summary.account_id,
        account_name: summary.name,
        currency: summary.currency,
        bucket: "",
        bucket_id: 0,
        ticker: addingRow.ticker,
        name: addingRow.name || "",
        quantity: 0,
        average_buy_price: 0,
        current_price: "-",
        days_held: "-",
        pnl_krw: 0,
        return_pct: 0,
        weight_pct: 0,
        buy_amount_krw: 0,
        valuation_krw: 0,
        target_ratio: 0,
      } as GridRow,
      ...baseRows,
    ];
  }, [addingRow, rows, summary.account_id, summary.currency, summary.name]);

  const hasPendingAdd = Boolean(addingRow);
  const hasSelectedRows = selectedRowIds.length > 0;
  const hasPendingSave = hasPendingAdd || dirtyRowIds.length > 0 || isReorderDirty;

  const isDirtyEditableCell = useCallback(
    (rowId: string | undefined, field: string) => Boolean(rowId && dirtyCellKeys.includes(buildDirtyCellKey(rowId, field))),
    [dirtyCellKeys],
  );

  const handleValidateTicker = useCallback(async (tickerToUse?: string) => {
    const ticker = tickerToUse || addingRow?.ticker;
    if (!ticker || addingRow?.isValidatingTicker) {
      return;
    }

    setAddingRow((previous) =>
      previous
        ? {
            ...previous,
            ticker,
            isValidatingTicker: true,
          }
        : null,
    );

    try {
      const response = await fetch("/api/assets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "validate",
          account_id: summary.account_id,
          ticker,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "검증 실패");
      }
      setAddingRow((previous) =>
        previous
          ? {
              ...previous,
              ticker: payload.ticker,
              name: payload.name,
              bucketId: payload.bucket_id,
              isValidated: true,
              isValidatingTicker: false,
            }
          : null,
      );
      toast.success(`조회 성공: ${payload.name}`);
    } catch (error) {
      setAddingRow((previous) =>
        previous
          ? {
              ...previous,
              isValidatingTicker: false,
            }
          : null,
      );
      toast.error(error instanceof Error ? error.message : "검증 실패");
    }
  }, [addingRow?.isValidatingTicker, addingRow?.ticker, summary.account_id, toast]);

  const processAddingRow = useCallback(async () => {
    if (!addingRow?.isValidated) {
      throw new Error("먼저 종목 확인을 완료해 주세요.");
    }

    const rawQuantity = qtyRef.current?.value ?? "";
    const rawPrice = priceRef.current?.value ?? "";
    const rawTargetRatio = targetRatioRef.current?.value ?? "0";

    const quantity = parseInt(parseRawPrice(rawQuantity), 10);
    const averageBuyPrice = safeParseFloat(rawPrice);
    const targetRatio = parseFloat(rawTargetRatio);

    if (Number.isNaN(quantity) || quantity < 0 || Number.isNaN(averageBuyPrice) || averageBuyPrice < 0) {
      throw new Error("수량과 매입 단가를 확인해 주세요.");
    }
    if (Number.isNaN(targetRatio) || targetRatio < 0 || targetRatio > 100) {
      throw new Error("목표비중은 0~100 사이여야 합니다.");
    }

    const response = await fetch("/api/assets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        account_id: summary.account_id,
        ticker: addingRow.ticker,
        quantity,
        average_buy_price: averageBuyPrice,
        target_ratio: parseFloat(targetRatio.toFixed(1)),
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "종목 추가에 실패했습니다.");
    }
  }, [addingRow, summary.account_id]);

  const processRowUpdate = useCallback(async (row: GridRow) => {
    const quantity = parseEditableQuantity(row.quantity);
    const averageBuyPrice = safeParseFloat(row.average_buy_price);
    const targetRatio = Number(row.target_ratio ?? 0);

    if (Number.isNaN(quantity) || quantity < 0 || Number.isNaN(averageBuyPrice) || averageBuyPrice < 0) {
      throw new Error("입력값이 올바르지 않습니다.");
    }
    if (Number.isNaN(targetRatio) || targetRatio < 0 || targetRatio > 100) {
      throw new Error("목표비중은 0~100 사이여야 합니다.");
    }

    const response = await fetch("/api/assets", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        account_id: summary.account_id,
        ticker: row.ticker.replace("ASX:", ""),
        quantity,
        average_buy_price: averageBuyPrice,
        target_ratio: parseFloat(targetRatio.toFixed(1)),
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "종목 수정에 실패했습니다.");
    }
  }, [summary.account_id]);

  const clearDirtyRowState = useCallback((rowId: string) => {
    setDirtyRowIds((previous) => previous.filter((id) => id !== rowId));
    setDirtyCellKeys((previous) => previous.filter((key) => !key.startsWith(`${rowId}::`)));
  }, []);

  const silentlySaveRow = useCallback(async (rowId: string) => {
    if (childSavingRowIdsRef.current.has(rowId)) {
      childQueuedRowIdsRef.current.add(rowId);
      return;
    }

    const sourceRow = rowsRef.current.find((row) => buildGridRowId(row) === rowId);
    if (!sourceRow) {
      clearDirtyRowState(rowId);
      return;
    }

    childSavingRowIdsRef.current.add(rowId);
    try {
      await processRowUpdate({
        ...sourceRow,
        id: rowId,
        quantity: typeof sourceRow.quantity === "number" ? sourceRow.quantity : parseInt(String(sourceRow.quantity), 10) || 0,
        average_buy_price: safeParseFloat(sourceRow.average_buy_price),
        target_ratio: sourceRow.target_ratio ?? 0,
      });
      clearDirtyRowState(rowId);
    } catch (error) {
      await onReload();
      toast.error(error instanceof Error ? error.message : "변경사항 저장에 실패했습니다.");
    } finally {
      childSavingRowIdsRef.current.delete(rowId);
      if (childQueuedRowIdsRef.current.has(rowId)) {
        childQueuedRowIdsRef.current.delete(rowId);
        const nextTimer = setTimeout(() => {
          childSaveTimersRef.current.delete(rowId);
          void silentlySaveRow(rowId);
        }, 400);
        childSaveTimersRef.current.set(rowId, nextTimer);
      }
    }
  }, [clearDirtyRowState, onReload, processRowUpdate, toast]);

  const scheduleSilentRowSave = useCallback((rowId: string) => {
    const currentTimer = childSaveTimersRef.current.get(rowId);
    if (currentTimer) {
      clearTimeout(currentTimer);
    }
    const nextTimer = setTimeout(() => {
      childSaveTimersRef.current.delete(rowId);
      void silentlySaveRow(rowId);
    }, 700);
    childSaveTimersRef.current.set(rowId, nextTimer);
  }, [silentlySaveRow]);

  const persistRowOrder = useCallback(async (orderedRows: HoldingsRow[]) => {
    const orderedTickers = orderedRows
      .map((row) => String(row.ticker || "").trim().toUpperCase())
      .filter((ticker) => ticker && ticker !== "IS");

    if (!orderedTickers.length) {
      return;
    }

    if (reorderSavingRef.current) {
      reorderQueuedRef.current = true;
      return;
    }

    reorderSavingRef.current = true;
    try {
      const response = await fetch("/api/assets", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "reorder",
          account_id: summary.account_id,
          ordered_tickers: orderedTickers,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "순서 저장에 실패했습니다.");
      }
      setIsReorderDirty(false);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "순서 저장에 실패했습니다.");
    } finally {
      reorderSavingRef.current = false;
      if (reorderQueuedRef.current) {
        reorderQueuedRef.current = false;
        const nextRows = rowsRef.current;
        const nextTimer = setTimeout(() => {
          reorderSaveTimerRef.current = null;
          void persistRowOrder(nextRows);
        }, 400);
        reorderSaveTimerRef.current = nextTimer;
      }
    }
  }, [onReload, summary.account_id, toast]);

  const scheduleSilentReorderSave = useCallback((orderedRows: HoldingsRow[]) => {
    if (reorderSaveTimerRef.current) {
      clearTimeout(reorderSaveTimerRef.current);
    }
    const nextTimer = setTimeout(() => {
      reorderSaveTimerRef.current = null;
      void persistRowOrder(orderedRows);
    }, 700);
    reorderSaveTimerRef.current = nextTimer;
  }, [persistRowOrder]);

  const flushPendingSaves = useCallback(() => {
    childSaveTimersRef.current.forEach((timer) => clearTimeout(timer));
    childSaveTimersRef.current.clear();
    if (reorderSaveTimerRef.current) {
      clearTimeout(reorderSaveTimerRef.current);
      reorderSaveTimerRef.current = null;
    }
  }, []);

  const handleSaveChanges = useCallback(async () => {
    if (processingId === "__adding__" || processingId === "__deleting__") {
      return;
    }

    flushPendingSaves();

    try {
      const dirtyRows = rowsRef.current
        .map((row) => ({
          ...row,
          id: buildGridRowId(row),
          quantity: typeof row.quantity === "number" ? row.quantity : parseInt(String(row.quantity), 10) || 0,
          average_buy_price: safeParseFloat(row.average_buy_price),
          target_ratio: row.target_ratio ?? 0,
        }))
        .filter((row) => dirtyRowIds.includes(row.id));

      for (const row of dirtyRows) {
        await processRowUpdate(row);
        clearDirtyRowState(row.id);
      }

      if (isReorderDirty) {
        await persistRowOrder(rowsRef.current);
      }

      if (addingRow) {
        setProcessingId("__adding__");
        await processAddingRow();
        await onReload();
        toast.success("종목 추가 완료");
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "변경사항 저장에 실패했습니다.");
    } finally {
      setProcessingId(null);
    }
  }, [
    addingRow,
    clearDirtyRowState,
    dirtyRowIds,
    flushPendingSaves,
    isReorderDirty,
    onReload,
    persistRowOrder,
    processAddingRow,
    processRowUpdate,
    processingId,
    toast,
  ]);

  const handleDeleteSelected = useCallback(async () => {
    if (!selectedRowIds.length) {
      return;
    }
    const selectedRows = gridRows.filter((row) => selectedRowIds.includes(row.id) && row.id !== "__adding__");
    if (!selectedRows.length) {
      return;
    }

    const summaryText =
      selectedRows.length === 1
        ? `${selectedRows[0].name}(${selectedRows[0].ticker}) 종목을 삭제하시겠습니까?`
        : `${selectedRows.length}개 종목을 삭제하시겠습니까?`;
    if (!confirm(summaryText)) {
      return;
    }

    setProcessingId("__deleting__");
    try {
      for (const row of selectedRows) {
        const params = new URLSearchParams({
          account: summary.account_id,
          ticker: row.ticker.replace("ASX:", ""),
        });
        const response = await fetch(`/api/assets?${params.toString()}`, { method: "DELETE" });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "삭제 실패");
        }
      }
      await onReload();
      toast.success("삭제 완료");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "삭제 실패");
    } finally {
      setProcessingId(null);
    }
  }, [gridRows, onReload, selectedRowIds, summary.account_id, toast]);

  const handleCellValueChanged = useCallback((row: GridRow | undefined, field: string | undefined) => {
    if (!row || !isEditableHoldingRow(row)) {
      return;
    }

    setRows((previous) =>
      previous.map((currentRow) => {
        if (buildGridRowId(currentRow) !== row.id) {
          return currentRow;
        }
        return {
          ...currentRow,
          quantity: parseEditableQuantity(row.quantity),
          average_buy_price: safeParseFloat(row.average_buy_price),
          target_ratio: Number(row.target_ratio ?? 0),
        };
      }),
    );
    setDirtyRowIds((previous) => (previous.includes(row.id) ? previous : [...previous, row.id]));
    if (field) {
      const dirtyCellKey = buildDirtyCellKey(row.id, field);
      setDirtyCellKeys((previous) => (previous.includes(dirtyCellKey) ? previous : [...previous, dirtyCellKey]));
    }
    scheduleSilentRowSave(row.id);
  }, [isEditableHoldingRow, scheduleSilentRowSave]);

  const columns = useMemo<ColDef<GridRow>[]>(() => [
    {
      colId: "drag",
      headerName: "",
      width: 42,
      maxWidth: 42,
      pinned: "left",
      sortable: false,
      resizable: false,
      suppressMovable: true,
      rowDrag: (params) => Boolean(params.data && params.data.id !== "__adding__" && params.data.ticker !== "IS"),
      cellClass: "assetsDragCell",
      valueGetter: () => "",
    },
    {
      field: "bucket",
      headerName: "버킷",
      width: 96,
      cellClass: (params) => getBucketCellClass(params.data?.bucket_id ?? 0),
    },
    {
      field: "ticker",
      headerName: "종목코드",
      width: 118,
      cellRenderer: (params: { data?: GridRow; value?: string }) => {
        const row = params.data;
        if (!row) {
          return null;
        }
        if (row.id === "__adding__") {
          if (addingRow?.isValidated) {
            return (
              <div className="d-flex gap-2 align-items-center">
                <span>{addingRow.ticker}</span>
                <button
                  className="btn btn-sm btn-link p-0 assetsInlineLinkButton"
                  onClick={() =>
                    setAddingRow((previous) =>
                      previous
                        ? {
                            ...previous,
                            ticker: "",
                            isValidated: false,
                            name: "",
                          }
                        : null,
                    )
                  }
                >
                  변경
                </button>
              </div>
            );
          }

          return (
            <div className="assetsTickerLookup">
              <StableInlineInput
                className="form-control form-control-sm assetsInlineInput assetsInlineInputTicker"
                initialValue={addingRow?.ticker ?? ""}
                disabled={addingRow?.isValidatingTicker}
                onChange={(value) =>
                  setAddingRow((previous) =>
                    previous
                      ? {
                          ...previous,
                          ticker: value,
                        }
                      : null,
                  )
                }
                onSave={handleValidateTicker}
              />
              <button
                className="btn btn-outline-primary btn-sm assetsInlineButton d-inline-flex align-items-center gap-1"
                disabled={addingRow?.isValidatingTicker}
                onMouseDown={(event) => {
                  event.stopPropagation();
                }}
                onClick={(event) => {
                  event.stopPropagation();
                  void handleValidateTicker();
                }}
              >
                {addingRow?.isValidatingTicker ? (
                  <IconLoader2 size={14} style={{ animation: "spin 1s linear infinite" }} />
                ) : null}
                {addingRow?.isValidatingTicker ? "확인중" : "확인"}
              </button>
            </div>
          );
        }
        return <span className="appCodeText">{params.value}</span>;
      },
    },
    {
      field: "name",
      headerName: "종목명",
      minWidth: 280,
      flex: 1.6,
      cellRenderer: (params: { value?: string | null }) => {
        const value = String(params.value ?? "-");
        return (
          <span className="assetsNameCellText" title={value}>
            {value}
          </span>
        );
      },
    },
    {
      field: "daily_change_pct",
      headerName: "일간(%)",
      width: 92,
      type: "rightAligned",
      cellRenderer: (params: { value?: number | null }) => (
        <span className={getSignedClass(params.value ?? 0)}>
          {params.value === null || params.value === undefined
            ? "-"
            : `${(params.value ?? 0) > 0 ? "+" : ""}${params.value.toFixed(2)}%`}
        </span>
      ),
    },
    {
      field: "current_price",
      headerName: "현재가",
      width: 116,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: string }) => (
        <span>{formatPrice(safeParseFloat(params.value), params.data?.currency || "KRW")}</span>
      ),
    },
    {
      field: "quantity",
      headerName: "수량",
      width: 88,
      type: "rightAligned",
      editable: (params) => isEditableHoldingRow(params.data) && processingId !== params.data?.id,
      cellClass: (params) => {
        if (!isEditableHoldingRow(params.data)) {
          return undefined;
        }
        return isDirtyEditableCell(params.data?.id, "quantity")
          ? "assetsEditableCell assetsDirtyCell"
          : "assetsEditableCell";
      },
      valueParser: (params) => {
        const parsed = parseInt(parseRawPrice(params.newValue), 10);
        if (Number.isNaN(parsed) || parsed < 0) {
          return params.oldValue;
        }
        return parsed;
      },
      cellRenderer: (params: { data?: GridRow; value?: number }) => {
        const row = params.data;
        if (!row) {
          return null;
        }
        if (row.id === "__adding__") {
          return (
            <input
              type="number"
              step="1"
              ref={qtyRef}
              className="form-control form-control-sm assetsInlineInput"
              defaultValue="0"
              disabled={!addingRow?.isValidated}
            />
          );
        }
        return <span>{new Intl.NumberFormat("ko-KR").format(params.value ?? 0)}</span>;
      },
    },
    {
      field: "average_buy_price",
      headerName: "매입 단가",
      width: 128,
      type: "rightAligned",
      editable: (params) => isEditableHoldingRow(params.data) && processingId !== params.data?.id,
      cellClass: (params) => {
        if (!isEditableHoldingRow(params.data)) {
          return undefined;
        }
        return isDirtyEditableCell(params.data?.id, "average_buy_price")
          ? "assetsEditableCell assetsDirtyCell"
          : "assetsEditableCell";
      },
      valueParser: (params) => {
        const parsed = parseFloat(parseRawPrice(params.newValue));
        if (Number.isNaN(parsed) || parsed < 0) {
          return params.oldValue;
        }
        return parsed;
      },
      cellRenderer: (params: { data?: GridRow; value?: string | number }) => {
        const row = params.data;
        if (!row) {
          return null;
        }
        if (row.id === "__adding__") {
          return (
            <input
              type="number"
              step="any"
              ref={priceRef}
              className="form-control form-control-sm assetsInlineInput"
              defaultValue="0"
              disabled={!addingRow?.isValidated}
            />
          );
        }
        return <span>{formatPrice(safeParseFloat(params.value), row.currency || "KRW")}</span>;
      },
    },
    {
      field: "return_pct",
      headerName: "수익률",
      width: 88,
      type: "rightAligned",
      cellRenderer: (params: { value?: number }) => (
        <span className={getSignedClass(params.value ?? 0)}>
          {(params.value ?? 0) > 0 ? "+" : ""}
          {(params.value ?? 0).toFixed(2)}%
        </span>
      ),
    },
    {
      field: "pnl_krw",
      headerName: "평가손익",
      width: 136,
      type: "rightAligned",
      cellRenderer: (params: { value?: number }) => (
        <span className={getSignedClass(params.value ?? 0)}>{formatKrw(params.value ?? 0)}</span>
      ),
    },
    {
      field: "valuation_krw",
      headerName: "평가 금액",
      width: 136,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow }) =>
        params.data ? formatKrw(getPreviewValuationKrw(params.data)) : "-",
    },
    {
      field: "weight_pct",
      headerName: "비중",
      width: 80,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow }) =>
        params.data ? (
          <span style={{ color: ASSETS_WEIGHT_TEXT_COLOR, fontWeight: 700 }}>
            {getPreviewWeightPct(params.data, rows, summary).toFixed(1)}%
          </span>
        ) : (
          "-"
        ),
    },
    {
      field: "target_ratio",
      headerName: "목표비중",
      width: 100,
      type: "rightAligned",
      editable: (params) => isEditableHoldingRow(params.data) && processingId !== params.data?.id,
      cellClass: (params) => {
        if (!isEditableHoldingRow(params.data)) {
          return undefined;
        }
        return isDirtyEditableCell(params.data?.id, "target_ratio")
          ? "assetsEditableCell assetsDirtyCell"
          : "assetsEditableCell";
      },
      valueParser: (params) => {
        const parsed = Number(params.newValue);
        if (Number.isNaN(parsed) || parsed < 0 || parsed > 100) {
          return params.oldValue;
        }
        return parsed;
      },
      cellRenderer: (params: { data?: GridRow; value?: number | null }) => {
        const row = params.data;
        if (!row) {
          return null;
        }
        if (row.id === "__adding__") {
          return (
            <input
              type="number"
              step="0.1"
              min="0"
              max="100"
              ref={targetRatioRef}
              className="form-control form-control-sm assetsInlineInput"
              defaultValue={addingRow?.target_ratio ?? "0"}
              disabled={!addingRow?.isValidated}
            />
          );
        }
        return (
          <span style={{ color: ASSETS_WEIGHT_TEXT_COLOR, fontWeight: 700 }}>
            {params.value === null || params.value === undefined ? "-" : `${params.value.toFixed(1)}%`}
          </span>
        );
      },
    },
    {
      field: "target_quantity",
      headerName: "목표수량",
      width: 124,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow }) => {
        if (!params.data) {
          return "-";
        }
        return (
          <span>{formatTargetQuantity(getPreviewTargetQuantity(params.data, summary, rows), params.data.currency || "KRW")}</span>
        );
      },
    },
    {
      colId: "expected_change_quantity",
      headerName: "예상변경수량",
      width: 138,
      type: "rightAligned",
      sortable: false,
      cellRenderer: (params: { data?: GridRow }) => {
        if (!params.data) {
          return "-";
        }
        const targetQuantity = getPreviewTargetQuantity(params.data, summary, rows);
        const currentQuantity = getPreviewQuantity(params.data);
        return (
          <span className={getSignedNullableClass(
            targetQuantity === null || targetQuantity === undefined ? null : targetQuantity - currentQuantity,
          )}>
            {formatExpectedChangeQuantity(targetQuantity, currentQuantity, params.data.currency || "KRW")}
          </span>
        );
      },
    },
    {
      field: "target_amount",
      headerName: "목표금액",
      width: 136,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow }) => {
        if (!params.data) {
          return "-";
        }
        return <span>{formatTargetAmount(getPreviewTargetAmount(params.data, summary, rows), params.data.currency || "KRW")}</span>;
      },
    },
    { field: "days_held", headerName: "보유일", width: 76 },
  ], [addingRow, handleValidateTicker, isDirtyEditableCell, isEditableHoldingRow, processingId, rows, summary]);

  return (
    <div className="assetsDetailPanel">
      <div className="appActionHeader">
        <div className="appActionHeaderInner">
          <button
            className="btn btn-primary btn-sm px-3 fw-bold"
            onClick={() =>
              setAddingRow({
                ticker: "",
                quantity: "",
                average_buy_price: "",
                target_ratio: "0",
                isValidated: false,
              })
            }
            disabled={hasPendingAdd}
          >
            <IconPlus size={16} /> 추가
          </button>
          <button
            className="btn btn-success btn-sm px-3 fw-bold"
            onClick={() => void handleSaveChanges()}
            disabled={!hasPendingSave || processingId === "__adding__" || processingId === "__deleting__"}
          >
            <IconCheck size={16} /> 저장
          </button>
          <button
            className="btn btn-outline-danger btn-sm px-3 fw-bold"
            onClick={() => void handleDeleteSelected()}
            disabled={!hasSelectedRows || processingId === "__adding__" || processingId === "__deleting__"}
          >
            <IconTrash size={16} /> 삭제
          </button>
        </div>
      </div>
      <div className="assetsDetailGridWrap">
        <AppAgGrid
          rowData={gridRows}
          columnDefs={columns}
          loading={processingId === "__adding__" || processingId === "__deleting__"}
          minHeight="100%"
          className="assetsAgGrid assetsChildAgGrid"
          theme={assetsGridTheme}
          getRowClass={(params: RowClassParams<GridRow>) => {
            const classes: string[] = [];
            if (params.data?.ticker === "IS") {
              return classes.join(" ");
            }
            if (Number(params.data?.quantity ?? 0) > 0) {
              classes.push("appHeldRow");
            }
            return classes.join(" ");
          }}
          gridOptions={{
            suppressMovableColumns: true,
            ensureDomOrder: true,
            stopEditingWhenCellsLoseFocus: true,
            rowDragManaged: true,
            animateRows: true,
            rowSelection: {
              mode: "multiRow",
              checkboxes: (params) => Boolean(params.data && params.data.id !== "__adding__" && params.data.ticker !== "IS"),
              headerCheckbox: true,
              hideDisabledCheckboxes: true,
              enableClickSelection: false,
              isRowSelectable: (params) =>
                Boolean(params.data && params.data.id !== "__adding__" && params.data.ticker !== "IS"),
            },
            selectionColumnDef: {
              width: 52,
              minWidth: 52,
              maxWidth: 52,
              pinned: "left",
              sortable: false,
              resizable: false,
              suppressMovable: true,
              headerName: "",
              cellClass: "assetsSelectCell",
            },
            onSelectionChanged: (params) => {
              setSelectedRowIds(
                params.api
                  .getSelectedRows()
                  .map((row) => row.id)
                  .filter((rowId): rowId is string => Boolean(rowId)),
              );
            },
            onCellEditingStarted: (params) => {
              if (params.data && isEditableHoldingRow(params.data)) {
                setEditingRowId(params.data.id);
              }
            },
            onCellEditingStopped: () => {
              setEditingRowId(null);
            },
            onCellValueChanged: (params) => {
              if (params.newValue === params.oldValue) {
                return;
              }
              handleCellValueChanged(params.data, params.colDef.field);
            },
            onRowDragEnd: (params) => {
              const orderedTickers: string[] = [];
              params.api.forEachNode((node) => {
                const ticker = String(node.data?.ticker || "").trim().toUpperCase();
                if (!ticker || ticker === "IS") {
                  return;
                }
                orderedTickers.push(ticker);
              });
              if (!orderedTickers.length) {
                return;
              }
              setRows((previous) => {
                const nextRows = reorderRowsByTickers(previous, orderedTickers);
                rowsRef.current = nextRows;
                return nextRows;
              });
              const nextRows = reorderRowsByTickers(rowsRef.current, orderedTickers);
              rowsRef.current = nextRows;
              setIsReorderDirty(true);
              scheduleSilentReorderSave(nextRows);
            },
            rowClassRules: {
              assetsAddingRow: (params) => params.data?.id === "__adding__",
              assetsEditingRow: (params) => Boolean(params.data?.id && params.data.id === editingRowId),
            },
          }}
        />
      </div>
    </div>
  );
}

export function AssetsManager({ onHeaderSummaryChange }: { onHeaderSummaryChange?: (summary: AssetsHeaderSummary) => void }) {
  const toast = useToast();
  const [allRows, setAllRows] = useState<HoldingsRow[]>([]);
  const [summaries, setSummaries] = useState<AccountSummary[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [parentDirtyCellKeys, setParentDirtyCellKeys] = useState<string[]>([]);
  const [editingParentId, setEditingParentId] = useState<string | null>(null);
  const [previewTargetRatioTotals, setPreviewTargetRatioTotals] = useState<Record<string, number>>({});
  const summariesRef = useRef<AccountSummary[]>([]);
  const parentSaveTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const parentSavingAccountIdsRef = useRef<Set<string>>(new Set());
  const parentQueuedAccountIdsRef = useRef<Set<string>>(new Set());

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/assets", { cache: "no-store" });
      const payload = (await response.json()) as HoldingsResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "자산 정보를 불러오지 못했습니다.");
      }
      setAllRows(payload.rows ?? []);
      setSummaries(payload.account_summaries ?? []);
      setExpandedId((current) => current ?? payload.account_summaries?.[0]?.account_id ?? null);
      setPreviewTargetRatioTotals({});
      setParentDirtyCellKeys([]);
      setEditingParentId(null);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "자산 정보를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    summariesRef.current = summaries;
  }, [summaries]);

  useEffect(() => {
    return () => {
      parentSaveTimersRef.current.forEach((timer) => clearTimeout(timer));
      parentSaveTimersRef.current.clear();
    };
  }, []);

  const groupedRows = useMemo(() => {
    const grouped = new Map<string, HoldingsRow[]>();
    for (const row of allRows) {
      const key = String(row.account_id || "").trim();
      if (!grouped.has(key)) {
        grouped.set(key, []);
      }
      grouped.get(key)?.push(row);
    }
    return grouped;
  }, [allRows]);

  const parentRows = useMemo<ParentGridRow[]>(() => {
    return summaries.flatMap((summary) => {
      const mainRow: ParentGridRow = {
        ...summary,
        id: summary.account_id,
        rowType: "main",
        cash_edit_value:
          String(summary.currency || "KRW").toUpperCase() === "AUD"
            ? Number(summary.cash_balance_native ?? 0)
            : Number(summary.cash_balance_krw ?? 0),
      };

      if (expandedId !== summary.account_id) {
        return [mainRow];
      }

      return [
        mainRow,
        {
          id: `${summary.account_id}__detail`,
          rowType: "detail",
          parentId: summary.account_id,
          summary,
          rows: groupedRows.get(summary.account_id) ?? [],
        },
      ];
    });
  }, [expandedId, groupedRows, summaries]);

  const totalAssets = useMemo(
    () => summaries.reduce((sum, summary) => sum + Number(summary.total_assets_krw ?? 0), 0),
    [summaries],
  );
  const totalValuation = useMemo(
    () => summaries.reduce((sum, summary) => sum + Number(summary.valuation_krw ?? 0), 0),
    [summaries],
  );
  const totalCash = useMemo(
    () => summaries.reduce((sum, summary) => sum + Number(summary.cash_balance_krw ?? 0), 0),
    [summaries],
  );

  useEffect(() => {
    onHeaderSummaryChange?.({
      totalAssets,
      totalValuation,
      totalCash,
      accountCount: summaries.length,
    });
  }, [onHeaderSummaryChange, summaries.length, totalAssets, totalCash, totalValuation]);

  const isDirtyParentCell = useCallback(
    (rowId: string | undefined, field: string) => Boolean(rowId && parentDirtyCellKeys.includes(buildDirtyCellKey(rowId, field))),
    [parentDirtyCellKeys],
  );

  const clearDirtyParentState = useCallback((accountId: string) => {
    setParentDirtyCellKeys((previous) => previous.filter((key) => !key.startsWith(`${accountId}::`)));
  }, []);

  const silentlySaveParent = useCallback(async (accountId: string) => {
    if (parentSavingAccountIdsRef.current.has(accountId)) {
      parentQueuedAccountIdsRef.current.add(accountId);
      return;
    }

    const summary = summariesRef.current.find((item) => item.account_id === accountId);
    if (!summary) {
      clearDirtyParentState(accountId);
      return;
    }

    parentSavingAccountIdsRef.current.add(accountId);
    try {
      const isAud = String(summary.currency || "KRW").toUpperCase() === "AUD";
      const response = await fetch("/api/assets", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          account_id: summary.account_id,
          total_principal: summary.total_principal,
          cash_balance_krw: isAud ? 0 : summary.cash_balance_krw,
          cash_balance_native: isAud ? summary.cash_balance_native : summary.cash_balance_krw,
          cash_currency: summary.cash_currency,
          intl_shares_value: summary.account_id === "aus_account" ? summary.intl_shares_value : null,
          intl_shares_change: summary.account_id === "aus_account" ? summary.intl_shares_change : null,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "계좌 저장에 실패했습니다.");
      }
      clearDirtyParentState(accountId);
    } catch (error) {
      await load();
      toast.error(error instanceof Error ? error.message : "계좌 저장에 실패했습니다.");
    } finally {
      parentSavingAccountIdsRef.current.delete(accountId);
      if (parentQueuedAccountIdsRef.current.has(accountId)) {
        parentQueuedAccountIdsRef.current.delete(accountId);
        const nextTimer = setTimeout(() => {
          parentSaveTimersRef.current.delete(accountId);
          void silentlySaveParent(accountId);
        }, 400);
        parentSaveTimersRef.current.set(accountId, nextTimer);
      }
    }
  }, [clearDirtyParentState, load, toast]);

  const scheduleSilentParentSave = useCallback((accountId: string) => {
    const currentTimer = parentSaveTimersRef.current.get(accountId);
    if (currentTimer) {
      clearTimeout(currentTimer);
    }
    const nextTimer = setTimeout(() => {
      parentSaveTimersRef.current.delete(accountId);
      void silentlySaveParent(accountId);
    }, 700);
    parentSaveTimersRef.current.set(accountId, nextTimer);
  }, [silentlySaveParent]);

  const handleParentCellValueChanged = useCallback((row: ParentGridRow | undefined, field: string | undefined) => {
    if (!row || isDetailRow(row) || !field) {
      return;
    }

    setSummaries((previous) =>
      previous.map((summary) => {
        if (summary.account_id !== row.account_id) {
          return summary;
        }

        const cashEditValue = Number((row as Extract<ParentGridRow, { rowType: "main" }>).cash_edit_value ?? 0);
        const isAud = String(summary.currency || "KRW").toUpperCase() === "AUD";
        return {
          ...summary,
          total_principal: Number(row.total_principal ?? summary.total_principal),
          cash_balance_krw: isAud ? summary.cash_balance_krw : cashEditValue,
          cash_balance_native: isAud ? cashEditValue : summary.cash_balance_native,
          intl_shares_value: row.account_id === "aus_account" ? Number(row.intl_shares_value ?? 0) : summary.intl_shares_value,
          intl_shares_change: row.account_id === "aus_account" ? Number(row.intl_shares_change ?? 0) : summary.intl_shares_change,
          total_assets_krw: summary.valuation_krw + (isAud ? summary.cash_balance_krw : cashEditValue),
        };
      }),
    );
    const dirtyCellKey = buildDirtyCellKey(row.account_id, field);
    setParentDirtyCellKeys((previous) => (previous.includes(dirtyCellKey) ? previous : [...previous, dirtyCellKey]));
    scheduleSilentParentSave(row.account_id);
  }, [scheduleSilentParentSave]);

  const handlePreviewTargetRatioTotalChange = useCallback((accountId: string, total: number) => {
    setPreviewTargetRatioTotals((previous) => {
      if (Math.abs(Number(previous[accountId] ?? NaN) - total) < 0.05) {
        return previous;
      }
      return {
        ...previous,
        [accountId]: total,
      };
    });
  }, []);

  const DetailRenderer = useCallback(
    (params: { data?: ParentGridRow }) => {
      const data = params.data;
      if (!data || !isDetailRow(data)) {
        return null;
      }
      return (
        <AccountHoldingsDetailPanel
          summary={data.summary}
          initialRows={data.rows}
          onReload={load}
          onPreviewTargetRatioTotalChange={handlePreviewTargetRatioTotalChange}
        />
      );
    },
    [handlePreviewTargetRatioTotalChange, load],
  );

  const parentColumns = useMemo<ColDef<ParentGridRow>[]>(() => [
    {
      field: "name",
      headerName: "계좌",
      minWidth: 220,
      flex: 1.2,
      cellRenderer: (params: { data?: ParentGridRow; value?: string }) => {
        const data = params.data;
        if (!data || isDetailRow(data)) {
          return "";
        }
        return (
          <div className="snapshotsExpandCell">
            <span className="snapshotsExpandIcon" aria-hidden="true">
              {data.account_id === expandedId ? "▾" : "▸"}
            </span>
            <span>{data.icon} {params.value}</span>
          </div>
        );
      },
    },
    {
      field: "total_assets_krw",
      headerName: "총 자산",
      minWidth: 132,
      flex: 1,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) =>
        params.data && !isDetailRow(params.data) ? formatKrw(params.value ?? 0) : "",
    },
    {
      field: "valuation_krw",
      headerName: "평가액",
      minWidth: 132,
      flex: 1,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) =>
        params.data && !isDetailRow(params.data) ? formatKrw(params.value ?? 0) : "",
    },
    {
      field: "total_principal",
      headerName: "원금",
      minWidth: 124,
      flex: 1,
      type: "rightAligned",
      editable: (params) => Boolean(params.data && !isDetailRow(params.data)),
      cellClass: (params) => {
        if (!params.data || isDetailRow(params.data)) {
          return undefined;
        }
        return isDirtyParentCell(params.data.account_id, "total_principal")
          ? "assetsEditableCell assetsDirtyCell"
          : "assetsEditableCell";
      },
      valueParser: (params) => {
        const parsed = parseFloat(parseRawPrice(params.newValue));
        if (Number.isNaN(parsed) || parsed < 0) {
          return params.oldValue;
        }
        return parsed;
      },
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) =>
        params.data && !isDetailRow(params.data) ? formatKrw(params.value ?? 0) : "",
    },
    {
      field: "cash_edit_value",
      headerName: "현금",
      minWidth: 124,
      flex: 1,
      type: "rightAligned",
      editable: (params) => Boolean(params.data && !isDetailRow(params.data)),
      cellClass: (params) => {
        if (!params.data || isDetailRow(params.data)) {
          return undefined;
        }
        return isDirtyParentCell(params.data.account_id, "cash_edit_value")
          ? "assetsEditableCell assetsDirtyCell"
          : "assetsEditableCell";
      },
      valueParser: (params) => {
        const parsed = parseFloat(parseRawPrice(params.newValue));
        if (Number.isNaN(parsed) || parsed < 0) {
          return params.oldValue;
        }
        return parsed;
      },
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) => {
        if (!params.data || isDetailRow(params.data)) {
          return "";
        }
        return formatPrice(params.value ?? 0, params.data.currency);
      },
    },
    {
      field: "target_ratio_total",
      headerName: "목표비중합",
      minWidth: 108,
      flex: 0.8,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) => {
        if (!params.data || isDetailRow(params.data)) {
          return "";
        }
        const previewValue = previewTargetRatioTotals[params.data.account_id];
        const value = Number(previewValue ?? params.value ?? 0);
        const colorClass = Math.abs(value - 100) < 0.05 ? "is-success" : "is-danger";
        return <span className={`appHeaderMetricValue ${colorClass}`}>{value.toFixed(1)}%</span>;
      },
    },
    {
      field: "holdings_count",
      headerName: "종목수",
      minWidth: 76,
      flex: 0.6,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) =>
        params.data && !isDetailRow(params.data) ? formatNumber(params.value) : "",
    },
    {
      field: "intl_shares_value",
      headerName: "Intl Value",
      minWidth: 120,
      flex: 0.9,
      type: "rightAligned",
      editable: (params) => Boolean(params.data && !isDetailRow(params.data) && params.data.account_id === "aus_account"),
      cellClass: (params) => {
        if (!params.data || isDetailRow(params.data) || params.data.account_id !== "aus_account") {
          return undefined;
        }
        return isDirtyParentCell(params.data.account_id, "intl_shares_value")
          ? "assetsEditableCell assetsDirtyCell"
          : "assetsEditableCell";
      },
      valueParser: (params) => {
        const parsed = parseFloat(parseRawPrice(params.newValue));
        if (Number.isNaN(parsed) || parsed < 0) {
          return params.oldValue;
        }
        return parsed;
      },
      cellRenderer: (params: { data?: ParentGridRow; value?: number | null }) => {
        if (!params.data || isDetailRow(params.data) || params.data.account_id !== "aus_account") {
          return "-";
        }
        return formatPrice(params.value, "AUD");
      },
    },
    {
      field: "intl_shares_change",
      headerName: "Intl Change",
      minWidth: 124,
      flex: 0.9,
      type: "rightAligned",
      editable: (params) => Boolean(params.data && !isDetailRow(params.data) && params.data.account_id === "aus_account"),
      cellClass: (params) => {
        if (!params.data || isDetailRow(params.data) || params.data.account_id !== "aus_account") {
          return undefined;
        }
        return isDirtyParentCell(params.data.account_id, "intl_shares_change")
          ? "assetsEditableCell assetsDirtyCell"
          : "assetsEditableCell";
      },
      valueParser: (params) => {
        const parsed = parseFloat(parseRawPrice(params.newValue));
        if (Number.isNaN(parsed)) {
          return params.oldValue;
        }
        return parsed;
      },
      cellRenderer: (params: { data?: ParentGridRow; value?: number | null }) => {
        if (!params.data || isDetailRow(params.data) || params.data.account_id !== "aus_account") {
          return "-";
        }
        return <span className={getSignedNullableClass(params.value)}>{formatPrice(params.value, "AUD")}</span>;
      },
    },
  ], [expandedId, isDirtyParentCell, previewTargetRatioTotals]);

  const gridOptions = useMemo<GridOptions<ParentGridRow>>(
    () => ({
      suppressMovableColumns: true,
      ensureDomOrder: true,
      stopEditingWhenCellsLoseFocus: true,
      isFullWidthRow: (params) => isDetailRow(params.rowNode.data),
      fullWidthCellRenderer: DetailRenderer,
      getRowHeight: (params) => (isDetailRow(params.data) ? 722 : 38),
      onCellClicked: (params) => {
        if (!params.data || isDetailRow(params.data)) {
          return;
        }
        if (params.colDef.field !== "name") {
          return;
        }
        const accountId = params.data.account_id;
        setExpandedId((current) => (current === accountId ? null : accountId));
      },
      onCellEditingStarted: (params) => {
        if (params.data && !isDetailRow(params.data)) {
          setEditingParentId(params.data.account_id);
        }
      },
      onCellEditingStopped: () => {
        setEditingParentId(null);
      },
      onCellValueChanged: (params) => {
        if (params.newValue === params.oldValue) {
          return;
        }
        handleParentCellValueChanged(params.data, params.colDef.field);
      },
      rowClassRules: {
        assetsEditingRow: (params) => Boolean(params.data && !isDetailRow(params.data) && params.data.account_id === editingParentId),
        snapshotsExpandedMainRow: (params) =>
          Boolean(params.data && !isDetailRow(params.data) && params.data.account_id === expandedId),
      },
    }),
    [DetailRenderer, editingParentId, expandedId, handleParentCellValueChanged],
  );

  if (loading && !summaries.length) {
    return (
      <div className="appPageStack">
        <div className="appPageLoading">
          <AppLoadingState label="자산 정보를 불러오는 중..." />
        </div>
      </div>
    );
  }

  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection appSectionFill">
        <div className="card appCard shadow-sm appTableCardFill">
          <div className="card-body p-2 appTableCardBodyFill">
            <AppAgGrid
              rowData={parentRows}
              columnDefs={parentColumns}
              loading={loading}
              minHeight="100%"
              className="assetsAgGrid assetsParentAgGrid"
              theme={assetsGridTheme}
              getRowClass={(params: RowClassParams<ParentGridRow>) => {
                if (isDetailRow(params.data)) {
                  return "assetsDetailFullRow";
                }
                return "";
              }}
              gridOptions={gridOptions}
            />
          </div>
        </div>
      </section>
      <div
        dangerouslySetInnerHTML={{
          __html:
            "<style>.assetsInlineLinkButton{font-weight:700;text-decoration:none;} .css-label{font-size:0.7rem;color:#666;display:block;margin-bottom:2px;} @keyframes spin{from{transform:rotate(0deg);}to{transform:rotate(360deg);}}</style>",
        }}
      />
    </div>
  );
}
