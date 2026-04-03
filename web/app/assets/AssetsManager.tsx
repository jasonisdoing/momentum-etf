"use client";

import { useEffect, useMemo, useState, useTransition, useCallback, useRef } from "react";
import { iconSetQuartzBold, themeQuartz } from "ag-grid-community";
import type { ColDef, RowClassParams } from "ag-grid-community";
import {
  IconPlus,
  IconTrash,
  IconLoader2,
  IconCheck,
} from "@tabler/icons-react";

import { AppAgGrid } from "../components/AppAgGrid";
import { AppLoadingState } from "../components/AppLoadingState";
import { useToast } from "../components/ToastProvider";
import {
  readRememberedMomentumEtfAccountId,
  writeRememberedMomentumEtfAccountId,
} from "../components/account-selection";

// --- Types ---
type AccountConfig = {
  account_id: string;
  name: string;
  icon: string;
};

type CashInfo = {
  account_id: string;
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
};

type HoldingsRow = {
  account_name: string;
  currency: string;
  bucket: string;
  bucket_id: number;
  ticker: string;
  name: string;
  quantity: number;
  average_buy_price: number;
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
};

type GridRow = HoldingsRow & { id: string };

type AddingRowState = {
  ticker: string;
  quantity: string;
  average_buy_price: string;
  target_ratio: string;
  isValidatingTicker?: boolean;
  validationError?: string;
  name?: string;
  bucketId?: number;
  isValidated?: boolean;
};

type CashEditingState = {
  total_principal: string;
  cash_value: string;
  intl_shares_value: string;
  intl_shares_change: string;
};

function buildGridRowId(row: Pick<HoldingsRow, "ticker" | "account_name">, index: number): string {
  return `${row.ticker}-${row.account_name}-${index}`;
}

function buildDirtyCellKey(rowId: string, field: string): string {
  return `${rowId}::${field}`;
}

const assetsGridTheme = themeQuartz
  .withPart(iconSetQuartzBold)
  .withParams({
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

// --- Helpers ---
function formatKrw(value: number): string {
  return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
}

function formatPrice(value: number | null, currency: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (currency === "AUD") {
    return `A$${new Intl.NumberFormat("en-AU", { minimumFractionDigits: 2, maximumFractionDigits: 4 }).format(value)}`;
  }
  if (currency === "USD") {
    return `$${new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 4 }).format(value)}`;
  }
  return `${new Intl.NumberFormat("ko-KR").format(value)}원`;
}

function formatNumber(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 2 }).format(value);
}

function formatTargetAmount(value: number | null | undefined, currency: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (currency === "AUD") {
    return `A$${new Intl.NumberFormat("en-AU", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)}`;
  }
  return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
}

function formatTargetQuantity(value: number | null | undefined, currency: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (currency === "AUD") {
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
  if (currency === "AUD") {
    return `${sign}${new Intl.NumberFormat("en-AU", {
      minimumFractionDigits: 0,
      maximumFractionDigits: 4,
    }).format(changeQuantity)}`;
  }
  return `${sign}${new Intl.NumberFormat("ko-KR", {
    maximumFractionDigits: 0,
  }).format(changeQuantity)}`;
}

function getSignedNullableClass(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) {
    return "";
  }
  return value > 0 ? "metricPositive" : "metricNegative";
}

function getSignedClass(value: number): string {
  if (value === 0) return "";
  return value > 0 ? "metricPositive" : "metricNegative";
}

function getBucketCellClass(bucketId: number): string {
  if (!bucketId) return "appBucketCell";
  return `appBucketCell appBucketCell${bucketId}`;
}

function parseRawPrice(formatted: unknown): string {
  if (formatted === null || formatted === undefined) return "0";
  // A$, $, ₩, 원, 콤마 제거
  return String(formatted).replace(/A\$|\$|₩|원|,|\s/g, "");
}

function safeParseFloat(val: unknown): number {
  const cleaned = parseRawPrice(val);
  const parsed = parseFloat(cleaned);
  return isNaN(parsed) ? 0 : parsed;
}

function parseEditableQuantity(value: unknown): number {
  const parsed = parseInt(parseRawPrice(value), 10);
  return isNaN(parsed) ? 0 : parsed;
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

  const currentQuantity = Number(row.quantity ?? 0);
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

  const currentQuantity = Number(row.quantity ?? 0);
  const currentAverageBuyPrice = safeParseFloat(row.average_buy_price);
  if (currentQuantity > 0 && currentAverageBuyPrice > 0) {
    const fxFactor = Number(row.buy_amount_krw ?? 0) / (currentQuantity * currentAverageBuyPrice);
    return averageBuyPrice * quantity * fxFactor;
  }

  return Number(row.buy_amount_krw ?? 0);
}

function getCurrentPriceNumber(row: GridRow): number {
  const currentPriceNum = Number(row.current_price_num ?? NaN);
  if (!Number.isNaN(currentPriceNum) && currentPriceNum > 0) {
    return currentPriceNum;
  }
  return safeParseFloat(row.current_price);
}

function getPreviewTargetRatio(row: GridRow): number | null {
  const parsed = Number(row.target_ratio ?? null);
  if (Number.isNaN(parsed) || parsed < 0) {
    return 0;
  }
  return parsed;
}

function roundTargetQuantity(quantity: number, currency: string): number {
  if (String(currency || "KRW").toUpperCase() === "AUD") {
    return Math.round(quantity * 10000) / 10000;
  }
  return Math.max(Math.floor(quantity), 0);
}

function getPreviewTargetAmount(row: GridRow, totalAssetsNative: number): number | null {
  const targetRatio = getPreviewTargetRatio(row);
  if (targetRatio === null) {
    return null;
  }
  return Math.round(totalAssetsNative * (targetRatio / 100) * 100) / 100;
}

function getPreviewTargetQuantity(row: GridRow, totalAssetsNative: number): number | null {
  const targetAmount = getPreviewTargetAmount(row, totalAssetsNative);
  if (targetAmount === null) {
    return null;
  }

  const currentPrice = getCurrentPriceNumber(row);
  if (currentPrice <= 0) {
    return null;
  }

  return roundTargetQuantity(targetAmount / currentPrice, row.currency);
}

// --- Components ---
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
  style?: React.CSSProperties;
  placeholder?: string;
  autoFocus?: boolean;
  disabled?: boolean;
}) {
  const [localValue, setLocalValue] = useState(initialValue);

  useEffect(() => {
    if (initialValue !== localValue) setLocalValue(initialValue);
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
      onChange={(e) => {
        setLocalValue(e.target.value);
        onChange?.(e.target.value);
      }}
      onKeyDown={(e) => {
        if (e.nativeEvent.isComposing) return;
        if (e.key === "Enter") onSave?.(localValue);
        else if (e.key === "Escape") { setLocalValue(initialValue); onCancel?.(); }
      }}
      onBlur={() => { if (localValue !== initialValue) onSave?.(localValue); }}
    />
  );
}

let holdingsDataCache: Record<string, any> = {};

export function AssetsManager() {
  const [accounts, setAccounts] = useState<AccountConfig[]>(holdingsDataCache["__ACCOUNTS__"]?.accounts ?? []);
  const [selectedAccountId, setSelectedAccountId] = useState(readRememberedMomentumEtfAccountId() ?? "");
  const [rows, setRows] = useState<any[]>(holdingsDataCache[readRememberedMomentumEtfAccountId() ?? ""]?.rows ?? []);
  const [cash, setCash] = useState<CashInfo | null>(null);
  const [loading, setLoading] = useState(!holdingsDataCache[readRememberedMomentumEtfAccountId() ?? ""]);
  const [addingRow, setAddingRow] = useState<AddingRowState | null>(null);
  const [cashEditing, setCashEditing] = useState<CashEditingState | null>(null);
  const [editingRowId, setEditingRowId] = useState<string | null>(null);
  const [dirtyRowIds, setDirtyRowIds] = useState<string[]>([]);
  const [dirtyCellKeys, setDirtyCellKeys] = useState<string[]>([]);
  const [selectedRowIds, setSelectedRowIds] = useState<string[]>([]);
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [savingDirtyRows, setSavingDirtyRows] = useState(false);
  const [, startTransition] = useTransition();
  const toast = useToast();

  const load = useCallback(async (accountId: string | null = null, silent = false) => {
    const targetId = accountId ?? selectedAccountId;
    if (!targetId && !silent) return;

    if (!silent) setLoading(true);

    try {
      const search = targetId ? `?account=${encodeURIComponent(targetId)}` : "";
      const response = await fetch(`/api/assets${search}`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? "로드 실패");

      setAccounts(payload.accounts ?? []);
      setRows(payload.rows ?? []);
      setCash(payload.cash ?? null);
      setDirtyRowIds([]);
      setDirtyCellKeys([]);
      setSelectedRowIds([]);
      setEditingRowId(null);

      holdingsDataCache[payload.account_id ?? ""] = payload;
      holdingsDataCache["__ACCOUNTS__"] = { accounts: payload.accounts ?? [] };

      if (accountId === null) {
        setSelectedAccountId(payload.account_id ?? "");
        writeRememberedMomentumEtfAccountId(payload.account_id ?? "");
      }
    } catch (err: any) {
      toast.error(err.message || "보유 종목 로드 실패");
    } finally {
      if (!silent) setLoading(false);
    }
  }, [selectedAccountId, toast]);

  useEffect(() => { void load(readRememberedMomentumEtfAccountId() ?? ""); }, []);

  const handleAccountChange = (nextId: string) => {
    setSelectedAccountId(nextId);
    writeRememberedMomentumEtfAccountId(nextId);
    setEditingRowId(null);
    setDirtyRowIds([]);
    setDirtyCellKeys([]);
    setSelectedRowIds([]);
    setAddingRow(null);
    setCashEditing(null);
    void load(nextId);
  };

  const handleDeleteSelected = useCallback(() => {
    if (!selectedRowIds.length) {
      return;
    }

    const selectedRows = rows
      .map((row, index) => ({
        ...row,
        id: buildGridRowId(row, index),
      }))
      .filter((row) => selectedRowIds.includes(row.id));
    if (!selectedRows.length) {
      return;
    }

    const summary =
      selectedRows.length === 1
        ? `${selectedRows[0].name}(${selectedRows[0].ticker}) 종목을 삭제하시겠습니까?`
        : `${selectedRows.length}개 종목을 삭제하시겠습니까?`;

    if (!confirm(summary)) {
      return;
    }

    setProcessingId("__deleting__");
    startTransition(async () => {
      try {
        for (const row of selectedRows) {
          const params = new URLSearchParams({ account: selectedAccountId, ticker: row.ticker.replace("ASX:", "") });
          const res = await fetch(`/api/assets?${params.toString()}`, { method: "DELETE" });
          if (!res.ok) {
            throw new Error("삭제 실패");
          }
        }
        delete holdingsDataCache[selectedAccountId];
        await load(selectedAccountId, true);
        setSelectedRowIds([]);
        toast.success("삭제 완료");
      } catch {
        toast.error("삭제 실패");
      } finally {
        setProcessingId(null);
      }
    });
  }, [load, rows, selectedAccountId, selectedRowIds, toast]);

  const processRowUpdate = async (newRow: GridRow, options?: { reloadAfterSave?: boolean; showToast?: boolean }) => {
    if (newRow.id === "__adding__") return newRow;
    const reloadAfterSave = options?.reloadAfterSave ?? true;
    const showToast = options?.showToast ?? true;
    const quantity = parseInt(String(newRow.quantity), 10);
    const avgPrice = safeParseFloat(newRow.average_buy_price);
    const targetRatio = Number(newRow.target_ratio ?? 0);

    if (isNaN(quantity) || quantity < 0 || isNaN(avgPrice) || avgPrice < 0) {
      toast.error("입력값이 올바르지 않습니다.");
      throw new Error("Invalid Input");
    }
    if (Number.isNaN(targetRatio) || targetRatio < 0 || targetRatio > 100) {
      toast.error("목표비중은 0%에서 100% 사이여야 합니다.");
      throw new Error("Invalid Target Ratio");
    }

    setProcessingId(newRow.id);
    try {
      const normalizedTicker = newRow.ticker.replace("ASX:", "");
      const normalizedTargetRatio = parseFloat(targetRatio.toFixed(1));
      const holdingsResponse = await fetch("/api/assets", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          account_id: selectedAccountId,
          ticker: normalizedTicker,
          quantity,
          average_buy_price: avgPrice,
          target_ratio: normalizedTargetRatio,
        }),
      });
      if (!holdingsResponse.ok) {
        throw new Error("보유 종목 수정에 실패했습니다.");
      }

      delete holdingsDataCache[selectedAccountId];
      setEditingRowId(null);
      if (reloadAfterSave) {
        startTransition(() => { void load(selectedAccountId, true); });
      }
      if (showToast) {
        toast.success("수정 완료");
      }
      return { ...newRow, quantity, average_buy_price: avgPrice, target_ratio: normalizedTargetRatio };
    } catch (error) {
      if (showToast) {
        toast.error(error instanceof Error ? error.message : "수정 실패");
      }
      throw new Error("Fail");
    }
    finally { setProcessingId(null); }
  };

  const handleValidateTicker = (tickerToUse?: string) => {
    const tickerStr = tickerToUse || addingRow?.ticker;
    if (!tickerStr || addingRow?.isValidatingTicker) return; // 중복 요청 차단

    setAddingRow(p => p ? { ...p, ticker: tickerStr, isValidatingTicker: true } : null);
    startTransition(async () => {
      try {
        const res = await fetch("/api/assets", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "validate", account_id: selectedAccountId, ticker: tickerStr }),
        });
        const data = await res.json();
        if (!res.ok) {
          toast.error(data.error || "검증 실패");
          setAddingRow(p => p ? { ...p, isValidatingTicker: false } : null);
          return;
        }
        setAddingRow(p => p ? { ...p, isValidatingTicker: false, isValidated: true, name: data.name, bucketId: data.bucket_id, ticker: data.ticker } : null);
        toast.success(`조회 성공: ${data.name}`);
      } catch {
        toast.error("오류 발생");
        setAddingRow(p => p ? { ...p, isValidatingTicker: false } : null);
      }
    });
  };

  // 입력 필드 직접 참조를 위한 Ref (상태 엇박자 방지)
  const qtyRef = useRef<HTMLInputElement>(null);
  const priceRef = useRef<HTMLInputElement>(null);
  const targetRatioRef = useRef<HTMLInputElement>(null);

  const processAddingRow = useCallback(async (options?: { reloadAfterSave?: boolean; showToast?: boolean }) => {
    if (!addingRow?.isValidated) {
      toast.error("먼저 종목 확인 버튼을 눌러주세요.");
      throw new Error("Add row not validated");
    }

    const reloadAfterSave = options?.reloadAfterSave ?? true;
    const showToast = options?.showToast ?? true;

    const rawQty = qtyRef.current?.value ?? "";
    const rawPrice = priceRef.current?.value ?? "";
    const rawTargetRatio = targetRatioRef.current?.value ?? "0";

    const quantity = parseInt(parseRawPrice(rawQty), 10);
    const avgPrice = safeParseFloat(rawPrice);
    const targetRatio = parseFloat(rawTargetRatio);

    const isQtyValid = !isNaN(quantity) && quantity >= 0 && rawQty !== "";
    const isPriceValid = !isNaN(avgPrice) && avgPrice >= 0 && rawPrice !== "";
    const isTargetRatioValid = !isNaN(targetRatio) && targetRatio >= 0 && targetRatio <= 100;

    if (!isQtyValid || !isPriceValid || !isTargetRatioValid) {
      const dbg = `[수량:'${rawQty}', 단가:'${rawPrice}', 목표비중:'${rawTargetRatio}']`;
      let detail = "";
      if (!isQtyValid) detail += "수량 ";
      if (!isPriceValid) detail += "단가 ";
      if (!isTargetRatioValid) detail += "목표비중 ";
      toast.error(`${detail.trim()} 형식이 올바르지 않습니다. ${dbg}`);
      throw new Error("Invalid add row");
    }

    setProcessingId("__adding__");
    try {
      const res = await fetch("/api/assets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          account_id: selectedAccountId,
          ticker: addingRow.ticker,
          quantity,
          average_buy_price: avgPrice,
          target_ratio: parseFloat(targetRatio.toFixed(1)),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "추가 실패");

      setAddingRow(null);
      delete holdingsDataCache[selectedAccountId];
      if (reloadAfterSave) {
        await load(selectedAccountId, true);
      }
      if (showToast) {
        toast.success(`${addingRow.ticker} 추가 완료`);
      }
    } catch (err: any) {
      if (showToast) {
        toast.error(err.message || "종목 추가에 실패했습니다.");
      }
      throw err instanceof Error ? err : new Error("종목 추가에 실패했습니다.");
    } finally {
      setProcessingId(null);
    }
  }, [addingRow?.isValidated, addingRow?.ticker, selectedAccountId, load, toast]);

  const handleCashSave = () => {
    if (!cashEditing || !cash) return;
    startTransition(async () => {
      try {
        const isKrw = cash.currency === "KRW";
        const cashValue = parseFloat(cashEditing.cash_value) || 0;
        const res = await fetch("/api/assets", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: cash.account_id,
            total_principal: parseFloat(cashEditing.total_principal) || 0,
            cash_balance_krw: isKrw ? cashValue : 0,
            cash_balance_native: cashValue,
            cash_currency: cash.cash_currency,
            intl_shares_value: parseFloat(cashEditing.intl_shares_value) || 0,
            intl_shares_change: parseFloat(cashEditing.intl_shares_change) || 0,
          }),
        });
        if (!res.ok) throw new Error();
        setCashEditing(null);
        delete holdingsDataCache[selectedAccountId];
        await load(selectedAccountId, true);
        toast.success("저장 완료");
      } catch { toast.error("저장 실패"); }
    });
  };

  const isEditableHoldingRow = useCallback(
    (row: GridRow | undefined | null) => Boolean(row && row.id !== "__adding__" && row.ticker !== "IS"),
    [],
  );

  const handleCellValueChanged = useCallback((row: GridRow | undefined, field: string | undefined) => {
    if (!row || !isEditableHoldingRow(row)) {
      return;
    }

    setRows((prev) =>
      prev.map((currentRow, index) => {
        if (buildGridRowId(currentRow, index) !== row.id) {
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
    setDirtyRowIds((prev) => (prev.includes(row.id) ? prev : [...prev, row.id]));
    if (field) {
      const dirtyCellKey = buildDirtyCellKey(row.id, field);
      setDirtyCellKeys((prev) => (prev.includes(dirtyCellKey) ? prev : [...prev, dirtyCellKey]));
    }
  }, [isEditableHoldingRow]);

  const handleSaveChanges = useCallback(() => {
    if ((!dirtyRowIds.length && !addingRow) || savingDirtyRows) {
      return;
    }

    setSavingDirtyRows(true);
    startTransition(async () => {
      try {
        if (addingRow) {
          await processAddingRow({ reloadAfterSave: false, showToast: false });
        }

        const currentGridRows = rows
          .map((row, index) => ({
            ...row,
            id: buildGridRowId(row, index),
            quantity: typeof row.quantity === "number" ? row.quantity : parseInt(String(row.quantity), 10) || 0,
            average_buy_price: safeParseFloat(row.average_buy_price),
            target_ratio: row.target_ratio ?? 0,
          }))
          .filter((row) => dirtyRowIds.includes(row.id));

        for (const row of currentGridRows) {
          await processRowUpdate(row as GridRow, { reloadAfterSave: false, showToast: false });
        }

        delete holdingsDataCache[selectedAccountId];
        await load(selectedAccountId, true);
        setDirtyRowIds([]);
        setDirtyCellKeys([]);
        setSelectedRowIds([]);
        toast.success("변경사항 저장 완료");
      } catch {
        delete holdingsDataCache[selectedAccountId];
        await load(selectedAccountId, true);
        toast.error("변경사항 저장에 실패했습니다.");
      } finally {
        setSavingDirtyRows(false);
      }
    });
  }, [addingRow, dirtyRowIds, load, processAddingRow, processRowUpdate, rows, savingDirtyRows, selectedAccountId, toast]);

  const accountCurrency = useMemo(
    () => String(cash?.currency || rows[0]?.currency || "KRW").trim().toUpperCase(),
    [cash?.currency, rows],
  );

  const accountTotalAssetsNative = useMemo(() => {
    if (accountCurrency === "AUD") {
      const holdingsNative = rows.reduce(
        (sum, row) => sum + (getCurrentPriceNumber(row as GridRow) * Number(row.quantity ?? 0)),
        0,
      );
      return holdingsNative + Number(cash?.cash_balance_native ?? 0);
    }

    const holdingsKrw = rows.reduce((sum, row) => sum + Number(row.valuation_krw ?? 0), 0);
    return holdingsKrw + Number(cash?.cash_balance_krw ?? 0);
  }, [accountCurrency, cash?.cash_balance_krw, cash?.cash_balance_native, rows]);

  const gridRows = useMemo<GridRow[]>(() => {
    const baseRows = rows.map((row, i) => ({
      ...row,
      id: buildGridRowId(row, i),
      quantity: typeof row.quantity === "number" ? row.quantity : parseInt(String(row.quantity), 10) || 0,
      average_buy_price: safeParseFloat(row.average_buy_price),
      target_ratio: row.target_ratio ?? 0,
    })).sort((a, b) => {
      if (a.bucket_id !== b.bucket_id) {
        return a.bucket_id - b.bucket_id;
      }
      if (a.weight_pct !== b.weight_pct) {
        return b.weight_pct - a.weight_pct;
      }
      const dailyA = a.daily_change_pct ?? -999;
      const dailyB = b.daily_change_pct ?? -999;
      if (dailyA !== dailyB) {
        return dailyB - dailyA;
      }
      return a.ticker.localeCompare(b.ticker);
    });

    if (addingRow) {
      return [{
        id: "__adding__", account_name: "", currency: "", bucket: "", bucket_id: 0,
        ticker: addingRow.ticker, name: addingRow.name || "",
        quantity: 0, average_buy_price: 0,
        current_price: "-", days_held: "-", pnl_krw: 0, return_pct: 0, weight_pct: 0, buy_amount_krw: 0, valuation_krw: 0, target_ratio: 0,
      } as GridRow, ...baseRows];
    }
    return baseRows;
  }, [rows, addingRow?.ticker, addingRow?.name]);

  const hasDirtyChanges = dirtyRowIds.length > 0;
  const hasPendingAdd = Boolean(addingRow);
  const hasSelectedRows = selectedRowIds.length > 0;
  const isDirtyEditableCell = useCallback(
    (rowId: string | undefined, field: string) => Boolean(rowId && dirtyCellKeys.includes(buildDirtyCellKey(rowId, field))),
    [dirtyCellKeys],
  );

  const columns = useMemo<ColDef<GridRow>[]>(() => [
    {
      colId: "select",
      headerName: "",
      width: 52,
      sortable: false,
      resizable: false,
      pinned: "left",
      checkboxSelection: (params) => Boolean(params.data && params.data.id !== "__adding__" && params.data.ticker !== "IS"),
      headerCheckboxSelection: true,
      showDisabledCheckboxes: false,
      cellClass: "assetsSelectCell",
    },
    { field: "bucket", headerName: "버킷", width: 96, cellClass: (params) => getBucketCellClass(params.data?.bucket_id ?? 0) },
    {
      field: "ticker",
      headerName: "종목코드",
      width: 118,
      cellRenderer: (params: { data: GridRow; value: string }) => {
        const row = params.data;
        if (row?.id === "__adding__") {
          if (addingRow?.isValidated) {
            return (
              <div className="d-flex gap-2 align-items-center">
                <span>{addingRow.ticker}</span>
                <button className="btn btn-sm btn-link p-0 assetsInlineLinkButton" onClick={() => setAddingRow((prev) => prev ? { ...prev, ticker: "", isValidated: false, name: "" } : null)}>
                  변경
                </button>
              </div>
            );
          }
          return (
            <div className="assetsTickerLookup">
              <StableInlineInput className="form-control form-control-sm assetsInlineInput assetsInlineInputTicker" initialValue={addingRow?.ticker ?? ""} onSave={handleValidateTicker} />
              <button className="btn btn-outline-primary btn-sm assetsInlineButton" onClick={() => handleValidateTicker()}>확인</button>
            </div>
          );
        }
        return <span className="appCodeText">{params.value}</span>;
      },
    },
    { field: "name", headerName: "종목명", minWidth: 280, flex: 1.6 },
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
        const raw = parseRawPrice(params.newValue);
        const parsed = parseInt(raw, 10);
        if (Number.isNaN(parsed) || parsed < 0) {
          return params.oldValue;
        }
        return parsed;
      },
      cellRenderer: (params: { data: GridRow; value: number }) => {
        const row = params.data;
        if (!row) return null;
        if (row.id === "__adding__") {
          return <input type="number" step="1" ref={qtyRef} className="form-control form-control-sm assetsInlineInput" defaultValue="0" disabled={!addingRow?.isValidated} />;
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
        const raw = parseRawPrice(params.newValue);
        const parsed = parseFloat(raw);
        if (Number.isNaN(parsed) || parsed < 0) {
          return params.oldValue;
        }
        return parsed;
      },
      cellRenderer: (params: { data: GridRow; value: number }) => {
        const row = params.data;
        if (!row) return null;
        if (row.id === "__adding__") {
          return <input type="number" step="any" ref={priceRef} className="form-control form-control-sm assetsInlineInput" defaultValue="0" disabled={!addingRow?.isValidated} />;
        }
        return <span>{formatPrice(safeParseFloat(params.value), row.currency || "KRW")}</span>;
      },
    },
    {
      field: "weight_pct",
      headerName: "비중",
      width: 80,
      type: "rightAligned",
      cellRenderer: (params: { value: number }) => <span style={{ color: "#0d6efd" }}>{params.value?.toFixed(1)}%</span>,
    },
    {
      field: "return_pct",
      headerName: "수익률",
      width: 88,
      type: "rightAligned",
      cellRenderer: (params: { value: number }) => <span className={getSignedClass(params.value ?? 0)}>{(params.value ?? 0) > 0 ? "+" : ""}{params.value?.toFixed(2)}%</span>,
    },
    {
      field: "daily_change_pct",
      headerName: "일간(%)",
      width: 92,
      type: "rightAligned",
      cellRenderer: (params: { value: number | null }) => <span className={getSignedClass(params.value ?? 0)}>{params.value === null || params.value === undefined ? "-" : `${(params.value ?? 0) > 0 ? "+" : ""}${params.value.toFixed(2)}%`}</span>,
    },
    {
      field: "current_price",
      headerName: "현재가",
      width: 116,
      type: "rightAligned",
      cellRenderer: (params: { data: GridRow; value: string }) => <span>{formatPrice(safeParseFloat(params.value), params.data?.currency || "KRW")}</span>,
    },
    {
      field: "pnl_krw",
      headerName: "평가손익",
      width: 136,
      type: "rightAligned",
      cellRenderer: (params: { value: number }) => <span className={getSignedClass(params.value ?? 0)}>{formatKrw(params.value ?? 0)}</span>,
    },
    {
      field: "valuation_krw",
      headerName: "평가 금액",
      width: 136,
      type: "rightAligned",
      cellRenderer: (params: { data: GridRow }) => formatKrw(getPreviewValuationKrw(params.data)),
    },
    {
      field: "buy_amount_krw",
      headerName: "매입 금액",
      width: 136,
      type: "rightAligned",
      cellRenderer: (params: { data: GridRow }) => formatKrw(getPreviewBuyAmountKrw(params.data)),
    },
    { field: "days_held", headerName: "보유일", width: 76 },
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
      cellRenderer: (params: { data: GridRow; value: number | null }) => {
        const row = params.data;
        if (!row) return null;
        if (row.id === "__adding__") {
          return <input type="number" step="0.1" min="0" max="100" ref={targetRatioRef} className="form-control form-control-sm assetsInlineInput" defaultValue={addingRow?.target_ratio ?? "0"} disabled={!addingRow?.isValidated} />;
        }
        return <span style={{ color: "#0d6efd", fontWeight: 700 }}>{params.value === null || params.value === undefined ? "-" : `${params.value.toFixed(1)}%`}</span>;
      },
    },
    {
      field: "target_quantity",
      headerName: "목표수량",
      width: 124,
      type: "rightAligned",
      cellRenderer: (params: { data: GridRow; value: number | null }) => (
        <span>{formatTargetQuantity(getPreviewTargetQuantity(params.data, accountTotalAssetsNative), params.data?.currency || "KRW")}</span>
      ),
    },
    {
      colId: "expected_change_quantity",
      headerName: "예상변경수량",
      width: 138,
      type: "rightAligned",
      sortable: false,
      cellRenderer: (params: { data: GridRow }) => (
        <span className={getSignedNullableClass(
          (() => {
            const targetQuantity = getPreviewTargetQuantity(params.data, accountTotalAssetsNative);
            if (targetQuantity === null || targetQuantity === undefined) {
              return null;
            }
            return targetQuantity - getPreviewQuantity(params.data);
          })(),
        )}>
          {formatExpectedChangeQuantity(
            getPreviewTargetQuantity(params.data, accountTotalAssetsNative),
            getPreviewQuantity(params.data),
            params.data?.currency || "KRW",
          )}
        </span>
      ),
    },
    {
      field: "target_amount",
      headerName: "목표금액",
      width: 136,
      type: "rightAligned",
      cellRenderer: (params: { data: GridRow; value: number | null }) => (
        <span>{formatTargetAmount(getPreviewTargetAmount(params.data, accountTotalAssetsNative), params.data?.currency || "KRW")}</span>
      ),
    },
  ], [
    accountTotalAssetsNative,
    addingRow,
    isDirtyEditableCell,
    isEditableHoldingRow,
    processingId,
  ]);

  if (loading && !rows.length) return <div className="appPageStack"><div className="appPageLoading"><AppLoadingState label="보유 종목 로드 중..." /></div></div>;

  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection appSectionFill">
        <div className="card appCard shadow-sm appTableCardFill">
          <div className="card-header d-flex justify-content-between align-items-center bg-white py-3 flex-shrink-0">
            <div className="d-flex gap-2">
              <select className="form-select w-auto fw-bold" value={selectedAccountId} onChange={(e) => handleAccountChange(e.target.value)}>
                {accounts.map(a => <option key={a.account_id} value={a.account_id}>{a.icon} {a.name}</option>)}
              </select>
            </div>
            {(() => {
              const totalValuation = rows.reduce((s, r) => s + (r.valuation_krw || 0), 0);
              const totalCash = cash?.cash_balance_krw || 0;
              const totalAssets = totalValuation + totalCash;
              const targetRatioTotal = rows.reduce((s, r) => s + (r.target_ratio || 0), 0);
              const valuationPct = totalAssets > 0 ? (totalValuation / totalAssets * 100).toFixed(1) : "0.0";
              const cashPct = totalAssets > 0 ? (totalCash / totalAssets * 100).toFixed(1) : "0.0";
              
              return (
                <div className="d-flex gap-5 align-items-center">
                  <div className="fw-bold fs-3">총 자산: <span className="fw-bold text-primary fs-2 ms-2">{formatKrw(totalAssets)}</span></div>
                  <div className="fw-bold fs-3">평가액: <span className="fw-bold text-dark fs-2 ms-2">{formatKrw(totalValuation)} <span className="text-secondary ms-2">({valuationPct}%)</span></span></div>
                  <div className="fw-bold fs-3">현금: <span className="fw-bold text-dark fs-2 ms-2">{formatKrw(totalCash)} <span className="text-secondary ms-2">({cashPct}%)</span></span></div>
                  <div className="fw-bold fs-3">목표비중합: <span className={`fw-bold fs-2 ms-2 ${Math.abs(targetRatioTotal - 100) < 0.05 ? "text-success" : "text-danger"}`}>{targetRatioTotal.toFixed(1)}%</span></div>
                </div>
              );
            })()}
          </div>
          {cash && (
            <div className="px-3 py-2 border-bottom bg-light flex-shrink-0">
              {cashEditing ? (
                <div className="d-flex gap-3 align-items-end">
                  <div><label className="css-label">투자 원금</label><input type="number" className="form-control form-control-sm" value={cashEditing.total_principal} onChange={e => setCashEditing({...cashEditing, total_principal: e.target.value})} /></div>
                  <div><label className="css-label">보유 현금</label><input type="number" className="form-control form-control-sm" value={cashEditing.cash_value} onChange={e => setCashEditing({...cashEditing, cash_value: e.target.value})} /></div>
                  <div className="d-flex gap-1"><button className="btn btn-primary btn-sm" onClick={handleCashSave}>저장</button><button className="btn btn-outline-secondary btn-sm" onClick={() => setCashEditing(null)}>취소</button></div>
                </div>
              ) : (
                <div className="d-flex gap-5 align-items-center">
                  <div className="fw-bold fs-3">투자원금: <span className="fw-bold text-dark fs-2 ms-2">{formatNumber(cash.total_principal)}원</span></div>
                  <div className="fw-bold fs-3">보유현금: <span className="fw-bold text-dark fs-2 ms-2">{formatPrice(cash.currency === "KRW" ? cash.cash_balance_krw : cash.cash_balance_native, cash.currency)}</span></div>
                  <button className="btn btn-outline-secondary btn-sm ms-3 fw-bold" onClick={() => { if (cash) setCashEditing({ total_principal: String(cash.total_principal), cash_value: String(cash.currency === "KRW" ? cash.cash_balance_krw : cash.cash_balance_native), intl_shares_value: String(cash.intl_shares_value), intl_shares_change: String(cash.intl_shares_change) }); }}>정보 수정</button>
                  <div className="ms-auto d-flex align-items-center gap-2">
                    <button
                      className="btn btn-primary btn-sm px-3 fw-bold"
                      onClick={() => setAddingRow({ ticker: "", quantity: "", average_buy_price: "", target_ratio: "0", isValidated: false })}
                      disabled={hasPendingAdd}
                    >
                      <IconPlus size={16} /> 추가
                    </button>
                    <button
                      className="btn btn-success btn-sm px-3 fw-bold"
                      onClick={handleSaveChanges}
                      disabled={(!hasDirtyChanges && !hasPendingAdd) || savingDirtyRows || processingId === "__deleting__"}
                    >
                      {savingDirtyRows ? <IconLoader2 size={16} style={{ animation: "spin 1s linear infinite" }} /> : <IconCheck size={16} />} 저장
                    </button>
                    <button
                      className="btn btn-outline-danger btn-sm px-3 fw-bold"
                      onClick={handleDeleteSelected}
                      disabled={!hasSelectedRows || savingDirtyRows || processingId === "__deleting__"}
                    >
                      {processingId === "__deleting__" ? <IconLoader2 size={16} style={{ animation: "spin 1s linear infinite" }} /> : <IconTrash size={16} />} 삭제
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
          <div className="card-body p-2 appTableCardBodyFill">
            <AppAgGrid
              rowData={gridRows}
              columnDefs={columns}
              loading={loading}
              minHeight="100%"
              className="assetsAgGrid"
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
                rowSelection: "multiple",
                suppressRowClickSelection: true,
                ensureDomOrder: true,
                stopEditingWhenCellsLoseFocus: true,
                isRowSelectable: (params) => Boolean(params.data && params.data.id !== "__adding__" && params.data.ticker !== "IS"),
                onSelectionChanged: (params) => {
                  setSelectedRowIds(
                    params.api
                      .getSelectedRows()
                      .map((row) => row.id)
                      .filter((rowId): rowId is string => Boolean(rowId)),
                  );
                },
                onCellEditingStarted: (params) => {
                  const row = params.data;
                  if (row && isEditableHoldingRow(row)) {
                    setEditingRowId(row.id);
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
                rowClassRules: {
                  assetsAddingRow: (params) => params.data?.id === "__adding__",
                  assetsEditingRow: (params) => Boolean(params.data?.id && params.data.id === editingRowId),
                },
              }}
            />
          </div>
        </div>
      </section>
      <div dangerouslySetInnerHTML={{ __html: `<style> .css-label { font-size: 0.7rem; color: #666; display: block; margin-bottom: 2px; } @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } } </style>` }} />
    </div>
  );
}
