"use client";

import { useEffect, useMemo, useState, useTransition, useCallback, useRef } from "react";
import {
  type GridRowModesModel,
  GridRowModes,
  GridActionsCellItem,
  type GridColDef,
  type GridRenderCellParams,
  GridRowEditStopReasons,
  type GridEventListener,
} from "@mui/x-data-grid";
import {
  IconPlus,
  IconPencil,
  IconDeviceFloppy,
  IconTrash,
  IconX,
  IconLoader2,
  IconCheck,
} from "@tabler/icons-react";

import { AppDataGrid } from "../components/AppDataGrid";
import { AppLoadingState } from "../components/AppLoadingState";
import { useToast } from "../components/ToastProvider";
import {
  readRememberedMomentumEtfAccountId,
  writeRememberedMomentumEtfAccountId,
} from "../components/account-selection";

// --- Hooks ---
function useEffectRef<T>(value: T) {
  const ref = useRef(value);
  useEffect(() => {
    ref.current = value;
  }, [value]);
  return ref;
}

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
  days_held: string;
  pnl_krw: number;
  return_pct: number;
  weight_pct: number;
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
  const [rowModesModel, setRowModesModel] = useState<GridRowModesModel>({});
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
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
    setRowModesModel({});
    setAddingRow(null);
    setCashEditing(null);
    void load(nextId);
  };

  const handleDelete = (ticker: string, id: string) => {
    if (!confirm(`${ticker} 종목을 삭제하시겠습니까?`)) return;
    setProcessingId(id);
    startTransition(async () => {
      try {
        const params = new URLSearchParams({ account: selectedAccountId, ticker: ticker.replace("ASX:", "") });
        const res = await fetch(`/api/assets?${params.toString()}`, { method: "DELETE" });
        if (!res.ok) throw new Error("삭제 실패");
        delete holdingsDataCache[selectedAccountId];
        await load(selectedAccountId, true);
        toast.success("삭제 완료");
      } catch { toast.error("삭제 실패"); }
      finally { setProcessingId(null); }
    });
  };

  const processRowUpdate = async (newRow: GridRow) => {
    if (newRow.id === "__adding__") return newRow;
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
      startTransition(() => { void load(selectedAccountId, true); });
      toast.success("수정 완료");
      return { ...newRow, quantity, average_buy_price: avgPrice, target_ratio: normalizedTargetRatio };
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "수정 실패");
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

  const handleAddRowSave = useCallback(() => {
    if (!addingRow?.isValidated) {
      toast.error("먼저 종목 확인 버튼을 눌러주세요.");
      return;
    }

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
      return;
    }

    setProcessingId("__adding__");
    startTransition(async () => {
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
        await load(selectedAccountId, true);
        toast.success(`${addingRow.ticker} 추가 완료`);
      } catch (err: any) {
        toast.error(err.message || "종목 추가에 실패했습니다.");
      } finally {
        setProcessingId(null);
      }
    });
  }, [addingRow?.isValidated, addingRow?.ticker, selectedAccountId, load, toast]);

  const handleAddRowCancel = useCallback(() => setAddingRow(null), []);

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

  const gridRows = useMemo<GridRow[]>(() => {
    const baseRows = rows.map((row, i) => ({
      ...row,
      id: `${row.ticker}-${row.account_name}-${i}`,
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

  const columns = useMemo<GridColDef<GridRow>[]>(() => [
    {
      field: "actions", type: "actions", headerName: "작업", width: 100,
      getActions: ({ id, row }) => {
        if (processingId === id) return [<GridActionsCellItem key="l" icon={<IconLoader2 size={18} style={{ animation: "spin 1s linear infinite" }} />} label="처리 중..." disabled />];
        if (id === "__adding__") return [
          <GridActionsCellItem
            key="s"
            icon={<IconCheck size={20} />}
            label="저장"
            color="primary"
            onClick={handleAddRowSave}
            disabled={!addingRow?.isValidated || isPending}
          />,
          <GridActionsCellItem
            key="c"
            icon={<IconX size={20} />}
            label="취소"
            onClick={handleAddRowCancel}
          />
        ];
        if (row.ticker === "IS") return [];
        if (rowModesModel[id]?.mode === GridRowModes.Edit) return [
          <GridActionsCellItem key="s" icon={<IconDeviceFloppy size={20} />} label="저장" color="primary" onClick={() => setRowModesModel(p => ({ ...p, [id]: { mode: GridRowModes.View } }))} />,
          <GridActionsCellItem key="c" icon={<IconX size={20} />} label="취소" onClick={() => setRowModesModel(p => ({ ...p, [id]: { mode: GridRowModes.View, ignoreModifications: true } }))} />
        ];
        return [
          <GridActionsCellItem key="e" icon={<IconPencil size={20} />} label="수정" onClick={() => setRowModesModel(p => ({ ...p, [id]: { mode: GridRowModes.Edit } }))} />,
          <GridActionsCellItem key="d" icon={<IconTrash size={20} />} label="삭제" onClick={() => handleDelete(row.ticker, String(id))} />
        ];
      }
    },
    { field: "bucket", headerName: "버킷", width: 90, cellClassName: (p) => getBucketCellClass(p.row.bucket_id) },
    {
      field: "ticker", headerName: "종목코드", width: 110,
      renderCell: (p) => {
        if (p.row.id === "__adding__") {
          if (addingRow?.isValidated) return <div className="d-flex gap-2"><span>{addingRow.ticker}</span><button className="btn btn-sm btn-link p-0" key="ch" onClick={() => setAddingRow(prev => prev ? { ...prev, ticker: "", isValidated: false, name: "" } : null)}>변경</button></div>;
          return <div className="d-flex gap-1"><StableInlineInput key="ti" className="form-control form-control-sm" style={{ width: "70px" }} initialValue={addingRow?.ticker ?? ""} onSave={handleValidateTicker} /> <button className="btn btn-link btn-sm p-0" key="btn" onClick={() => handleValidateTicker()}>확인</button></div>;
        }
        return <span className="appCodeText">{String(p.value)}</span>;
      }
    },
    { field: "name", headerName: "종목명", minWidth: 300, flex: 2 },
    {
      field: "quantity", headerName: "수량", type: "number", width: 80, 
      editable: true,
      valueFormatter: (p: any) => {
        if (!p || p.value === null || p.value === undefined) return "";
        return new Intl.NumberFormat("ko-KR").format(p.value);
      },
      renderCell: (p: any) => {
        if (!p) return null;
        if (p.row?.id === "__adding__") {
          return (
            <input
              type="number"
              step="any"
              ref={qtyRef}
              className="form-control form-control-sm"
              defaultValue=""
              disabled={!addingRow?.isValidated}
            />
          );
        }
        return <span>{new Intl.NumberFormat("ko-KR").format(p.value ?? 0)}</span>;
      }
    },
    {
      field: "average_buy_price", headerName: "매입 단가", type: "number", width: 120, 
      editable: true,
      valueFormatter: (p: any) => {
        if (!p || p.value === null || p.value === undefined) return "";
        return formatPrice(p.value, p.row?.currency || "KRW");
      },
      renderCell: (p: any) => {
        if (!p) return null;
        if (p.row?.id === "__adding__") {
          return (
            <input
              type="number"
              step="any"
              ref={priceRef}
              className="form-control form-control-sm"
              defaultValue=""
              disabled={!addingRow?.isValidated}
            />
          );
        }
        return <span>{formatPrice(p.value, p.row?.currency || "KRW")}</span>;
      },
      // 수정 모드에서 소수점 정밀도 보장을 위해 step="any" 적용
      renderEditCell: (p: any) => (
        <input
          type="number"
          step="any"
          className="form-control form-control-sm h-100"
          autoFocus
          value={p.value}
          onChange={(e) => p.api.setEditCellValue({ id: p.id, field: p.field, value: e.target.value })}
        />
      )
    },
    { field: "weight_pct", headerName: "비중", width: 70, renderCell: (p: any) => <span style={{ color: "#0d6efd" }}>{p?.value?.toFixed(1)}%</span> },
    { field: "return_pct", headerName: "수익률", width: 80, renderCell: (p: any) => <span className={getSignedClass(p?.value ?? 0)}>{(p?.value ?? 0) > 0 ? "+" : ""}{p?.value?.toFixed(1)}%</span> },
    { field: "daily_change_pct", headerName: "일간(%)", width: 84, renderCell: (p: any) => <span className={getSignedClass(p?.value ?? 0)}>{p?.value === null || p?.value === undefined ? "-" : `${(p?.value ?? 0) > 0 ? "+" : ""}${p?.value?.toFixed(2)}%`}</span> },
    { 
      field: "current_price", headerName: "현재가", width: 110,
      renderCell: (p: any) => <span>{formatPrice(safeParseFloat(p.value), p.row?.currency || "KRW")}</span>
    },
    { field: "pnl_krw", headerName: "평가손익", width: 130, renderCell: (p: any) => <span className={getSignedClass(p?.value ?? 0)}>{formatKrw(p?.value ?? 0)}</span> },
    { field: "valuation_krw", headerName: "평가 금액", width: 130, renderCell: (p: any) => formatKrw(p?.value ?? 0) },
    { field: "buy_amount_krw", headerName: "매입 금액", width: 130, renderCell: (p: any) => formatKrw(p?.value ?? 0) },
    { field: "days_held", headerName: "보유일", width: 65, align: "center" },
    {
      field: "target_ratio",
      headerName: "목표비중",
      width: 92,
      editable: true,
      align: "right",
      headerAlign: "right",
      renderCell: (p: any) => (
        p.row?.id === "__adding__" ? (
          <input
            type="number"
            step="0.1"
            min="0"
            max="100"
            ref={targetRatioRef}
            className="form-control form-control-sm"
            defaultValue={addingRow?.target_ratio ?? "0"}
            disabled={!addingRow?.isValidated}
          />
        ) : (
          <span style={{ color: "#0d6efd", fontWeight: 700 }}>
            {p?.value === null || p?.value === undefined ? "-" : `${p.value.toFixed(1)}%`}
          </span>
        )
      ),
      renderEditCell: (p: any) => (
        <input
          type="number"
          step="0.1"
          min="0"
          max="100"
          className="form-control form-control-sm h-100"
          value={p.value ?? 0}
          onChange={(e) => p.api.setEditCellValue({ id: p.id, field: p.field, value: e.target.value })}
        />
      ),
    },
    {
      field: "target_quantity",
      headerName: "목표수량",
      width: 120,
      align: "right",
      headerAlign: "right",
      renderCell: (p: any) => <span>{formatTargetQuantity(p?.value ?? null, p.row?.currency || "KRW")}</span>,
    },
    {
      field: "expected_change_quantity",
      headerName: "예상변경수량",
      width: 132,
      align: "right",
      headerAlign: "right",
      renderCell: (p: any) => (
        <span className={getSignedNullableClass(
          (p.row?.target_quantity ?? null) !== null && (p.row?.target_quantity ?? null) !== undefined
            ? (p.row.target_quantity ?? 0) - (p.row?.quantity ?? 0)
            : null,
        )}>
          {formatExpectedChangeQuantity(
            p.row?.target_quantity ?? null,
            p.row?.quantity ?? null,
            p.row?.currency || "KRW",
          )}
        </span>
      ),
    },
    {
      field: "target_amount",
      headerName: "목표금액",
      width: 130,
      align: "right",
      headerAlign: "right",
      renderCell: (p: any) => <span>{formatTargetAmount(p?.value ?? null, p.row?.currency || "KRW")}</span>,
    },
  ], [addingRow, isPending, rowModesModel, processingId, handleAddRowSave, handleAddRowCancel]);

  if (loading && !rows.length) return <div className="appPageStack"><div className="appPageLoading"><AppLoadingState label="보유 종목 로드 중..." /></div></div>;

  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection appSectionFill">
        <div className="card appCard shadow-sm h-100 d-flex flex-column">
          <div className="card-header d-flex justify-content-between align-items-center bg-white py-3">
            <div className="d-flex gap-2">
              <select className="form-select w-auto fw-bold" value={selectedAccountId} onChange={(e) => handleAccountChange(e.target.value)}>
                {accounts.map(a => <option key={a.account_id} value={a.account_id}>{a.icon} {a.name}</option>)}
              </select>
              <button className="btn btn-primary btn-sm px-3 fw-bold" onClick={() => setAddingRow({ ticker: "", quantity: "", average_buy_price: "", target_ratio: "0", isValidated: false })}><IconPlus size={16} /> 종목 추가</button>
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
            <div className="card-body py-2 border-bottom bg-light">
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
                </div>
              )}
            </div>
          )}
          <div className="card-body p-0" style={{ minHeight: 0 }}>
            <AppDataGrid 
              rows={gridRows} 
              columns={columns} 
              loading={loading} 
              editMode="row" 
              rowModesModel={rowModesModel} 
              onRowModesModelChange={setRowModesModel} 
              onRowEditStop={(p, e) => { if (p.reason === GridRowEditStopReasons.rowFocusOut) e.defaultMuiPrevented = true; }} 
              processRowUpdate={processRowUpdate}
              isCellEditable={(p) => p.row.id !== "__adding__"}
              minHeight="calc(100vh - 15rem)"
            />
          </div>
        </div>
      </section>
      <div dangerouslySetInnerHTML={{ __html: `<style> .css-label { font-size: 0.7rem; color: #666; display: block; margin-bottom: 2px; } @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } } </style>` }} />
    </div>
  );
}
