"use client";

import { useEffect, useMemo, useState, useTransition } from "react";
import { type GridColDef, type GridRenderCellParams } from "@mui/x-data-grid";
import { IconPlus } from "@tabler/icons-react";

import { AppDataGrid } from "../components/AppDataGrid";
import { AppLoadingState } from "../components/AppLoadingState";
import { useToast } from "../components/ToastProvider";
import {
  readRememberedMomentumEtfAccountId,
  writeRememberedMomentumEtfAccountId,
} from "../components/account-selection";

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
  average_buy_price: string;
  current_price: string;
  days_held: string;
  pnl_krw: number;
  return_pct: number;
  buy_amount_krw: number;
  valuation_krw: number;
  memo: string;
};

type HoldingsResponse = {
  accounts?: AccountConfig[];
  account_id?: string;
  cash?: CashInfo;
  rows?: HoldingsRow[];
  error?: string;
};

type GridRow = HoldingsRow & { id: string };

type EditingState = {
  ticker: string;
  quantity: string;
  average_buy_price: string;
  memo: string;
};

type AddingRowState = {
  ticker: string;
  quantity: string;
  average_buy_price: string;
  isValidatingTicker?: boolean;
  validationError?: string;
  name?: string;
  bucketId?: number;
  isValidated?: boolean;
  memo: string;
};

type CashEditingState = {
  total_principal: string;
  cash_value: string;
  intl_shares_value: string;
  intl_shares_change: string;
};

function formatKrw(value: number): string {
  return `${new Intl.NumberFormat("ko-KR").format(value)}원`;
}

function formatNumber(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 2 }).format(value);
}

function getSignedClass(value: number): string {
  if (value === 0) return "";
  return value > 0 ? "metricPositive" : "metricNegative";
}

function getBucketCellClass(bucketId: number): string {
  if (!bucketId) return "appBucketCell";
  return `appBucketCell appBucketCell${bucketId}`;
}

function parseRawPrice(formatted: string): string {
  return formatted.replace(/[A$₩원,\s]/g, "");
}

/**
 * IME(한글 등) 입력 시 리렌더링으로 인한 조합 분리를 방지하기 위한 독립형 입력 컴포넌트
 */
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

  // 부모 값이 외부 요인으로 명시적으로 바뀌었을 때만 동기화
  useEffect(() => {
    if (initialValue !== localValue) {
      setLocalValue(initialValue);
    }
  }, [initialValue]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.nativeEvent.isComposing) return;
    if (e.key === "Enter") {
      onSave?.(localValue);
    } else if (e.key === "Escape") {
      setLocalValue(initialValue);
      onCancel?.();
    }
  };

  const handleBlur = () => {
    if (localValue !== initialValue) {
      onSave?.(localValue);
    }
  };

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
      onKeyDown={handleKeyDown}
      onBlur={handleBlur}
    />
  );
}

let holdingsDataCache: Record<string, HoldingsResponse> = {};

export function AssetsManager() {
  const [accounts, setAccounts] = useState<AccountConfig[]>(holdingsDataCache["__ACCOUNTS__"]?.accounts ?? []);
  const [selectedAccountId, setSelectedAccountId] = useState(readRememberedMomentumEtfAccountId() ?? "");
  const [rows, setRows] = useState<HoldingsRow[]>(holdingsDataCache[readRememberedMomentumEtfAccountId() ?? ""]?.rows ?? []);
  const [cash, setCash] = useState<CashInfo | null>(null);
  const [loading, setLoading] = useState(!holdingsDataCache[readRememberedMomentumEtfAccountId() ?? ""]);
  const [editing, setEditing] = useState<EditingState | null>(null);
  const [addingRow, setAddingRow] = useState<AddingRowState | null>(null);
  const [cashEditing, setCashEditing] = useState<CashEditingState | null>(null);
  const [isPending, startTransition] = useTransition();
  const toast = useToast();

  async function load(accountId: string | null = null, silent = false) {
    const targetId = accountId ?? selectedAccountId;

    const cached = holdingsDataCache[targetId];
    if (cached) {
      if (cached.accounts) setAccounts(cached.accounts);
      if (cached.rows) setRows(cached.rows);
      if (cached.cash) setCash(cached.cash);
    } else if (!silent) {
      setLoading(true);
    }

    try {
      const search = targetId !== null ? `?account=${encodeURIComponent(targetId)}` : "";
      const response = await fetch(`/api/assets${search}`, { cache: "no-store" });
      const payload = (await response.json()) as HoldingsResponse;

      if (!response.ok) {
        throw new Error(payload.error ?? "보유 종목을 불러오지 못했습니다.");
      }

      const nextAccounts = payload.accounts ?? [];
      const nextRows = payload.rows ?? [];
      const returnedId = payload.account_id ?? "";

      setAccounts(nextAccounts);
      setRows(nextRows);
      setCash(payload.cash ?? null);

      holdingsDataCache[returnedId] = payload;
      holdingsDataCache["__ACCOUNTS__"] = { accounts: nextAccounts };

      if (accountId === null) {
        setSelectedAccountId(returnedId);
        writeRememberedMomentumEtfAccountId(returnedId);
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "보유 종목을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load(readRememberedMomentumEtfAccountId() ?? "");
  }, []);

  function handleAccountChange(nextId: string) {
    setSelectedAccountId(nextId);
    writeRememberedMomentumEtfAccountId(nextId);
    setEditing(null);
    setAddingRow(null);
    setCashEditing(null);

    const cached = holdingsDataCache[nextId];
    if (cached) {
      setRows(cached.rows ?? []);
      setCash(cached.cash ?? null);
      void load(nextId, true);
    } else {
      void load(nextId);
    }
  }

  function handleDelete(ticker: string) {
    if (!confirm(`${ticker} 종목을 삭제하시겠습니까?`)) return;

    startTransition(async () => {
      try {
        const params = new URLSearchParams({ account: selectedAccountId, ticker });
        const response = await fetch(`/api/assets?${params.toString()}`, { method: "DELETE" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error ?? "삭제 실패");

        delete holdingsDataCache[selectedAccountId];
        await load(selectedAccountId, true);
        toast.success(`${ticker} 삭제 완료`);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "삭제에 실패했습니다.");
      }
    });
  }

  function handleAddRowStart() {
    setAddingRow({
      ticker: "",
      quantity: "",
      average_buy_price: "",
      memo: "",
    });
  }

  function handleAddRowCancel() {
    setAddingRow(null);
  }

  function handleValidateTicker() {
    if (!addingRow?.ticker) return;

    setAddingRow((prev) => prev ? { ...prev, isValidatingTicker: true, validationError: undefined } : null);

    startTransition(async () => {
      try {
        const response = await fetch("/api/assets", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "validate",
            account_id: selectedAccountId,
            ticker: addingRow.ticker,
          }),
        });
        const payload = await response.json();

        if (!response.ok) {
          const errorMsg = payload.error ?? "검증 실패";
          setAddingRow((prev) => prev ? { ...prev, isValidatingTicker: false, validationError: errorMsg } : null);
          toast.error(errorMsg);
          return;
        }

        setAddingRow((prev) => prev ? {
          ...prev,
          isValidatingTicker: false,
          isValidated: true,
          name: payload.name ?? "",
          bucketId: payload.bucket_id ?? 1,
          ticker: payload.ticker ?? addingRow.ticker,
          validationError: undefined,
        } : null);
        
        toast.success(`조회 성공: ${payload.name}`);
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "검증 중 오류 발생";
        setAddingRow((prev) => prev ? { ...prev, isValidatingTicker: false, validationError: errorMsg } : null);
        toast.error(errorMsg);
      }
    });
  }

  function handleAddRowSave() {
    if (!addingRow) return;

    const quantity = parseInt(addingRow.quantity, 10);
    const avgPrice = parseFloat(addingRow.average_buy_price);

    if (Number.isNaN(quantity) || quantity < 0) {
      toast.error("수량이 올바르지 않습니다.");
      return;
    }
    if (Number.isNaN(avgPrice) || avgPrice < 0) {
      toast.error("매입 단가가 올바르지 않습니다.");
      return;
    }

    startTransition(async () => {
      try {
        const response = await fetch("/api/assets", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            ticker: addingRow.ticker,
            quantity,
            average_buy_price: avgPrice,
            memo: addingRow.memo,
          }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error ?? "추가 실패");

        setAddingRow(null);
        delete holdingsDataCache[selectedAccountId];
        await load(selectedAccountId, true);
        toast.success(`${addingRow.ticker} 추가 완료`);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "종목 추가에 실패했습니다.");
      }
    });
  }

  function startEditing(row: GridRow) {
    setEditing({
      ticker: row.ticker,
      quantity: String(row.quantity),
      average_buy_price: parseRawPrice(row.average_buy_price),
      memo: row.memo || "",
    });
  }

  function cancelEditing() {
    setEditing(null);
  }

  function handleSave() {
    if (!editing) return;

    const quantity = parseInt(editing.quantity, 10);
    const avgPrice = parseFloat(editing.average_buy_price);

    if (Number.isNaN(quantity) || quantity < 0) {
      toast.error("수량이 올바르지 않습니다.");
      return;
    }
    if (Number.isNaN(avgPrice) || avgPrice < 0) {
      toast.error("매입 단가가 올바르지 않습니다.");
      return;
    }

    startTransition(async () => {
      try {
        const response = await fetch("/api/assets", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            ticker: editing.ticker,
            quantity,
            average_buy_price: avgPrice,
            memo: editing.memo,
          }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error ?? "수정 실패");

        setEditing(null);
        delete holdingsDataCache[selectedAccountId];
        await load(selectedAccountId, true);
        toast.success(`${editing.ticker} 수정 완료`);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "수정에 실패했습니다.");
      }
    });
  }

  // --- Cash editing ---
  function startCashEditing() {
    if (!cash) return;
    const isKrw = cash.currency === "KRW";
    setCashEditing({
      total_principal: String(cash.total_principal ?? 0),
      cash_value: String(isKrw ? (cash.cash_balance_krw ?? 0) : (cash.cash_balance_native ?? 0)),
      intl_shares_value: String(cash.intl_shares_value ?? 0),
      intl_shares_change: String(cash.intl_shares_change ?? 0),
    });
  }

  function cancelCashEditing() {
    setCashEditing(null);
  }

  function handleCashSave() {
    if (!cash || !cashEditing) return;

    startTransition(async () => {
      try {
        const isKrw = cash.currency === "KRW";
        const cashValue = parseFloat(cashEditing.cash_value) || 0;

        const body: Record<string, unknown> = {
          account_id: cash.account_id,
          total_principal: parseFloat(cashEditing.total_principal) || 0,
          cash_balance_krw: isKrw ? cashValue : 0,
          cash_balance_native: isKrw ? cashValue : cashValue,
          cash_currency: cash.cash_currency,
        };

        if (cash.account_id === "aus_account") {
          body.intl_shares_value = parseFloat(cashEditing.intl_shares_value) || 0;
          body.intl_shares_change = parseFloat(cashEditing.intl_shares_change) || 0;
        }

        const response = await fetch("/api/assets", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error ?? "자산 정보 저장 실패");

        setCashEditing(null);
        delete holdingsDataCache[selectedAccountId];
        await load(selectedAccountId, true);
        toast.success("자산 정보 저장 완료");
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "자산 정보 저장에 실패했습니다.");
      }
    });
  }

  const gridRows = useMemo<GridRow[]>(
    () => {
      // 데이터를 넘겨주기 전에 미리 버킷(ASC), 종목코드(ASC) 순으로 정렬함
      const sortedBase = [...rows].sort((a, b) => {
        if (a.bucket_id !== b.bucket_id) {
          return a.bucket_id - b.bucket_id;
        }
        return a.ticker.localeCompare(b.ticker);
      });

      const baseRows = sortedBase.map((row, i) => ({ ...row, id: `${row.ticker}-${row.account_name}-${i}` }));
      if (addingRow) {
        return [
          {
            id: "__adding__",
            account_name: accounts.find((a) => a.account_id === selectedAccountId)?.name ?? "",
            currency: "",
            bucket: "",
            bucket_id: 0,
            ticker: addingRow.ticker,
            name: addingRow.name || "",
            quantity: parseInt(addingRow.quantity, 10) || 0,
            average_buy_price: addingRow.average_buy_price,
            current_price: "-",
            days_held: "-",
            pnl_krw: 0,
            return_pct: 0,
            buy_amount_krw: 0,
            valuation_krw: 0,
            memo: addingRow.memo,
          } as GridRow,
          ...baseRows,
        ];
      }
      return baseRows;
    },
    [rows, addingRow, selectedAccountId, accounts],
  );

  const columns = useMemo<GridColDef<GridRow>[]>(
    () => [
      {
        field: "__edit__",
        headerName: "",
        width: 58,
        sortable: false,
        filterable: false,
        disableColumnMenu: true,
        renderCell: (params: GridRenderCellParams<GridRow>) => {
          if (addingRow && params.row.id === "__adding__") {
            return (
              <span className="d-flex gap-1">
                <button type="button" className="btn btn-link btn-sm p-0 appEditLink" onClick={handleAddRowSave} disabled={isPending || !addingRow.isValidated || !addingRow.quantity || !addingRow.average_buy_price}>저장</button>
                <button type="button" className="btn btn-link btn-sm p-0" style={{ color: "#6c757d" }} onClick={handleAddRowCancel} disabled={isPending}>취소</button>
              </span>
            );
          }
          const isEditing = editing?.ticker === params.row.ticker;
          if (isEditing) {
            return (
              <span className="d-flex gap-1">
                <button type="button" className="btn btn-link btn-sm p-0 appEditLink" onClick={handleSave} disabled={isPending}>저장</button>
                <button type="button" className="btn btn-link btn-sm p-0" style={{ color: "#6c757d" }} onClick={cancelEditing} disabled={isPending}>취소</button>
              </span>
            );
          }
          return (
            <button type="button" className="btn btn-link btn-sm p-0 appEditLink" onClick={() => startEditing(params.row)} disabled={isPending}>수정</button>
          );
        },
      },
      { field: "currency", headerName: "환종", minWidth: 60, width: 60, align: "center", headerAlign: "center" },
      {
        field: "bucket",
        headerName: "버킷",
        minWidth: 90,
        width: 90,
        sortable: false,
        cellClassName: (params) => {
          if (params.row.id === "__adding__") return "";
          return getBucketCellClass(params.row.bucket_id);
        },
        renderCell: (params) => {
          if (params.row.id === "__adding__") return "";
          return <span>{String(params.value ?? "-")}</span>;
        },
      },
      {
        field: "ticker",
        headerName: "종목코드",
        minWidth: 110,
        width: 110,
        renderCell: (params: GridRenderCellParams<GridRow>) => {
          if (addingRow && params.row.id === "__adding__") {
            if (addingRow.isValidated) {
              return (
                <div className="d-flex gap-2 align-items-center">
                  <span className="appCodeText" style={{ fontWeight: 600 }}>{addingRow.ticker}</span>
                  <button
                    type="button"
                    className="btn btn-link btn-sm p-0"
                    style={{ fontSize: "0.75rem", color: "#6c757d" }}
                    onClick={() => setAddingRow({ ticker: "", quantity: "", average_buy_price: "", memo: "" })}
                  >
                    변경
                  </button>
                </div>
              );
            }
            return (
              <div className="d-flex flex-column gap-1">
                <div className="d-flex gap-1 align-items-center">
                  <StableInlineInput
                    className="form-control form-control-sm"
                    style={{ width: "80px", textAlign: "center" }}
                    placeholder="티커"
                    initialValue={addingRow.ticker}
                    onChange={(val) => setAddingRow({ ...addingRow, ticker: val, validationError: undefined })}
                    onSave={handleValidateTicker}
                    disabled={addingRow.isValidatingTicker}
                  />
                  <button
                    type="button"
                    className="btn btn-link btn-sm p-0"
                    style={{ fontSize: "0.75rem" }}
                    onClick={handleValidateTicker}
                    disabled={addingRow.isValidatingTicker || !addingRow.ticker}
                  >
                    {addingRow.isValidatingTicker ? "확인 중..." : "확인"}
                  </button>
                </div>
              </div>
            );
          }
          return <span className="appCodeText">{String(params.value ?? "-")}</span>;
        },
      },
      { field: "name", headerName: "종목명", minWidth: 180, flex: 1, renderCell: (params: GridRenderCellParams<GridRow>) => {
        if (addingRow && params.row.id === "__adding__") {
          return addingRow.name ? <span>{addingRow.name}</span> : <span style={{ color: "#6c757d", fontSize: "0.85rem" }}>―</span>;
        }
        return <span>{params.value ?? "-"}</span>;
      } },
      {
        field: "quantity",
        headerName: "수량",
        minWidth: 80,
        width: 80,
        align: "right",
        headerAlign: "right",
        renderCell: (params: GridRenderCellParams<GridRow>) => {
          if (addingRow && params.row.id === "__adding__") {
            return (
              <input
                type="number"
                className="form-control form-control-sm"
                style={{ width: "80px", textAlign: "right" }}
                value={addingRow.quantity}
                onChange={(e) => setAddingRow({ ...addingRow, quantity: e.target.value })}
                onKeyDown={(e) => { if (e.key === "Enter") handleAddRowSave(); if (e.key === "Escape") handleAddRowCancel(); }}
                disabled={!addingRow.isValidated || isPending}
              />
            );
          }
          if (editing?.ticker === params.row.ticker) {
            return (
              <input
                type="number"
                className="form-control form-control-sm"
                style={{ width: "80px", textAlign: "right" }}
                value={editing.quantity}
                onChange={(e) => setEditing({ ...editing, quantity: e.target.value })}
                onKeyDown={(e) => { if (e.key === "Enter") handleSave(); if (e.key === "Escape") cancelEditing(); }}
              />
            );
          }
          return new Intl.NumberFormat("ko-KR").format(params.value ?? 0);
        },
      },
      {
        field: "average_buy_price",
        headerName: "매입 단가",
        minWidth: 110,
        width: 110,
        align: "right",
        headerAlign: "right",
        renderCell: (params: GridRenderCellParams<GridRow>) => {
          if (addingRow && params.row.id === "__adding__") {
            return (
              <input
                type="number"
                step="0.0001"
                className="form-control form-control-sm"
                style={{ width: "110px", textAlign: "right" }}
                value={addingRow.average_buy_price}
                onChange={(e) => setAddingRow({ ...addingRow, average_buy_price: e.target.value })}
                onKeyDown={(e) => { if (e.key === "Enter") handleAddRowSave(); if (e.key === "Escape") handleAddRowCancel(); }}
                disabled={!addingRow.isValidated || isPending}
              />
            );
          }
          if (editing?.ticker === params.row.ticker) {
            return (
              <input
                type="number"
                step="0.0001"
                className="form-control form-control-sm"
                style={{ width: "110px", textAlign: "right" }}
                value={editing.average_buy_price}
                onChange={(e) => setEditing({ ...editing, average_buy_price: e.target.value })}
                onKeyDown={(e) => { if (e.key === "Enter") handleSave(); if (e.key === "Escape") cancelEditing(); }}
              />
            );
          }
          return String(params.value ?? "-");
        },
      },
      {
        field: "current_price",
        headerName: "현재가",
        minWidth: 100,
        width: 100,
        align: "right",
        headerAlign: "right",
      },
      {
        field: "days_held",
        headerName: "보유일",
        minWidth: 65,
        width: 65,
        align: "center",
        headerAlign: "center",
      },
      {
        field: "pnl_krw",
        headerName: "평가손익",
        minWidth: 120,
        width: 120,
        align: "right",
        headerAlign: "right",
        renderCell: (params: GridRenderCellParams<GridRow, number>) => (
          <span className={getSignedClass(params.value ?? 0)}>{formatKrw(params.value ?? 0)}</span>
        ),
      },
      {
        field: "return_pct",
        headerName: "수익률",
        minWidth: 80,
        width: 80,
        align: "right",
        headerAlign: "right",
        renderCell: (params: GridRenderCellParams<GridRow, number>) => {
          const value = params.value ?? 0;
          return <span className={getSignedClass(value)}>{value > 0 ? "+" : ""}{value.toFixed(2)}%</span>;
        },
      },
      {
        field: "buy_amount_krw",
        headerName: "매입 금액",
        minWidth: 120,
        width: 120,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatKrw(params.value ?? 0),
      },
      {
        field: "valuation_krw",
        headerName: "평가 금액",
        minWidth: 120,
        width: 120,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatKrw(params.value ?? 0),
      },
      {
        field: "memo",
        headerName: "메모",
        minWidth: 200,
        flex: 1.5,
        renderCell: (params: GridRenderCellParams<GridRow>) => {
          if (addingRow && params.row.id === "__adding__") {
            return (
              <StableInlineInput
                className="form-control form-control-sm"
                placeholder="메모 입력"
                initialValue={addingRow.memo}
                onChange={(val) => setAddingRow({ ...addingRow, memo: val })}
                onSave={handleAddRowSave}
                onCancel={handleAddRowCancel}
                disabled={!addingRow.isValidated || isPending}
              />
            );
          }
          if (editing?.ticker === params.row.ticker) {
            return (
              <StableInlineInput
                className="form-control form-control-sm"
                placeholder="메모 수정"
                initialValue={editing.memo}
                onChange={(val) => setEditing({ ...editing, memo: val })}
                onSave={handleSave}
                onCancel={cancelEditing}
                disabled={isPending}
              />
            );
          }
          return <span className="text-secondary" style={{ fontSize: "0.85rem" }}>{params.value || "-"}</span>;
        },
      },
      {
        field: "__delete__",
        headerName: "",
        width: 58,
        sortable: false,
        filterable: false,
        disableColumnMenu: true,
        renderCell: (params: GridRenderCellParams<GridRow>) => (
          <button type="button" className="btn btn-link btn-sm p-0" style={{ color: "#dc3545" }} onClick={() => handleDelete(params.row.ticker)} disabled={isPending}>삭제</button>
        ),
      },
    ],
    [editing, isPending, addingRow],
  );

  if (loading) {
    return (
      <div className="appPageStack">
        <div className="appPageLoading">
          <AppLoadingState label="보유 종목을 불러오는 중..." />
        </div>
      </div>
    );
  }

  const isKrw = cash?.currency === "KRW";
  const isAus = cash?.account_id === "aus_account";
  const cashLabel = isKrw ? "보유 현금 (KRW)" : `보유 현금 (${cash?.currency ?? ""})`;

  return (
    <div className="appPageStack">
      <section className="appSection appSectionFill">
        <div className="card appCard">
          <div className="card-header">
            <div className="tickerTypeToolbar w-100" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div className="tickerTypeToolbarLeft" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <div className="accountSelect">
                  <select
                    className="form-select"
                    style={{ width: "auto", minWidth: "220px", fontWeight: 600 }}
                    value={selectedAccountId}
                    onChange={(e) => handleAccountChange(e.target.value)}
                    disabled={loading}
                  >
                    {accounts.map((acc) => (
                      <option key={acc.account_id} value={acc.account_id}>
                        {acc.icon} {acc.name}
                      </option>
                    ))}
                  </select>
                </div>
                <button className="btn btn-primary d-flex align-items-center gap-1" type="button" onClick={handleAddRowStart} disabled={loading} style={{ fontWeight: 600 }}>
                  <IconPlus size={18} stroke={2} />
                  <span>종목 추가</span>
                </button>
              </div>
              <div className="tickerTypeToolbarRight">
                <div className="stocksSummary d-flex align-items-center gap-3">
                  <div className="d-flex align-items-center gap-1">
                    <span style={{ color: "#6c757d", fontSize: "0.85rem", fontWeight: 600 }}>총 종목 수:</span>
                    <span style={{ fontWeight: 700 }}>{rows.filter(r => r.quantity > 0).length}개</span>
                  </div>
                  <div className="d-flex align-items-center gap-1">
                    <span style={{ color: "#6c757d", fontSize: "0.85rem", fontWeight: 600 }}>총 평가액:</span>
                    <span style={{ fontWeight: 700 }}>{formatKrw(rows.reduce((acc, row) => acc + (row.valuation_krw || 0), 0))}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Cash 인라인 편집 영역 */}
          {cash ? (
            <div className="card-body" style={{ borderBottom: "1px solid var(--bs-border-color)", paddingTop: "0.6rem", paddingBottom: "0.6rem" }}>
              {cashEditing ? (
                <div className="d-flex align-items-end gap-3 flex-wrap">
                  <div>
                    <label className="form-label mb-0" style={{ fontSize: "0.78rem", color: "#6c757d" }}>투자 원금 (KRW)</label>
                    <input
                      type="number"
                      className="form-control form-control-sm"
                      style={{ width: "140px" }}
                      value={cashEditing.total_principal}
                      onChange={(e) => setCashEditing({ ...cashEditing, total_principal: e.target.value })}
                    />
                  </div>
                  <div>
                    <label className="form-label mb-0" style={{ fontSize: "0.78rem", color: "#6c757d" }}>{cashLabel}</label>
                    <input
                      type="number"
                      className="form-control form-control-sm"
                      style={{ width: "140px" }}
                      value={cashEditing.cash_value}
                      onChange={(e) => setCashEditing({ ...cashEditing, cash_value: e.target.value })}
                    />
                  </div>
                  {isAus ? (
                    <>
                      <div>
                        <label className="form-label mb-0" style={{ fontSize: "0.78rem", color: "#6c757d" }}>Intl Shares Value</label>
                        <input
                          type="number"
                          className="form-control form-control-sm"
                          style={{ width: "140px" }}
                          value={cashEditing.intl_shares_value}
                          onChange={(e) => setCashEditing({ ...cashEditing, intl_shares_value: e.target.value })}
                        />
                      </div>
                      <div>
                        <label className="form-label mb-0" style={{ fontSize: "0.78rem", color: "#6c757d" }}>Intl Shares Change</label>
                        <input
                          type="number"
                          className="form-control form-control-sm"
                          style={{ width: "140px" }}
                          value={cashEditing.intl_shares_change}
                          onChange={(e) => setCashEditing({ ...cashEditing, intl_shares_change: e.target.value })}
                        />
                      </div>
                    </>
                  ) : null}
                  <div className="d-flex gap-1">
                    <button type="button" className="btn btn-sm btn-primary" onClick={handleCashSave} disabled={isPending}>
                      {isPending ? "저장 중..." : "저장"}
                    </button>
                    <button type="button" className="btn btn-sm btn-outline-secondary" onClick={cancelCashEditing} disabled={isPending}>
                      취소
                    </button>
                  </div>
                </div>
              ) : (
                <div className="d-flex align-items-center gap-4 flex-wrap">
                  <div className="d-flex align-items-center gap-1">
                    <span style={{ color: "#6c757d", fontSize: "0.85rem" }}>투자 원금:</span>
                    <span style={{ fontWeight: 600 }}>{formatNumber(cash.total_principal)}원</span>
                  </div>
                  <div className="d-flex align-items-center gap-1">
                    <span style={{ color: "#6c757d", fontSize: "0.85rem" }}>{cashLabel}:</span>
                    <span style={{ fontWeight: 600 }}>
                      {formatNumber(isKrw ? cash.cash_balance_krw : cash.cash_balance_native)}
                      {isKrw ? "원" : ""}
                    </span>
                  </div>
                  {isAus ? (
                    <>
                      <div className="d-flex align-items-center gap-1">
                        <span style={{ color: "#6c757d", fontSize: "0.85rem" }}>Intl Shares:</span>
                        <span style={{ fontWeight: 600 }}>{formatNumber(cash.intl_shares_value)}</span>
                      </div>
                      <div className="d-flex align-items-center gap-1">
                        <span style={{ color: "#6c757d", fontSize: "0.85rem" }}>Intl Change:</span>
                        <span style={{ fontWeight: 600 }}>{formatNumber(cash.intl_shares_change)}</span>
                      </div>
                    </>
                  ) : null}
                  <button
                    type="button"
                    className="btn btn-sm btn-outline-secondary"
                    onClick={startCashEditing}
                    disabled={isPending}
                  >
                    자산 정보 수정
                  </button>
                </div>
              )}
            </div>
          ) : null}

          <div className="card-body appCardBodyTight">
            <AppDataGrid
              className="appDataGrid"
              rows={gridRows}
              columns={columns}
              loading={loading}
              initialState={{
                sorting: {
                  sortModel: [
                    { field: "bucket_id", sort: "asc" },
                  ],
                },
              }}
              getRowClassName={(params) => {
                const pnl = params.row.pnl_krw ?? 0;
                return pnl > 0 ? "appHeldRow" : "";
              }}
              minHeight="75vh"
            />
          </div>
        </div>
      </section>
    </div>
  );
}
