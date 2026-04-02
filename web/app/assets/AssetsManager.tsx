"use client";

import { useEffect, useMemo, useState, useTransition, useCallback } from "react";
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
  memo: string;
};

type GridRow = HoldingsRow & { id: string };

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

// --- Helpers ---
function formatKrw(value: number): string {
  return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
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

function parseRawPrice(formatted: unknown): string {
  if (formatted === null || formatted === undefined) return "0";
  return String(formatted).replace(/[A$₩원,\s]/g, "");
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

    if (isNaN(quantity) || quantity < 0 || isNaN(avgPrice) || avgPrice < 0) {
      toast.error("입력값이 올바르지 않습니다.");
      throw new Error("Invalid Input");
    }

    setProcessingId(newRow.id);
    try {
      const res = await fetch("/api/assets", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          account_id: selectedAccountId,
          ticker: newRow.ticker.replace("ASX:", ""),
          quantity,
          average_buy_price: avgPrice,
          memo: newRow.memo,
        }),
      });
      if (!res.ok) throw new Error();
      delete holdingsDataCache[selectedAccountId];
      startTransition(() => { void load(selectedAccountId, true); });
      toast.success("수정 완료");
      return { ...newRow, quantity, average_buy_price: avgPrice };
    } catch { toast.error("수정 실패"); throw new Error("Fail"); }
    finally { setProcessingId(null); }
  };

  const handleValidateTicker = (tickerToUse?: string) => {
    const tickerStr = tickerToUse || addingRow?.ticker;
    if (!tickerStr) return;
    setAddingRow(p => p ? { ...p, ticker: tickerStr, isValidatingTicker: true } : null);
    startTransition(async () => {
      try {
        const res = await fetch("/api/assets", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "validate", account_id: selectedAccountId, ticker: tickerStr }),
        });
        const data = await res.json();
        if (!res.ok) { toast.error(data.error || "검증 실패"); setAddingRow(p => p ? { ...p, isValidatingTicker: false } : null); return; }
        setAddingRow(p => p ? { ...p, isValidatingTicker: false, isValidated: true, name: data.name, bucketId: data.bucket_id, ticker: data.ticker } : null);
        toast.success(`조회 성공: ${data.name}`);
      } catch { toast.error("오류 발생"); setAddingRow(p => p ? { ...p, isValidatingTicker: false } : null); }
    });
  };

  const handleAddRowSave = useCallback(() => {
    if (!addingRow?.isValidated) return;
    const quantity = parseInt(addingRow.quantity, 10);
    const avgPrice = safeParseFloat(addingRow.average_buy_price);
    if (isNaN(quantity) || quantity < 0 || isNaN(avgPrice) || avgPrice < 0) { toast.error("정보를 정확히 입력하세요."); return; }

    setProcessingId("__adding__");
    startTransition(async () => {
      try {
        const res = await fetch("/api/assets", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ account_id: selectedAccountId, ticker: addingRow.ticker, quantity, average_buy_price: avgPrice, memo: addingRow.memo }),
        });
        if (!res.ok) throw new Error();
        setAddingRow(null);
        delete holdingsDataCache[selectedAccountId];
        await load(selectedAccountId, true);
        toast.success("추가 완료");
      } catch { toast.error("추가 실패"); }
      finally { setProcessingId(null); }
    });
  }, [addingRow, selectedAccountId, load, toast]);

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
    })).sort((a, b) => a.bucket_id - b.bucket_id || a.ticker.localeCompare(b.ticker));

    if (addingRow) {
      return [{
        id: "__adding__", account_name: "", currency: "", bucket: "", bucket_id: 0,
        ticker: addingRow.ticker, name: addingRow.name || "",
        quantity: parseInt(addingRow.quantity, 10) || 0,
        average_buy_price: safeParseFloat(addingRow.average_buy_price),
        current_price: "-", days_held: "-", pnl_krw: 0, return_pct: 0, weight_pct: 0, buy_amount_krw: 0, valuation_krw: 0, memo: addingRow.memo,
      } as GridRow, ...baseRows];
    }
    return baseRows;
  }, [rows, addingRow]);

  const columns = useMemo<GridColDef<GridRow>[]>(() => [
    {
      field: "actions", type: "actions", headerName: "작업", width: 100,
      getActions: ({ id, row }) => {
        if (processingId === id) return [<GridActionsCellItem key="l" icon={<IconLoader2 size={18} style={{ animation: "spin 1s linear infinite" }} />} label="L" disabled />];
        if (id === "__adding__") return [
          <GridActionsCellItem key="s" icon={<IconCheck size={20} color={addingRow?.isValidated ? "primary" : "disabled"} />} label="S" onClick={handleAddRowSave} disabled={!addingRow?.isValidated || isPending} />,
          <GridActionsCellItem key="c" icon={<IconX size={20} />} label="C" onClick={handleAddRowCancel} />
        ];
        if (row.ticker === "IS") return [];
        if (rowModesModel[id]?.mode === GridRowModes.Edit) return [
          <GridActionsCellItem key="s" icon={<IconDeviceFloppy size={20} />} label="S" color="primary" onClick={() => setRowModesModel(p => ({ ...p, [id]: { mode: GridRowModes.View } }))} />,
          <GridActionsCellItem key="c" icon={<IconX size={20} />} label="C" onClick={() => setRowModesModel(p => ({ ...p, [id]: { mode: GridRowModes.View, ignoreModifications: true } }))} />
        ];
        return [
          <GridActionsCellItem key="e" icon={<IconPencil size={20} />} label="E" onClick={() => setRowModesModel(p => ({ ...p, [id]: { mode: GridRowModes.Edit } }))} />,
          <GridActionsCellItem key="d" icon={<IconTrash size={20} />} label="D" onClick={() => handleDelete(row.ticker, String(id))} />
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
    { field: "name", headerName: "종목명", minWidth: 150, flex: 1 },
    {
      field: "quantity", headerName: "수량", type: "number", width: 80, editable: true,
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
              className="form-control form-control-sm"
              value={addingRow?.quantity ?? ""}
              onChange={(e) => setAddingRow(v => v ? { ...v, quantity: e.target.value } : null)}
              disabled={!addingRow?.isValidated}
            />
          );
        }
        return <span>{new Intl.NumberFormat("ko-KR").format(p.value ?? 0)}</span>;
      }
    },
    {
      field: "average_buy_price", headerName: "매입 단가", type: "number", width: 110, editable: true,
      valueFormatter: (p: any) => {
        if (!p || p.value === null || p.value === undefined) return "";
        return `${new Intl.NumberFormat("ko-KR").format(p.value)}원`;
      },
      renderCell: (p: any) => {
        if (!p) return null;
        if (p.row?.id === "__adding__") {
          return (
            <input
              type="number"
              className="form-control form-control-sm"
              value={addingRow?.average_buy_price ?? ""}
              onChange={(e) => setAddingRow(v => v ? { ...v, average_buy_price: e.target.value } : null)}
              disabled={!addingRow?.isValidated}
            />
          );
        }
        return <span>{new Intl.NumberFormat("ko-KR").format(p.value ?? 0)}원</span>;
      }
    },
    { field: "weight_pct", headerName: "비중", width: 70, renderCell: (p: any) => <span style={{ color: "#0d6efd" }}>{p?.value?.toFixed(1)}%</span> },
    { field: "return_pct", headerName: "수익률", width: 80, renderCell: (p: any) => <span className={getSignedClass(p?.value ?? 0)}>{(p?.value ?? 0) > 0 ? "+" : ""}{p?.value?.toFixed(1)}%</span> },
    { field: "current_price", headerName: "현재가", width: 110 },
    { field: "days_held", headerName: "보유일", width: 65, align: "center" },
    { field: "pnl_krw", headerName: "평가손익", width: 130, renderCell: (p: any) => <span className={getSignedClass(p?.value ?? 0)}>{formatKrw(p?.value ?? 0)}</span> },
    { field: "buy_amount_krw", headerName: "매입 금액", width: 130, renderCell: (p: any) => formatKrw(p?.value ?? 0) },
    { field: "valuation_krw", headerName: "평가 금액", width: 130, renderCell: (p: any) => formatKrw(p?.value ?? 0) },
    {
      field: "memo", headerName: "메모", minWidth: 150, flex: 1.5, editable: true,
      renderCell: (p) => p.row.id === "__adding__" ? <StableInlineInput className="form-control form-control-sm" initialValue={addingRow?.memo ?? ""} onChange={v => setAddingRow(p => p ? {...p, memo: v} : null)} onSave={handleAddRowSave} disabled={!addingRow?.isValidated} /> : <span>{p.value}</span>
    }
  ], [addingRow, isPending, rowModesModel, processingId, handleAddRowSave, handleAddRowCancel]);

  if (loading && !rows.length) return <div className="appPageStack"><div className="appPageLoading"><AppLoadingState label="보유 종목 로드 중..." /></div></div>;

  return (
    <div className="appPageStack">
      <section className="appSection appSectionFill">
        <div className="card appCard shadow-sm">
          <div className="card-header d-flex justify-content-between align-items-center bg-white py-3">
            <div className="d-flex gap-2">
              <select className="form-select w-auto fw-bold" value={selectedAccountId} onChange={(e) => handleAccountChange(e.target.value)}>
                {accounts.map(a => <option key={a.account_id} value={a.account_id}>{a.icon} {a.name}</option>)}
              </select>
              <button className="btn btn-primary btn-sm px-3 fw-bold" onClick={() => setAddingRow({ ticker: "", quantity: "", average_buy_price: "", memo: "" })}><IconPlus size={16} /> 종목 추가</button>
            </div>
            <div className="d-flex gap-4 small">
              <div className="text-muted">총 자산: <span className="fw-bold text-primary fs-6">{formatKrw(rows.reduce((s, r) => s + (r.valuation_krw || 0), 0) + (cash?.cash_balance_krw || 0))}</span></div>
              <div className="text-muted">평가액: <span className="fw-bold text-dark">{formatKrw(rows.reduce((s, r) => s + (r.valuation_krw || 0), 0))}</span></div>
              <div className="text-muted">종목수: <span className="fw-bold text-dark">{rows.length}개</span></div>
            </div>
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
                  <div className="small">투자원금: <b className="ms-1">{formatNumber(cash.total_principal)}원</b></div>
                  <div className="small">보유현금: <b className="ms-1">{formatNumber(cash.currency === "KRW" ? cash.cash_balance_krw : cash.cash_balance_native)}원</b></div>
                  <button className="btn btn-link btn-sm p-0" onClick={() => { if (cash) setCashEditing({ total_principal: String(cash.total_principal), cash_value: String(cash.currency === "KRW" ? cash.cash_balance_krw : cash.cash_balance_native), intl_shares_value: String(cash.intl_shares_value), intl_shares_change: String(cash.intl_shares_change) }); }}>수정</button>
                </div>
              )}
            </div>
          )}
          <div className="card-body p-0" style={{ minHeight: "500px" }}>
            <AppDataGrid rows={gridRows} columns={columns} loading={loading} editMode="row" rowModesModel={rowModesModel} onRowModesModelChange={setRowModesModel} onRowEditStop={(p, e) => { if (p.reason === GridRowEditStopReasons.rowFocusOut) e.defaultMuiPrevented = true; }} processRowUpdate={processRowUpdate} />
          </div>
        </div>
      </section>
      <div dangerouslySetInnerHTML={{ __html: `<style> .css-label { font-size: 0.7rem; color: #666; display: block; margin-bottom: 2px; } @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } } </style>` }} />
    </div>
  );
}
