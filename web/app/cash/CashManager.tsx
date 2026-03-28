"use client";

import { useEffect, useState } from "react";
import { type GridColDef } from "@mui/x-data-grid";

import { AppDataGrid } from "../components/AppDataGrid";
import { AppModal } from "../components/AppModal";
import { useToast } from "../components/ToastProvider";

type CashAccountItem = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
  country_code: string;
  currency: string;
  total_principal: number;
  cash_balance_krw: number;
  cash_balance_native: number | null;
  cash_currency: string;
  intl_shares_value: number | null;
  intl_shares_change: number | null;
  updated_at: string | null;
};

type CashAccountsResponse = {
  accounts?: CashAccountItem[];
  rates?: Record<string, number>;
  error?: string;
};

type CashGridRow = CashAccountItem & {
  id: string;
};

function formatUpdatedAt(value: string | null): string {
  if (!value) {
    return "저장 이력 없음";
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

function normalizeInputValue(value: number | null): string {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value);
}

function formatNumber(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 2 }).format(value);
}

function getLatestUpdatedAt(accounts: CashAccountItem[]): string | null {
  const timestamps = accounts
    .map((account) => account.updated_at)
    .filter((value): value is string => Boolean(value))
    .map((value) => new Date(value).getTime())
    .filter((value) => !Number.isNaN(value));

  if (timestamps.length === 0) {
    return null;
  }

  return new Date(Math.max(...timestamps)).toISOString();
}

function cloneAccount(account: CashAccountItem): CashAccountItem {
  return {
    ...account,
  };
}

export function CashManager() {
  const [accounts, setAccounts] = useState<CashAccountItem[]>([]);
  const [rates, setRates] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [editingAccount, setEditingAccount] = useState<CashAccountItem | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const latestUpdatedAt = getLatestUpdatedAt(accounts);
  const toast = useToast();
  const gridRows: CashGridRow[] = accounts.map((account) => ({
    ...account,
    id: account.account_id,
  }));
  const columns: GridColDef<CashGridRow>[] = [
    {
      field: "__edit__",
      headerName: "",
      width: 58,
      sortable: false,
      filterable: false,
      disableColumnMenu: true,
      renderCell: (params) => (
        <button type="button" className="btn btn-link btn-sm p-0 appEditLink" onClick={() => openEditModal(params.row)}>
          Edit
        </button>
      ),
    },
    {
      field: "name",
      headerName: "계좌",
      minWidth: 146,
      width: 146,
      renderCell: (params) => <strong>{params.row.order}. {params.row.icon} {params.row.name}</strong>,
    },
    {
      field: "total_principal",
      headerName: "투자 원금 (KRW)",
      minWidth: 132,
      width: 132,
      align: "right",
      headerAlign: "right",
      renderCell: (params) => <span className="tablePlainValue">{formatNumber(params.row.total_principal)}</span>,
    },
    {
      field: "cash_value",
      headerName: "보유 현금",
      minWidth: 116,
      width: 116,
      align: "right",
      headerAlign: "right",
      sortable: false,
      valueGetter: (_, row) => (row.currency === "KRW" ? row.cash_balance_krw : row.cash_balance_native),
      renderCell: (params) => (
        <span className="tablePlainValue">
          {formatNumber(params.row.currency === "KRW" ? params.row.cash_balance_krw : params.row.cash_balance_native)}
        </span>
      ),
    },
    {
      field: "cash_balance_krw",
      headerName: "저장값 (KRW)",
      minWidth: 116,
      width: 116,
      align: "right",
      headerAlign: "right",
      renderCell: (params) => <span className="tablePlainValue">{formatNumber(params.row.cash_balance_krw)}</span>,
    },
    { field: "cash_currency", headerName: "현금 통화", minWidth: 92, width: 92 },
    {
      field: "intl_shares_value",
      headerName: "Intl Shares Value",
      minWidth: 124,
      width: 124,
      align: "right",
      headerAlign: "right",
      renderCell: (params) => (
        <span className="tablePlainValue">
          {params.row.account_id === "aus_account" ? formatNumber(params.row.intl_shares_value) : "-"}
        </span>
      ),
    },
    {
      field: "intl_shares_change",
      headerName: "Intl Shares Change",
      minWidth: 132,
      width: 132,
      align: "right",
      headerAlign: "right",
      renderCell: (params) => (
        <span className="tablePlainValue">
          {params.row.account_id === "aus_account" ? formatNumber(params.row.intl_shares_change) : "-"}
        </span>
      ),
    },
  ];

  useEffect(() => {
    let alive = true;

    async function load() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch("/api/cash/accounts", { cache: "no-store" });
        const payload = (await response.json()) as CashAccountsResponse;
        if (!response.ok) {
          throw new Error(payload.error ?? "자산관리 데이터를 불러오지 못했습니다.");
        }
        if (alive) {
          setAccounts(payload.accounts ?? []);
          setRates(payload.rates ?? {});
        }
      } catch (loadError) {
        if (alive) {
          setError(loadError instanceof Error ? loadError.message : "자산관리 데이터를 불러오지 못했습니다.");
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

  function updateEditingAccount(field: keyof CashAccountItem, rawValue: string) {
    setEditingAccount((current) => {
      if (!current) {
        return current;
      }

      const numericValue = rawValue === "" ? null : Number(rawValue);
      if (field === "cash_balance_native" && current.currency !== "KRW") {
        const rate = rates[current.currency] ?? 0;
        return {
          ...current,
          cash_balance_native: numericValue,
          cash_balance_krw: numericValue !== null && rate > 0 ? numericValue * rate : current.cash_balance_krw,
        };
      }
      if (field === "cash_balance_krw" && current.currency === "KRW") {
        return {
          ...current,
          cash_balance_krw: numericValue ?? 0,
          cash_balance_native: numericValue,
        };
      }
      return {
        ...current,
        [field]: numericValue,
      };
    });
  }

  function openEditModal(account: CashAccountItem) {
    setEditingAccount(cloneAccount(account));
  }

  function closeEditModal() {
    if (isSaving) {
      return;
    }
    setEditingAccount(null);
  }

  async function handleSave() {
    if (!editingAccount) {
      return;
    }

    try {
      setError(null);
      setIsSaving(true);

      const rate = rates[editingAccount.currency] ?? 0;
      const normalizedCashKrw =
        editingAccount.currency === "KRW"
          ? Number(editingAccount.cash_balance_krw ?? 0)
          : Number(editingAccount.cash_balance_native ?? 0) * rate;

      const payloadAccount = {
        account_id: editingAccount.account_id,
        total_principal: Number(editingAccount.total_principal ?? 0),
        cash_balance_krw: normalizedCashKrw,
        cash_balance_native:
          editingAccount.currency === "KRW"
            ? Number(editingAccount.cash_balance_krw ?? 0)
            : Number(editingAccount.cash_balance_native ?? 0),
        cash_currency: editingAccount.cash_currency,
        intl_shares_value: editingAccount.intl_shares_value,
        intl_shares_change: editingAccount.intl_shares_change,
      };

      const response = await fetch("/api/cash/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ accounts: [payloadAccount] }),
      });
      const payload = (await response.json()) as { error?: string };
      if (!response.ok) {
        throw new Error(payload.error ?? "저장에 실패했습니다.");
      }

      const savedAt = new Date().toISOString();
      setAccounts((current) =>
        current.map((account) =>
          account.account_id === editingAccount.account_id
            ? {
                ...account,
                ...editingAccount,
                cash_balance_krw: normalizedCashKrw,
                updated_at: savedAt,
              }
            : account,
        ),
      );
      toast.success(`[자산-자산 관리] ${editingAccount.name} 저장 완료`);
      setEditingAccount(null);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "저장에 실패했습니다.");
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <div className="appPageStack">
      {error || (rates.AUD ?? 0) <= 0 ? (
        <div className="appBannerStack">
          {error ? <div className="bannerError">{error}</div> : null}
          {(rates.AUD ?? 0) <= 0 ? (
            <div className="bannerWarn">AUD/KRW 환율을 불러오지 못했습니다. 호주 계좌 저장 전에 확인이 필요합니다.</div>
          ) : null}
        </div>
      ) : null}
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBody">
            <AppDataGrid
              rows={gridRows}
              columns={columns}
              loading={loading}
              minHeight="26rem"
              rowHeight={40}
              wrapClassName="appDataGridWrapScrollable"
              fitContentRows
            />
            <div className="tableFooterMeta">마지막 저장: {formatUpdatedAt(latestUpdatedAt)}</div>
          </div>
        </div>
      </section>
      <AppModal
        open={editingAccount !== null}
        title="자산 관리"
        subtitle={editingAccount ? `${editingAccount.icon} ${editingAccount.name} · ${editingAccount.account_id}` : undefined}
        onClose={closeEditModal}
        footer={
          <>
            <button type="button" className="btn btn-outline-secondary" onClick={closeEditModal} disabled={isSaving}>
              취소
            </button>
            <button type="button" className="btn btn-primary" onClick={handleSave} disabled={isSaving || !editingAccount}>
              {isSaving ? "저장 중..." : "저장"}
            </button>
          </>
        }
      >
        {editingAccount ? (
          <div className="appPageStack">
            <div className="row g-3">
              <div className="col-md-6">
                <label className="form-label">투자 원금 (KRW)</label>
                <input
                  className="form-control"
                  type="number"
                  value={normalizeInputValue(editingAccount.total_principal)}
                  onChange={(event) => updateEditingAccount("total_principal", event.target.value)}
                />
              </div>
              <div className="col-md-6">
                <label className="form-label">현금 통화</label>
                <input className="form-control" type="text" value={editingAccount.cash_currency} readOnly />
              </div>
              <div className="col-md-6">
                <label className="form-label">
                  {editingAccount.currency === "KRW" ? "보유 현금 (KRW)" : `보유 현금 (${editingAccount.currency})`}
                </label>
                {editingAccount.currency === "KRW" ? (
                  <input
                    className="form-control"
                    type="number"
                    value={normalizeInputValue(editingAccount.cash_balance_krw)}
                    onChange={(event) => updateEditingAccount("cash_balance_krw", event.target.value)}
                  />
                ) : (
                  <>
                    <input
                      className="form-control"
                      type="number"
                      value={normalizeInputValue(editingAccount.cash_balance_native)}
                      onChange={(event) => updateEditingAccount("cash_balance_native", event.target.value)}
                    />
                  </>
                )}
              </div>
              <div className="col-md-6">
                <label className="form-label">저장값 (KRW)</label>
                <input
                  className="form-control"
                  type="text"
                  value={formatNumber(editingAccount.cash_balance_krw)}
                  readOnly
                />
              </div>
              {editingAccount.account_id === "aus_account" ? (
                <>
                  <div className="col-md-6">
                    <label className="form-label">Intl Shares Value</label>
                    <input
                      className="form-control"
                      type="number"
                      value={normalizeInputValue(editingAccount.intl_shares_value)}
                      onChange={(event) => updateEditingAccount("intl_shares_value", event.target.value)}
                    />
                  </div>
                  <div className="col-md-6">
                    <label className="form-label">Intl Shares Change</label>
                    <input
                      className="form-control"
                      type="number"
                      value={normalizeInputValue(editingAccount.intl_shares_change)}
                      onChange={(event) => updateEditingAccount("intl_shares_change", event.target.value)}
                    />
                  </div>
                </>
              ) : null}
            </div>
          </div>
        ) : null}
      </AppModal>
    </div>
  );
}
