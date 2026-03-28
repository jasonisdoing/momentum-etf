"use client";

import { useEffect, useState } from "react";

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

  if (loading) {
    return (
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBody">
            <p>자산관리 데이터를 불러오는 중...</p>
          </div>
        </div>
      </section>
    );
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
          <div className="tableWrap">
            <table className="erpTable">
              <thead>
                <tr>
                  <th />
                  <th>계좌</th>
                  <th>투자 원금 (KRW)</th>
                  <th>보유 현금</th>
                  <th>저장값 (KRW)</th>
                  <th>현금 통화</th>
                  <th>Intl Shares Value</th>
                  <th>Intl Shares Change</th>
                </tr>
              </thead>
              <tbody>
                {accounts.map((account) => (
                  <tr key={account.account_id}>
                    <td>
                      <button
                        type="button"
                        className="btn btn-link btn-sm p-0 appEditLink"
                        onClick={() => openEditModal(account)}
                      >
                        Edit
                      </button>
                    </td>
                    <td>
                      <div className="tableAccountCell">
                        <strong>
                          {account.order}. {account.icon} {account.name}
                        </strong>
                        <span>{account.account_id}</span>
                      </div>
                    </td>
                    <td>
                      <span className="tablePlainValue">{formatNumber(account.total_principal)}</span>
                    </td>
                    <td>
                      {account.currency === "KRW" ? (
                        <span className="tablePlainValue">{formatNumber(account.cash_balance_krw)}</span>
                      ) : (
                        <span className="tablePlainValue">{formatNumber(account.cash_balance_native)}</span>
                      )}
                    </td>
                    <td>
                      <span className="tablePlainValue">{formatNumber(account.cash_balance_krw)}</span>
                    </td>
                    <td>{account.cash_currency}</td>
                    <td>
                      {account.account_id === "aus_account" ? (
                        <span className="tablePlainValue">{formatNumber(account.intl_shares_value)}</span>
                      ) : (
                        <span className="tableMuted">-</span>
                      )}
                    </td>
                    <td>
                      {account.account_id === "aus_account" ? (
                        <span className="tablePlainValue">{formatNumber(account.intl_shares_change)}</span>
                      ) : (
                        <span className="tableMuted">-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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
