"use client";

import { useEffect, useState, useTransition } from "react";

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

export function CashManager() {
  const [accounts, setAccounts] = useState<CashAccountItem[]>([]);
  const [rates, setRates] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [isPending, startTransition] = useTransition();
  const latestUpdatedAt = getLatestUpdatedAt(accounts);

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

  function updateAccount(accountId: string, field: keyof CashAccountItem, rawValue: string) {
    setAccounts((current) =>
      current.map((account) => {
        if (account.account_id !== accountId) {
          return account;
        }

        const numericValue = rawValue === "" ? null : Number(rawValue);
        if (field === "cash_balance_native" && account.currency !== "KRW") {
          const rate = rates[account.currency] ?? 0;
          return {
            ...account,
            cash_balance_native: numericValue,
            cash_balance_krw: numericValue !== null && rate > 0 ? numericValue * rate : account.cash_balance_krw,
          };
        }
        if (field === "cash_balance_krw" && account.currency === "KRW") {
          return {
            ...account,
            cash_balance_krw: numericValue ?? 0,
            cash_balance_native: numericValue,
          };
        }
        return {
          ...account,
          [field]: numericValue,
        };
      }),
    );
  }

  function handleSave() {
    startTransition(async () => {
      try {
        setError(null);
        setNotice(null);

        const payloadAccounts = accounts.map((account) => {
          const rate = rates[account.currency] ?? 0;
          const normalizedCashKrw =
            account.currency === "KRW"
              ? Number(account.cash_balance_krw ?? 0)
              : Number(account.cash_balance_native ?? 0) * rate;

          return {
            account_id: account.account_id,
            total_principal: Number(account.total_principal ?? 0),
            cash_balance_krw: normalizedCashKrw,
            cash_balance_native:
              account.currency === "KRW"
                ? Number(account.cash_balance_krw ?? 0)
                : Number(account.cash_balance_native ?? 0),
            cash_currency: account.cash_currency,
            intl_shares_value: account.intl_shares_value,
            intl_shares_change: account.intl_shares_change,
          };
        });

        const response = await fetch("/api/cash/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ accounts: payloadAccounts }),
        });
        const payload = (await response.json()) as { error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "저장에 실패했습니다.");
        }
        setNotice("저장 완료");
      } catch (saveError) {
        setError(saveError instanceof Error ? saveError.message : "저장에 실패했습니다.");
      }
    });
  }

  if (loading) {
    return <section className="section"><p>자산관리 데이터를 불러오는 중...</p></section>;
  }

  return (
    <section className="section">
      {error ? <div className="bannerError">{error}</div> : null}
      {notice ? <div className="bannerSuccess">{notice}</div> : null}
      {(rates.AUD ?? 0) <= 0 ? (
        <div className="bannerWarn">AUD/KRW 환율을 불러오지 못했습니다. 호주 계좌 저장 전에 확인이 필요합니다.</div>
      ) : null}

      <div className="tableToolbar">
        <div />
        <button className="primaryButton" type="button" onClick={handleSave} disabled={isPending}>
          {isPending ? "저장 중..." : "전체 계좌 저장"}
        </button>
      </div>

      <div className="tableWrap">
        <table className="erpTable">
          <thead>
            <tr>
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
                  <div className="tableAccountCell">
                    <strong>
                      {account.order}. {account.icon} {account.name}
                    </strong>
                    <span>{account.account_id}</span>
                  </div>
                </td>
                <td>
                  <input
                    className="tableField"
                    type="number"
                    value={normalizeInputValue(account.total_principal)}
                    onChange={(event) => updateAccount(account.account_id, "total_principal", event.target.value)}
                  />
                </td>
                <td>
                  {account.currency === "KRW" ? (
                    <input
                      className="tableField"
                      type="number"
                      value={normalizeInputValue(account.cash_balance_krw)}
                      onChange={(event) => {
                        updateAccount(account.account_id, "cash_balance_krw", event.target.value);
                        updateAccount(account.account_id, "cash_balance_native", event.target.value);
                      }}
                    />
                  ) : (
                    <div className="tableCellStack">
                      <input
                        className="tableField"
                        type="number"
                        value={normalizeInputValue(account.cash_balance_native)}
                        onChange={(event) => updateAccount(account.account_id, "cash_balance_native", event.target.value)}
                      />
                      <span className="tableSubtext">
                        {account.currency}/KRW {formatNumber(rates[account.currency] ?? 0)}
                      </span>
                    </div>
                  )}
                </td>
                <td>
                  <div className="tableReadonly">{formatNumber(account.cash_balance_krw)}</div>
                </td>
                <td>{account.cash_currency}</td>
                <td>
                  {account.account_id === "aus_account" ? (
                    <input
                      className="tableField"
                      type="number"
                      value={normalizeInputValue(account.intl_shares_value)}
                      onChange={(event) => updateAccount(account.account_id, "intl_shares_value", event.target.value)}
                    />
                  ) : (
                    <span className="tableMuted">-</span>
                  )}
                </td>
                <td>
                  {account.account_id === "aus_account" ? (
                    <input
                      className="tableField"
                      type="number"
                      value={normalizeInputValue(account.intl_shares_change)}
                      onChange={(event) => updateAccount(account.account_id, "intl_shares_change", event.target.value)}
                    />
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
    </section>
  );
}
