"use client";

import { useEffect, useMemo, useState, useTransition } from "react";

import { useToast } from "../components/ToastProvider";

type DeletedStocksAccountItem = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
};

type DeletedStocksRowItem = {
  ticker: string;
  name: string;
  bucket_id: number;
  bucket_name: string;
  listing_date: string;
  week_volume: number | null;
  return_1w: number | null;
  return_2w: number | null;
  return_1m: number | null;
  return_3m: number | null;
  return_6m: number | null;
  return_12m: number | null;
  deleted_date: string;
  deleted_reason: string;
};

type DeletedStocksResponse = {
  accounts?: DeletedStocksAccountItem[];
  rows?: DeletedStocksRowItem[];
  account_id?: string;
  error?: string;
};

function formatNumber(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR").format(value);
}

function formatPercent(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${value.toFixed(2)}%`;
}

function getSignedMetricClass(value: number | null): string | undefined {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return undefined;
  }
  if (value > 0) {
    return "metricPositive";
  }
  if (value < 0) {
    return "metricNegative";
  }
  return undefined;
}

export function DeletedStocksManager() {
  const [accounts, setAccounts] = useState<DeletedStocksAccountItem[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState("");
  const [rows, setRows] = useState<DeletedStocksRowItem[]>([]);
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [isPending, startTransition] = useTransition();
  const toast = useToast();

  async function load(accountId?: string) {
    setLoading(true);
    setError(null);

    try {
      const search = accountId ? `?account=${encodeURIComponent(accountId)}` : "";
      const response = await fetch(`/api/deleted${search}`, { cache: "no-store" });
      const payload = (await response.json()) as DeletedStocksResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "삭제된 종목 데이터를 불러오지 못했습니다.");
      }

      setAccounts(payload.accounts ?? []);
      setSelectedAccountId(payload.account_id ?? "");
      setRows(payload.rows ?? []);
      setSelectedTickers([]);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "삭제된 종목 데이터를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  const selectedAccount = useMemo(
    () => accounts.find((account) => account.account_id === selectedAccountId) ?? null,
    [accounts, selectedAccountId],
  );

  const selectedTickerSet = useMemo(() => new Set(selectedTickers), [selectedTickers]);
  const allSelected = rows.length > 0 && selectedTickers.length === rows.length;

  function handleAccountChange(nextAccountId: string) {
    void load(nextAccountId);
  }

  function toggleTicker(ticker: string) {
    setSelectedTickers((current) => {
      const normalized = ticker.trim().toUpperCase();
      if (current.includes(normalized)) {
        return current.filter((item) => item !== normalized);
      }
      return [...current, normalized];
    });
  }

  function toggleAll() {
    if (allSelected) {
      setSelectedTickers([]);
      return;
    }
    setSelectedTickers(rows.map((row) => row.ticker.trim().toUpperCase()));
  }

  function handleRestore() {
    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/deleted", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            tickers: selectedTickers,
          }),
        });
        const payload = (await response.json()) as { error?: string; restored_count?: number };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 복구에 실패했습니다.");
        }
        const restoredCount = Number(payload.restored_count ?? 0);
        setRows((current) => current.filter((row) => !selectedTickerSet.has(row.ticker.trim().toUpperCase())));
        setSelectedTickers([]);
        toast.success(`[Momentum ETF-삭제된 종목] ${restoredCount}개 종목 복구 완료`);
      } catch (restoreError) {
        setError(restoreError instanceof Error ? restoreError.message : "종목 복구에 실패했습니다.");
      }
    });
  }

  function handleHardDelete() {
    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/deleted", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            tickers: selectedTickers,
          }),
        });
        const payload = (await response.json()) as { error?: string; deleted_count?: number };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 완전 삭제에 실패했습니다.");
        }
        const deletedCount = Number(payload.deleted_count ?? 0);
        setRows((current) => current.filter((row) => !selectedTickerSet.has(row.ticker.trim().toUpperCase())));
        setSelectedTickers([]);
        toast.success(`[Momentum ETF-삭제된 종목] ${deletedCount}개 종목 영구 삭제 완료`);
      } catch (deleteError) {
        setError(deleteError instanceof Error ? deleteError.message : "종목 완전 삭제에 실패했습니다.");
      }
    });
  }

  if (loading) {
    return (
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBody">
            <p>삭제된 종목 데이터를 불러오는 중...</p>
          </div>
        </div>
      </section>
    );
  }

  return (
    <div className="appPageStack">
      {error ? (
        <div className="appBannerStack">
          {error ? <div className="bannerError">{error}</div> : null}
        </div>
      ) : null}
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBody">
          <div className="tableToolbar">
            <div className="toolbarActions">
              <select
                className="field compactField"
                value={selectedAccountId}
                onChange={(event) => handleAccountChange(event.target.value)}
              >
                {accounts.map((account) => (
                  <option key={account.account_id} value={account.account_id}>
                    {account.order}. {account.name}
                  </option>
                ))}
              </select>
              <button className="secondaryButton" type="button" onClick={toggleAll} disabled={rows.length === 0 || isPending}>
                {allSelected ? "전체 해제" : "전체 선택"}
              </button>
              <button
                className="primaryButton"
                type="button"
                onClick={handleRestore}
                disabled={selectedTickers.length === 0 || isPending}
              >
                선택 복구
              </button>
              <button
                className="secondaryButton dangerButton"
                type="button"
                onClick={handleHardDelete}
                disabled={selectedTickers.length === 0 || isPending}
              >
                완전 삭제
              </button>
            </div>
            <div className="tableMeta">
              {selectedAccount ? (
                <span>
                  {selectedAccount.icon} {selectedAccount.name}
                </span>
              ) : null}
              <span>총 {new Intl.NumberFormat("ko-KR").format(rows.length)}개 종목</span>
              <span>선택 {new Intl.NumberFormat("ko-KR").format(selectedTickers.length)}개</span>
            </div>
          </div>

          <div className="tableWrap">
            <table className="erpTable">
              <thead>
                <tr>
                  <th className="tableCheckboxCell">
                    <input type="checkbox" checked={allSelected} onChange={toggleAll} />
                  </th>
                  <th>버킷</th>
                  <th>티커</th>
                  <th>종목명</th>
                  <th className="tableAlignRight">주간거래량</th>
                  <th className="tableAlignRight">1주(%)</th>
                  <th className="tableAlignRight">2주(%)</th>
                  <th className="tableAlignRight">1달(%)</th>
                  <th className="tableAlignRight">3달(%)</th>
                  <th className="tableAlignRight">6달(%)</th>
                  <th className="tableAlignRight">12달(%)</th>
                  <th>상장일</th>
                  <th>삭제일</th>
                  <th>삭제 사유</th>
                </tr>
              </thead>
              <tbody>
                {rows.length === 0 ? (
                  <tr>
                    <td colSpan={14} className="tableEmpty">
                      삭제된 종목이 없습니다.
                    </td>
                  </tr>
                ) : (
                  rows.map((row) => {
                    const isChecked = selectedTickerSet.has(row.ticker.trim().toUpperCase());
                    return (
                      <tr key={row.ticker} className={isChecked ? "tableRowSelected" : undefined}>
                        <td className="tableCheckboxCell">
                          <input
                            type="checkbox"
                            checked={isChecked}
                            onChange={() => toggleTicker(row.ticker)}
                            disabled={isPending}
                          />
                        </td>
                        <td>{row.bucket_name}</td>
                        <td>{row.ticker}</td>
                        <td>{row.name}</td>
                        <td className="tableAlignRight">{formatNumber(row.week_volume)}</td>
                        <td className={`tableAlignRight ${getSignedMetricClass(row.return_1w) ?? ""}`.trim()}>
                          {formatPercent(row.return_1w)}
                        </td>
                        <td className={`tableAlignRight ${getSignedMetricClass(row.return_2w) ?? ""}`.trim()}>
                          {formatPercent(row.return_2w)}
                        </td>
                        <td className={`tableAlignRight ${getSignedMetricClass(row.return_1m) ?? ""}`.trim()}>
                          {formatPercent(row.return_1m)}
                        </td>
                        <td className={`tableAlignRight ${getSignedMetricClass(row.return_3m) ?? ""}`.trim()}>
                          {formatPercent(row.return_3m)}
                        </td>
                        <td className={`tableAlignRight ${getSignedMetricClass(row.return_6m) ?? ""}`.trim()}>
                          {formatPercent(row.return_6m)}
                        </td>
                        <td className={`tableAlignRight ${getSignedMetricClass(row.return_12m) ?? ""}`.trim()}>
                          {formatPercent(row.return_12m)}
                        </td>
                        <td>{row.listing_date}</td>
                        <td>{row.deleted_date}</td>
                        <td>{row.deleted_reason}</td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
          </div>
        </div>
      </section>
    </div>
  );
}
