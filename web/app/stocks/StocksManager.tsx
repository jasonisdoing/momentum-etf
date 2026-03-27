"use client";

import { useEffect, useMemo, useState, useTransition } from "react";

type StocksAccountItem = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
};

type StocksRowItem = {
  ticker: string;
  name: string;
  bucket_id: number;
  bucket_name: string;
  added_date: string;
  listing_date: string;
  week_volume: number | null;
  return_1w: number | null;
  return_2w: number | null;
  return_1m: number | null;
  return_3m: number | null;
  return_6m: number | null;
  return_12m: number | null;
};

type StocksResponse = {
  accounts?: StocksAccountItem[];
  rows?: StocksRowItem[];
  account_id?: string;
  error?: string;
};

const BUCKET_OPTIONS = [
  { id: 1, name: "1. 모멘텀" },
  { id: 2, name: "2. 혁신기술" },
  { id: 3, name: "3. 시장지수" },
  { id: 4, name: "4. 배당방어" },
  { id: 5, name: "5. 대체헷지" },
];

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

export function StocksManager() {
  const [accounts, setAccounts] = useState<StocksAccountItem[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState("");
  const [rows, setRows] = useState<StocksRowItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [isPending, startTransition] = useTransition();

  async function load(accountId?: string) {
    setLoading(true);
    setError(null);

    try {
      const search = accountId ? `?account=${encodeURIComponent(accountId)}` : "";
      const response = await fetch(`/api/stocks${search}`, { cache: "no-store" });
      const payload = (await response.json()) as StocksResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "종목 관리 데이터를 불러오지 못했습니다.");
      }

      setAccounts(payload.accounts ?? []);
      setSelectedAccountId(payload.account_id ?? "");
      setRows(payload.rows ?? []);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "종목 관리 데이터를 불러오지 못했습니다.");
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

  function handleAccountChange(nextAccountId: string) {
    void load(nextAccountId);
  }

  function handleBucketChange(ticker: string, bucketId: number) {
    startTransition(async () => {
      try {
        setError(null);
        setNotice(null);
        const response = await fetch("/api/stocks", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            ticker,
            bucket_id: bucketId,
          }),
        });
        const payload = (await response.json()) as { error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "버킷 변경에 실패했습니다.");
        }
        setRows((current) =>
          current.map((row) =>
            row.ticker === ticker
              ? {
                  ...row,
                  bucket_id: bucketId,
                  bucket_name: BUCKET_OPTIONS.find((bucket) => bucket.id === bucketId)?.name ?? row.bucket_name,
                }
              : row,
          ),
        );
        setNotice(`${ticker} 버킷 변경 완료`);
      } catch (updateError) {
        setError(updateError instanceof Error ? updateError.message : "버킷 변경에 실패했습니다.");
      }
    });
  }

  function handleDelete(ticker: string) {
    startTransition(async () => {
      try {
        setError(null);
        setNotice(null);
        const response = await fetch("/api/stocks", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            ticker,
          }),
        });
        const payload = (await response.json()) as { error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 삭제에 실패했습니다.");
        }
        setRows((current) => current.filter((row) => row.ticker !== ticker));
        setNotice(`${ticker} 삭제 완료`);
      } catch (deleteError) {
        setError(deleteError instanceof Error ? deleteError.message : "종목 삭제에 실패했습니다.");
      }
    });
  }

  if (loading) {
    return (
      <section className="section">
        <p>종목 관리 데이터를 불러오는 중...</p>
      </section>
    );
  }

  return (
    <section className="section">
      {error ? <div className="bannerError">{error}</div> : null}
      {notice ? <div className="bannerSuccess">{notice}</div> : null}

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
        </div>
        <div className="tableMeta">
          {selectedAccount ? (
            <span>
              {selectedAccount.icon} {selectedAccount.name}
            </span>
          ) : null}
          <span>총 {new Intl.NumberFormat("ko-KR").format(rows.length)}개 종목</span>
        </div>
      </div>

      <div className="tableWrap">
        <table className="erpTable">
          <thead>
            <tr>
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
              <th>추가일자</th>
              <th>작업</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={13} className="tableEmpty">
                  종목 데이터가 없습니다.
                </td>
              </tr>
            ) : (
              rows.map((row) => (
                <tr key={row.ticker}>
                  <td>
                    <select
                      className="tableField"
                      value={row.bucket_id}
                      onChange={(event) => handleBucketChange(row.ticker, Number(event.target.value))}
                      disabled={isPending}
                    >
                      {BUCKET_OPTIONS.map((bucket) => (
                        <option key={bucket.id} value={bucket.id}>
                          {bucket.name}
                        </option>
                      ))}
                    </select>
                  </td>
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
                  <td>{row.added_date}</td>
                  <td>
                    <button
                      className="secondaryButton"
                      type="button"
                      onClick={() => handleDelete(row.ticker)}
                      disabled={isPending}
                    >
                      삭제
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
