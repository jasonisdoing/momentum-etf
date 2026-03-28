"use client";

import {
  IconArrowBackUp,
  IconChecks,
  IconLayoutGrid,
  IconPlaylistX,
  IconTrash,
} from "@tabler/icons-react";
import { useEffect, useMemo, useState, useTransition } from "react";

import { AppModal } from "../components/AppModal";
import { useToast } from "../components/ToastProvider";

type StocksAccountItem = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
};

type ActiveStocksRowItem = {
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

type StocksResponse = {
  accounts?: StocksAccountItem[];
  rows?: ActiveStocksRowItem[];
  account_id?: string;
  error?: string;
};

type DeletedStocksResponse = {
  accounts?: StocksAccountItem[];
  rows?: DeletedStocksRowItem[];
  account_id?: string;
  error?: string;
};

type ViewMode = "active" | "deleted";

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

function getSignedMetricClass(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) {
    return "";
  }
  return value > 0 ? "stocksPositive" : "stocksNegative";
}

function getBucketClass(bucketId: number): string {
  return `stocksBucket stocksBucket${bucketId}`;
}

export function StocksManager() {
  const [accounts, setAccounts] = useState<StocksAccountItem[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("active");
  const [activeRows, setActiveRows] = useState<ActiveStocksRowItem[]>([]);
  const [deletedRows, setDeletedRows] = useState<DeletedStocksRowItem[]>([]);
  const [selectedDeletedTickers, setSelectedDeletedTickers] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [isPending, startTransition] = useTransition();
  const [editingRow, setEditingRow] = useState<ActiveStocksRowItem | null>(null);
  const [editingBucketId, setEditingBucketId] = useState<number>(1);
  const toast = useToast();

  async function load(mode: ViewMode, accountId?: string) {
    setLoading(true);
    setError(null);

    try {
      const search = accountId ? `?account=${encodeURIComponent(accountId)}` : "";
      const apiPath = mode === "active" ? `/api/stocks${search}` : `/api/deleted${search}`;
      const response = await fetch(apiPath, { cache: "no-store" });
      const payload = (await response.json()) as StocksResponse | DeletedStocksResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "종목 관리 데이터를 불러오지 못했습니다.");
      }

      setAccounts(payload.accounts ?? []);
      setSelectedAccountId(payload.account_id ?? "");
      setSelectedDeletedTickers([]);

      if (mode === "active") {
        setActiveRows((payload.rows as ActiveStocksRowItem[] | undefined) ?? []);
      } else {
        setDeletedRows((payload.rows as DeletedStocksRowItem[] | undefined) ?? []);
      }
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "종목 관리 데이터를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load(viewMode);
  }, []);

  const selectedAccount = useMemo(
    () => accounts.find((account) => account.account_id === selectedAccountId) ?? null,
    [accounts, selectedAccountId],
  );

  const selectedDeletedTickerSet = useMemo(() => new Set(selectedDeletedTickers), [selectedDeletedTickers]);
  const allDeletedSelected = deletedRows.length > 0 && selectedDeletedTickers.length === deletedRows.length;

  function handleAccountChange(nextAccountId: string) {
    void load(viewMode, nextAccountId);
  }

  function handleViewModeChange(nextMode: ViewMode) {
    if (nextMode === viewMode) {
      return;
    }
    setViewMode(nextMode);
    void load(nextMode, selectedAccountId);
  }

  function handleBucketChange(ticker: string, bucketId: number) {
    startTransition(async () => {
      try {
        setError(null);
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
        setActiveRows((current) =>
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
        const targetRow = activeRows.find((row) => row.ticker === ticker);
        const label = targetRow ? `${targetRow.name}(${targetRow.ticker})` : ticker;
        toast.success(`[Momentum ETF-종목 관리] ${label} 변경 완료`);
      } catch (updateError) {
        setError(updateError instanceof Error ? updateError.message : "버킷 변경에 실패했습니다.");
      }
    });
  }

  function handleDelete(ticker: string) {
    startTransition(async () => {
      try {
        setError(null);
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
        setActiveRows((current) => current.filter((row) => row.ticker !== ticker));
        const targetRow = activeRows.find((row) => row.ticker === ticker);
        const label = targetRow ? `${targetRow.name}(${targetRow.ticker})` : ticker;
        toast.success(`[Momentum ETF-종목 관리] ${label} 삭제 완료`);
      } catch (deleteError) {
        setError(deleteError instanceof Error ? deleteError.message : "종목 삭제에 실패했습니다.");
      }
    });
  }

  function openEditModal(row: ActiveStocksRowItem) {
    setEditingRow(row);
    setEditingBucketId(row.bucket_id);
  }

  function closeEditModal() {
    if (isPending) {
      return;
    }
    setEditingRow(null);
  }

  function handleSaveFromModal() {
    if (!editingRow) {
      return;
    }

    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/stocks", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            ticker: editingRow.ticker,
            bucket_id: editingBucketId,
          }),
        });
        const payload = (await response.json()) as { error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "버킷 변경에 실패했습니다.");
        }
        setActiveRows((current) =>
          current.map((row) =>
            row.ticker === editingRow.ticker
              ? {
                  ...row,
                  bucket_id: editingBucketId,
                  bucket_name: BUCKET_OPTIONS.find((bucket) => bucket.id === editingBucketId)?.name ?? row.bucket_name,
                }
              : row,
          ),
        );
        toast.success(`[Momentum ETF-종목 관리] ${editingRow.name}(${editingRow.ticker}) 변경 완료`);
        setEditingRow(null);
      } catch (updateError) {
        setError(updateError instanceof Error ? updateError.message : "버킷 변경에 실패했습니다.");
      }
    });
  }

  function handleDeleteFromModal() {
    if (!editingRow) {
      return;
    }

    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/stocks", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            ticker: editingRow.ticker,
          }),
        });
        const payload = (await response.json()) as { error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 삭제에 실패했습니다.");
        }
        setActiveRows((current) => current.filter((row) => row.ticker !== editingRow.ticker));
        toast.success(`[Momentum ETF-종목 관리] ${editingRow.name}(${editingRow.ticker}) 삭제 완료`);
        setEditingRow(null);
      } catch (deleteError) {
        setError(deleteError instanceof Error ? deleteError.message : "종목 삭제에 실패했습니다.");
      }
    });
  }

  function toggleDeletedTicker(ticker: string) {
    setSelectedDeletedTickers((current) => {
      const normalized = ticker.trim().toUpperCase();
      if (current.includes(normalized)) {
        return current.filter((item) => item !== normalized);
      }
      return [...current, normalized];
    });
  }

  function toggleAllDeleted() {
    if (allDeletedSelected) {
      setSelectedDeletedTickers([]);
      return;
    }
    setSelectedDeletedTickers(deletedRows.map((row) => row.ticker.trim().toUpperCase()));
  }

  function handleRestoreDeleted() {
    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/deleted", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            tickers: selectedDeletedTickers,
          }),
        });
        const payload = (await response.json()) as { error?: string; restored_count?: number };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 복구에 실패했습니다.");
        }
        const restoredCount = Number(payload.restored_count ?? 0);
        setDeletedRows((current) =>
          current.filter((row) => !selectedDeletedTickerSet.has(row.ticker.trim().toUpperCase())),
        );
        setSelectedDeletedTickers([]);
        toast.success(`[Momentum ETF-종목 관리] ${restoredCount}개 종목 복구 완료`);
      } catch (restoreError) {
        setError(restoreError instanceof Error ? restoreError.message : "종목 복구에 실패했습니다.");
      }
    });
  }

  function handleHardDeleteDeleted() {
    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/deleted", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            tickers: selectedDeletedTickers,
          }),
        });
        const payload = (await response.json()) as { error?: string; deleted_count?: number };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 완전 삭제에 실패했습니다.");
        }
        const deletedCount = Number(payload.deleted_count ?? 0);
        setDeletedRows((current) =>
          current.filter((row) => !selectedDeletedTickerSet.has(row.ticker.trim().toUpperCase())),
        );
        setSelectedDeletedTickers([]);
        toast.success(`[Momentum ETF-종목 관리] ${deletedCount}개 종목 영구 삭제 완료`);
      } catch (deleteError) {
        setError(deleteError instanceof Error ? deleteError.message : "종목 완전 삭제에 실패했습니다.");
      }
    });
  }

  return (
    <div className="appPageStack appPageStackFill">
      {error ? (
        <div className="appBannerStack">
          {error ? <div className="alert alert-danger mb-0">{error}</div> : null}
        </div>
      ) : null}

      <section className="appSection appSectionFill stocksPage">
        <div className="card appCard stocksCard">
          <div className="card-header">
            <div className="stocksToolbar w-100">
              <div className="stocksToolbarLeft">
                <div className="stocksSelect">
                  <select
                    className="form-select"
                    aria-label="계좌 선택"
                    value={selectedAccountId}
                    onChange={(event) => handleAccountChange(event.target.value)}
                    disabled={loading}
                  >
                    {accounts.map((account) => (
                      <option key={account.account_id} value={account.account_id}>
                        {account.order}. {account.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="stocksToolbarModes">
                  <button
                    className={
                      viewMode === "active" ? "btn stocksModeButton is-active" : "btn btn-outline-secondary stocksModeButton"
                    }
                    type="button"
                    onClick={() => handleViewModeChange("active")}
                  >
                    <IconLayoutGrid size={16} stroke={1.75} />
                    <span>등록된 종목</span>
                  </button>
                  <button
                    className={
                      viewMode === "deleted" ? "btn stocksModeButton is-active" : "btn btn-outline-secondary stocksModeButton"
                    }
                    type="button"
                    onClick={() => handleViewModeChange("deleted")}
                  >
                    <IconPlaylistX size={16} stroke={1.75} />
                    <span>삭제된 종목</span>
                  </button>
                </div>
              </div>

              <div className="stocksToolbarRight">
                <div className="stocksSummary">
                  {selectedAccount ? (
                    <span className="badge stocksMetricBadge">
                      {selectedAccount.icon} {selectedAccount.name}
                    </span>
                  ) : null}
                  <span className="badge stocksMetricBadge">
                    {viewMode === "active"
                      ? `총 ${new Intl.NumberFormat("ko-KR").format(activeRows.length)}개`
                      : `총 ${new Intl.NumberFormat("ko-KR").format(deletedRows.length)}개`}
                  </span>
                  {viewMode === "deleted" ? (
                    <span className="badge stocksMetricBadge">
                      선택 {new Intl.NumberFormat("ko-KR").format(selectedDeletedTickers.length)}개
                    </span>
                  ) : null}
                </div>

                {viewMode === "deleted" ? (
                  <div className="btn-list">
                    <button
                      className="btn btn-outline-secondary"
                      type="button"
                      onClick={toggleAllDeleted}
                      disabled={deletedRows.length === 0 || isPending}
                    >
                      <IconChecks size={16} stroke={1.75} />
                      <span>{allDeletedSelected ? "전체 해제" : "전체 선택"}</span>
                    </button>
                    <button
                      className="btn btn-primary"
                      type="button"
                      onClick={handleRestoreDeleted}
                      disabled={selectedDeletedTickers.length === 0 || isPending}
                    >
                      <IconArrowBackUp size={16} stroke={1.75} />
                      <span>선택 복구</span>
                    </button>
                    <button
                      className="btn btn-outline-danger"
                      type="button"
                      onClick={handleHardDeleteDeleted}
                      disabled={selectedDeletedTickers.length === 0 || isPending}
                    >
                      <IconTrash size={16} stroke={1.75} />
                      <span>완전 삭제</span>
                    </button>
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          <div className="stocksTableWrap">
            <table className="table table-vcenter card-table table-nowrap stocksTable">
              <thead>
                {viewMode === "active" ? (
                  <tr>
                    <th className="w-1"></th>
                    <th>버킷</th>
                    <th>티커</th>
                    <th>종목명</th>
                    <th className="text-end">주간거래량</th>
                    <th className="text-end">1주(%)</th>
                    <th className="text-end">2주(%)</th>
                    <th className="text-end">1달(%)</th>
                    <th className="text-end">3달(%)</th>
                    <th className="text-end">6달(%)</th>
                    <th className="text-end">12달(%)</th>
                    <th className="stocksDateCol">상장일</th>
                    <th className="stocksDateCol">추가일자</th>
                  </tr>
                ) : (
                  <tr>
                    <th className="stocksCheckboxCell">
                      <input type="checkbox" checked={allDeletedSelected} onChange={toggleAllDeleted} />
                    </th>
                    <th>버킷</th>
                    <th>티커</th>
                    <th>종목명</th>
                    <th className="text-end">주간거래량</th>
                    <th className="text-end">1주(%)</th>
                    <th className="text-end">2주(%)</th>
                    <th className="text-end">1달(%)</th>
                    <th className="text-end">3달(%)</th>
                    <th className="text-end">6달(%)</th>
                    <th className="text-end">12달(%)</th>
                    <th className="stocksDateCol">상장일</th>
                    <th className="stocksDateCol">삭제일</th>
                    <th>삭제 사유</th>
                  </tr>
                )}
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={viewMode === "active" ? 13 : 14} className="stocksEmpty">
                      데이터를 불러오는 중...
                    </td>
                  </tr>
                ) : null}

                {!loading && viewMode === "active" && activeRows.length === 0 ? (
                  <tr>
                    <td colSpan={13} className="stocksEmpty">
                      등록된 종목이 없습니다.
                    </td>
                  </tr>
                ) : null}

                {!loading && viewMode === "deleted" && deletedRows.length === 0 ? (
                  <tr>
                    <td colSpan={14} className="stocksEmpty">
                      삭제된 종목이 없습니다.
                    </td>
                  </tr>
                ) : null}

                {!loading && viewMode === "active"
                  ? activeRows.map((row) => (
                      <tr key={row.ticker}>
                        <td className="text-secondary">
                          <button
                            className="btn btn-link btn-sm p-0 appEditLink"
                            type="button"
                            onClick={() => openEditModal(row)}
                          >
                            Edit
                          </button>
                        </td>
                        <td>
                          <span className={getBucketClass(row.bucket_id)}>{row.bucket_name}</span>
                        </td>
                        <td className="appCodeText">{row.ticker}</td>
                        <td>{row.name}</td>
                        <td className="text-end">{formatNumber(row.week_volume)}</td>
                        <td className={`text-end ${getSignedMetricClass(row.return_1w)}`.trim()}>
                          {formatPercent(row.return_1w)}
                        </td>
                        <td className={`text-end ${getSignedMetricClass(row.return_2w)}`.trim()}>
                          {formatPercent(row.return_2w)}
                        </td>
                        <td className={`text-end ${getSignedMetricClass(row.return_1m)}`.trim()}>
                          {formatPercent(row.return_1m)}
                        </td>
                        <td className={`text-end ${getSignedMetricClass(row.return_3m)}`.trim()}>
                          {formatPercent(row.return_3m)}
                        </td>
                        <td className={`text-end ${getSignedMetricClass(row.return_6m)}`.trim()}>
                          {formatPercent(row.return_6m)}
                        </td>
                        <td className={`text-end ${getSignedMetricClass(row.return_12m)}`.trim()}>
                          {formatPercent(row.return_12m)}
                        </td>
                        <td className="stocksDateCol">{row.listing_date}</td>
                        <td className="stocksDateCol">{row.added_date}</td>
                      </tr>
                    ))
                  : null}

                {!loading && viewMode === "deleted"
                  ? deletedRows.map((row) => {
                      const isChecked = selectedDeletedTickerSet.has(row.ticker.trim().toUpperCase());
                      return (
                        <tr key={row.ticker} className={isChecked ? "stocksSelectedRow" : undefined}>
                          <td className="stocksCheckboxCell">
                            <input
                              type="checkbox"
                              checked={isChecked}
                              onChange={() => toggleDeletedTicker(row.ticker)}
                              disabled={isPending}
                            />
                          </td>
                          <td>
                            <span className={getBucketClass(row.bucket_id)}>{row.bucket_name}</span>
                          </td>
                          <td className="appCodeText">{row.ticker}</td>
                          <td>{row.name}</td>
                          <td className="text-end">{formatNumber(row.week_volume)}</td>
                          <td className={`text-end ${getSignedMetricClass(row.return_1w)}`.trim()}>
                            {formatPercent(row.return_1w)}
                          </td>
                          <td className={`text-end ${getSignedMetricClass(row.return_2w)}`.trim()}>
                            {formatPercent(row.return_2w)}
                          </td>
                          <td className={`text-end ${getSignedMetricClass(row.return_1m)}`.trim()}>
                            {formatPercent(row.return_1m)}
                          </td>
                          <td className={`text-end ${getSignedMetricClass(row.return_3m)}`.trim()}>
                            {formatPercent(row.return_3m)}
                          </td>
                          <td className={`text-end ${getSignedMetricClass(row.return_6m)}`.trim()}>
                            {formatPercent(row.return_6m)}
                          </td>
                          <td className={`text-end ${getSignedMetricClass(row.return_12m)}`.trim()}>
                            {formatPercent(row.return_12m)}
                          </td>
                          <td className="stocksDateCol">{row.listing_date}</td>
                          <td className="stocksDateCol">{row.deleted_date}</td>
                          <td className="stocksMuted">{row.deleted_reason}</td>
                        </tr>
                      );
                    })
                  : null}
              </tbody>
            </table>
          </div>
        </div>
      </section>
      <AppModal
        open={Boolean(editingRow)}
        title="종목 편집"
        onClose={closeEditModal}
        footer={
          <>
            <button type="button" className="btn me-auto btn-outline-danger" onClick={handleDeleteFromModal} disabled={isPending}>
              <IconTrash size={16} stroke={1.9} />
              <span>삭제</span>
            </button>
            <button type="button" className="btn btn-link link-secondary" onClick={closeEditModal} disabled={isPending}>
              취소
            </button>
            <button type="button" className="btn btn-primary" onClick={handleSaveFromModal} disabled={isPending}>
              저장
            </button>
          </>
        }
      >
        {editingRow ? (
          <>
            <div className="mb-3">
              <label className="form-label">티커</label>
              <div className="form-control-plaintext appCodeText">{editingRow.ticker}</div>
            </div>
            <div className="mb-3">
              <label className="form-label">종목명</label>
              <div className="form-control-plaintext">{editingRow.name}</div>
            </div>
            <div className="mb-3">
              <label className="form-label">버킷</label>
              <select
                className="form-select"
                value={editingBucketId}
                onChange={(event) => setEditingBucketId(Number(event.target.value))}
                disabled={isPending}
              >
                {BUCKET_OPTIONS.map((bucket) => (
                  <option key={bucket.id} value={bucket.id}>
                    {bucket.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="row g-2 text-secondary small">
              <div className="col-6">상장일: {editingRow.listing_date}</div>
              <div className="col-6">추가일자: {editingRow.added_date}</div>
            </div>
          </>
        ) : null}
      </AppModal>
    </div>
  );
}
