"use client";

import {
  IconPlus,
  IconSearch,
} from "@tabler/icons-react";
import { useEffect, useMemo, useState, useTransition } from "react";
import { type GridColDef, type GridRenderCellParams } from "@mui/x-data-grid";

import { AppDataGrid } from "../components/AppDataGrid";
import { AppModal } from "../components/AppModal";
import {
  readRememberedMomentumEtfAccountId,
  writeRememberedMomentumEtfAccountId,
} from "../components/account-selection";
import { useToast } from "../components/ToastProvider";

type AccountConfig = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
};

type TickerDef = {
  [key: string]: string | number | null;
  ticker: string;
  name: string;
  bucket_id: number;
  bucket_name: string;
  added_date: string;
  listing_date: string;
  week_volume: number | null;
  return_1d: number | null;
};

type TargetItemRow = TickerDef & {
  ratio: number;
  id: string;
};

type AccountStocksPayload = {
  accounts: AccountConfig[];
  account_id: string;
  monthly_return_labels?: string[];
  available_tickers: TickerDef[];
  rows: TargetItemRow[];
  error?: string;
};

type StockValidationState = {
  ticker: string;
  name: string;
  listing_date: string;
  status: "active" | "deleted" | "new";
  is_deleted: boolean;
  deleted_reason: string;
  bucket_id: number;
};

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
  return value > 0 ? "metricPositive" : "metricNegative";
}

export function AccountStocksManager() {
  const [accounts, setAccounts] = useState<AccountConfig[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState(readRememberedMomentumEtfAccountId() ?? "");
  const [availableTickers, setAvailableTickers] = useState<TickerDef[]>([]);
  const [monthlyReturnLabels, setMonthlyReturnLabels] = useState<string[]>([]);
  const [rows, setRows] = useState<TargetItemRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [isPending, startTransition] = useTransition();

  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [addTickerInput, setAddTickerInput] = useState("");
  const [validatedCandidate, setValidatedCandidate] = useState<StockValidationState | null>(null);
  const [isValidatingTicker, setIsValidatingTicker] = useState(false);
  const [addRatio, setAddRatio] = useState<number | "">(0);

  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [editingRow, setEditingRow] = useState<TargetItemRow | null>(null);
  const [editingRatio, setEditingRatio] = useState<number>(0);

  const toast = useToast();

  async function load(accountId?: string) {
    setLoading(true);
    setError(null);

    try {
      const search = accountId ? `?account=${encodeURIComponent(accountId)}` : "";
      const response = await fetch(`/api/account-stocks${search}`, { cache: "no-store" });
      const payload = (await response.json()) as AccountStocksPayload;
      if (!response.ok) {
        throw new Error(payload.error ?? "타겟 비중 데이터를 불러오지 못했습니다.");
      }

      setAccounts(payload.accounts ?? []);
      const nextAccountId = payload.account_id ?? "";
      setSelectedAccountId(nextAccountId);
      writeRememberedMomentumEtfAccountId(nextAccountId);
      setAvailableTickers(payload.available_tickers ?? []);
      setMonthlyReturnLabels(payload.monthly_return_labels ?? []);

      const sortedRows = (payload.rows || []).map(r => ({ 
        ...r, 
        id: r.ticker.trim().toUpperCase() 
      })).sort((a, b) => {
        // 1. 버킷 (오름차순)
        if (a.bucket_id !== b.bucket_id) {
          return a.bucket_id - b.bucket_id;
        }
        // 2. 비중 (내림차순)
        if (a.ratio !== b.ratio) {
          return b.ratio - a.ratio;
        }
        // 3. 일간(%) (내림차순)
        const retA = a.return_1d ?? -999;
        const retB = b.return_1d ?? -999;
        return retB - retA;
      });

      setRows(sortedRows);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "데이터를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load(readRememberedMomentumEtfAccountId() ?? undefined);
  }, []);

  const selectedAccountItem = useMemo(
    () => accounts.find((a) => a.account_id === selectedAccountId) ?? null,
    [accounts, selectedAccountId],
  );

  function handleAccountChange(nextAccountId: string) {
    setSelectedAccountId(nextAccountId);
    writeRememberedMomentumEtfAccountId(nextAccountId);
    void load(nextAccountId);
  }

  function handleRatioChange(ticker: string, idx: number, ratioStr: string) {
    const val = parseFloat(ratioStr);
    const num = isNaN(val) ? 0 : val;
    setRows(current => current.map(row => row.ticker === ticker ? { ...row, ratio: num } : row));
  }

  function handleRatioSave(ticker: string, name: string, ratio: number) {
    if (ratio <= 0 || ratio > 100) {
      toast.error("비중은 0.1%에서 100% 사이여야 합니다.");
      return;
    }

    startTransition(async () => {
      try {
        setError(null);
        const finalRatio = parseFloat(ratio.toFixed(1));
        const response = await fetch("/api/account-stocks", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            ticker,
            ratio: finalRatio,
          }),
        });
        const payload = (await response.json()) as { error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "비중 변경에 실패했습니다.");
        }
        toast.success(`[포트폴리오 비중] ${name}(${ticker}) 변경 완료: ${finalRatio.toFixed(1)}%`);
        void load(selectedAccountId);
      } catch (updateError) {
        setError(updateError instanceof Error ? updateError.message : "비중 변경에 실패했습니다.");
      }
    });
  }

  function handleDeleteTarget(ticker: string, name: string) {
    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/account-stocks", {
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
        toast.success(`[포트폴리오 비중] ${name}(${ticker}) 삭제 완료 (Hard Delete)`);
      } catch (deleteError) {
        setError(deleteError instanceof Error ? deleteError.message : "종목 삭제에 실패했습니다.");
      }
    });
  }

  function openEditModal(row: TargetItemRow) {
    setEditingRow(row);
    setEditingRatio(row.ratio);
    setIsEditModalOpen(true);
  }

  function closeEditModal() {
    setIsEditModalOpen(false);
    setEditingRow(null);
  }

  async function handleBatchEditSave() {
    if (!editingRow) return;
    handleRatioSave(editingRow.ticker, editingRow.name, editingRatio);
    closeEditModal();
  }

  const columns = useMemo<GridColDef<TargetItemRow>[]>(
    () => [
      {
        field: "__edit__",
        headerName: "",
        width: 58,
        sortable: false,
        filterable: false,
        renderCell: (params: GridRenderCellParams<TargetItemRow>) => (
          <button className="btn btn-link btn-sm p-0 appEditLink" type="button" onClick={() => openEditModal(params.row)}>
            Edit
          </button>
        ),
      },
      {
        field: "bucket_name",
        headerName: "버킷",
        width: 112,
        minWidth: 112,
        sortable: false,
        cellClassName: (params) => `appBucketCell appBucketCell${params.row.bucket_id}`,
      },
      {
        field: "ticker",
        headerName: "티커",
        width: 100,
        minWidth: 100,
        renderCell: (params: GridRenderCellParams<TargetItemRow>) => <span className="appCodeText" style={{ fontWeight: 600 }}>{params.row.ticker}</span>,
      },
      { field: "name", headerName: "종목명", minWidth: 220, flex: 1 },
      {
        field: "ratio",
        headerName: "비중(%)",
        width: 100,
        minWidth: 100,
        sortable: true,
        headerAlign: "right",
        align: "right",
        renderCell: (params: GridRenderCellParams<TargetItemRow>) => (
          <span style={{ fontWeight: 800, fontSize: "0.94rem" }}>
            {params.row.ratio.toFixed(1)}
          </span>
        ),
      },
      {
        field: "return_1d",
        headerName: "일간(%)",
        width: 88,
        minWidth: 88,
        align: "right",
        headerAlign: "right",
        renderCell: (params: GridRenderCellParams<TargetItemRow>) => (
          <span className={getSignedMetricClass(params.row.return_1d)}>
             {formatPercent(params.row.return_1d)}
          </span>
        ),
      },
      ...monthlyReturnLabels.map((label) => ({
        field: label,
        headerName: label,
        width: 108,
        minWidth: 108,
        align: "right" as const,
        headerAlign: "right" as const,
        renderCell: (params: GridRenderCellParams<TargetItemRow>) => (
          <span className={getSignedMetricClass((params.row[label] as number | null) ?? null)}>
            {formatPercent((params.row[label] as number | null) ?? null)}
          </span>
        ),
      })),
      { field: "listing_date", headerName: "상장일", width: 112, minWidth: 112 },
    ],
    [monthlyReturnLabels]
  );

  function openAddModal() {
    setIsAddModalOpen(true);
    setAddTickerInput("");
    setValidatedCandidate(null);
    setAddRatio("");
  }

  function closeAddModal() {
    if (isPending || isValidatingTicker) return;
    setIsAddModalOpen(false);
    setAddTickerInput("");
    setValidatedCandidate(null);
    setAddRatio("");
  }

  async function handleValidateTicker() {
    try {
      setError(null);
      setIsValidatingTicker(true);
      
      const ticker = addTickerInput.trim().toUpperCase();
      const def = availableTickers.find(t => t.ticker.toUpperCase() === ticker);
      if (!def) {
        throw new Error("해당 계좌의 풀에 존재하지 않는 티커입니다. 먼저 /stocks 종목 관리에 이 티커를 추가해주세요.");
      }

      setValidatedCandidate({
        ticker: def.ticker,
        name: def.name,
        listing_date: def.listing_date || "-",
        status: "active",
        is_deleted: false,
        deleted_reason: "",
        bucket_id: def.bucket_id,
      });

    } catch (validationError) {
      setValidatedCandidate(null);
      setError(validationError instanceof Error ? validationError.message : "티커 확인에 실패했습니다.");
    } finally {
      setIsValidatingTicker(false);
    }
  }

  function handleCreateTarget() {
    if (!validatedCandidate) return;

    startTransition(async () => {
      try {
        setError(null);
        const rVal = typeof addRatio === "number" ? addRatio : parseFloat(addRatio);
        if (isNaN(rVal) || rVal <= 0 || rVal > 100) {
           toast.error("비중은 0.1%에서 100% 사이여야 합니다.");
           return;
        }

        const finalRatio = parseFloat(rVal.toFixed(1));
        
        const response = await fetch("/api/account-stocks", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            ticker: validatedCandidate.ticker,
            ratio: finalRatio,
            name: validatedCandidate.name
          }),
        });

        const payload = (await response.json()) as { error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 추가에 실패했습니다.");
        }
        
        toast.success(`[포트폴리오 비중] ${validatedCandidate.name}(${validatedCandidate.ticker}) 추가 완료: ${finalRatio.toFixed(1)}%`);
        closeAddModal();
        void load(selectedAccountId);

      } catch (createError) {
        setError(createError instanceof Error ? createError.message : "종목 추가에 실패했습니다.");
      }
    });
  }

  const totalRatio = rows.reduce((acc, cur) => acc + (cur.ratio || 0), 0);
  const isRatioValid = Math.abs(totalRatio - 100) < 0.05;

  return (
    <>
      <div className="appPageStack appPageStackFill">
        {error ? (
          <div className="appBannerStack">
            <div className="alert alert-danger mb-0">{error}</div>
          </div>
        ) : null}

        <section className="appSection appSectionFill stocksPage">
          <div className="card appCard stocksCard">
            <div className="card-header">
              <div className="tickerTypeToolbar w-100" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div className="tickerTypeToolbarLeft" style={{ display: "flex", gap: "0.4rem", alignItems: "center" }}>
                  <div className="accountSelect">
                    <select
                      className="form-select"
                      style={{ width: "auto", minWidth: "180px", fontWeight: 600 }}
                      aria-label="계좌 선택"
                      value={selectedAccountId}
                      onChange={(event) => handleAccountChange(event.target.value)}
                      disabled={loading}
                    >
                      {accounts.map((account) => (
                        <option key={account.account_id} value={account.account_id}>
                          {account.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <button className="btn btn-primary d-flex align-items-center gap-1" type="button" onClick={openAddModal} disabled={loading}>
                    <IconPlus size={18} stroke={2} />
                    <span style={{ fontWeight: 600 }}>종목 추가</span>
                  </button>
                </div>

                <div className="tickerTypeToolbarRight" style={{ display: "flex", alignItems: "center", gap: "1.25rem" }}>
                  <div className="stocksSummary d-flex align-items-center gap-3">
                    {selectedAccountItem ? (
                      <div className="d-flex align-items-center gap-1">
                        <span style={{ color: "#6c757d", fontSize: "0.85rem", fontWeight: 600 }}>계좌명:</span>
                        <span style={{ fontWeight: 700 }}>{selectedAccountItem.icon} {selectedAccountItem.name}</span>
                      </div>
                    ) : null}
                    <div className="d-flex align-items-center gap-1">
                      <span style={{ color: "#6c757d", fontSize: "0.85rem", fontWeight: 600 }}>총 개수:</span>
                      <span style={{ fontWeight: 700 }}>{new Intl.NumberFormat("ko-KR").format(rows.length)}개</span>
                    </div>
                    <div className="d-flex align-items-center gap-1">
                      <span style={{ color: "#6c757d", fontSize: "0.85rem", fontWeight: 600 }}>비중 합:</span>
                      <span style={{ 
                        fontWeight: 800, 
                        color: isRatioValid ? '#2fb344' : '#d63939',
                        fontSize: "1rem"
                      }}>
                        {totalRatio.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="appDataGridWrap">
              <AppDataGrid
                className="stocksTable"
                rows={rows}
                columns={columns}
                loading={loading}
                minHeight="70vh"
              />
            </div>
          </div>
        </section>
      </div>

      <AppModal
        open={isAddModalOpen}
        onClose={closeAddModal}
        title="포트폴리오 비중 대상 추가"
        footer={
          <div className="d-flex justify-content-end gap-2 w-100">
            <button className="btn btn-link link-secondary" type="button" onClick={closeAddModal}>
              취소
            </button>
            <button 
              className="btn btn-primary" 
              type="button" 
              onClick={handleCreateTarget} 
              disabled={isPending || !validatedCandidate}
              style={{ minWidth: "100px" }}
            >
              저장
            </button>
          </div>
        }
      >
        <div className="appModalBody">
          <div className="mb-3">
            <label className="form-label">티커</label>
            <div className="row g-2">
              <div className="col">
                <input
                  type="text"
                  className="form-control"
                  placeholder="예: SCHD"
                  value={addTickerInput}
                  onChange={(e) => {
                    setAddTickerInput(e.target.value);
                    if (validatedCandidate) setValidatedCandidate(null);
                  }}
                  onKeyDown={(e) => e.key === "Enter" && handleValidateTicker()}
                />
              </div>
              <div className="col-auto">
                <button
                  className="btn btn-outline-secondary d-flex align-items-center gap-1"
                  type="button"
                  onClick={handleValidateTicker}
                  disabled={isValidatingTicker || !addTickerInput.trim()}
                >
                  <IconSearch size={16} stroke={1.75} />
                  <span>확인</span>
                </button>
              </div>
            </div>
            {isValidatingTicker ? (
              <div className="mt-1 small text-muted">티커 확인 중...</div>
            ) : validatedCandidate ? (
              <div className="mt-2 p-2 bg-success-lt rounded border border-success-subtle">
                <div className="d-flex align-items-center gap-1 text-success fw-bold">
                  <span className="appCodeText">{validatedCandidate.ticker}</span>
                  <span>-</span>
                  <span>{validatedCandidate.name}</span>
                </div>
              </div>
            ) : null}
          </div>

            <div className="mb-3">
                <label className="form-label">비중(%) (0.1 ~ 100.0%)</label>
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  max="100"
                  className="form-control"
                  placeholder="예: 10.0"
                  value={addRatio}
                  onChange={(e) => setAddRatio(e.target.value ? parseFloat(e.target.value) : "")}
                  disabled={isPending}
                />
              </div>
        </div>
      </AppModal>

      <AppModal
        open={isEditModalOpen}
        onClose={closeEditModal}
        title="비중 수정"
        footer={
          <div className="d-flex justify-content-between w-100">
            <button
              className="btn btn-outline-danger"
              type="button"
              onClick={() => {
                if (editingRow && window.confirm(`${editingRow.name}(${editingRow.ticker}) 종목을 이 계좌에서 삭제하시겠습니까?`)) {
                  handleDeleteTarget(editingRow.ticker, editingRow.name);
                  closeEditModal();
                }
              }}
              disabled={isPending}
            >
              종목 삭제
            </button>
            <div className="d-flex gap-2">
              <button className="btn btn-link link-secondary" type="button" onClick={closeEditModal}>
                취소
              </button>
              <button className="btn btn-primary" type="button" onClick={handleBatchEditSave} disabled={isPending} style={{ minWidth: "100px" }}>
                저장 완료
              </button>
            </div>
          </div>
        }
      >
        <div className="appModalBody">
          {editingRow && (
            <>
              <div className="mb-3">
                <div className="fw-bold text-secondary mb-1">대상 종목</div>
                <div className="appCodeText" style={{ fontSize: '1.2rem' }}>{editingRow.ticker}</div>
                <div style={{ fontSize: '1.1rem' }}>{editingRow.name}</div>
              </div>
              <div className="mb-3">
                <label className="form-label">비중(%) 수정 (0.1 ~ 100.0%)</label>
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  max="100"
                  className="form-control form-control-lg"
                  value={editingRatio}
                  onChange={(e) => setEditingRatio(parseFloat(e.target.value) || 0)}
                  autoFocus
                />
              </div>
            </>
          )}
        </div>
      </AppModal>
    </>
  );
}
