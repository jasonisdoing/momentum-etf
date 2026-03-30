"use client";

import { useEffect, useMemo, useState } from "react";
import { type GridColDef, type GridRenderCellParams } from "@mui/x-data-grid";

import { AppDataGrid } from "../components/AppDataGrid";
import { AppLoadingState } from "../components/AppLoadingState";
import {
  readRememberedMomentumEtfAccountId,
  writeRememberedMomentumEtfAccountId,
} from "../components/account-selection";

type AccountConfig = {
  account_id: string;
  name: string;
  icon: string;
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
  pnl_krw: number;
  return_pct: number;
  buy_amount_krw: number;
  valuation_krw: number;
};

type HoldingsResponse = {
  accounts?: AccountConfig[];
  account_id?: string;
  rows?: HoldingsRow[];
  error?: string;
};

type GridRow = HoldingsRow & { id: string };

function formatKrw(value: number): string {
  return `${new Intl.NumberFormat("ko-KR").format(value)}원`;
}

function getSignedClass(value: number): string {
  if (value === 0) return "";
  return value > 0 ? "metricPositive" : "metricNegative";
}

function getBucketCellClass(bucketId: number): string {
  if (!bucketId) return "appBucketCell";
  return `appBucketCell appBucketCell${bucketId}`;
}

// 모듈 수준 캐시 변수로 페이지 이동 간에도 데이터를 유지합니다. (브라우저 새로고침 전까지 유효)
let holdingsDataCache: Record<string, HoldingsResponse> = {};

export function HoldingsManager() {
  const [accounts, setAccounts] = useState<AccountConfig[]>(holdingsDataCache["__ACCOUNTS__"]?.accounts ?? []);
  const [selectedAccountId, setSelectedAccountId] = useState(readRememberedMomentumEtfAccountId() ?? "");
  const [rows, setRows] = useState<HoldingsRow[]>(holdingsDataCache[readRememberedMomentumEtfAccountId() ?? ""]?.rows ?? []);
  const [loading, setLoading] = useState(!holdingsDataCache[readRememberedMomentumEtfAccountId() ?? ""]);
  const [error, setError] = useState<string | null>(null);

  async function load(accountId: string | null = null, silent = false) {
    const targetId = accountId ?? selectedAccountId;
    
    // 1. 캐시가 있으면 즉시 반영 (Stale-While-Revalidate)
    const cached = holdingsDataCache[targetId];
    if (cached) {
      if (cached.accounts) setAccounts(cached.accounts);
      if (cached.rows) setRows(cached.rows);
      // 캐시가 있으면 로딩 바를 굳이 보여주지 않거나, silent로 처리
    } else if (!silent) {
      setLoading(true);
    }
    
    setError(null);

    try {
      const search = targetId !== null ? `?account=${encodeURIComponent(targetId)}` : "";
      const response = await fetch(`/api/holdings${search}`, { cache: "no-store" });
      const payload = (await response.json()) as HoldingsResponse;

      if (!response.ok) {
        throw new Error(payload.error ?? "보유 종목을 불러오지 못했습니다.");
      }

      // 서버 응답 저장
      const nextAccounts = payload.accounts ?? [];
      const nextRows = payload.rows ?? [];
      const returnedId = payload.account_id ?? "";

      setAccounts(nextAccounts);
      setRows(nextRows);
      
      // 전역 캐시 업데이트
      holdingsDataCache[returnedId] = payload;
      holdingsDataCache["__ACCOUNTS__"] = { accounts: nextAccounts };

      if (accountId === null) {
        setSelectedAccountId(returnedId);
        writeRememberedMomentumEtfAccountId(returnedId);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "보유 종목을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load(readRememberedMomentumEtfAccountId() ?? "");
  }, []);

  function handleAccountChange(nextId: string) {
    // 낙관적 업데이트
    setSelectedAccountId(nextId);
    writeRememberedMomentumEtfAccountId(nextId);

    // 캐시 확인 및 즉시 반영
    const cached = holdingsDataCache[nextId];
    if (cached) {
      setRows(cached.rows ?? []);
      // 캐시가 있으면 백그라운드에서만 로드
      void load(nextId, true);
    } else {
      // 캐시가 없으면 로딩 표시와 함께 로드
      void load(nextId);
    }
  }

  const gridRows = useMemo<GridRow[]>(
    () => rows.map((row, i) => ({ ...row, id: `${row.ticker}-${row.account_name}-${i}` })),
    [rows],
  );

  const columns = useMemo<GridColDef<GridRow>[]>(
    () => [
      { field: "account_name", headerName: "계좌", minWidth: 130, width: 130 },
      { field: "currency", headerName: "환종", minWidth: 70, width: 70, align: "center", headerAlign: "center" },
      {
        field: "bucket",
        headerName: "버킷",
        minWidth: 108,
        width: 108,
        sortable: false,
        cellClassName: (params) => getBucketCellClass(params.row.bucket_id),
        renderCell: (params) => <span>{String(params.value ?? "-")}</span>,
      },
      {
        field: "ticker",
        headerName: "종목코드",
        minWidth: 110,
        width: 110,
        renderCell: (params) => <span className="appCodeText">{String(params.value ?? "-")}</span>,
      },
      { field: "name", headerName: "종목명", minWidth: 220, flex: 1 },
      {
        field: "quantity",
        headerName: "수량",
        minWidth: 80,
        width: 80,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => new Intl.NumberFormat("ko-KR").format(params.value ?? 0),
      },
      {
        field: "average_buy_price",
        headerName: "매입 단가",
        minWidth: 120,
        width: 120,
        align: "right",
        headerAlign: "right",
      },
      {
        field: "current_price",
        headerName: "현재가",
        minWidth: 120,
        width: 120,
        align: "right",
        headerAlign: "right",
      },
      {
        field: "pnl_krw",
        headerName: "평가손익",
        minWidth: 130,
        width: 130,
        align: "right",
        headerAlign: "right",
        renderCell: (params: GridRenderCellParams<GridRow, number>) => (
          <span className={getSignedClass(params.value ?? 0)}>{formatKrw(params.value ?? 0)}</span>
        ),
      },
      {
        field: "return_pct",
        headerName: "수익률",
        minWidth: 90,
        width: 90,
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
        minWidth: 140,
        width: 140,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatKrw(params.value ?? 0),
      },
      {
        field: "valuation_krw",
        headerName: "평가 금액",
        minWidth: 140,
        width: 140,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatKrw(params.value ?? 0),
      },
    ],
    [],
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

  return (
    <div className="appPageStack">
      {error ? (
        <div className="appBannerStack">
          <div className="bannerError">{error}</div>
        </div>
      ) : null}

      <section className="appSection appSectionFill">
        <div className="card appCard">
          <div className="card-header">
            <div className="tickerTypeToolbar w-100" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div className="tickerTypeToolbarLeft">
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
              </div>
              <div className="tickerTypeToolbarRight">
                <div className="stocksSummary d-flex align-items-center gap-3">
                  <div className="d-flex align-items-center gap-1">
                    <span style={{ color: "#6c757d", fontSize: "0.85rem", fontWeight: 600 }}>총 종목 수:</span>
                    <span style={{ fontWeight: 700 }}>{rows.length}개</span>
                  </div>
                  <div className="d-flex align-items-center gap-1">
                    <span style={{ color: "#6c757d", fontSize: "0.85rem", fontWeight: 600 }}>총 평가액:</span>
                    <span style={{ fontWeight: 700 }}>{formatKrw(rows.reduce((acc, row) => acc + (row.valuation_krw || 0), 0))}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="card-body appCardBodyTight">
            <AppDataGrid
              className="appDataGrid"
              rows={gridRows}
              columns={columns}
              loading={loading}
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
