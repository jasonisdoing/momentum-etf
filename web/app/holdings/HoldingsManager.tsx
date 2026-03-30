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

export function HoldingsManager() {
  const [accounts, setAccounts] = useState<AccountConfig[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState(readRememberedMomentumEtfAccountId() ?? "");
  const [rows, setRows] = useState<HoldingsRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load(accountId: string | null = null, silent = false) {
    if (!silent) {
      setLoading(true);
    }
    setError(null);

    try {
      const search = accountId !== null ? `?account=${encodeURIComponent(accountId)}` : "";
      const response = await fetch(`/api/holdings${search}`, { cache: "no-store" });
      const payload = (await response.json()) as HoldingsResponse;

      if (!response.ok) {
        throw new Error(payload.error ?? "보유 종목을 불러오지 못했습니다.");
      }

      setAccounts(payload.accounts ?? []);
      const nextId = payload.account_id ?? "";
      if (accountId === null) {
        // 초기 로드 시만 서버에서 준 ID로 업데이트 (TOTAL일 수 있음)
        setSelectedAccountId(nextId);
        writeRememberedMomentumEtfAccountId(nextId);
      }
      setRows(payload.rows ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "보유 종목을 불러오지 못했습니다.");
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }

  useEffect(() => {
    void load(readRememberedMomentumEtfAccountId() ?? "");
  }, []);

  function handleAccountChange(nextId: string) {
    // 낙관적 업데이트
    setSelectedAccountId(nextId);
    writeRememberedMomentumEtfAccountId(nextId);

    // 실제 데이터 로드
    void load(nextId, true); // silent로 하면 프로그레스바가 안 보임 (사용자 요청에 따라 결정 가능)
    // 하지만 사용자가 '프로그레스바가 보여야 하는거 아닌가?'라고 했으므로 
    // 여기서는 setLoading(true)가 포함된 기본 load를 호출하되, 
    // 이미 ID는 업데이트된 상태이므로 UI는 드롭다운만 먼저 바뀝니다.
    setLoading(true);
    void load(nextId);
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
