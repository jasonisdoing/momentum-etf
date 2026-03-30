"use client";

import { useEffect, useMemo, useState } from "react";
import { type GridColDef, type GridRenderCellParams } from "@mui/x-data-grid";

import { AppDataGrid } from "../components/AppDataGrid";
import { AppLoadingState } from "../components/AppLoadingState";

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
  const [rows, setRows] = useState<HoldingsRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const response = await fetch("/api/holdings", { cache: "no-store" });
        const payload = (await response.json()) as { rows?: HoldingsRow[]; error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "보유 종목을 불러오지 못했습니다.");
        }
        setRows(payload.rows ?? []);
      } catch (err) {
        setError(err instanceof Error ? err.message : "보유 종목을 불러오지 못했습니다.");
      } finally {
        setLoading(false);
      }
    }
    void load();
  }, []);

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
