"use client";

import { useEffect, useMemo, useState } from "react";
import { type GridColDef } from "@mui/x-data-grid";

import { AppDataGrid } from "../components/AppDataGrid";

type SnapshotAccountItem = {
  account_id: string;
  account_name: string;
  order: number;
  total_assets: number;
  total_principal: number;
  cash_balance: number;
  valuation_krw: number;
};

type SnapshotListItem = {
  id: string;
  snapshot_date: string;
  total_assets: number;
  total_principal: number;
  cash_balance: number;
  valuation_krw: number;
  account_count: number;
  accounts: SnapshotAccountItem[];
};

type SnapshotListResponse = {
  snapshots?: SnapshotListItem[];
  error?: string;
};

type SnapshotGridRow = SnapshotListItem & { id: string };
type SnapshotDetailGridRow = SnapshotAccountItem & { id: string };

function formatKrw(value: number): string {
  return new Intl.NumberFormat("ko-KR").format(value);
}

export function SnapshotsManager() {
  const [snapshots, setSnapshots] = useState<SnapshotListItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;

    async function load() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch("/api/snapshots", { cache: "no-store" });
        const payload = (await response.json()) as SnapshotListResponse;
        if (!response.ok) {
          throw new Error(payload.error ?? "스냅샷 목록을 불러오지 못했습니다.");
        }

        if (!alive) {
          return;
        }

        const nextSnapshots = payload.snapshots ?? [];
        setSnapshots(nextSnapshots);
        setSelectedId((current) => current ?? nextSnapshots[0]?.id ?? null);
      } catch (loadError) {
        if (alive) {
          setError(loadError instanceof Error ? loadError.message : "스냅샷 목록을 불러오지 못했습니다.");
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

  const selectedSnapshot = useMemo(
    () => snapshots.find((snapshot) => snapshot.id === selectedId) ?? snapshots[0] ?? null,
    [selectedId, snapshots],
  );
  const listRows = useMemo<SnapshotGridRow[]>(
    () => snapshots.map((snapshot) => ({ ...snapshot, id: snapshot.id })),
    [snapshots],
  );
  const detailRows = useMemo<SnapshotDetailGridRow[]>(
    () =>
      (selectedSnapshot?.accounts ?? []).map((account) => ({
        ...account,
        id: `${selectedSnapshot?.id ?? "snapshot"}-${account.account_id}`,
      })),
    [selectedSnapshot],
  );
  const listColumns = useMemo<GridColDef<SnapshotGridRow>[]>(
    () => [
      { field: "snapshot_date", headerName: "날짜", minWidth: 120, width: 120 },
      {
        field: "total_assets",
        headerName: "총 자산",
        minWidth: 120,
        width: 120,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatKrw(params.row.total_assets),
      },
      {
        field: "total_principal",
        headerName: "원금",
        minWidth: 120,
        width: 120,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatKrw(params.row.total_principal),
      },
      {
        field: "cash_balance",
        headerName: "현금",
        minWidth: 120,
        width: 120,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatKrw(params.row.cash_balance),
      },
      {
        field: "valuation_krw",
        headerName: "평가액",
        minWidth: 120,
        width: 120,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatKrw(params.row.valuation_krw),
      },
      {
        field: "account_count",
        headerName: "계좌수",
        minWidth: 88,
        width: 88,
        align: "right",
        headerAlign: "right",
      },
    ],
    [],
  );
  const detailColumns = useMemo<GridColDef<SnapshotDetailGridRow>[]>(
    () => [
      { field: "account_name", headerName: "계좌", minWidth: 180, flex: 1 },
      {
        field: "total_assets",
        headerName: "총 자산",
        minWidth: 120,
        width: 120,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatKrw(params.row.total_assets),
      },
      {
        field: "total_principal",
        headerName: "원금",
        minWidth: 120,
        width: 120,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatKrw(params.row.total_principal),
      },
      {
        field: "cash_balance",
        headerName: "현금",
        minWidth: 120,
        width: 120,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatKrw(params.row.cash_balance),
      },
      {
        field: "valuation_krw",
        headerName: "평가액",
        minWidth: 120,
        width: 120,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatKrw(params.row.valuation_krw),
      },
    ],
    [],
  );

  return (
    <div className="appPageStack">
      {error ? (
        <div className="appBannerStack">
          <div className="bannerError">{error}</div>
        </div>
      ) : null}
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBody">
            <AppDataGrid
              rows={listRows}
              columns={listColumns}
              loading={loading}
              minHeight="22rem"
              getRowClassName={(params) => (params.row.id === selectedSnapshot?.id ? "tableRowSelected" : "")}
              onRowClick={(params) => setSelectedId(String(params.id))}
            />
          </div>
        </div>
      </section>

      <section className="appSection">
        <div className="card appCard">
          <div className="card-header appCardHeader">
            <div className="sectionHeaderCompact w-100">
              <h2>선택일 계좌별 상세</h2>
              <span className="tableMuted">{selectedSnapshot?.snapshot_date ?? "-"}</span>
            </div>
          </div>
          <div className="card-body appCardBodyTight">
            <AppDataGrid rows={detailRows} columns={detailColumns} loading={loading} minHeight="20rem" />
          </div>
        </div>
      </section>
    </div>
  );
}
