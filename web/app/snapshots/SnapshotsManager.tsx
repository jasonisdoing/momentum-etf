"use client";

import { useEffect, useMemo, useState } from "react";

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

  if (loading) {
    return (
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBody">
            <p>스냅샷 목록을 불러오는 중...</p>
          </div>
        </div>
      </section>
    );
  }

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
            <div className="tableWrap">
              <table className="erpTable snapshotsTable">
                <thead>
                  <tr>
                    <th>날짜</th>
                    <th>총 자산</th>
                    <th>원금</th>
                    <th>현금</th>
                    <th>평가액</th>
                    <th>계좌수</th>
                  </tr>
                </thead>
                <tbody>
                  {snapshots.length === 0 ? (
                    <tr>
                      <td colSpan={6}>
                        <div className="tableEmpty">저장된 스냅샷이 없습니다.</div>
                      </td>
                    </tr>
                  ) : (
                    snapshots.map((snapshot) => {
                      const isSelected = snapshot.id === selectedSnapshot?.id;
                      return (
                        <tr
                          key={snapshot.id}
                          className={isSelected ? "tableRowSelected" : undefined}
                          onClick={() => setSelectedId(snapshot.id)}
                        >
                          <td>{snapshot.snapshot_date}</td>
                          <td>{formatKrw(snapshot.total_assets)}</td>
                          <td>{formatKrw(snapshot.total_principal)}</td>
                          <td>{formatKrw(snapshot.cash_balance)}</td>
                          <td>{formatKrw(snapshot.valuation_krw)}</td>
                          <td>{snapshot.account_count}</td>
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

      <section className="appSection">
        <div className="card appCard">
          <div className="card-header appCardHeader">
            <div className="sectionHeaderCompact w-100">
              <h2>선택일 계좌별 상세</h2>
              <span className="tableMuted">{selectedSnapshot?.snapshot_date ?? "-"}</span>
            </div>
          </div>
          <div className="card-body appCardBodyTight">
            <div className="tableWrap">
              <table className="erpTable">
                <thead>
                  <tr>
                    <th>계좌</th>
                    <th>총 자산</th>
                    <th>원금</th>
                    <th>현금</th>
                    <th>평가액</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedSnapshot?.accounts.length ? (
                    selectedSnapshot.accounts.map((account) => (
                      <tr key={`${selectedSnapshot.id}-${account.account_id}`}>
                        <td>{account.account_name}</td>
                        <td>{formatKrw(account.total_assets)}</td>
                        <td>{formatKrw(account.total_principal)}</td>
                        <td>{formatKrw(account.cash_balance)}</td>
                        <td>{formatKrw(account.valuation_krw)}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={5}>
                        <div className="tableEmpty">선택한 스냅샷의 계좌 상세 데이터가 없습니다.</div>
                      </td>
                    </tr>
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
