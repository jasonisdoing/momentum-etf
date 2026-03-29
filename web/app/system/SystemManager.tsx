"use client";

import { useEffect, useState, useTransition } from "react";
import { type GridColDef } from "@mui/x-data-grid";

import { AppDataGrid } from "../components/AppDataGrid";
import { useToast } from "../components/ToastProvider";

type SystemSummaryRow = {
  category: string;
  count: number;
  target: string;
};

type SystemScheduleRow = {
  job: string;
  target: string;
  cadence: string;
  command: string;
};

type SystemResponse = {
  summary_rows?: SystemSummaryRow[];
  schedule_rows?: SystemScheduleRow[];
  schedule_note?: string;
  error?: string;
};

type SystemSummaryGridRow = SystemSummaryRow & { id: string };
type SystemScheduleGridRow = SystemScheduleRow & { id: string };

function formatCount(value: number): string {
  return new Intl.NumberFormat("ko-KR").format(value);
}

export function SystemManager() {
  const [summaryRows, setSummaryRows] = useState<SystemSummaryRow[]>([]);
  const [scheduleRows, setScheduleRows] = useState<SystemScheduleRow[]>([]);
  const [scheduleNote, setScheduleNote] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runningAction, setRunningAction] = useState<"asset_summary" | null>(null);
  const [isPending, startTransition] = useTransition();
  const toast = useToast();
  const summaryGridRows: SystemSummaryGridRow[] = summaryRows.map((row) => ({ ...row, id: row.category }));
  const scheduleGridRows: SystemScheduleGridRow[] = scheduleRows.map((row) => ({ ...row, id: row.job }));
  const summaryColumns: GridColDef<SystemSummaryGridRow>[] = [
    { field: "category", headerName: "구분", minWidth: 160, flex: 1 },
    { field: "count", headerName: "개수", minWidth: 96, width: 96, align: "right", headerAlign: "right", renderCell: (params) => formatCount(params.row.count) },
    { field: "target", headerName: "대상", minWidth: 220, flex: 1.2 },
  ];
  const scheduleColumns: GridColDef<SystemScheduleGridRow>[] = [
    { field: "job", headerName: "작업", minWidth: 180, flex: 1 },
    { field: "target", headerName: "대상", minWidth: 180, flex: 1 },
    { field: "cadence", headerName: "자동 주기", minWidth: 140, width: 140 },
    { field: "command", headerName: "실행 명령", minWidth: 320, flex: 1.6, renderCell: (params) => <span className="appCodeText">{params.row.command}</span> },
  ];

  useEffect(() => {
    let alive = true;

    async function load() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch("/api/system", { cache: "no-store" });
        const payload = (await response.json()) as SystemResponse;
        if (!response.ok) {
          throw new Error(payload.error ?? "시스템정보 데이터를 불러오지 못했습니다.");
        }

        if (!alive) {
          return;
        }

        setSummaryRows(payload.summary_rows ?? []);
        setScheduleRows(payload.schedule_rows ?? []);
        setScheduleNote(payload.schedule_note ?? "");
      } catch (loadError) {
        if (alive) {
          setError(loadError instanceof Error ? loadError.message : "시스템정보 데이터를 불러오지 못했습니다.");
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

  function handleAction(action: "asset_summary") {
    startTransition(async () => {
      try {
        setError(null);
        setRunningAction(action);
        const response = await fetch("/api/system", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action }),
        });
        const payload = (await response.json()) as { message?: string; error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "시스템 작업 실행에 실패했습니다.");
        }
        toast.success(String(payload.message ?? "[시스템-정보] 작업 시작"));
      } catch (actionError) {
        setError(actionError instanceof Error ? actionError.message : "시스템 작업 실행에 실패했습니다.");
      } finally {
        setRunningAction(null);
      }
    });
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
          <div className="card-header appCardHeader">
            <div className="sectionHeaderCompact w-100">
              <h2>수동 실행</h2>
            </div>
          </div>
          <div className="card-body appCardBody">
            <div className="tableToolbar">
              <div className="toolbarActions">
                <button
                  className="secondaryButton"
                  type="button"
                  onClick={() => handleAction("asset_summary")}
                  disabled={isPending}
                >
                  {runningAction === "asset_summary" ? "실행 중..." : "전체 자산 요약 알림 전송"}
                </button>
              </div>
              <div className="tableMeta">
                <span>메타데이터/가격 캐시는 종목별로 관리됩니다. (종목 편집 모달에서 새로고침 가능)</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="appSection">
        <div className="card appCard">
          <div className="card-header appCardHeader">
            <div className="sectionHeaderCompact w-100">
              <h2>계좌 요약</h2>
            </div>
          </div>
          <div className="card-body appCardBodyTight">
            <AppDataGrid rows={summaryGridRows} columns={summaryColumns} loading={loading} minHeight="18rem" />
          </div>
        </div>
      </section>

      <section className="appSection">
        <div className="card appCard">
          <div className="card-header appCardHeader">
            <div className="sectionHeaderCompact w-100">
              <h2>자동 작업</h2>
            </div>
          </div>
          <div className="card-body appCardBodyTight">
            <AppDataGrid rows={scheduleGridRows} columns={scheduleColumns} loading={loading} minHeight="18rem" />
            {scheduleNote ? <div className="tableFooterMeta">{scheduleNote}</div> : null}
          </div>
        </div>
      </section>
    </div>
  );
}
