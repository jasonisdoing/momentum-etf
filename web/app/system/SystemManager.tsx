"use client";

import { useEffect, useState, useTransition } from "react";
import type { CellStyle, ColDef } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";

type SystemSummaryRow = {
  category: string;
  count: number;
  target: string;
};

type SystemScheduleRow = {
  key: string;
  job: string;
  target: string;
  cadence: string;
  command: string;
};

type SystemLastRunInfo = {
  status?: string | null;
  display?: string | null;
};

type SystemRunningJobDetail = {
  started_at?: string | null;
  estimated_seconds?: number | null;
  remaining_seconds?: number | null;
  estimated_display?: string | null;
  remaining_display?: string | null;
  owner_app_type?: string | null;
  is_mine?: boolean;
};

type SystemNextRunInfo = {
  at?: string | null;
  display?: string | null;
};

type SystemJobKey =
  | "data_aggregate"
  | "cache_refresh"
  | "market_hours_analysis"
  | "metadata_updater"
  | "asset_summary"
  | "us_market_stocks";

type SystemResponse = {
  summary_rows?: SystemSummaryRow[];
  schedule_rows?: SystemScheduleRow[];
  schedule_note?: string;
  running_jobs?: string[];
  is_deploying?: boolean;
  last_run_by_job?: Record<string, SystemLastRunInfo>;
  running_job_details?: Record<string, SystemRunningJobDetail>;
  next_run_by_job?: Record<string, SystemNextRunInfo>;
  error?: string;
};

type SystemSummaryGridRow = SystemSummaryRow & { id: string };
type SystemScheduleGridRow = SystemScheduleRow & {
  id: string;
  running: boolean;
  anyRunning: boolean;
  isDeploying: boolean;
  lastRunDisplay: string;
  runningCommandPrefix: string;
  nextRunAt: string | null;
  nextRunDisplay: string;
};

function formatRelativeUntil(iso: string | null | undefined, nowMs: number): string | null {
  if (!iso) return null;
  const target = new Date(iso).getTime();
  if (Number.isNaN(target)) return null;
  const deltaSec = Math.floor((target - nowMs) / 1000);
  if (deltaSec <= 0) return "곧";
  if (deltaSec < 60) return `${deltaSec}초 후`;
  if (deltaSec < 3600) return `${Math.floor(deltaSec / 60)}분 후`;
  if (deltaSec < 86400) {
    const hours = Math.floor(deltaSec / 3600);
    const minutes = Math.floor((deltaSec % 3600) / 60);
    return minutes ? `${hours}시간 ${minutes}분 후` : `${hours}시간 후`;
  }
  const days = Math.floor(deltaSec / 86400);
  const hours = Math.floor((deltaSec % 86400) / 3600);
  return hours ? `${days}일 ${hours}시간 후` : `${days}일 후`;
}

function formatDurationSeconds(seconds: number): string {
  const totalSec = Math.max(0, Math.round(seconds));
  if (totalSec < 60) return `${totalSec}초`;
  const minutes = Math.floor(totalSec / 60);
  const remainSeconds = totalSec % 60;
  if (minutes < 60) return remainSeconds ? `${minutes}분 ${remainSeconds}초` : `${minutes}분`;
  const hours = Math.floor(minutes / 60);
  const remainMinutes = minutes % 60;
  return remainMinutes ? `${hours}시간 ${remainMinutes}분` : `${hours}시간`;
}

function formatRunningCommandPrefix(detail: SystemRunningJobDetail | undefined, nowMs: number): string {
  // 다른 인스턴스가 락을 잡고 있으면 간단히 "[그 인스턴스]에서 작업중" 만 표시.
  if (detail && detail.is_mine === false) {
    const owner = (detail.owner_app_type ?? "다른 인스턴스").toUpperCase();
    return `▶ ${owner}에서 작업중... `;
  }
  const estimatedSeconds = detail?.estimated_seconds;
  const estimatedDisplay = detail?.estimated_display;
  if (typeof estimatedSeconds !== "number" || estimatedSeconds <= 0 || !detail?.started_at) {
    return "▶ 실행 중... ";
  }
  const startedMs = new Date(detail.started_at).getTime();
  if (Number.isNaN(startedMs)) {
    return estimatedDisplay
      ? `▶ 실행 중(예상시간 ${estimatedDisplay})... `
      : "▶ 실행 중... ";
  }
  const elapsedSeconds = Math.max(0, Math.floor((nowMs - startedMs) / 1000));
  const remainingSeconds = Math.max(0, Math.round(estimatedSeconds - elapsedSeconds));
  // 예상시간 초과 시: "+12초 초과" 처럼 얼만큼 초과했는지 표시. 2배 초과시 곧 lock 자동 제거됨.
  const overrunSeconds = Math.max(0, Math.round(elapsedSeconds - estimatedSeconds));
  const remainingText =
    remainingSeconds > 0
      ? `${formatDurationSeconds(remainingSeconds)} 남음`
      : `+${formatDurationSeconds(overrunSeconds)} 초과`;
  return `▶ 실행 중(${remainingText}, 예상시간 ${formatDurationSeconds(estimatedSeconds)})... `;
}

function formatCount(value: number): string {
  return new Intl.NumberFormat("ko-KR").format(value);
}

const appGridTheme = createAppGridTheme();

const summaryColumns: ColDef<SystemSummaryGridRow>[] = [
  {
    field: "count",
    headerName: "순서",
    minWidth: 72,
    width: 72,
    type: "rightAligned",
    cellRenderer: (params: { value: number }) => formatCount(params.value),
  },
  { field: "target", headerName: "계좌 ID", minWidth: 200, flex: 1 },
  { field: "category", headerName: "계좌", minWidth: 200, flex: 1.2 },
];

const scheduleColumns: ColDef<SystemScheduleGridRow>[] = [
  { field: "job", headerName: "작업", minWidth: 140, width: 150 },
  { field: "target", headerName: "대상", minWidth: 140, width: 180 },
  {
    field: "cadence",
    headerName: "자동 주기",
    minWidth: 260,
    width: 300,
    cellRenderer: (params: { value?: string }) => {
      const text = params.value ?? "";
      return text || "-";
    },
  },
  {
    field: "nextRunDisplay",
    headerName: "다음 실행",
    minWidth: 140,
    width: 160,
    cellRenderer: (params: { value: string }) => params.value || "-",
  },
  {
    field: "lastRunDisplay",
    headerName: "마지막 실행",
    minWidth: 140,
    width: 160,
    cellRenderer: (params: { value: string }) => params.value || "-",
  },
  {
    field: "command",
    headerName: "실행 명령 (클릭하여 백그라운드 실행)",
    minWidth: 320,
    flex: 1.6,
    cellStyle: (params): CellStyle => {
      const row = params.data as SystemScheduleGridRow | undefined;
      if (!row) return { cursor: "default" };
      if (row.running) return { cursor: "default", backgroundColor: "#fff8e1" };
      if (row.anyRunning) return { cursor: "not-allowed", color: "#9aa4b1" };
      return { cursor: "pointer" };
    },
    tooltipValueGetter: (params) => {
      const row = params.data as SystemScheduleGridRow | undefined;
      if (!row) return "";
      if (row.running) return "현재 실행 중입니다.";
      if (row.anyRunning) return "다른 배치가 실행 중이라 시작할 수 없습니다.";
      return `클릭 시 "${row.job}" 배치를 백그라운드로 실행합니다.`;
    },
    cellRenderer: (params: { value: string; data?: SystemScheduleGridRow }) => (
      <span className="appCodeText">
        {params.data?.running ? params.data.runningCommandPrefix : ""}
        {params.value}
      </span>
    ),
  },
];

export function SystemManager({
  onHeaderSummaryChange,
}: {
  onHeaderSummaryChange?: (summary: { accountCount: number; scheduleCount: number }) => void;
}) {
  const [summaryRows, setSummaryRows] = useState<SystemSummaryRow[]>([]);
  const [scheduleRows, setScheduleRows] = useState<SystemScheduleRow[]>([]);
  const [scheduleNote, setScheduleNote] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runningJobs, setRunningJobs] = useState<string[]>([]);
  const [isDeploying, setIsDeploying] = useState(false);
  const [lastRunByJob, setLastRunByJob] = useState<Record<string, SystemLastRunInfo>>({});
  const [runningJobDetails, setRunningJobDetails] = useState<Record<string, SystemRunningJobDetail>>({});
  const [nextRunByJob, setNextRunByJob] = useState<Record<string, SystemNextRunInfo>>({});
  const [nowTick, setNowTick] = useState(() => Date.now());
  const [, startTransition] = useTransition();
  const toast = useToast();
  const runningSet = new Set(runningJobs);
  const anyRunning = runningSet.size > 0;
  const summaryGridRows: SystemSummaryGridRow[] = summaryRows.map((row) => ({ ...row, id: row.category }));
  const scheduleGridRows: SystemScheduleGridRow[] = scheduleRows.map((row) => {
    const nextRunAt = nextRunByJob[row.key]?.at ?? null;
    const fallbackDisplay = String(nextRunByJob[row.key]?.display ?? "-");
    return {
      ...row,
      id: row.key,
      running: runningSet.has(row.key),
      anyRunning,
      isDeploying,
      lastRunDisplay: String(lastRunByJob[row.key]?.display ?? "-"),
      runningCommandPrefix: formatRunningCommandPrefix(runningJobDetails[row.key], nowTick),
      nextRunAt,
      nextRunDisplay: formatRelativeUntil(nextRunAt, nowTick) ?? fallbackDisplay,
    };
  });

  useEffect(() => {
    const id = window.setInterval(() => setNowTick(Date.now()), anyRunning ? 1000 : 30_000);
    return () => window.clearInterval(id);
  }, [anyRunning]);

  useEffect(() => {
    onHeaderSummaryChange?.({
      accountCount: summaryRows.length,
      scheduleCount: scheduleRows.length,
    });
  }, [onHeaderSummaryChange, scheduleRows.length, summaryRows.length]);

  useEffect(() => {
    let alive = true;

    async function load(initial: boolean) {
      try {
        if (initial) setLoading(true);
        const response = await fetch("/api/system", { cache: "no-store" });
        const payload = (await response.json()) as SystemResponse;
        if (!response.ok) {
          throw new Error(payload.error ?? "시스템정보 데이터를 불러오지 못했습니다.");
        }

        if (!alive) return;

        setSummaryRows(payload.summary_rows ?? []);
        setScheduleRows(payload.schedule_rows ?? []);
        setScheduleNote(payload.schedule_note ?? "");
        setRunningJobs(payload.running_jobs ?? []);
        setIsDeploying(Boolean(payload.is_deploying));
        setLastRunByJob(payload.last_run_by_job ?? {});
        setRunningJobDetails(payload.running_job_details ?? {});
        setNextRunByJob(payload.next_run_by_job ?? {});
        if (initial) setError(null);
      } catch (loadError) {
        if (alive && initial) {
          setError(loadError instanceof Error ? loadError.message : "시스템정보 데이터를 불러오지 못했습니다.");
        }
      } finally {
        if (alive && initial) setLoading(false);
      }
    }

    load(true);
    const intervalId = window.setInterval(() => load(false), 3000);
    return () => {
      alive = false;
      window.clearInterval(intervalId);
    };
  }, []);

  function handleTriggerJob(action: SystemJobKey, label: string) {
    if (runningSet.has(action)) {
      // 이미 이 배치가 실행 중인 경우 — 조용히 무시 (표시만 확인하라는 의미)
      return;
    }
    if (anyRunning) {
      toast.error("다른 배치가 실행 중입니다. 완료 후 다시 시도해주세요.");
      return;
    }
    startTransition(async () => {
      setError(null);
      // 낙관적 UI: 즉시 running 표시 → 다음 폴링에서 서버 상태로 교체됨
      setRunningJobs((prev) => (prev.includes(action) ? prev : [...prev, action]));
      try {
        const response = await fetch("/api/system", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action }),
        });
        const payload = (await response.json()) as { message?: string; error?: string };
        if (!response.ok) {
          // 트리거 실패 시 낙관적 표시 롤백
          setRunningJobs((prev) => prev.filter((k) => k !== action));
          throw new Error(payload.error ?? "배치 실행에 실패했습니다.");
        }
        toast.success(String(payload.message ?? `[배치] ${label} 실행 시작`));
      } catch (actionError) {
        const msg = actionError instanceof Error ? actionError.message : "배치 실행에 실패했습니다.";
        setError(msg);
        toast.error(msg);
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
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <span className="appHeaderMetricValue">배치</span>
              </div>
            </div>
          </div>
          <div className="card-body appCardBodyTight">
            <AppAgGrid
              rowData={scheduleGridRows}
              columnDefs={scheduleColumns}
              loading={loading}
              minHeight="18rem"
              theme={appGridTheme}
              gridOptions={{
                suppressMovableColumns: true,
                domLayout: "autoHeight",
                onCellClicked: (event) => {
                  if (event.colDef.field !== "command") return;
                  const row = event.data as SystemScheduleGridRow | undefined;
                  if (!row?.key) return;
                  if (row.running || row.anyRunning) return;
                  handleTriggerJob(row.key as SystemJobKey, row.job);
                },
              }}
            />
            {scheduleNote ? <div className="tableFooterMeta">{scheduleNote}</div> : null}
          </div>
        </div>
      </section>

      <section className="appSection">
        <div className="card appCard">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <span className="appHeaderMetricValue">계좌 요약</span>
              </div>
            </div>
          </div>
          <div className="card-body appCardBodyTight">
            <AppAgGrid
              rowData={summaryGridRows}
              columnDefs={summaryColumns}
              loading={loading}
              minHeight="18rem"
              theme={appGridTheme}
              gridOptions={{ suppressMovableColumns: true, domLayout: "autoHeight" }}
            />
          </div>
        </div>
      </section>
    </div>
  );
}
