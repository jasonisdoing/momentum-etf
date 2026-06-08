"use client";

import { useEffect, useState, useTransition } from "react";
import type { CellStyle, ColDef } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";

type SystemPoolRow = {
  id: string;
  order: number;
  pool: string;
  ticker_type: string;
  country_code: string;
  stock_count: number;
  rising_count: number;
  rising_ratio: number;
  etf_count: number;
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
  owner_app_type?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  is_clickable?: boolean;
};

type SystemRunningJobDetail = {
  started_at?: string | null;
  estimated_seconds?: number | null;
  remaining_seconds?: number | null;
  estimated_display?: string | null;
  remaining_display?: string | null;
  owner_app_type?: string | null;
  is_mine?: boolean;
  cancel_requested?: boolean;
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

type BatchQueueItem = {
  id: string;
  job_name: string;
  status: "pending" | "running" | "done" | "failed";
  triggered_by: "manual" | "schedule" | string;
  triggered_at: string | null;
  started_at: string | null;
  ended_at: string | null;
  exit_code: number | null;
  error: string | null;
};

type SystemResponse = {
  pool_rows?: SystemPoolRow[];
  schedule_rows?: SystemScheduleRow[];
  schedule_note?: string;
  running_jobs?: string[];
  is_deploying?: boolean;
  last_run_by_job?: Record<string, SystemLastRunInfo>;
  running_job_details?: Record<string, SystemRunningJobDetail>;
  next_run_by_job?: Record<string, SystemNextRunInfo>;
  estimated_by_job?: Record<string, { seconds: number | null; display: string | null }>;
  batch_queue?: BatchQueueItem[];
  error?: string;
};

type SystemPoolGridRow = SystemPoolRow;
type SystemScheduleGridRow = SystemScheduleRow & {
  id: string;
  running: boolean;
  anyRunning: boolean;
  isDeploying: boolean;
  lastRunDisplay: string;
  lastRunOwner: string | null;       // "Server" | "Local" | null — 어디서 실행됐나
  lastRunClickable: boolean;          // 현재 인스턴스에서 다운로드 가능?
  lastRunStartedAt: string | null;    // ISO — 로그 다운로드 파라미터
  lastRunEndedAt: string | null;
  estimatedDisplay: string; // "4분 7초" 또는 "-" (이력 없음)
  runningCommandPrefix: string;
  runningCancellable: boolean; // 같은 인스턴스의 worker 가 처리 중이라 중단 가능
  runningCancelRequested: boolean; // 이미 중단 요청 보냈는지
  nextRunAt: string | null;
  nextRunDisplay: string;
  pendingPosition: number; // 0 = 대기 없음, 1~ = N번째 대기
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
  // 어느 인스턴스(서버 scheduler / 로컬 worker)가 처리 중인지를 항상 표시한다.
  // owner_app_type 이 있으면 is_mine 여부와 무관하게 [SERVER] / [LOCAL] 라벨을 붙인다.
  const ownerLabel =
    detail && detail.owner_app_type
      ? `[${detail.owner_app_type.toUpperCase()}] `
      : "";
  // 취소 요청이 들어온 상태는 라벨을 "중단 중" 으로 바꿔 사용자가 명확히 인지하도록 한다.
  const stateLabel = detail?.cancel_requested ? "중단 중" : "실행 중";

  const estimatedSeconds = detail?.estimated_seconds;
  const estimatedDisplay = detail?.estimated_display;
  if (typeof estimatedSeconds !== "number" || estimatedSeconds <= 0 || !detail?.started_at) {
    return `▶ ${ownerLabel}${stateLabel}... `;
  }
  const startedMs = new Date(detail.started_at).getTime();
  if (Number.isNaN(startedMs)) {
    return estimatedDisplay
      ? `▶ ${ownerLabel}${stateLabel}(예상시간 ${estimatedDisplay})... `
      : `▶ ${ownerLabel}${stateLabel}... `;
  }
  const elapsedSeconds = Math.max(0, Math.floor((nowMs - startedMs) / 1000));
  const remainingSeconds = Math.max(0, Math.round(estimatedSeconds - elapsedSeconds));
  // 예상시간 초과 시: "+12초 초과" 처럼 얼만큼 초과했는지 표시. 2배 초과시 곧 lock 자동 제거됨.
  const overrunSeconds = Math.max(0, Math.round(elapsedSeconds - estimatedSeconds));
  const remainingText =
    remainingSeconds > 0
      ? `${formatDurationSeconds(remainingSeconds)} 남음`
      : `+${formatDurationSeconds(overrunSeconds)} 초과`;
  return `▶ ${ownerLabel}${stateLabel}(${remainingText}, 예상시간 ${formatDurationSeconds(estimatedSeconds)})... `;
}

function formatCount(value: number): string {
  return new Intl.NumberFormat("ko-KR").format(value);
}

function formatPercent(value: number): string {
  return `${new Intl.NumberFormat("ko-KR", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)}%`;
}

const appGridTheme = createAppGridTheme();

const poolColumns: ColDef<SystemPoolGridRow>[] = [
  {
    field: "order",
    headerName: "순서",
    minWidth: 72,
    flex: 0.45,
    type: "rightAligned",
    cellRenderer: (params: { value: number }) => formatCount(params.value),
  },
  { field: "pool", headerName: "종목풀", minWidth: 180, flex: 1.8 },
  { field: "ticker_type", headerName: "ID", minWidth: 100, flex: 0.7 },
  { field: "country_code", headerName: "국가", minWidth: 82, flex: 0.55 },
  {
    field: "stock_count",
    headerName: "종목수",
    minWidth: 100,
    flex: 0.65,
    type: "rightAligned",
    cellRenderer: (params: { value: number }) => formatCount(params.value),
  },
  {
    field: "rising_count",
    headerName: "상승수",
    minWidth: 100,
    flex: 0.75,
    type: "rightAligned",
    cellRenderer: (params: { value: number; data?: SystemPoolGridRow }) => {
      const total = params.data?.stock_count ?? 0;
      return `${formatCount(params.value)}/${formatCount(total)}`;
    },
  },
  {
    field: "rising_ratio",
    headerName: "상승비율",
    minWidth: 100,
    flex: 0.75,
    type: "rightAligned",
    cellStyle: { color: "#dc2626" },
    cellRenderer: (params: { value: number }) => formatPercent(params.value),
  },
  {
    field: "etf_count",
    headerName: "ETF",
    minWidth: 82,
    flex: 0.55,
    type: "rightAligned",
    cellRenderer: (params: { value: number }) => formatCount(params.value),
  },
];

const scheduleColumns: ColDef<SystemScheduleGridRow>[] = [
  {
    headerName: "#",
    minWidth: 48,
    width: 56,
    maxWidth: 64,
    sortable: false,
    suppressMovable: true,
    type: "rightAligned",
    valueGetter: (params) => (params.node ? (params.node.rowIndex ?? -1) + 1 : ""),
  },
  { field: "job", headerName: "작업", minWidth: 140, width: 150 },
  { field: "target", headerName: "대상", minWidth: 140, width: 180 },
  {
    field: "cadence",
    headerName: "자동 주기",
    minWidth: 220,
    width: 240,
    cellRenderer: (params: { value?: string }) => {
      const text = params.value ?? "";
      return text || "-";
    },
  },
  {
    field: "nextRunDisplay",
    headerName: "다음 실행",
    minWidth: 100,
    width: 110,
    cellRenderer: (params: { value: string }) => params.value || "-",
  },
  {
    field: "lastRunDisplay",
    headerName: "마지막 실행",
    minWidth: 180,
    width: 200,
    tooltipValueGetter: (params) => {
      const row = params.data as SystemScheduleGridRow | undefined;
      if (!row || !row.lastRunOwner) return "";
      if (row.lastRunClickable) {
        return `클릭하면 [${row.lastRunOwner.toUpperCase()}] 의 실행 로그를 다운로드합니다.`;
      }
      return `이 작업은 [${row.lastRunOwner.toUpperCase()}] 에서 실행되었습니다. 해당 인스턴스의 시스템 페이지에서 다운로드할 수 있습니다.`;
    },
    cellStyle: (params): CellStyle => {
      const row = params.data as SystemScheduleGridRow | undefined;
      return row && row.lastRunClickable && row.lastRunStartedAt
        ? { cursor: "pointer", textDecoration: "underline" }
        : { cursor: "default" };
    },
    onCellClicked: async (event) => {
      const row = event.data as SystemScheduleGridRow | undefined;
      if (!row || !row.lastRunClickable || !row.lastRunStartedAt) return;
      const params = new URLSearchParams({ key: row.key, started_at: row.lastRunStartedAt });
      if (row.lastRunEndedAt) params.set("ended_at", row.lastRunEndedAt);
      try {
        const resp = await fetch(`/api/system/job-logs?${params.toString()}`, { cache: "no-store" });
        if (!resp.ok) {
          const payload = await resp.json().catch(() => ({}));
          alert(`로그 다운로드 실패: ${payload.error ?? resp.status}`);
          return;
        }
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        const disposition = resp.headers.get("Content-Disposition") || "";
        const match = disposition.match(/filename="([^"]+)"/);
        a.download = match?.[1] ?? `${row.key}.log`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } catch (err) {
        alert(`로그 다운로드 실패: ${err instanceof Error ? err.message : String(err)}`);
      }
    },
    cellRenderer: (params: { value: string; data?: SystemScheduleGridRow }) => {
      const row = params.data;
      if (!row || !row.lastRunDisplay || row.lastRunDisplay === "-") return "-";
      const ownerLabel = row.lastRunOwner ? ` [${row.lastRunOwner.toUpperCase()}]` : "";
      return `${row.lastRunDisplay}${ownerLabel}`;
    },
  },
  {
    field: "estimatedDisplay",
    headerName: "예상시간",
    minWidth: 85,
    width: 95,
    tooltipValueGetter: () => "최근 성공한 5건의 평균 소요시간 (성공 이력 없으면 실패 포함). 출처: logs/cron/{job}.log",
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
      // 큐 기반: 다른 배치가 실행 중이어도 클릭 가능 (대기 큐에 추가됨)
      return { cursor: "pointer" };
    },
    tooltipValueGetter: (params) => {
      const row = params.data as SystemScheduleGridRow | undefined;
      if (!row) return "";
      if (row.running) return "현재 실행 중입니다.";
      return `클릭 시 "${row.job}" 배치를 큐에 추가합니다 (워커가 순서대로 실행).`;
    },
    cellRenderer: (params: { value: string; data?: SystemScheduleGridRow }) => {
      const row = params.data;
      let badge: React.ReactNode = null;
      let cancelBtn: React.ReactNode = null;
      if (row?.running) {
        badge = <span style={{ color: "#d97706", fontWeight: 700, marginRight: 6 }}>{row.runningCommandPrefix}</span>;
        if (row.runningCancellable) {
          const alreadyRequested = row.runningCancelRequested;
          cancelBtn = (
            <button
              type="button"
              style={{
                marginLeft: 8,
                padding: "0 4px",
                fontSize: "20px",
                fontWeight: 700,
                lineHeight: 1,
                color: alreadyRequested ? "#9ca3af" : "#dc2626",
                background: "transparent",
                border: "none",
                cursor: alreadyRequested ? "not-allowed" : "pointer",
              }}
              disabled={alreadyRequested}
              title={alreadyRequested ? "이미 중단 요청을 보냈습니다." : "이 배치를 중단합니다."}
              onClick={async (e) => {
                e.stopPropagation();
                if (!window.confirm(`"${row.job}" 배치를 중단할까요? (현재 진행 중인 데이터가 일부만 반영됩니다)`)) return;
                try {
                  const resp = await fetch("/api/system/cancel", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ key: row.key }),
                  });
                  const data = await resp.json().catch(() => ({}));
                  if (!resp.ok) {
                    window.alert(`중단 실패: ${data.error ?? `HTTP ${resp.status}`}`);
                  } else {
                    window.alert(data.message ?? "중단 요청을 보냈습니다.");
                  }
                } catch (err) {
                  window.alert(`중단 실패: ${err instanceof Error ? err.message : String(err)}`);
                }
              }}
            >
              {alreadyRequested ? "⏳" : "✕"}
            </button>
          );
        }
      } else if (row && row.pendingPosition > 0) {
        badge = (
          <span style={{ color: "#2563eb", fontWeight: 700, marginRight: 6 }}>
            ⏳ 대기 {row.pendingPosition} ▶
          </span>
        );
      }
      return (
        <span className="appCodeText">
          {badge}
          {cancelBtn}
          {params.value}
        </span>
      );
    },
  },
];

export function SystemManager({
  onHeaderSummaryChange,
}: {
  onHeaderSummaryChange?: (summary: { poolCount: number; scheduleCount: number }) => void;
}) {
  const [poolRows, setPoolRows] = useState<SystemPoolRow[]>([]);
  const [scheduleRows, setScheduleRows] = useState<SystemScheduleRow[]>([]);
  const [scheduleNote, setScheduleNote] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runningJobs, setRunningJobs] = useState<string[]>([]);
  const [isDeploying, setIsDeploying] = useState(false);
  const [lastRunByJob, setLastRunByJob] = useState<Record<string, SystemLastRunInfo>>({});
  const [runningJobDetails, setRunningJobDetails] = useState<Record<string, SystemRunningJobDetail>>({});
  const [nextRunByJob, setNextRunByJob] = useState<Record<string, SystemNextRunInfo>>({});
  const [estimatedByJob, setEstimatedByJob] = useState<Record<string, { seconds: number | null; display: string | null }>>({});
  const [batchQueue, setBatchQueue] = useState<BatchQueueItem[]>([]);
  const [nowTick, setNowTick] = useState(() => Date.now());
  const [, startTransition] = useTransition();
  const toast = useToast();
  const runningSet = new Set(runningJobs);
  const anyRunning = runningSet.size > 0;
  // 대기 중 (pending) 큐 항목 — 오래된 순. FIFO 처리 순번 매핑.
  const pendingOrder = new Map<string, number>();
  batchQueue
    .filter((q) => q.status === "pending")
    .sort((a, b) => String(a.triggered_at ?? "").localeCompare(String(b.triggered_at ?? "")))
    .forEach((q, idx) => {
      if (!pendingOrder.has(q.job_name)) {
        pendingOrder.set(q.job_name, idx + 1);
      }
    });
  const poolGridRows: SystemPoolGridRow[] = poolRows;
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
      lastRunOwner: lastRunByJob[row.key]?.owner_app_type ?? null,
      lastRunClickable: Boolean(lastRunByJob[row.key]?.is_clickable),
      lastRunStartedAt: lastRunByJob[row.key]?.started_at ?? null,
      lastRunEndedAt: lastRunByJob[row.key]?.ended_at ?? null,
      estimatedDisplay: String(estimatedByJob[row.key]?.display ?? "-"),
      runningCommandPrefix: formatRunningCommandPrefix(runningJobDetails[row.key], nowTick),
      runningCancellable:
        runningSet.has(row.key) && Boolean(runningJobDetails[row.key]?.is_mine),
      runningCancelRequested: Boolean(runningJobDetails[row.key]?.cancel_requested),
      nextRunAt,
      nextRunDisplay: formatRelativeUntil(nextRunAt, nowTick) ?? fallbackDisplay,
      pendingPosition: pendingOrder.get(row.key) ?? 0,
    };
  });

  useEffect(() => {
    const id = window.setInterval(() => setNowTick(Date.now()), anyRunning ? 1000 : 30_000);
    return () => window.clearInterval(id);
  }, [anyRunning]);

  useEffect(() => {
    onHeaderSummaryChange?.({
      poolCount: poolRows.length,
      scheduleCount: scheduleRows.length,
    });
  }, [onHeaderSummaryChange, poolRows.length, scheduleRows.length]);

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

        setPoolRows(payload.pool_rows ?? []);
        setScheduleRows(payload.schedule_rows ?? []);
        setScheduleNote(payload.schedule_note ?? "");
        setRunningJobs(payload.running_jobs ?? []);
        setBatchQueue(payload.batch_queue ?? []);
        setIsDeploying(Boolean(payload.is_deploying));
        setLastRunByJob(payload.last_run_by_job ?? {});
        setRunningJobDetails(payload.running_job_details ?? {});
        setNextRunByJob(payload.next_run_by_job ?? {});
        setEstimatedByJob(payload.estimated_by_job ?? {});
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
      toast.success(`${label} 은(는) 현재 실행 중입니다.`);
      return;
    }
    const pendingPos = pendingOrder.get(action) ?? 0;
    if (pendingPos > 0) {
      toast.success(`${label} 은(는) 이미 큐에서 ${pendingPos}번째로 대기 중입니다.`);
      return;
    }
    // 큐 기반: 다른 배치 실행 중이어도 거부하지 않고 enqueue (백엔드가 중복 enqueue 만 차단)
    startTransition(async () => {
      setError(null);
      try {
        const response = await fetch("/api/system", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action }),
        });
        const payload = (await response.json()) as { message?: string; error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "배치 큐 추가에 실패했습니다.");
        }
        toast.success(String(payload.message ?? `[배치] ${label} 큐에 추가됨`));
      } catch (actionError) {
        const msg = actionError instanceof Error ? actionError.message : "배치 큐 추가에 실패했습니다.";
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
                  // 큐 기반: 모든 작업 클릭 허용. running/pending 중복은 handleTriggerJob 내부에서 토스트 안내.
                  handleTriggerJob(row.key as SystemJobKey, row.job);
                },
              }}
            />
            {(() => {
              // 백엔드 schedule_note 한 문장 + 30분 timeout 안내 한 문장.
              // 모두 한국어 문장이라 "다. " 단위로 split 하여 한 문장 한 줄로 표시.
              const timeoutLine = "배치 실행이 30분을 초과하면 hang 으로 간주하여 자동 종료(SIGKILL)되고 Slack 알림이 전송됩니다.";
              const combined = scheduleNote ? `${scheduleNote} ${timeoutLine}` : timeoutLine;
              // "다. " 뒤에서 split 후 종결 마침표 복원.
              const sentences = combined
                .split(/(?<=다\.) +/)
                .map((s) => s.trim())
                .filter((s) => s.length > 0);
              return (
                <div
                  className="tableFooterMeta"
                  style={{
                    color: "#000",
                    display: "flex",
                    flexDirection: "column",
                    gap: 4,
                    lineHeight: 1.6,
                  }}
                >
                  {sentences.map((line, idx) => (
                    <div key={idx}>{line}</div>
                  ))}
                </div>
              );
            })()}
          </div>
        </div>
      </section>

      <section className="appSection">
        <div className="card appCard">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <span className="appHeaderMetricValue">종목풀</span>
              </div>
            </div>
          </div>
          <div className="card-body appCardBodyTight">
            <AppAgGrid
              rowData={poolGridRows}
              columnDefs={poolColumns}
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
