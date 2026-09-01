import { fetchFastApiJson } from "./internal-api";

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
  /** 화면 표시 번호 — 백엔드 배치 정의(`SCHEDULE_ROWS`)의 `no` 가 단일 소스다. */
  no?: number | null;
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
  elapsed_seconds?: number | null;
  remaining_seconds?: number | null;
  estimated_display?: string | null;
  remaining_display?: string | null;
};

type SystemData = {
  pool_rows: SystemPoolRow[];
  schedule_rows: SystemScheduleRow[];
  schedule_note: string;
  running_jobs: string[];
  last_run_by_job?: Record<string, SystemLastRunInfo>;
  running_job_details?: Record<string, SystemRunningJobDetail>;
};

/** 배치 키. 목록은 백엔드(`utils/system_service`)가 단일 소스이며 화면은 그 응답으로 받는다.
 *  여기에 union 으로 복사해 두면 배치를 추가할 때마다 빠뜨린다. */
type SystemAction = string;

export async function loadSystemData(): Promise<SystemData> {
  return fetchFastApiJson<SystemData>("/internal/system");
}

export async function triggerSystemAction(action: SystemAction): Promise<string> {
  const payload = await fetchFastApiJson<{ message: string }>("/internal/system", {
    method: "POST",
    body: JSON.stringify({ action }),
  });
  return payload.message;
}

export type {
  SystemAction,
  SystemData,
  SystemLastRunInfo,
  SystemRunningJobDetail,
  SystemScheduleRow,
  SystemPoolRow,
};
