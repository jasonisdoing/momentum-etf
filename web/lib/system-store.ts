import { loadAccountConfigs } from "./accounts";
import { spawnPythonScript } from "./python-runtime";

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

type SystemData = {
  summary_rows: SystemSummaryRow[];
  schedule_rows: SystemScheduleRow[];
  schedule_note: string;
};

type SystemAction = "meta_all" | "cache_all" | "asset_summary";

const SCHEDULE_ROWS: SystemScheduleRow[] = [
  {
    job: "종목 메타데이터 업데이트",
    target: "모든 계좌",
    cadence: "매일 09:00 KST",
    command: "python scripts/stock_meta_updater.py",
  },
  {
    job: "가격 캐시 업데이트",
    target: "모든 계좌",
    cadence: "매시 정각 KST",
    command: "python scripts/update_price_cache.py",
  },
  {
    job: "전체 자산 요약 알림",
    target: "전체 계좌 요약",
    cadence: "매일 11:00, 18:00, 23:00, 06:00 KST",
    command: "python scripts/slack_asset_summary.py",
  },
];

export async function loadSystemData(): Promise<SystemData> {
  const accounts = await loadAccountConfigs();
  const accountIds = accounts.map((account) => account.account_id);

  return {
    summary_rows: [
      {
        category: "계좌",
        count: accountIds.length,
        target: accountIds.length > 0 ? accountIds.join(", ") : "-",
      },
    ],
    schedule_rows: SCHEDULE_ROWS,
    schedule_note: "자동 주기는 현재 `.github/workflows` 기준입니다.",
  };
}

export async function triggerSystemAction(action: SystemAction): Promise<string> {
  if (action === "meta_all") {
    await spawnPythonScript(["scripts/stock_meta_updater.py"]);
    return "[시스템-정보] 전체 메타데이터 업데이트 시작";
  }

  if (action === "cache_all") {
    await spawnPythonScript(["scripts/update_price_cache.py"]);
    return "[시스템-정보] 전체 가격 캐시 업데이트 시작";
  }

  if (action === "asset_summary") {
    await spawnPythonScript(["scripts/slack_asset_summary.py"]);
    return "[시스템-정보] 전체 자산 요약 알림 전송 시작";
  }

  throw new Error("지원하지 않는 시스템 작업입니다.");
}

export type { SystemAction, SystemData, SystemScheduleRow, SystemSummaryRow };
