"use client";

import { useEffect, useState, useTransition } from "react";

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

function formatCount(value: number): string {
  return new Intl.NumberFormat("ko-KR").format(value);
}

export function SystemManager() {
  const [summaryRows, setSummaryRows] = useState<SystemSummaryRow[]>([]);
  const [scheduleRows, setScheduleRows] = useState<SystemScheduleRow[]>([]);
  const [scheduleNote, setScheduleNote] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runningAction, setRunningAction] = useState<"meta_all" | "cache_all" | "asset_summary" | null>(null);
  const [isPending, startTransition] = useTransition();
  const toast = useToast();

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

  if (loading) {
    return (
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBody">
            <p>시스템정보 데이터를 불러오는 중...</p>
          </div>
        </div>
      </section>
    );
  }

  function handleAction(action: "meta_all" | "cache_all" | "asset_summary") {
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
                  className="primaryButton"
                  type="button"
                  onClick={() => handleAction("meta_all")}
                  disabled={isPending}
                >
                  {runningAction === "meta_all" ? "실행 중..." : "모든 메타데이터 업데이트"}
                </button>
                <button
                  className="secondaryButton"
                  type="button"
                  onClick={() => handleAction("cache_all")}
                  disabled={isPending}
                >
                  {runningAction === "cache_all" ? "실행 중..." : "모든 가격 캐시 업데이트"}
                </button>
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
                <span>백그라운드 작업은 Python 스크립트로 시작된다.</span>
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
            <div className="tableWrap">
              <table className="erpTable">
                <thead>
                  <tr>
                    <th>구분</th>
                    <th className="tableAlignRight">개수</th>
                    <th>대상</th>
                  </tr>
                </thead>
                <tbody>
                  {summaryRows.map((row) => (
                    <tr key={row.category}>
                      <td>{row.category}</td>
                      <td className="tableAlignRight">{formatCount(row.count)}</td>
                      <td>{row.target}</td>
                    </tr>
                  ))}
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
              <h2>자동 작업</h2>
            </div>
          </div>
          <div className="card-body appCardBodyTight">
            <div className="tableWrap">
              <table className="erpTable">
                <thead>
                  <tr>
                    <th>작업</th>
                    <th>대상</th>
                    <th>자동 주기</th>
                    <th>실행 명령</th>
                  </tr>
                </thead>
                <tbody>
                  {scheduleRows.map((row) => (
                    <tr key={row.job}>
                      <td>{row.job}</td>
                      <td>{row.target}</td>
                      <td>{row.cadence}</td>
                      <td className="appCodeText">{row.command}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {scheduleNote ? <div className="tableFooterMeta">{scheduleNote}</div> : null}
          </div>
        </div>
      </section>
    </div>
  );
}
