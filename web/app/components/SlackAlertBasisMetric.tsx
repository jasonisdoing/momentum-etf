"use client";

import { useEffect, useState } from "react";

/** 헤더의 「슬랙 기준」 — 미등록 알림 대상 명단 설명. 고정 문구라 종목 데이터 로딩(실패 포함)과 따로 한 번 받는다. */
export function SlackAlertBasisMetric({ apiPath }: { apiPath: string }) {
  const [alertBasis, setAlertBasis] = useState<string | null>(null);
  useEffect(() => {
    void fetch(apiPath, { cache: "no-store" })
      .then((response) => response.json() as Promise<{ alert_basis?: string }>)
      .then((payload) => setAlertBasis(payload.alert_basis ?? null))
      .catch(() => setAlertBasis(null));
  }, [apiPath]);

  return (
    <div className="appHeaderMetric" title="이 명단 중 종목풀에 없는 종목을 매일 슬랙으로 알립니다.">
      <span>슬랙 기준:</span>
      <span className="appHeaderMetricValue">{alertBasis ?? "-"}</span>
    </div>
  );
}
