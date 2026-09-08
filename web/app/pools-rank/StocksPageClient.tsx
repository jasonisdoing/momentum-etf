"use client";

import { useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { StocksManager } from "./StocksManager";

type StocksHeaderSummary = {
  /** 진입 가능 종목 수 — 진입 기준(0선 + 진입 문턱) 충족, 회색 아닌 행. */
  entryCount: number;
  entryPct: number;
  totalCount: number;
  ruleSummary: string;
  /** 실계좌 보유 종목 수 — 표의 녹색 행 수와 같다. */
  heldCount: number;
  /** 시장 ADR(모멘텀 ADR 게이트와 같은 소스) — 레짐 지수 없는 풀은 null. */
  adr: { market: string; value: number; floor: number | null } | null;
};

const DEFAULT_SUMMARY: StocksHeaderSummary = {
  entryCount: 0,
  entryPct: 0,
  totalCount: 0,
  ruleSummary: "-",
  heldCount: 0,
  adr: null,
};

export function StocksPageClient() {
  const [summary, setSummary] = useState<StocksHeaderSummary>(DEFAULT_SUMMARY);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric" title="진입 기준(장기·단기 0선 + 진입 문턱) 통과 종목 — 회색 아닌 행과 같은 판정">
          <span>진입가능:</span>
          <span className="appHeaderMetricValue is-danger">
            {summary.entryCount}개 ({summary.entryPct}%)
          </span>
        </div>
        <div className="appHeaderMetric">
          <span>기준:</span>
          <span className="appHeaderMetricValue">{summary.ruleSummary}</span>
        </div>
        {summary.adr ? (
          <div className="appHeaderMetric" title={`시장(${summary.adr.market})의 20일 등락 비율 — 모멘텀 ADR 게이트와 같은 값`}>
            <span>ADR:</span>
            <span className="appHeaderMetricValue is-danger">
              {summary.adr.value.toFixed(1)}
              {summary.adr.floor != null ? ` (하한 ${summary.adr.floor})` : ""}
            </span>
          </div>
        ) : null}
        <div className="appHeaderMetric" title="실계좌 보유 종목 수 — 표의 녹색(티커·종목명 칸) 행과 같다">
          <span>보유:</span>
          <span className="appHeaderMetricValue is-success">{summary.heldCount}개</span>
        </div>
        <div className="appHeaderMetric">
          <span>총 개수:</span>
          <span className="appHeaderMetricValue">{new Intl.NumberFormat("ko-KR").format(summary.totalCount)}개</span>
        </div>
      </div>
    ),
    [summary],
  );

  return (
    <PageFrame title="순위" fullHeight fullWidth titleRight={titleRight}>
      <StocksManager onHeaderSummaryChange={setSummary} />
    </PageFrame>
  );
}
