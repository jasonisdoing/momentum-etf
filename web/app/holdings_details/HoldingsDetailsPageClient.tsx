"use client";

import React, { useCallback, useEffect, useState } from "react";
import { PageFrame } from "../components/PageFrame";

// ─── 타입 ────────────────────────────────────────────────────────────────────
type AccountOption = {
  account_id: string;
  name: string;
};

type ComponentSource = {
  etf_ticker: string;
  etf_name: string;
  weight: number;
};

type ComponentRow = {
  ticker: string;
  name: string;
  total_weight: number;
  sources: ComponentSource[];
};

type EtfDetail = {
  ticker: string;
  name: string;
  quantity: number;
  component_count: number;
  has_components: boolean;
};

type HoldingsComponentsData = {
  account_id: string;
  account_name: string;
  held_etf_count: number;
  components: ComponentRow[];
  etf_details: EtfDetail[];
};

// ─── 유틸 ────────────────────────────────────────────────────────────────────
function formatWeight(w: number): string {
  return `${w.toFixed(2)}%`;
}

// ─── 컴포넌트 ─────────────────────────────────────────────────────────────────
export function HoldingsDetailsPageClient() {
  const [accounts, setAccounts] = useState<AccountOption[]>([]);
  const [selectedAccount, setSelectedAccount] = useState<string>("");
  const [data, setData] = useState<HoldingsComponentsData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());

  // 계좌 목록 로드
  useEffect(() => {
    async function fetchAccounts() {
      try {
        const res = await fetch("/api/holdings-components/accounts", { cache: "no-store" });
        if (!res.ok) throw new Error("계좌 목록을 불러오지 못했습니다.");
        const list = (await res.json()) as AccountOption[];
        setAccounts(list);
        if (list.length > 0) setSelectedAccount(list[0].account_id);
      } catch (e) {
        setError(e instanceof Error ? e.message : "오류가 발생했습니다.");
      }
    }
    void fetchAccounts();
  }, []);

  // 선택된 계좌 데이터 로드
  const loadData = useCallback(async (accountId: string) => {
    if (!accountId) return;
    setIsLoading(true);
    setError(null);
    setData(null);
    setExpandedRows(new Set());
    try {
      const res = await fetch(`/api/holdings-components?account_id=${encodeURIComponent(accountId)}`, {
        cache: "no-store",
      });
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? "데이터를 불러오지 못했습니다.");
      }
      const result = (await res.json()) as HoldingsComponentsData;
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selectedAccount) void loadData(selectedAccount);
  }, [selectedAccount, loadData]);

  function toggleRow(ticker: string) {
    setExpandedRows((prev) => {
      const next = new Set(prev);
      if (next.has(ticker)) next.delete(ticker);
      else next.add(ticker);
      return next;
    });
  }

  // 헤더 우측 정보
  const titleRight = data ? (
    <div className="appHeaderMetrics rankToolbarMeta">
      <div className="appHeaderMetric">
        <span>보유 ETF:</span>
        <span className="appHeaderMetricValue">{data.held_etf_count}개</span>
      </div>
      <div className="appHeaderMetric">
        <span>구성종목:</span>
        <span className="appHeaderMetricValue">{data.components.length}개</span>
      </div>
    </div>
  ) : null;

  return (
    <PageFrame title="보유종목 상세" fullHeight fullWidth titleRight={titleRight}>
      {/* ── 에러 및 로딩 배너 ── */}
      {error && (
        <div className="appBannerStack">
          <div className="bannerError alert alert-danger mb-0">{error}</div>
        </div>
      )}

      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft rankMainHeaderLeft">
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">계좌</span>
                  <select
                    id="holdings-details-account-select"
                    className="form-select"
                    value={selectedAccount}
                    onChange={(e) => setSelectedAccount(e.target.value)}
                    disabled={accounts.length === 0}
                  >
                    {accounts.length === 0 ? (
                      <option value="">계좌 불러오는 중...</option>
                    ) : (
                      accounts.map((acc) => (
                        <option key={acc.account_id} value={acc.account_id}>
                          {acc.name}
                        </option>
                      ))
                    )}
                  </select>
                </label>
              </div>
            </div>
          </div>
          
          <div className="card-table table-responsive">
            {isLoading && (
              <div className="pageLoading" style={{ padding: "40px 0", textAlign: "center", color: "#888" }}>
                불러오는 중...
              </div>
            )}
            
            {data && !isLoading && data.held_etf_count === 0 && (
              <div style={{ padding: "40px 0", textAlign: "center", color: "#aaa" }}>
                이 계좌에 보유 중인 ETF가 없습니다.
              </div>
            )}

            {data && !isLoading && data.components.length > 0 && (
              <table className="table table-vcenter table-hover table-sticky-header mb-0">
                <thead>
                  <tr>
                    <th style={{ width: "40px" }} />
                    <th style={{ width: "120px" }}>종목코드</th>
                    <th>종목명</th>
                    <th style={{ width: "120px", textAlign: "right" }}>합산 비중</th>
                  </tr>
                </thead>
                <tbody>
                  {data.components.map((row, idx) => {
                    const isExpanded = expandedRows.has(row.ticker);
                    return (
                      <React.Fragment key={row.ticker}>
                        <tr
                          style={{ cursor: row.sources.length > 1 ? "pointer" : "default" }}
                          onClick={() => row.sources.length > 1 && toggleRow(row.ticker)}
                        >
                          <td style={{ color: "#aaa", textAlign: "center" }}>
                            {row.sources.length > 1 ? (isExpanded ? "▾" : "▸") : ""}
                          </td>
                          <td style={{ fontFamily: "monospace" }}>{row.ticker}</td>
                          <td>
                            <span style={{ fontWeight: idx < 10 ? 600 : 400 }}>{row.name}</span>
                          </td>
                          <td style={{ textAlign: "right", fontWeight: idx < 10 ? 600 : 400 }}>
                            {formatWeight(row.total_weight)}
                          </td>
                        </tr>
                        {/* 출처 ETF 상세 펼침 */}
                        {isExpanded &&
                          row.sources.map((src) => (
                            <tr
                              key={`${row.ticker}-${src.etf_ticker}`}
                              style={{ background: "#f8f9fa" }}
                            >
                              <td />
                              <td />
                              <td style={{ paddingLeft: "32px", color: "#868e96", fontSize: "13px" }}>
                                └ {src.etf_name}
                              </td>
                              <td style={{ textAlign: "right", color: "#868e96", fontSize: "13px" }}>
                                {formatWeight(src.weight)}
                              </td>
                            </tr>
                          ))}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </section>
    </PageFrame>
  );
}
