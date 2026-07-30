"use client";

import { useCallback, useEffect, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";

type StrategyConfig = {
  entry_drop_pct: number;
  add_drop_pct: number;
  take_profit_pct: number;
  rounds: number;
  index_name: string;
};

type IndexStatus = {
  name: string;
  as_of: string;
  close: number;
  buy_trigger: number;
};

type Status = {
  held_count: number;
  waiting_first_entry: boolean;
  next_round: number | null;
  next_ticker: string | null;
  next_name: string | null;
  last_buy_price: number | null;
  invested_amount: number;
  valuation_amount: number;
  profit_amount: number | null;
  profit_pct: number | null;
};

type RoundRow = {
  round: number;
  fixed_round: number;
  ticker: string;
  name: string;
  held: boolean;
  avg_price: number | null;
  close: number | null;
  profit_pct: number | null;
  buy_limit: number | null;
  buy_index: number | null;
  sell_limit: number | null;
  sell_index: number | null;
  buy_reached: boolean;
  sell_reached: boolean;
  is_next: boolean;
};

type View = {
  account_id: string;
  config: StrategyConfig;
  slack_enabled: boolean;
  index: IndexStatus;
  status: Status;
  rounds: RoundRow[];
};

const cardStyle: React.CSSProperties = {
  padding: 14,
  display: "flex",
  flexDirection: "column",
  gap: 10,
};

const thStyle: React.CSSProperties = { padding: "6px 8px", textAlign: "right" };
const tdStyle: React.CSSProperties = { padding: "6px 8px", textAlign: "right" };

function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return value.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatSigned(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

function signColor(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value) || value === 0) return "inherit";
  return value > 0 ? "var(--up-color, #d64545)" : "var(--down-color, #2f6fd0)";
}

function ReachedBadge() {
  return (
    <span
      style={{
        marginLeft: 6,
        padding: "1px 5px",
        borderRadius: 4,
        fontSize: "0.74rem",
        fontWeight: 700,
        color: "var(--up-color, #d64545)",
        border: "1px solid var(--up-color, #d64545)",
      }}
    >
      도달
    </span>
  );
}

export function StrategyTradeClient() {
  const toast = useToast();
  const [view, setView] = useState<View | null>(null);
  const [loading, setLoading] = useState(true);
  const [slackSaving, setSlackSaving] = useState(false);
  const [slackTesting, setSlackTesting] = useState(false);
  const applyView = useCallback((data: View) => {
    setView(data);
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await fetch("/api/strategy-trade", { cache: "no-store" });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "운용 현황을 불러오지 못했습니다.");
      applyView(payload as View);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "운용 현황을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [applyView, toast]);

  useEffect(() => {
    void load();
  }, [load]);

  const putSettings = useCallback(
    async (body: Record<string, unknown>) => {
      const resp = await fetch("/api/strategy-trade", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "설정을 저장하지 못했습니다.");
      applyView(payload as View);
    },
    [applyView],
  );

  const toggleSlack = useCallback(
    async (enabled: boolean) => {
      setSlackSaving(true);
      try {
        await putSettings({ slack_enabled: enabled });
        toast.success(enabled ? "슬랙 알림을 켰습니다." : "슬랙 알림을 껐습니다.");
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "설정을 저장하지 못했습니다.");
      } finally {
        setSlackSaving(false);
      }
    },
    [putSettings, toast],
  );

  const testSlack = useCallback(async () => {
    setSlackTesting(true);
    try {
      const resp = await fetch("/api/strategy-trade/slack-test", { method: "POST" });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "슬랙 발송에 실패했습니다.");
      if (payload?.sent) toast.success(payload?.message ?? "슬랙을 발송했습니다.");
      else toast.warning(payload?.message ?? "슬랙을 발송하지 못했습니다.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "슬랙 발송에 실패했습니다.");
    } finally {
      setSlackTesting(false);
    }
  }, [toast]);

  if (loading && !view) {
    return (
      <PageFrame title="전략 사고팔기">
        <div style={{ color: "var(--text-muted)", padding: 20 }}>불러오는 중…</div>
      </PageFrame>
    );
  }
  if (!view) {
    return (
      <PageFrame title="전략 사고팔기">
        <div style={{ color: "var(--text-muted)", padding: 20 }}>데이터가 없습니다.</div>
      </PageFrame>
    );
  }

  const { config, status, index } = view;

  return (
    <PageFrame title="전략 사고팔기">
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {/* 전략 규칙 + 회차당 투입금 */}
        <div className="card appCard" style={cardStyle}>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "center", fontSize: "0.86rem" }}>
            <span style={{ fontWeight: 700 }}>전략 규칙</span>
            <span>
              1호 진입: {config.index_name} 일간 <b>−{config.entry_drop_pct}%</b>
            </span>
            <span>
              추가 진입: 마지막 매수가 <b>−{config.add_drop_pct}%</b>
            </span>
            <span>
              매도: 평균단가 <b>+{config.take_profit_pct}%</b>
            </span>
            <span>
              회차: <b>{config.rounds}회</b>
            </span>
            <span style={{ color: "var(--text-muted)" }}>계좌 {view.account_id}</span>
          </div>
        </div>

        {/* 슬랙 알림 */}
        <div className="card appCard" style={cardStyle}>
          <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ fontWeight: 700, fontSize: "0.95rem" }}>슬랙 알림</span>
            <label style={{ display: "flex", gap: 6, alignItems: "center", fontSize: "0.86rem" }}>
              <input
                type="checkbox"
                checked={view.slack_enabled}
                disabled={slackSaving}
                onChange={(event) => void toggleSlack(event.target.checked)}
              />
              매수·매도 조건 도달 시 알림
            </label>
            <button className="appButton" onClick={() => void testSlack()} disabled={slackTesting}>
              {slackTesting ? "발송 중…" : "테스트 발송"}
            </button>
            <span style={{ color: "var(--text-muted)", fontSize: "0.82rem" }}>
              평일 09:10~15:20 10분 간격 판정 · 같은 회차·동작은 하루 1회
            </span>
          </div>
        </div>

        {/* 현재 판단 */}
        <div className="card appCard" style={cardStyle}>
          <div style={{ display: "flex", gap: 14, alignItems: "baseline", flexWrap: "wrap" }}>
            <span style={{ fontWeight: 700, fontSize: "0.95rem" }}>현재 판단</span>
            <span style={{ color: "var(--text-muted)", fontSize: "0.82rem" }}>
              계좌 실제 보유 기준 · 회차는 평균단가 높은 순 · 코스피 환산은 참고값
            </span>
          </div>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: "0.86rem" }}>
            <span>
              {index.name} <b>{formatNumber(index.close, 2)}</b>
              <span style={{ color: "var(--text-muted)" }}> ({index.as_of})</span>
            </span>
            <span>
              보유 <b>{status.held_count}</b>/{config.rounds}회차
            </span>
            {status.last_buy_price != null ? (
              <span>
                마지막 매수가 <b>{formatNumber(status.last_buy_price)}</b>
              </span>
            ) : null}
            {status.waiting_first_entry ? (
              <span style={{ fontWeight: 700 }}>
                1호 진입 대기 — {index.name} {formatNumber(index.buy_trigger, 2)} 이하
              </span>
            ) : status.next_round != null ? (
              <span style={{ fontWeight: 700 }}>
                다음 진입: {status.next_round}호 {status.next_ticker} {status.next_name}
              </span>
            ) : (
              <span style={{ fontWeight: 700, color: "var(--text-muted)" }}>회차 소진 — 매도 대기</span>
            )}
          </div>
          {status.held_count > 0 ? (
            <div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: "0.86rem" }}>
              <span>
                매입 <b>{formatNumber(status.invested_amount)}</b>원
              </span>
              <span>
                평가 <b>{formatNumber(status.valuation_amount)}</b>원
              </span>
              <span>
                손익{" "}
                <b style={{ color: signColor(status.profit_amount) }}>
                  {formatNumber(status.profit_amount)}원 ({formatSigned(status.profit_pct)})
                </b>
              </span>
            </div>
          ) : null}
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", minWidth: 920, borderCollapse: "collapse", fontSize: "0.84rem" }}>
              <thead>
                <tr style={{ color: "var(--text-muted)" }}>
                  <th style={{ ...thStyle, textAlign: "left" }}>회차</th>
                  <th style={{ ...thStyle, textAlign: "left" }}>종목</th>
                  <th style={thStyle}>평균단가</th>
                  <th style={thStyle}>현재가</th>
                  <th style={thStyle}>손익</th>
                  <th style={thStyle}>매수 지정가</th>
                  <th style={thStyle}>매수 코스피</th>
                  <th style={thStyle}>매도 지정가</th>
                  <th style={thStyle}>매도 코스피</th>
                </tr>
              </thead>
              <tbody>
                {view.rounds.map((row) => (
                  <tr
                    key={row.ticker}
                    style={{ borderTop: "1px solid rgba(148,163,184,0.15)" }}
                  >
                    <td style={{ ...tdStyle, textAlign: "left", fontWeight: 700 }}>
                      {row.round}호
                      {row.is_next ? (
                        <span style={{ marginLeft: 5, color: "var(--text-muted)", fontSize: "0.74rem", fontWeight: 400 }}>
                          다음
                        </span>
                      ) : null}
                    </td>
                    <td style={{ ...tdStyle, textAlign: "left" }}>
                      {row.ticker} {row.name}
                    </td>
                    <td style={tdStyle}>{formatNumber(row.avg_price)}</td>
                    <td style={tdStyle}>{formatNumber(row.close)}</td>
                    <td style={{ ...tdStyle, color: signColor(row.profit_pct), fontWeight: 700 }}>
                      {formatSigned(row.profit_pct)}
                    </td>
                    <td style={{ ...tdStyle, fontWeight: row.is_next ? 700 : undefined }}>
                      {formatNumber(row.buy_limit)}
                      {row.buy_reached ? <ReachedBadge /> : null}
                    </td>
                    <td style={{ ...tdStyle, color: "var(--text-muted)" }}>
                      {formatNumber(row.buy_index, 2)}
                    </td>
                    <td style={{ ...tdStyle, fontWeight: row.sell_limit != null ? 700 : undefined }}>
                      {formatNumber(row.sell_limit)}
                      {row.sell_reached ? <ReachedBadge /> : null}
                    </td>
                    <td style={{ ...tdStyle, color: "var(--text-muted)" }}>
                      {formatNumber(row.sell_index, 2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </PageFrame>
  );
}
