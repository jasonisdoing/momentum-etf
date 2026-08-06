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

// 편집 대상 파라미터 (백엔드 EDITABLE_PCT_KEYS 와 같은 순서·같은 범위)
type PctKey = "entry_drop_pct" | "add_drop_pct" | "take_profit_pct";
type PctDraft = Record<PctKey, string>;

const PCT_FIELDS: ReadonlyArray<readonly [PctKey, string]> = [
  ["entry_drop_pct", "1호 진입 하락률"],
  ["add_drop_pct", "추가 진입 하락률"],
  ["take_profit_pct", "매도 목표 상승률"],
];
const PCT_MIN = 0.1;
const PCT_MAX = 50;

const pctInputStyle: React.CSSProperties = {
  width: 74,
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "3px 6px",
  fontSize: "var(--fs-sm)",
  textAlign: "right",
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
        fontSize: "var(--fs-sm)",
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
  // 파라미터 편집 초안 — 저장 전까지는 화면 값과 별개로 들고 있는다.
  const [draft, setDraft] = useState<PctDraft | null>(null);
  const [configSaving, setConfigSaving] = useState(false);
  const applyView = useCallback((data: View) => {
    setView(data);
    setDraft({
      entry_drop_pct: String(data.config.entry_drop_pct),
      add_drop_pct: String(data.config.add_drop_pct),
      take_profit_pct: String(data.config.take_profit_pct),
    });
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

  const saveConfig = useCallback(async () => {
    if (!draft) return;
    const parsed: Record<string, number> = {};
    for (const [key, label] of PCT_FIELDS) {
      const value = Number(draft[key]);
      if (!Number.isFinite(value) || value < PCT_MIN || value > PCT_MAX) {
        toast.error(`${label} 은(는) ${PCT_MIN}~${PCT_MAX} 사이 숫자여야 합니다.`);
        return;
      }
      parsed[key] = value;
    }
    setConfigSaving(true);
    try {
      await putSettings({ config: parsed });
      toast.success("전략 파라미터를 저장했습니다.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "설정을 저장하지 못했습니다.");
    } finally {
      setConfigSaving(false);
    }
  }, [draft, putSettings, toast]);

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
  // 저장된 값과 같으면 저장 버튼을 잠근다 — 같은 값을 다시 쓰는 호출을 막는다.
  const isDraftDirty = Boolean(draft) && PCT_FIELDS.some(([key]) => Number(draft?.[key]) !== config[key]);

  return (
    <PageFrame title="전략 사고팔기">
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {/* 전략 규칙 + 회차당 투입금 */}
        <div className="card appCard" style={cardStyle}>
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "center", fontSize: "var(--fs-sm)" }}>
            <span style={{ fontWeight: 700 }}>전략 규칙</span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
              1호 진입: {config.index_name} 일간 −
              <input
                type="number"
                step="0.1"
                min={PCT_MIN}
                max={PCT_MAX}
                style={pctInputStyle}
                value={draft?.entry_drop_pct ?? ""}
                disabled={configSaving}
                onChange={(event) => setDraft((d) => d && { ...d, entry_drop_pct: event.target.value })}
              />
              %
            </span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
              추가 진입: 마지막 매수가 −
              <input
                type="number"
                step="0.1"
                min={PCT_MIN}
                max={PCT_MAX}
                style={pctInputStyle}
                value={draft?.add_drop_pct ?? ""}
                disabled={configSaving}
                onChange={(event) => setDraft((d) => d && { ...d, add_drop_pct: event.target.value })}
              />
              %
            </span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
              매도: 평균단가 +
              <input
                type="number"
                step="0.1"
                min={PCT_MIN}
                max={PCT_MAX}
                style={pctInputStyle}
                value={draft?.take_profit_pct ?? ""}
                disabled={configSaving}
                onChange={(event) => setDraft((d) => d && { ...d, take_profit_pct: event.target.value })}
              />
              %
            </span>
            <button
              type="button"
              className="btn btn-sm btn-dark"
              disabled={configSaving || !isDraftDirty}
              onClick={() => void saveConfig()}
            >
              {configSaving ? "저장 중…" : "저장"}
            </button>
            <span>
              회차: <b>{config.rounds}회</b>
            </span>
            <span style={{ color: "var(--text-muted)" }}>계좌 {view.account_id}</span>
          </div>
        </div>

        {/* 슬랙 알림 */}
        <div className="card appCard" style={cardStyle}>
          <div style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>슬랙 알람</div>
          <p style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", lineHeight: 1.45, margin: 0 }}>
            켜두면 배치가 평일 09:10~15:20 을 10분 간격으로 판정해, 매수·매도 지정가에 닿은 회차가
            있을 때만 슬랙을 보냅니다. 같은 회차·동작은 하루 1회입니다.
          </p>
          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <div className="form-check form-switch" style={{ paddingLeft: "2.6em", marginBottom: 0 }}>
              <input
                className="form-check-input"
                type="checkbox"
                role="switch"
                id="strategyTradeSlackToggle"
                style={{ width: "2.2em", height: "1.2em" }}
                checked={view.slack_enabled}
                disabled={slackSaving}
                onChange={(event) => void toggleSlack(event.target.checked)}
              />
              <label
                className="form-check-label"
                htmlFor="strategyTradeSlackToggle"
                style={{ fontWeight: 700, marginLeft: 6 }}
              >
                슬랙 알람 {view.slack_enabled ? "켜짐" : "꺼짐"}
              </label>
            </div>
            <button
              type="button"
              className="btn btn-sm btn-outline-secondary"
              disabled={slackTesting}
              onClick={() => void testSlack()}
            >
              {slackTesting ? "발송 중…" : "지금 발송(테스트)"}
            </button>
          </div>
        </div>

        {/* 현재 판단 */}
        <div className="card appCard" style={cardStyle}>
          <div style={{ display: "flex", gap: 14, alignItems: "baseline", flexWrap: "wrap" }}>
            <span style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>현재 판단</span>
            <span style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
              계좌 실제 보유 기준 · 회차는 평균단가 높은 순 · 코스피 환산은 참고값
            </span>
          </div>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: "var(--fs-sm)" }}>
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
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", minWidth: 920, borderCollapse: "collapse", fontSize: "var(--fs-sm)" }}>
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
                        <span style={{ marginLeft: 5, color: "var(--text-muted)", fontSize: "var(--fs-sm)", fontWeight: 400 }}>
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
