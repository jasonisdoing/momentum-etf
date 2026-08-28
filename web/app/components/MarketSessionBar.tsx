"use client";

/** 상단 헤더의 시장 세션 표시 — 한국·미국·호주가 지금 어떤 장이고 다음 전환까지 얼마 남았는지.
 *
 *  세션 판정은 백엔드(`utils/market_session`)가 한다 — 시장 시간표가 `config` 한 곳에만
 *  있어야 화면과 서비스가 갈리지 않는다. 여기서는 남은 시간만 **매초 다시 센다**:
 *  서버 값을 그대로 쓰면 헤더 갱신 주기(10초)만큼 시간이 튄다.
 */

import { useEffect, useState } from "react";

export type MarketSession = {
  country: string;
  name: string;
  /** premarket | regular | aftermarket | closed */
  session: string;
  /** 다음 상태로 넘어가는 시각(ISO, 타임존 포함). */
  next_change_at: string;
};

// 세션 문구·색은 여기 두 상수만 고치면 된다(판정은 백엔드 utils/market_session).
const SESSION_LABELS: Record<string, string> = {
  premarket: "프리",
  regular: "본장",
  aftermarket: "애프터",
  daymarket: "데이",
  closed: "마감",
};

/** 세션별 색 — 열려 있을수록 진하다. 마감은 눌러 둔다. */
const SESSION_COLORS: Record<string, string> = {
  premarket: "#0ca678",
  regular: "#e03131",
  aftermarket: "#1971c2",
  daymarket: "#7048e8",
  closed: "var(--text-muted)",
};

const FLAGS: Record<string, string> = { kor: "🇰🇷", us: "🇺🇸", au: "🇦🇺" };

/** 남은 시간 — `17분` · `2시간 47분` · `2일 13시간 17분`. 0인 자리는 적지 않는다. */
function formatRemaining(untilMs: number): string {
  const totalMinutes = Math.max(0, Math.round(untilMs / 60000));
  const days = Math.floor(totalMinutes / (60 * 24));
  const hours = Math.floor((totalMinutes % (60 * 24)) / 60);
  const minutes = totalMinutes % 60;
  const parts: string[] = [];
  if (days > 0) parts.push(`${days}일`);
  if (hours > 0) parts.push(`${hours}시간`);
  if (minutes > 0 || parts.length === 0) parts.push(`${minutes}분`);
  return parts.join(" ");
}

export function MarketSessionBar({ sessions }: { sessions: MarketSession[] }) {
  // 1분마다 다시 그린다 — 표시 단위가 분이라 초 단위로 돌릴 이유가 없다.
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setTick((value) => value + 1), 60_000);
    return () => clearInterval(timer);
  }, []);
  void tick;

  if (!sessions.length) return null;
  const now = Date.now();

  return (
    <span className="marketSessionBar">
      {sessions.map((row) => {
        const label = SESSION_LABELS[row.session] ?? row.session;
        const until = new Date(row.next_change_at).getTime() - now;
        const remaining = Number.isFinite(until) ? formatRemaining(until) : null;
        // 마감이면 '열릴 때까지', 그 외에는 '이 세션이 끝날 때까지'다.
        const suffix = row.session === "closed" ? "후 개장" : "남음";
        const detail = remaining ? `${remaining} ${suffix}` : null;
        return (
          <span
            key={row.country}
            className="marketSessionItem"
            title={detail ? `${row.name} ${label} — ${detail}` : `${row.name} ${label}`}
          >
            <span aria-hidden>{FLAGS[row.country] ?? ""}</span>
            {/* 세션 이름과 괄호는 한 덩어리로 붙인다 — 「애프터(17분 남음)」 */}
            <span className="marketSessionText">
              <strong style={{ color: SESSION_COLORS[row.session] ?? "var(--text-muted)" }}>{label}</strong>
              {detail ? <span className="marketSessionRemaining">({detail})</span> : null}
            </span>
          </span>
        );
      })}
    </span>
  );
}
