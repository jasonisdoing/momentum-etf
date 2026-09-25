"use client";

import { useEffect, useMemo, useState } from "react";
import { IconChevronLeft, IconChevronRight } from "@tabler/icons-react";

import { PageFrame } from "../components/PageFrame";
import styles from "./market-calendar.module.css";

type CalendarDay = {
  key: string;
  day: number;
  inMonth: boolean;
};

type PricePoint = { close: number; change_pct: number | null; provisional: boolean; quote_at?: string };
type IndexIssue = { label: string; reason: string };
type AdrPoint = { date: string; adr: number | null; advance: number; decline: number; entry_allowed: boolean; gate_adr: number | null };
type DayData = {
  sessions: Record<"kor" | "us", "closed" | "closed_future" | "finished" | "open" | "scheduled">;
  indices: Record<string, PricePoint | null>;
  index_issues: Record<string, IndexIssue>;
  futures: Record<string, PricePoint> | null;
  fx: PricePoint | null;
  adr: Record<"kor_stock" | "us_stock", AdrPoint | null>;
};
type CalendarResponse = {
  days: Record<string, DayData>;
  adr_meta: Record<"kor_stock" | "us_stock", { floor: number | null; gate_market: string | null }>;
  warnings: string[];
  error?: string;
};

const WEEKDAYS = ["일", "월", "화", "수", "목", "금", "토"];
const VISIBLE_WEEKDAYS = WEEKDAYS.slice(1, 6);
const MARKET_ROWS = [
  { label: "코스피", country: "kor", ticker: "^KS11" },
  { label: "코스닥", country: "kor", ticker: "^KQ11" },
  { label: "한국 개별주", country: "kor", pool: "kor_stock" },
  { label: "S&P 500", country: "us", ticker: "^GSPC" },
  { label: "나스닥 100", country: "us", ticker: "^NDX" },
  { label: "미국 개별주", country: "us", pool: "us_stock" },
] as const;
const SESSION_LABELS: Record<DayData["sessions"]["kor"], string> = {
  closed: "휴장",
  closed_future: "휴장 예정",
  finished: "마감",
  open: "장중",
  scheduled: "개장 예정",
};

function formatChange(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function changeClass(value: number | null | undefined): string {
  if (value == null || value === 0) return "";
  return value > 0 ? styles.positive : styles.negative;
}

function indexChangeForBackground(data: DayData | undefined, country: "kor" | "us", ticker: string): number | null {
  const point = data?.indices[ticker];
  const future = country === "us" && point?.change_pct == null ? data?.futures?.[ticker] : null;
  if (data?.index_issues[ticker] && !future) return null;
  return (future ?? point)?.change_pct ?? null;
}

function adrDecisionTitle(point: AdrPoint | null | undefined, meta: CalendarResponse["adr_meta"]["kor_stock"] | undefined): string | undefined {
  if (!point || !meta) return undefined;
  if (meta.floor == null) return "ADR 하한 없음 · 신규 진입 허용";
  const reference = point.gate_adr == null ? "기준 ADR 데이터 없음" : `기준 ADR 약 ${point.gate_adr.toFixed(1)}`;
  return `${meta.gate_market ?? "기준 시장 없음"} · ${reference} / 하한 ${meta.floor} · 신규 진입 ${point.entry_allowed ? "허용" : "제한"}`;
}

function dateKey(year: number, month: number, day: number): string {
  return `${year}-${String(month + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function monthDays(year: number, month: number): CalendarDay[] {
  const firstWeekday = (new Date(Date.UTC(year, month, 1)).getUTCDay() + 6) % 7;
  const dayCount = new Date(Date.UTC(year, month + 1, 0)).getUTCDate();
  const cellCount = Math.ceil((firstWeekday + dayCount) / 7) * 7;
  return Array.from({ length: cellCount }, (_, index) => {
    const date = new Date(Date.UTC(year, month, index - firstWeekday + 1));
    return {
      key: dateKey(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()),
      day: date.getUTCDate(),
      inMonth: date.getUTCMonth() === month,
    };
  }).filter((_, index) => index % 7 < 5);
}

function latestVisibleDay(key: string): string {
  const date = new Date(`${key}T00:00:00Z`);
  const weekday = date.getUTCDay();
  if (weekday === 0 || weekday === 6) date.setUTCDate(date.getUTCDate() - (weekday === 0 ? 2 : 1));
  return dateKey(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
}

function dayLabel(key: string): string {
  const [year, month, day] = key.split("-").map(Number);
  const weekday = WEEKDAYS[new Date(Date.UTC(year, month - 1, day)).getUTCDay()];
  return `${year}년 ${month}월 ${day}일 (${weekday})`;
}

export function MarketCalendarClient({ today }: { today: string }) {
  const [todayYear, todayMonth] = today.split("-").map(Number);
  const [year, setYear] = useState(todayYear);
  const [month, setMonth] = useState(todayMonth - 1);
  const [selectedDay, setSelectedDay] = useState(() => latestVisibleDay(today));
  const [calendar, setCalendar] = useState<CalendarResponse | null>(null);
  const [calendarError, setCalendarError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const days = useMemo(() => monthDays(year, month), [year, month]);
  const firstDay = days[0].key;
  const lastDay = days[days.length - 1].key;

  useEffect(() => {
    const controller = new AbortController();
    async function loadCalendar() {
      setLoading(true);
      setCalendar(null);
      setCalendarError(null);
      const query = new URLSearchParams({ start: firstDay, end: lastDay });
      try {
        const response = await fetch(`/api/market-calendar?${query}`, { cache: "no-store", signal: controller.signal });
        const payload = (await response.json()) as CalendarResponse;
        if (!response.ok) throw new Error(payload.error ?? "시장 캘린더 데이터를 불러오지 못했습니다.");
        if (!controller.signal.aborted) setCalendar(payload);
      } catch (error) {
        if (controller.signal.aborted) return;
        setCalendar(null);
        setCalendarError(error instanceof Error ? error.message : "시장 캘린더 데이터를 불러오지 못했습니다.");
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void loadCalendar();
    return () => controller.abort();
  }, [firstDay, lastDay]);

  function moveMonth(offset: number) {
    const next = new Date(Date.UTC(year, month + offset, 1));
    setYear(next.getUTCFullYear());
    setMonth(next.getUTCMonth());
    const firstDayInMonth = monthDays(next.getUTCFullYear(), next.getUTCMonth()).find((day) => day.inMonth);
    if (firstDayInMonth) setSelectedDay(firstDayInMonth.key);
  }

  function goToday() {
    setYear(todayYear);
    setMonth(todayMonth - 1);
    setSelectedDay(latestVisibleDay(today));
  }

  return (
    <PageFrame title="시장 캘린더" fullWidth fullHeight>
      <div className={styles.root}>
        <div className={styles.toolbar}>
          <div className={styles.monthNav} aria-label="월 선택">
            <button className="btn btn-outline-secondary btn-sm" type="button" onClick={() => moveMonth(-1)} aria-label="이전 달">
              <IconChevronLeft size={18} />
            </button>
            <strong>{year}년 {month + 1}월</strong>
            <button className="btn btn-outline-secondary btn-sm" type="button" onClick={() => moveMonth(1)} aria-label="다음 달">
              <IconChevronRight size={18} />
            </button>
            <button className="btn btn-outline-secondary btn-sm" type="button" onClick={goToday}>{latestVisibleDay(today) === today ? "오늘" : "최근 평일"}</button>
          </div>
        </div>

        {calendarError ? <p className={styles.error} role="alert">{calendarError}</p> : null}
        {calendar?.warnings.length ? <p className={styles.error} role="status">{calendar.warnings.join(" ")}</p> : null}
        <p className={styles.notice}>
          {loading ? "날짜별 시장 데이터를 불러오는 중…" : "지수는 시장 현지 거래일·환율은 일봉 날짜 기준 · 장중·선물은 잠정값(*) · 미국 개장 전 지수 값이 없으면 오늘의 지연 선물 시세를 표시합니다."}
          {loading ? "" : " 한국·미국 개별주는 각 종목풀의 종가 ADR입니다. 빨강은 모멘텀 신규 진입 허용, 파랑은 ADR 하한 미달입니다."}
        </p>

        <div className={styles.calendarScroll}>
          <div className={styles.calendar} role="grid" aria-label={`${year}년 ${month + 1}월 시장 캘린더`}>
            {VISIBLE_WEEKDAYS.map((weekday) => <div key={weekday} className={styles.weekday} role="columnheader">{weekday}</div>)}
            {days.map((date) => {
              const data = calendar?.days[date.key];
              return (
                <button
                  key={date.key}
                  type="button"
                  role="gridcell"
                  aria-label={dayLabel(date.key)}
                  aria-selected={selectedDay === date.key}
                  className={[styles.day, !date.inMonth ? styles.outside : "", selectedDay === date.key ? styles.selected : ""].filter(Boolean).join(" ")}
                  onClick={() => {
                    setSelectedDay(date.key);
                    if (!date.inMonth) {
                      const [nextYear, nextMonth] = date.key.split("-").map(Number);
                      setYear(nextYear);
                      setMonth(nextMonth - 1);
                    }
                  }}
                >
                  <span className={styles.dayHeader}><strong>{date.day}</strong>{date.key === today ? <span className={styles.todayBadge}>오늘</span> : null}</span>
                  <span className={styles.indexGroups}>
                    {(["kor", "us"] as const).map((country) => {
                      const session = data?.sessions[country];
                      const holiday = session === "closed" || session === "closed_future";
                      const tickers = country === "kor" ? ["^KS11", "^KQ11"] : ["^GSPC", "^NDX"];
                      const first = indexChangeForBackground(data, country, tickers[0]);
                      const second = indexChangeForBackground(data, country, tickers[1]);
                      const sum = holiday || first == null || second == null ? null : first + second;
                      const background = sum == null || sum === 0 ? "" : sum > 0 ? styles.marketUp : styles.marketDown;
                      return (
                        <span
                          key={country}
                          className={[styles.indexRows, background].filter(Boolean).join(" ")}
                          title={sum == null ? undefined : `${country === "kor" ? "코스피 + 코스닥" : "S&P 500 + 나스닥 100"} 등락률 합 ${formatChange(sum)}`}
                        >
                        {MARKET_ROWS.filter((row) => row.country === country).map((row) => {
                          const session = data?.sessions[row.country];
                          const holiday = session === "closed" || session === "closed_future";
                          const point = "ticker" in row ? data?.indices[row.ticker] : null;
                          const issue = "ticker" in row ? data?.index_issues[row.ticker] : undefined;
                          const future = "ticker" in row && row.country === "us" && point?.change_pct == null
                            ? data?.futures?.[row.ticker] : null;
                          const displayPoint = future ?? point;
                          const adrPoint = "pool" in row ? data?.adr[row.pool] : null;
                          return (
                            <span key={row.label}>
                              <span>{holiday ? `${row.label} ${SESSION_LABELS[session]}` : future ? `${row.label} 선물` : row.label}</span>
                              {holiday ? null : "pool" in row
                                ? <span
                                    className={adrPoint?.adr == null ? "" : adrPoint.entry_allowed ? styles.positive : styles.negative}
                                    title={adrDecisionTitle(adrPoint, calendar?.adr_meta[row.pool])}
                                  >ADR {adrPoint?.adr == null ? "—" : adrPoint.adr.toFixed(1)}</span>
                                : <span className={issue && !future ? styles.dataIssue : changeClass(displayPoint?.change_pct)} title={issue?.reason ?? (future ? "Yahoo 선물 지연 시세" : undefined)}>{issue && !future ? issue.label : formatChange(displayPoint?.change_pct)}{displayPoint?.provisional && !issue ? "*" : ""}</span>}
                            </span>
                          );
                        })}
                        </span>
                      );
                    })}
                  </span>
                  <span className={styles.extraRow}><span>USD/KRW</span><span className={changeClass(data?.fx?.change_pct)}>{formatChange(data?.fx?.change_pct)}{data?.fx?.provisional ? "*" : ""}</span></span>
                </button>
              );
            })}
          </div>
        </div>

        <section className={styles.detail} aria-label="선택한 날짜의 상세 정보">
          <div><strong>{dayLabel(selectedDay)}</strong><span>선택한 날짜의 상세 정보</span></div>
          {calendar?.days[selectedDay] ? (
            <>
              <p>
                한국 {SESSION_LABELS[calendar.days[selectedDay].sessions.kor]} · 미국 {SESSION_LABELS[calendar.days[selectedDay].sessions.us]}
                {(["kor_stock", "us_stock"] as const).map((pool) => {
                  const point = calendar.days[selectedDay].adr[pool];
                  return point ? ` · ${pool === "kor_stock" ? "한국" : "미국"} 개별주 ADR ${point.adr?.toFixed(1) ?? "—"} (상승 ${point.advance} · 하락 ${point.decline})` : "";
                })}
                {calendar.days[selectedDay].fx ? ` · USD/KRW ${calendar.days[selectedDay].fx.close.toFixed(2)}원` : ""}
              </p>
              {MARKET_ROWS.flatMap((row) => {
                if (!("ticker" in row)) return [];
                const issue = calendar.days[selectedDay].index_issues[row.ticker];
                return issue ? [<p key={row.label} className={styles.dataIssue} role="status">{row.label}: {issue.reason}</p>] : [];
              })}
            </>
          ) : <p>이 날짜의 데이터가 없습니다.</p>}
        </section>
      </div>
    </PageFrame>
  );
}
