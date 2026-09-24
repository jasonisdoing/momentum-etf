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

type PricePoint = { close: number; change_pct: number | null; provisional: boolean };
type AdrPoint = { date: string; adr: number | null; advance: number; decline: number };
type DayData = {
  sessions: Record<"kor" | "us", "closed" | "closed_future" | "finished" | "open" | "scheduled">;
  indices: Record<string, PricePoint | null>;
  fx: PricePoint | null;
  adr: Record<"kor_stock" | "us_stock", AdrPoint | null>;
};
type CalendarResponse = {
  days: Record<string, DayData>;
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
    <PageFrame title="시장 캘린더" fullWidth>
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
          {loading ? "날짜별 시장 데이터를 불러오는 중…" : "지수는 시장 현지 거래일·환율은 일봉 날짜 기준 · 장중 수치는 잠정값(*) · ‘—’는 수집되지 않은 값입니다."}
          {loading ? "" : " 한국·미국 개별주는 각 종목풀의 종가 ADR이며 다음 거래일 진입 판단에 사용됩니다."}
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
                  <span className={styles.indexRows}>
                    {MARKET_ROWS.map((row) => {
                      const session = data?.sessions[row.country];
                      const holiday = session === "closed" || session === "closed_future";
                      const point = "ticker" in row ? data?.indices[row.ticker] : null;
                      const adr = "pool" in row ? data?.adr[row.pool]?.adr : null;
                      return (
                        <span key={row.label}>
                          <span>{holiday ? `${row.label} ${SESSION_LABELS[session]}` : row.label}</span>
                          {holiday ? null : "pool" in row
                            ? <span>ADR {adr == null ? "—" : adr.toFixed(1)}</span>
                            : <span className={changeClass(point?.change_pct)}>{formatChange(point?.change_pct)}{point?.provisional ? "*" : ""}</span>}
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
            <p>
              한국 {SESSION_LABELS[calendar.days[selectedDay].sessions.kor]} · 미국 {SESSION_LABELS[calendar.days[selectedDay].sessions.us]}
              {(["kor_stock", "us_stock"] as const).map((pool) => {
                const point = calendar.days[selectedDay].adr[pool];
                return point ? ` · ${pool === "kor_stock" ? "한국" : "미국"} 개별주 ADR ${point.adr?.toFixed(1) ?? "—"} (상승 ${point.advance} · 하락 ${point.decline})` : "";
              })}
              {calendar.days[selectedDay].fx ? ` · USD/KRW ${calendar.days[selectedDay].fx.close.toFixed(2)}원` : ""}
            </p>
          ) : <p>이 날짜의 데이터가 없습니다.</p>}
        </section>
      </div>
    </PageFrame>
  );
}
