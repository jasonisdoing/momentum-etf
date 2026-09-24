"use client";

import { useEffect, useMemo, useState } from "react";
import { IconChevronLeft, IconChevronRight } from "@tabler/icons-react";

import { PageFrame } from "../components/PageFrame";
import styles from "./market-calendar.module.css";

type PoolOption = {
  ticker_type: string;
  name: string;
  icon: string;
};

type PoolResponse = {
  ticker_types?: PoolOption[];
  error?: string;
};

type CalendarDay = {
  key: string;
  day: number;
  inMonth: boolean;
};

type PricePoint = { close: number; change_pct: number | null; provisional: boolean };
type AdrPoint = { date: string; adr: number | null; advance: number; decline: number };
type DayData = {
  sessions: Record<"kor" | "us", "closed" | "finished" | "open" | "scheduled">;
  indices: Record<string, PricePoint | null>;
  fx: PricePoint | null;
  adr: AdrPoint | null;
};
type CalendarResponse = {
  days: Record<string, DayData>;
  adr_meta: { market: string; floor: number | null } | null;
  warnings: string[];
  error?: string;
};

const WEEKDAYS = ["일", "월", "화", "수", "목", "금", "토"];
const INDEX_LABELS = ["코스피", "코스닥", "S&P 500", "나스닥 100"];
const FX_OPTIONS = ["USD/KRW", "AUD/KRW"];
const INDEX_TICKERS = ["^KS11", "^KQ11", "^GSPC", "^NDX"];
const SESSION_LABELS: Record<DayData["sessions"]["kor"], string> = {
  closed: "휴장",
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
  const firstWeekday = new Date(Date.UTC(year, month, 1)).getUTCDay();
  const dayCount = new Date(Date.UTC(year, month + 1, 0)).getUTCDate();
  const cellCount = Math.ceil((firstWeekday + dayCount) / 7) * 7;
  return Array.from({ length: cellCount }, (_, index) => {
    const date = new Date(Date.UTC(year, month, index - firstWeekday + 1));
    return {
      key: dateKey(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()),
      day: date.getUTCDate(),
      inMonth: date.getUTCMonth() === month,
    };
  });
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
  const [selectedDay, setSelectedDay] = useState(today);
  const [selectedPool, setSelectedPool] = useState("");
  const [selectedFx, setSelectedFx] = useState(FX_OPTIONS[0]);
  const [pools, setPools] = useState<PoolOption[]>([]);
  const [poolError, setPoolError] = useState<string | null>(null);
  const [calendar, setCalendar] = useState<CalendarResponse | null>(null);
  const [calendarError, setCalendarError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const days = useMemo(() => monthDays(year, month), [year, month]);
  const firstDay = days[0].key;
  const lastDay = days[days.length - 1].key;

  useEffect(() => {
    const controller = new AbortController();
    async function loadPools() {
      try {
        const response = await fetch("/api/rank-toolbar", { signal: controller.signal });
        const payload = (await response.json()) as PoolResponse;
        if (!response.ok) throw new Error(payload.error ?? "종목풀 목록을 불러오지 못했습니다.");
        setPools(payload.ticker_types ?? []);
        setPoolError(null);
      } catch (error) {
        if (controller.signal.aborted) return;
        setPoolError(error instanceof Error ? error.message : "종목풀 목록을 불러오지 못했습니다.");
      }
    }
    void loadPools();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    async function loadCalendar() {
      setLoading(true);
      setCalendar(null);
      setCalendarError(null);
      const query = new URLSearchParams({ start: firstDay, end: lastDay, fx: selectedFx });
      if (selectedPool) query.set("pool", selectedPool);
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
  }, [firstDay, lastDay, selectedPool, selectedFx]);

  function moveMonth(offset: number) {
    const next = new Date(Date.UTC(year, month + offset, 1));
    setYear(next.getUTCFullYear());
    setMonth(next.getUTCMonth());
    setSelectedDay(dateKey(next.getUTCFullYear(), next.getUTCMonth(), 1));
  }

  function goToday() {
    setYear(todayYear);
    setMonth(todayMonth - 1);
    setSelectedDay(today);
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
            <button className="btn btn-outline-secondary btn-sm" type="button" onClick={goToday}>오늘</button>
          </div>
          <div className={styles.filters}>
            <label>
              <span>종목풀 ADR</span>
              <select className="form-select form-select-sm" value={selectedPool} onChange={(event) => setSelectedPool(event.target.value)}>
                <option value="">종목풀 선택</option>
                {pools.map((pool) => (
                  <option key={pool.ticker_type} value={pool.ticker_type}>{pool.icon} {pool.name}</option>
                ))}
              </select>
            </label>
            <label>
              <span>환율</span>
              <select className="form-select form-select-sm" value={selectedFx} onChange={(event) => setSelectedFx(event.target.value)}>
                {FX_OPTIONS.map((fx) => <option key={fx} value={fx}>{fx}</option>)}
              </select>
            </label>
          </div>
        </div>

        {poolError ? <p className={styles.error} role="alert">{poolError}</p> : null}
        {calendarError ? <p className={styles.error} role="alert">{calendarError}</p> : null}
        {calendar?.warnings.length ? <p className={styles.error} role="status">{calendar.warnings.join(" ")}</p> : null}
        <p className={styles.notice}>
          {loading ? "날짜별 시장 데이터를 불러오는 중…" : "지수는 시장 현지 거래일·환율은 일봉 날짜 기준 · 장중 수치는 잠정값(*) · ‘—’는 수집되지 않은 값입니다."}
          {calendar?.adr_meta ? ` ADR 기준: ${calendar.adr_meta.market}${calendar.adr_meta.floor == null ? "" : ` · 진입 하한 ${calendar.adr_meta.floor}`} · 종가 ADR은 다음 거래일 진입 판단에 사용` : ""}
        </p>

        <div className={styles.calendarScroll}>
          <div className={styles.calendar} role="grid" aria-label={`${year}년 ${month + 1}월 시장 캘린더`}>
            {WEEKDAYS.map((weekday) => <div key={weekday} className={styles.weekday} role="columnheader">{weekday}</div>)}
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
                  <span className={styles.sessionRow}>
                    <span>한국</span><span>{data ? SESSION_LABELS[data.sessions.kor] : "—"}</span>
                    <span>미국</span><span>{data ? SESSION_LABELS[data.sessions.us] : "—"}</span>
                  </span>
                  <span className={styles.indexRows}>
                    {INDEX_LABELS.map((label, index) => {
                      const point = data?.indices[INDEX_TICKERS[index]];
                      return <span key={label}><span>{label}</span><span className={changeClass(point?.change_pct)}>{formatChange(point?.change_pct)}{point?.provisional ? "*" : ""}</span></span>;
                    })}
                  </span>
                  <span className={styles.extraRow}><span>{selectedFx}</span><span className={changeClass(data?.fx?.change_pct)}>{formatChange(data?.fx?.change_pct)}{data?.fx?.provisional ? "*" : ""}</span></span>
                  <span className={styles.extraRow}><span>ADR</span><span className={data?.adr?.adr != null && calendar?.adr_meta?.floor != null && data.adr.adr < calendar.adr_meta.floor ? styles.adrBelow : ""}>{data?.adr?.adr == null ? "—" : data.adr.adr.toFixed(1)}</span></span>
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
              {calendar.days[selectedDay].adr ? ` · ADR ${calendar.days[selectedDay].adr.adr?.toFixed(1) ?? "—"} (상승 ${calendar.days[selectedDay].adr.advance} · 하락 ${calendar.days[selectedDay].adr.decline})` : ""}
              {calendar.days[selectedDay].fx ? ` · ${selectedFx} ${calendar.days[selectedDay].fx.close.toFixed(2)}원` : ""}
            </p>
          ) : <p>이 날짜의 데이터가 없습니다.</p>}
        </section>
      </div>
    </PageFrame>
  );
}
