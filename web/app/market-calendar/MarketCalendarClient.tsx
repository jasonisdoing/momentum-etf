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

const WEEKDAYS = ["일", "월", "화", "수", "목", "금", "토"];
const INDEX_LABELS = ["코스피", "코스닥", "S&P 500", "나스닥 100"];
const FX_OPTIONS = ["USD/KRW", "AUD/KRW"];

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
  const days = useMemo(() => monthDays(year, month), [year, month]);

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
        <p className={styles.notice}>날짜별 시장·지수·환율·ADR 데이터 연결 전입니다. 값이 없는 칸은 실제 휴장이나 등락률 0%를 뜻하지 않습니다.</p>

        <div className={styles.calendarScroll}>
          <div className={styles.calendar} role="grid" aria-label={`${year}년 ${month + 1}월 시장 캘린더`}>
            {WEEKDAYS.map((weekday) => <div key={weekday} className={styles.weekday} role="columnheader">{weekday}</div>)}
            {days.map((date) => (
              <button
                key={date.key}
                type="button"
                role="gridcell"
                aria-label={dayLabel(date.key)}
                aria-selected={selectedDay === date.key}
                className={[styles.day, !date.inMonth ? styles.outside : "", selectedDay === date.key ? styles.selected : "", date.key === today ? styles.today : ""].filter(Boolean).join(" ")}
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
                <span className={styles.sessionRow}><span>한국</span><span>—</span><span>미국</span><span>—</span></span>
                <span className={styles.indexRows}>
                  {INDEX_LABELS.map((label) => <span key={label}><span>{label}</span><span>—</span></span>)}
                </span>
                <span className={styles.extraRow}><span>{selectedFx}</span><span>—</span></span>
                <span className={styles.extraRow}><span>ADR</span><span>—</span></span>
              </button>
            ))}
          </div>
        </div>

        <section className={styles.detail} aria-label="선택한 날짜의 상세 정보">
          <div><strong>{dayLabel(selectedDay)}</strong><span>선택한 날짜의 상세 정보</span></div>
          <p>시장 개장 여부와 확정된 지수·환율 등락률, 종목풀 ADR을 연결하면 이곳에 표시합니다.</p>
        </section>
      </div>
    </PageFrame>
  );
}
