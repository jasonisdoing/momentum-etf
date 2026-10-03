"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { IconChevronLeft, IconChevronRight } from "@tabler/icons-react";

import { PageFrame } from "../components/PageFrame";
import styles from "./market-calendar.module.css";

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

type WeekdayStat = { mean_pct: number; up_ratio: number; count: number };
type WeekdayStatsResponse = {
  start: string;
  end: string;
  months: number;
  /** 지수 티커 → 요일 번호(월=0 … 금=4) → 통계. 센 거래일이 없는 요일은 빠진다. */
  stats: Record<string, Record<string, WeekdayStat>>;
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
/** 요일별 통계에 넣는 지수 — 달력 칸의 지수 줄과 같은 4개. */
const STAT_INDICES = MARKET_ROWS.flatMap((row) => ("ticker" in row ? [{ label: row.label, ticker: row.ticker }] : []));
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

function marketBackground(sum: number | null): string {
  if (sum == null || Math.abs(sum) < 0.5) return "";
  const magnitude = Math.abs(sum);
  const level = magnitude >= 2 ? 4 : magnitude >= 1.5 ? 3 : magnitude >= 1 ? 2 : 1;
  const up = [styles.marketUp1, styles.marketUp2, styles.marketUp3, styles.marketUp4];
  const down = [styles.marketDown1, styles.marketDown2, styles.marketDown3, styles.marketDown4];
  return (sum > 0 ? up : down)[level - 1];
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

/** 오늘이 속한 달에서 과거로 몇 달까지, 미래로 몇 달까지 한 화면에 이어 붙일지. */
const MONTHS_BACK = 24;
const MONTHS_AHEAD = 1;

type DayCell = { key: string; day: number; monthKey: string; month: number; monthStart: boolean };
/** 월 컬럼 한 칸이 가리키는 묶음 — 연속된 주들. 한 주는 그 주 금요일이 속한 달에 넣는다. */
type MonthBlock = { key: string; year: number; month: number; weeks: DayCell[][] };

function monthKeyOf(year: number, month: number): string {
  return `${year}-${String(month + 1).padStart(2, "0")}`;
}

/**
 * 같은 날짜가 두 번 나오지 않는 연속 주 목록을 달 묶음으로 나눈다.
 * 한 주(월~금)는 **금요일이 속한 달**의 묶음에 들어간다 — 달이 바뀌는 주(예: 9/28~10/2)는 10월 묶음의
 * 첫 줄이 되고, 그 줄 안에서 달이 바뀌는 날에는 칸에 달 표시(`monthStart`)가 붙는다.
 */
function buildMonthBlocks(todayYear: number, todayMonth: number): MonthBlock[] {
  const startMonth = new Date(Date.UTC(todayYear, todayMonth - 1 - MONTHS_BACK, 1));
  const endMonth = new Date(Date.UTC(todayYear, todayMonth - 1 + MONTHS_AHEAD + 1, 0)); // 마지막 달의 말일
  const cursor = new Date(startMonth);
  cursor.setUTCDate(cursor.getUTCDate() - ((cursor.getUTCDay() + 6) % 7)); // 시작일이 속한 주의 월요일
  const blocks: MonthBlock[] = [];
  let previousMonthKey = "";
  while (cursor <= endMonth) {
    const weekCells: DayCell[] = Array.from({ length: 5 }, (_, offset) => {
      const date = new Date(cursor);
      date.setUTCDate(date.getUTCDate() + offset);
      const monthKey = monthKeyOf(date.getUTCFullYear(), date.getUTCMonth());
      const cell: DayCell = {
        key: dateKey(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()),
        day: date.getUTCDate(),
        monthKey,
        month: date.getUTCMonth(),
        monthStart: previousMonthKey !== "" && previousMonthKey !== monthKey,
      };
      previousMonthKey = monthKey;
      return cell;
    });
    const friday = weekCells[4];
    const last = blocks[blocks.length - 1];
    if (last && last.key === friday.monthKey) last.weeks.push(weekCells);
    else blocks.push({ key: friday.monthKey, year: Number(friday.monthKey.slice(0, 4)), month: friday.month, weeks: [weekCells] });
    cursor.setUTCDate(cursor.getUTCDate() + 7);
  }
  return blocks;
}

type MonthBlockViewProps = {
  block: MonthBlock;
  today: string;
  daysByKey: Record<string, DayData>;
  adrMeta: CalendarResponse["adr_meta"] | null;
  onVisible: (block: MonthBlock, first: string, last: string) => void;
  registerBlock: (key: string, element: HTMLElement | null) => void;
  scrollRoot: HTMLElement | null;
};

/** 한 달 묶음 — 왼쪽 좁은 월 컬럼이 이 묶음의 주들에 걸쳐 있다. 화면 가까이 오면 그때 데이터를 불러온다. */
function MonthBlockView({ block, today, daysByKey, adrMeta, onVisible, registerBlock, scrollRoot }: MonthBlockViewProps) {
  const cells = useMemo(() => block.weeks.flat(), [block]);
  const firstDay = cells[0].key;
  const lastDay = cells[cells.length - 1].key;
  const elementRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element || !scrollRoot) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) onVisible(block, firstDay, lastDay);
      },
      { root: scrollRoot, rootMargin: "600px 0px" },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [scrollRoot, block, firstDay, lastDay, onVisible]);

  return (
    <section
      ref={(element) => {
        elementRef.current = element;
        registerBlock(block.key, element);
      }}
      className={styles.monthBlock}
      aria-label={`${block.year}년 ${block.month + 1}월`}
    >
      <div className={styles.monthColumn} style={{ gridRow: `1 / span ${block.weeks.length}` }}>
        <span className={styles.monthLabel}>{block.year}년<br />{block.month + 1}월</span>
      </div>
      {cells.map((date) => {
        const data = daysByKey[date.key];
        return (
          <div
            key={date.key}
            id={`cal-${date.key}`}
            role="gridcell"
            aria-label={dayLabel(date.key)}
            className={[styles.day, date.month % 2 === 1 ? styles.monthAlt : ""].filter(Boolean).join(" ")}
          >
            <span className={styles.dayHeader}>
              <strong>{date.day}</strong>
              {date.monthStart ? <span className={styles.monthChip}>{date.month + 1}월</span> : null}
              {date.key === today ? <span className={styles.todayBadge}>오늘</span> : null}
            </span>
            <span className={styles.indexGroups}>
              {(["kor", "us"] as const).map((country) => {
                const session = data?.sessions[country];
                const holiday = session === "closed" || session === "closed_future";
                const tickers = country === "kor" ? ["^KS11", "^KQ11"] : ["^GSPC", "^NDX"];
                const first = indexChangeForBackground(data, country, tickers[0]);
                const second = indexChangeForBackground(data, country, tickers[1]);
                const sum = holiday || first == null || second == null ? null : first + second;
                const background = marketBackground(sum);
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
                              title={adrDecisionTitle(adrPoint, adrMeta?.[row.pool])}
                            >ADR {adrPoint?.adr == null ? "—" : adrPoint.adr.toFixed(1)}</span>
                          : <span className={issue && !future ? styles.dataIssue : changeClass(displayPoint?.change_pct)} title={issue?.reason ?? (future ? "Yahoo 선물 지연 시세" : undefined)}>{issue && !future ? issue.label : formatChange(displayPoint?.change_pct)}{displayPoint?.provisional && !issue ? "*" : ""}</span>}
                      </span>
                    );
                  })}
                  </span>
                );
              })}
            </span>
            <span className={styles.extraRow}>
              <span>USD/KRW</span>
              <span className={styles.fxValue}>
                {data?.fx ? <span>{data.fx.close.toFixed(2)}원</span> : null}
                <span className={changeClass(data?.fx?.change_pct)}>{formatChange(data?.fx?.change_pct)}{data?.fx?.provisional ? "*" : ""}</span>
              </span>
            </span>
          </div>
        );
      })}
    </section>
  );
}

export function MarketCalendarClient({ today }: { today: string }) {
  const [todayYear, todayMonth] = today.split("-").map(Number);
  const months = useMemo(() => buildMonthBlocks(todayYear, todayMonth), [todayYear, todayMonth]);
  const [weekdayStats, setWeekdayStats] = useState<WeekdayStatsResponse | null>(null);
  const [weekdayStatsError, setWeekdayStatsError] = useState<string | null>(null);
  const [visibleMonthKey, setVisibleMonthKey] = useState(monthKeyOf(todayYear, todayMonth - 1));
  const [daysByKey, setDaysByKey] = useState<Record<string, DayData>>({});
  const [adrMeta, setAdrMeta] = useState<CalendarResponse["adr_meta"] | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [calendarError, setCalendarError] = useState<string | null>(null);
  const [loadingCount, setLoadingCount] = useState(0);
  const [scrollRoot, setScrollRoot] = useState<HTMLDivElement | null>(null);
  const stickyRef = useRef<HTMLDivElement | null>(null);
  const sectionElements = useRef<Record<string, HTMLElement | null>>({});
  /** 이미 요청했거나 불러온 달 — 같은 달을 두 번 받지 않는다. 실패하면 빼서 다시 화면에 올 때 재시도한다. */
  const requestedMonths = useRef(new Set<string>());

  const registerSection = useCallback((key: string, element: HTMLElement | null) => {
    sectionElements.current[key] = element;
  }, []);

  const loadMonth = useCallback(async (monthRef: MonthBlock, first: string, last: string) => {
    if (requestedMonths.current.has(monthRef.key)) return;
    requestedMonths.current.add(monthRef.key);
    setLoadingCount((count) => count + 1);
    try {
      const query = new URLSearchParams({ start: first, end: last });
      const response = await fetch(`/api/market-calendar?${query}`, { cache: "no-store" });
      const payload = (await response.json()) as CalendarResponse;
      if (!response.ok) throw new Error(payload.error ?? "시장 캘린더 데이터를 불러오지 못했습니다.");
      setDaysByKey((prev) => ({ ...prev, ...payload.days }));
      setAdrMeta(payload.adr_meta);
      if (payload.warnings.length) setWarnings((prev) => [...new Set([...prev, ...payload.warnings])]);
      setCalendarError(null);
    } catch (error) {
      requestedMonths.current.delete(monthRef.key);
      setCalendarError(error instanceof Error ? error.message : "시장 캘린더 데이터를 불러오지 못했습니다.");
    } finally {
      setLoadingCount((count) => count - 1);
    }
  }, []);

  /** 스크롤 위치에서 지금 보고 있는 달 — 위쪽 고정 줄 바로 아래에 걸친 마지막 달. */
  const updateVisibleMonth = useCallback(() => {
    if (!scrollRoot) return;
    const rootTop = scrollRoot.getBoundingClientRect().top + (stickyRef.current?.offsetHeight ?? 0) + 24;
    let current = months[0].key;
    for (const month of months) {
      const element = sectionElements.current[month.key];
      if (element && element.getBoundingClientRect().top <= rootTop) current = month.key;
    }
    setVisibleMonthKey(current);
  }, [months, scrollRoot]);

  function scrollToMonth(key: string) {
    const element = sectionElements.current[key];
    if (!scrollRoot || !element) return;
    const offset = element.getBoundingClientRect().top - scrollRoot.getBoundingClientRect().top + scrollRoot.scrollTop;
    scrollRoot.scrollTo({ top: offset - (stickyRef.current?.offsetHeight ?? 0), behavior: "smooth" });
  }

  function moveMonth(offset: number) {
    const index = months.findIndex((month) => month.key === visibleMonthKey);
    const target = months[Math.min(Math.max(index + offset, 0), months.length - 1)];
    scrollToMonth(target.key);
  }

  function goToday() {
    document.getElementById(`cal-${latestVisibleDay(today)}`)?.scrollIntoView({ block: "center" });
  }

  // 처음에는 최근 평일이 화면 가운데에 오게 맞춘다 — 위로 스크롤하면 지난달이 이어진다.
  useEffect(() => {
    if (!scrollRoot) return;
    document.getElementById(`cal-${latestVisibleDay(today)}`)?.scrollIntoView({ block: "center" });
    updateVisibleMonth();
  }, [scrollRoot, today, updateVisibleMonth]);

  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      try {
        const response = await fetch("/api/market-calendar/weekday-stats", { cache: "no-store", signal: controller.signal });
        const payload = (await response.json()) as WeekdayStatsResponse;
        if (!response.ok || payload.error) throw new Error(payload.error ?? "요일별 통계를 불러오지 못했습니다.");
        setWeekdayStats(payload);
      } catch (error) {
        if (controller.signal.aborted) return;
        setWeekdayStatsError(error instanceof Error ? error.message : "요일별 통계를 불러오지 못했습니다.");
      }
    })();
    return () => controller.abort();
  }, []);

  const visibleMonth = months.find((month) => month.key === visibleMonthKey) ?? months[months.length - 1];
  const loading = loadingCount > 0;

  return (
    <PageFrame title="시장 캘린더" fullWidth fullHeight>
      <div className={styles.root}>
        <div className={styles.toolbar}>
          <div className={styles.monthNav} aria-label="월 선택">
            <button className="btn btn-outline-secondary btn-sm" type="button" onClick={() => moveMonth(-1)} aria-label="이전 달">
              <IconChevronLeft size={18} />
            </button>
            <strong>{visibleMonth.year}년 {visibleMonth.month + 1}월</strong>
            <button className="btn btn-outline-secondary btn-sm" type="button" onClick={() => moveMonth(1)} aria-label="다음 달">
              <IconChevronRight size={18} />
            </button>
            <button className="btn btn-outline-secondary btn-sm" type="button" onClick={goToday}>{latestVisibleDay(today) === today ? "오늘" : "최근 평일"}</button>
          </div>
        </div>

        {calendarError ? <p className={styles.error} role="alert">{calendarError}</p> : null}
        {warnings.length ? <p className={styles.error} role="status">{warnings.join(" ")}</p> : null}
        <p className={styles.notice}>
          {loading ? "날짜별 시장 데이터를 불러오는 중…" : "지수는 시장 현지 거래일·환율은 일봉 날짜 기준 · 장중·선물은 잠정값(*) · 미국 개장 전 지수 값이 없으면 오늘의 지연 선물 시세를 표시합니다."}
          {loading ? "" : " 한국·미국 개별주는 각 종목풀의 종가 ADR입니다. 빨강은 모멘텀 신규 진입 허용, 파랑은 ADR 하한 미달입니다."}
        </p>

        <div className={styles.calendarScroll} ref={setScrollRoot} onScroll={updateVisibleMonth}>
          <div className={styles.calendarInner}>
            <div className={styles.weekdayRow} ref={stickyRef} role="row">
              <div className={styles.weekdayCorner} />
              {VISIBLE_WEEKDAYS.map((weekday) => <div key={weekday} className={styles.weekday} role="columnheader">{weekday}</div>)}
            </div>
            {months.map((block) => (
              <MonthBlockView
                key={block.key}
                block={block}
                today={today}
                daysByKey={daysByKey}
                adrMeta={adrMeta}
                onVisible={loadMonth}
                registerBlock={registerSection}
                scrollRoot={scrollRoot}
              />
            ))}
          </div>
        </div>

        <section className={styles.stats} aria-label="요일별 평균 등락률">
          <div className={styles.statsHeader}>
            <strong>요일별 평균 등락률</strong>
            <span>
              최근 {weekdayStats?.months ?? 12}개월{weekdayStats ? ` (${weekdayStats.start} ~ ${weekdayStats.end})` : ""} · 확정 거래일만 · 칸 아래는 상승한 날의 비율과 거래일 수
            </span>
          </div>
          {weekdayStatsError ? <p className={styles.error} role="alert">{weekdayStatsError}</p> : null}
          <table className={styles.statsTable}>
            <thead>
              <tr>
                <th />
                {VISIBLE_WEEKDAYS.map((weekday) => <th key={weekday}>{weekday}</th>)}
              </tr>
            </thead>
            <tbody>
              {STAT_INDICES.map(({ label, ticker }) => (
                <tr key={ticker}>
                  <th>{label}</th>
                  {VISIBLE_WEEKDAYS.map((weekday, index) => {
                    const stat = weekdayStats?.stats[ticker]?.[String(index)];
                    return (
                      <td key={weekday}>
                        <span className={changeClass(stat?.mean_pct)}>{formatChange(stat?.mean_pct)}</span>
                        {stat ? <small>상승 {Math.round(stat.up_ratio * 100)}% · {stat.count}일</small> : null}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </PageFrame>
  );
}
