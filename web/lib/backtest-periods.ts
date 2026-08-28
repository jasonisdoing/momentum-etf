/** 백테스트 일별 수익률 → 기간별(주·월·연) 집계 — 전략 화면과 자산 헬퍼가 함께 쓴다.
 *
 *  입력은 **일간 변동률(%)** 시계열 하나뿐이다. 주·월·연은 전부 여기서 복리로 합성하므로
 *  화면마다 다른 기준으로 갈리지 않는다(같은 기간이 화면마다 -16.62% / -9.20% 로 어긋난
 *  적이 있어 한 곳으로 모았다).
 */

/** 일별 행 — 백엔드가 전략·벤치마크의 그날 변동률(%)을 내려준다. */
export type BacktestDayRow = {
  date: string;
  strategy_pct: number | null;
  benchmark_pct: number | null;
};

export type BacktestMonthRow = {
  month: string;
  strategy_pct: number | null;
  benchmark_pct: number | null;
};

/** 달력 주 행 — 기준일은 그 주의 마지막 거래일이다. */
export type BacktestCalendarWeekRow = {
  week_end: string;
  strategy_pct: number | null;
  benchmark_pct: number | null;
};

export type BacktestYearRow = {
  year: string;
  strategy_pct: number | null;
  benchmark_pct: number | null;
  strategy_partial: boolean;
  benchmark_partial: boolean;
};

/** 변동률(%)들을 복리로 합성한다. 값이 하나도 없으면 null. */
export function compoundPct(values: (number | null)[]): number | null {
  const usable = values.filter((v): v is number => v != null && Number.isFinite(v));
  if (usable.length === 0) return null;
  return (usable.reduce((acc, v) => acc * (1 + v / 100), 1) - 1) * 100;
}

/** 누적 수익률(%) 곡선 → 일간 변동률(%) 행. 첫날은 기준일이라 빠진다.
 *
 *  자산 헬퍼처럼 누적 곡선만 내려주는 엔진의 결과를 전략 화면과 같은 입력으로 맞춘다.
 */
export function cumulativeToDailyRows(
  dates: string[],
  strategyCumulative: number[],
  benchmarkCumulative: number[],
): BacktestDayRow[] {
  const step = (series: number[], index: number): number | null => {
    const prev = series[index - 1];
    const current = series[index];
    if (prev == null || current == null || !Number.isFinite(prev) || !Number.isFinite(current)) return null;
    const base = 1 + prev / 100;
    if (base === 0) return null;
    return ((1 + current / 100) / base - 1) * 100;
  };
  const rows: BacktestDayRow[] = [];
  for (let index = 1; index < dates.length; index += 1) {
    rows.push({
      date: dates[index],
      strategy_pct: step(strategyCumulative, index),
      benchmark_pct: step(benchmarkCumulative, index),
    });
  }
  return rows;
}

function groupCompound<T>(
  daily: BacktestDayRow[],
  keyOf: (row: BacktestDayRow) => string,
  build: (key: string, strategy: number | null, benchmark: number | null) => T,
): T[] {
  const buckets = new Map<string, { strategy: (number | null)[]; benchmark: (number | null)[] }>();
  for (const row of daily) {
    const key = keyOf(row);
    const bucket = buckets.get(key) ?? { strategy: [], benchmark: [] };
    bucket.strategy.push(row.strategy_pct);
    bucket.benchmark.push(row.benchmark_pct);
    buckets.set(key, bucket);
  }
  return [...buckets.entries()]
    .sort((a, b) => b[0].localeCompare(a[0])) // 최신이 위
    .map(([key, bucket]) => build(key, compoundPct(bucket.strategy), compoundPct(bucket.benchmark)));
}

/** 일간 변동률을 **달력 월**로 복리 합산한다. */
export function toCalendarMonthRows(daily: BacktestDayRow[]): BacktestMonthRow[] {
  return groupCompound(daily, (row) => row.date.slice(0, 7), (month, strategy, benchmark) => ({
    month,
    strategy_pct: strategy,
    benchmark_pct: benchmark,
  }));
}

/** 일간 변동률을 **달력 주**(월~일)로 복리 합산한다. 행의 기준일은 그 주의 마지막 거래일. */
export function toCalendarWeekRows(daily: BacktestDayRow[]): BacktestCalendarWeekRow[] {
  const lastDayOfWeek = new Map<string, string>();
  const weekKey = (date: string): string => {
    const parsed = new Date(`${date}T00:00:00Z`);
    // 월요일을 주의 시작으로 본다(getUTCDay: 일=0). 주 키는 그 주 월요일 날짜다.
    const offset = (parsed.getUTCDay() + 6) % 7;
    parsed.setUTCDate(parsed.getUTCDate() - offset);
    return parsed.toISOString().slice(0, 10);
  };
  for (const row of daily) {
    const key = weekKey(row.date);
    const known = lastDayOfWeek.get(key);
    // 입력 순서를 믿지 않고 실제로 가장 늦은 날짜를 기준일로 쓴다.
    if (!known || row.date > known) lastDayOfWeek.set(key, row.date);
  }
  return groupCompound(daily, (row) => weekKey(row.date), (key, strategy, benchmark) => ({
    week_end: lastDayOfWeek.get(key) ?? key,
    strategy_pct: strategy,
    benchmark_pct: benchmark,
  }));
}

/** 월별 행을 연도로 묶는다. 12개월이 다 차지 않은 해는 `partial` 로 표시한다
 *  (/compare 의 부분 기간 `*` 표기와 같은 규칙). */
export function toYearRows(monthly: BacktestMonthRow[]): BacktestYearRow[] {
  const byYear = new Map<string, BacktestMonthRow[]>();
  for (const row of monthly) {
    const year = row.month.slice(0, 4);
    byYear.set(year, [...(byYear.get(year) ?? []), row]);
  }
  const countOf = (rows: BacktestMonthRow[], key: "strategy_pct" | "benchmark_pct") =>
    rows.filter((row) => row[key] != null).length;

  return [...byYear.entries()]
    .sort((a, b) => b[0].localeCompare(a[0]))
    .map(([year, rows]) => ({
      year,
      strategy_pct: compoundPct(rows.map((row) => row.strategy_pct)),
      benchmark_pct: compoundPct(rows.map((row) => row.benchmark_pct)),
      strategy_partial: countOf(rows, "strategy_pct") < 12,
      benchmark_partial: countOf(rows, "benchmark_pct") < 12,
    }));
}
