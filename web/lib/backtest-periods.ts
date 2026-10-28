/** 백테스트 일별 누적 곡선 → 기간별(연·월·주·일) 수익률 — 모든 전략 화면이 함께 쓴다.
 *
 *  입력은 백엔드 백테스트가 내려주는 **일별 누적 수익률(%)** 하나다. 연·월·주·일은 전부 여기서
 *  같은 곡선의 두 점 비율로 자르므로 화면마다 기준이 갈리지 않는다(같은 기간이 화면마다
 *  -16.62% / -9.20% 로 어긋난 적이 있어 한 곳으로 모았다).
 */

/** 일별 누적 행 — 백엔드가 시작일 대비 누적(%)을 내려준다. 합성만 인출·슬리브 값을 함께 준다. */
export type BacktestCumulativeRow = {
  date: string;
  strategy_pct: number;
  benchmark_pct: number;
  /** 그날의 시장 ADR — 모멘텀 ADR 게이트 이해용(넣는 화면만 채운다). */
  adr?: number | null;
  /** 합성: 인출 누계(최초 기준금액 대비 %). */
  withdrawn_to_initial_pct?: number;
  /** 합성: 각 슬리브를 혼자 굴렸을 때 누적(%). 그날 값이 없으면 null. */
  slots?: Record<string, number | null>;
};

export type PeriodRow = {
  period: string;
  strategy_pct: number;
  benchmark_pct: number;
  /** 구간 마지막 날의 시장 ADR — 일간·주간(판정일) 컬럼이 쓴다. */
  adr: number | null;
  /** 구간 최저 시장 ADR — 월간·연간 컬럼이 쓴다. */
  adr_min: number | null;
  /** 합성: 구간 중 인출한 금액(최초 기준금액 대비 %p). */
  withdrawal_pct?: number;
  /** 합성: 슬리브를 혼자 굴렸을 때의 구간 수익률. 어느 한쪽 값이 없으면 null. */
  slots?: Record<string, number | null>;
};

/** 그 날짜가 속한 주의 월요일 — 주간 묶음 키. 로컬 기준으로 조립한다(UTC 파싱은 하루 밀린다). */
export function weekKeyOf(date: string): string {
  const parsed = new Date(`${date}T00:00:00`);
  parsed.setDate(parsed.getDate() - ((parsed.getDay() + 6) % 7));
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${parsed.getFullYear()}-${pad(parsed.getMonth() + 1)}-${pad(parsed.getDate())}`;
}

/** 누적(%) 시계열을 기간별 수익률로 자른다. 구간 양끝의 누적값 비로 계산한다.
 *  주간은 묶음 키(월요일)와 표시 라벨(그 주 마지막 거래일)이 달라 따로 담는다. */
export function toPeriodRows(
  daily: BacktestCumulativeRow[],
  keyOf: (date: string) => string,
  labelByLastDate = false,
): PeriodRow[] {
  if (daily.length === 0) return [];
  type Snapshot = {
    strategy: number;
    benchmark: number;
    lastDate: string;
    adr: number | null;
    adrMin: number | null;
    withdrawn: number;
    slots: Record<string, number | null>;
  };
  const hasWithdrawn = daily[0].withdrawn_to_initial_pct !== undefined;
  const slotKeys = Object.keys(daily[0].slots ?? {});
  const lastByPeriod = new Map<string, Snapshot>();
  const order: string[] = [];
  for (const point of daily) {
    const key = keyOf(point.date);
    if (!lastByPeriod.has(key)) order.push(key);
    const knownMin = lastByPeriod.get(key)?.adrMin ?? null;
    const adr = point.adr ?? null;
    lastByPeriod.set(key, {
      strategy: point.strategy_pct,
      benchmark: point.benchmark_pct,
      lastDate: point.date,
      adr,
      adrMin: adr == null ? knownMin : knownMin == null ? adr : Math.min(knownMin, adr),
      withdrawn: point.withdrawn_to_initial_pct ?? 0,
      slots: point.slots ?? {},
    });
  }
  // 첫 구간의 기준은 시작 시점(누적 0%)이다 — 슬리브도 모두 0 에서 시작한다.
  let prev: Snapshot = {
    strategy: 0,
    benchmark: 0,
    lastDate: "",
    adr: null,
    adrMin: null,
    withdrawn: 0,
    slots: Object.fromEntries(slotKeys.map((key) => [key, 0])),
  };
  // 누적(%) 두 점 사이의 구간 수익률. 슬리브는 어느 한쪽이라도 값이 없으면 null — 0 으로 채우면
  // 데이터가 없는 구간이 '보합' 으로 보인다.
  const step = (now: number, before: number) => ((1 + now / 100) / (1 + before / 100) - 1) * 100;
  const slotStep = (now: number | null, before: number | null): number | null =>
    now == null || before == null ? null : step(now, before);
  const rows: PeriodRow[] = [];
  for (const key of order) {
    const current = lastByPeriod.get(key)!;
    rows.push({
      period: labelByLastDate ? current.lastDate : key,
      strategy_pct: step(current.strategy, prev.strategy),
      benchmark_pct: step(current.benchmark, prev.benchmark),
      adr: current.adr,
      adr_min: current.adrMin,
      ...(hasWithdrawn ? { withdrawal_pct: current.withdrawn - prev.withdrawn } : {}),
      ...(slotKeys.length > 0
        ? {
            slots: Object.fromEntries(
              slotKeys.map((slot) => [slot, slotStep(current.slots[slot] ?? null, prev.slots[slot] ?? null)]),
            ),
          }
        : {}),
    });
    prev = current;
  }
  return rows.reverse();
}
