"use client";

/**
 * 연도별 값의 흐름을 보여주는 미니 막대 그래프 (그리드 셀용).
 *
 * `2/2` 같은 비율 표기는 **유지와 감소를 구분하지 못하고**(둘 다 "증가 아님"),
 * 얼마나 늘었는지도 안 보인다. 막대 높이로 값의 모양을, **막대마다의 색**으로 전년 대비
 * 방향을 준다 — 비율 표기가 담던 "몇 년 중 몇 번 늘었나"가 색으로 그대로 남는다.
 *
 * 색은 시스템 규칙(한국식)을 따른다: **증가 빨강 · 감소 파랑**
 * (`globals.css` 의 `.metricPositive` / `.metricNegative` 와 같은 값).
 * 서구식(증가 초록)으로 두면 같은 행의 기간 수익률 컬럼과 색 뜻이 뒤집힌다.
 *
 * AG Grid 의 스파크라인은 Enterprise 기능이라 쓸 수 없어 SVG 로 직접 그린다.
 */

export type TrendPoint = {
  /** 연도 등 가로축 라벨. 툴팁에 쓴다. */
  label: string;
  /** 값. 없으면(미공시) 막대를 그리지 않고 자리만 비운다. */
  value: number | null;
  /** 확정이 아닌 추정치(컨센서스)면 true — 옅게 그려 확정값과 구분한다. */
  estimated?: boolean;
};

const BAR_WIDTH = 7;
const BAR_GAP = 3;
const HEIGHT = 22;

// globals.css 의 .metricPositive / .metricNegative 와 같은 값 (SVG 는 클래스로 못 칠한다).
const UP_COLOR = "#d32f2f"; // 증가
const DOWN_COLOR = "#1d4ed8"; // 감소
const FLAT_COLOR = "#94a3b8"; // 유지 · 비교 불가

/** 전년 대비 방향 — 비교할 앞 값이 없으면 회색. */
function barColor(previous: number | null, current: number): string {
  if (previous === null) return FLAT_COLOR;
  if (current > previous) return UP_COLOR;
  if (current < previous) return DOWN_COLOR;
  return FLAT_COLOR;
}

export function TrendSparkline({
  points,
  format,
}: {
  /** 오래된 → 최신 순. */
  points: TrendPoint[];
  /** 툴팁에 쓸 값 표기. 생략하면 그대로 찍는다. */
  format?: (value: number) => string;
}) {
  const values = points.map((point) => point.value).filter((value): value is number => value !== null);
  if (values.length === 0) {
    return <span style={{ color: "var(--text-muted)" }}>-</span>;
  }

  // 0 을 기준선으로 둔다 — 적자(음수)가 아래로 내려가야 흑자 전환이 보인다.
  const max = Math.max(0, ...values);
  const min = Math.min(0, ...values);
  const span = max - min || 1;
  const baselineY = HEIGHT * (max / span);

  const width = points.length * BAR_WIDTH + (points.length - 1) * BAR_GAP;
  const tooltip = points
    .map((point) => {
      const text = point.value === null ? "미공시" : format ? format(point.value) : String(point.value);
      return `${point.label} ${text}${point.estimated ? " (예상)" : ""}`;
    })
    .join("  →  ");

  // 색 판정용 직전 값 — 미공시 연도는 건너뛰고 마지막으로 값이 있던 해와 견준다.
  let previous: number | null = null;

  return (
    <span style={{ display: "inline-flex", alignItems: "center", height: "100%" }} title={tooltip}>
      <svg width={width} height={HEIGHT} role="img" aria-label={tooltip}>
        {/* 값이 음수인 항목이 있을 때만 0 기준선을 그린다 */}
        {min < 0 ? <line x1={0} y1={baselineY} x2={width} y2={baselineY} stroke="#cbd5e1" strokeWidth={1} /> : null}
        {points.map((point, index) => {
          const x = index * (BAR_WIDTH + BAR_GAP);
          if (point.value === null) {
            // 미공시 — 자리를 비워 두면 연도가 밀려 보이므로 옅은 바닥선만 남긴다.
            return <rect key={point.label} x={x} y={HEIGHT - 1} width={BAR_WIDTH} height={1} fill="#e2e8f0" />;
          }
          const color = barColor(previous, point.value);
          previous = point.value;
          const valueY = HEIGHT * ((max - point.value) / span);
          const top = Math.min(valueY, baselineY);
          const barHeight = Math.max(1, Math.abs(baselineY - valueY));
          return (
            <rect
              key={point.label}
              x={x}
              y={top}
              width={BAR_WIDTH}
              height={barHeight}
              fill={color}
              // 추정치는 옅게 — 확정값과 같은 진하기로 두면 예상이 사실처럼 읽힌다.
              fillOpacity={point.estimated ? 0.35 : 1}
            />
          );
        })}
      </svg>
    </span>
  );
}
