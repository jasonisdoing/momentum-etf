import type { PortfolioChangeBreakdownItem } from "@/lib/portfolio-change";

type PortfolioChangeFxRate = {
  currency: string;
  change_pct?: number | null;
};

type PortfolioChangeBreakdownProps = {
  items: PortfolioChangeBreakdownItem[];
  /** 호환용 — 더 이상 사용하지 않음 (합산값은 item.adjusted_change_pct 에 포함). */
  fxRates?: PortfolioChangeFxRate[];
  variant: "detail" | "compact";
  emptyText: string;
};

function formatWeight(value: number): string {
  if (!Number.isFinite(value)) return "-";
  return `${new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 0 }).format(value)}%`;
}

function formatSignedPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function getSignedClass(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) return "";
  return value > 0 ? "metricPositive" : "metricNegative";
}

/** 다중 통화일 때 표시할 "합계" 정보를 계산.
 * 가중평균 = Σ(weight × adjusted_change_pct) / Σ(weight)
 * → "추적된 추세가 전체에 동일하게 적용된다고 가정한 추정 평균 수익률"
 */
function computeSummary(items: PortfolioChangeBreakdownItem[]): {
  totalWeight: number;
  weightedAvg: number;
} | null {
  if (items.length < 2) return null;
  const totalWeight = items.reduce((sum, it) => sum + it.weight, 0);
  if (totalWeight <= 0) return null;
  const weightedSum = items.reduce((sum, it) => sum + it.weight * it.adjusted_change_pct, 0);
  return {
    totalWeight,
    weightedAvg: weightedSum / totalWeight,
  };
}

export function PortfolioChangeBreakdown({
  items,
  variant,
  emptyText,
}: PortfolioChangeBreakdownProps) {
  if (items.length === 0) {
    return <span>{emptyText}</span>;
  }

  const summary = computeSummary(items);

  if (variant === "detail") {
    return (
      <div className="portfolioChangeBreakdownDetail" aria-label="포트폴리오 변동 상세">
        {summary ? (
          <>
            <div className="portfolioChangeBreakdownDetailRow portfolioChangeBreakdownTotal">
              <span className="portfolioChangeBreakdownRegion">
                <strong>합계</strong> <span style={{ color: "#5f6b82", fontWeight: 400 }}>(추적 {formatWeight(summary.totalWeight)})</span>
              </span>
              <strong className={getSignedClass(summary.weightedAvg)} style={{ fontSize: "1.2em" }}>
                {formatSignedPercent(summary.weightedAvg)}
              </strong>
            </div>
            <div className="portfolioChangeBreakdownDetailRow portfolioChangeBreakdownInline">
              {items.map((item, idx) => (
                <span key={item.currency}>
                  {idx > 0 ? ", " : ""}
                  <span>{item.label}({formatWeight(item.weight)})</span>{" "}
                  <span className={getSignedClass(item.adjusted_change_pct)}>
                    {formatSignedPercent(item.adjusted_change_pct)}
                  </span>
                </span>
              ))}
            </div>
          </>
        ) : (
          // 단일 통화 — 합계 라인과 동일한 강조 스타일
          items.map((item) => (
            <div
              key={item.currency}
              className="portfolioChangeBreakdownDetailRow portfolioChangeBreakdownTotal"
            >
              <span className="portfolioChangeBreakdownRegion">
                <strong>{item.label}</strong> <span style={{ color: "#5f6b82", fontWeight: 400 }}>({formatWeight(item.weight)})</span>
              </span>
              <strong className={getSignedClass(item.adjusted_change_pct)} style={{ fontSize: "1.2em" }}>
                {formatSignedPercent(item.adjusted_change_pct)}
              </strong>
            </div>
          ))
        )}
      </div>
    );
  }
  // compact variant
  return (
    <span className="portfolioChangeBreakdownCompact">
      {summary ? (
        <>
          <span className="portfolioChangeBreakdownCompactItem">
            <strong>합계</strong> <span style={{ color: "#5f6b82", fontWeight: 400 }}>(추적 {formatWeight(summary.totalWeight)})</span>{" "}
            <strong className={getSignedClass(summary.weightedAvg)} style={{ fontSize: "1.2em" }}>
              {formatSignedPercent(summary.weightedAvg)}
            </strong>
          </span>
          <span className="portfolioChangeBreakdownCompactItem portfolioChangeBreakdownInline">
            {items.map((item, idx) => (
              <span key={item.currency}>
                {idx > 0 ? ", " : ""}
                <span>{item.label}({formatWeight(item.weight)})</span>{" "}
                <span className={getSignedClass(item.adjusted_change_pct)}>
                  {formatSignedPercent(item.adjusted_change_pct)}
                </span>
              </span>
            ))}
          </span>
        </>
      ) : (
        // 단일 통화 — 합계 라인과 동일한 강조 스타일
        items.map((item) => (
          <span key={item.currency} className="portfolioChangeBreakdownCompactItem">
            <strong>{item.label}</strong> <span style={{ color: "#5f6b82", fontWeight: 400 }}>({formatWeight(item.weight)})</span>{" "}
            <strong className={getSignedClass(item.adjusted_change_pct)} style={{ fontSize: "1.2em" }}>
              {formatSignedPercent(item.adjusted_change_pct)}
            </strong>
          </span>
        ))
      )}
    </span>
  );
}
