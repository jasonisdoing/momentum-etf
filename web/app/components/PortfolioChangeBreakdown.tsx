import type { PortfolioChangeBreakdownItem } from "@/lib/portfolio-change";

type PortfolioChangeFxRate = {
  currency: string;
  change_pct?: number | null;
};

type PortfolioChangeBreakdownProps = {
  items: PortfolioChangeBreakdownItem[];
  fxRates: PortfolioChangeFxRate[];
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

function buildFxRateMap(fxRates: PortfolioChangeFxRate[]): Map<string, PortfolioChangeFxRate> {
  const fxRateByCurrency = new Map<string, PortfolioChangeFxRate>();
  for (const fx of fxRates) {
    const currency = String(fx.currency || "").trim().toUpperCase();
    if (currency) fxRateByCurrency.set(currency, fx);
  }
  return fxRateByCurrency;
}

export function PortfolioChangeBreakdown({
  items,
  fxRates,
  variant,
  emptyText,
}: PortfolioChangeBreakdownProps) {
  if (items.length === 0) {
    return <span>{emptyText}</span>;
  }

  const fxRateByCurrency = buildFxRateMap(fxRates);

  if (variant === "detail") {
    return (
      <div className="portfolioChangeBreakdownDetail" aria-label="포트폴리오 변동 상세">
        {items.map((item) => {
          const fx = fxRateByCurrency.get(item.currency);
          return (
            <div key={item.currency} className="portfolioChangeBreakdownDetailRow">
              <span className="portfolioChangeBreakdownRegion">{item.label}({formatWeight(item.weight)})</span>
              <span className={getSignedClass(item.change_pct)}>{formatSignedPercent(item.change_pct)}</span>
              {fx ? (
                <span className="portfolioChangeBreakdownDetailFx">
                  · 환율{" "}
                  <span className={getSignedClass(fx.change_pct ?? null)}>
                    {formatSignedPercent(fx.change_pct ?? null)}
                  </span>
                </span>
              ) : null}
            </div>
          );
        })}
      </div>
    );
  }
  return (
    <span className="portfolioChangeBreakdownCompact">
      {items.map((item) => {
        const fx = fxRateByCurrency.get(item.currency);
        return (
          <span key={item.currency} className="portfolioChangeBreakdownCompactItem">
            <span>{item.label}({formatWeight(item.weight)})</span>
            <span className={getSignedClass(item.change_pct)}>{formatSignedPercent(item.change_pct)}</span>
            {fx ? (
              <span className="portfolioChangeBreakdownCompactFx">
                · 환율{" "}
                <span className={getSignedClass(fx.change_pct ?? null)}>
                  {formatSignedPercent(fx.change_pct ?? null)}
                </span>
              </span>
            ) : null}
          </span>
        );
      })}
    </span>
  );
}
