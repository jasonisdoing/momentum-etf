"use client";

import { MarketTrendChart } from "../market-trend/MarketTrendChart";

// 홈 메인 — 시장지수 추세 4개 차트(2×2). 각 MarketTrendChart 는 ticker(yf 심볼)로 자체 history 를 조회한다.
// 배열: 좌상 코스피 · 우상 코스닥 · 좌하 나스닥100 · 우하 필라델피아 반도체.
const HOME_TREND_CHARTS: { ticker: string; name: string }[] = [
  { ticker: "^KS11", name: "코스피" },
  { ticker: "^KQ11", name: "코스닥" },
  { ticker: "^NDX", name: "나스닥 100" },
  { ticker: "^SOX", name: "필라델피아 반도체" },
];

export function HomeTrendCharts({ maDays, maType }: { maDays: number; maType: string }) {
  return (
    <div className="homeTrendGrid">
      {HOME_TREND_CHARTS.map((chart) => (
        <div key={chart.ticker} className="card appCard homeTrendCell">
          {/* key 를 차트 컴포넌트에도 줘서 티커별 인스턴스를 완전히 분리한다(상태 혼선 차단). */}
          <MarketTrendChart
            key={chart.ticker}
            ticker={chart.ticker}
            name={chart.name}
            maType={maType}
            maDays={maDays}
            compact
          />
        </div>
      ))}
      <style jsx global>{`
        .homeTrendGrid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 16px;
        }
        .homeTrendCell {
          min-width: 0;
          height: 440px;
          overflow: hidden;
          padding: 0;
          background: #fff;
          border: 1px solid rgba(148, 163, 184, 0.16);
          border-radius: 16px;
          box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 10px 28px rgba(15, 23, 42, 0.05);
          transition: box-shadow 180ms ease, border-color 180ms ease;
        }
        .homeTrendCell:hover {
          border-color: rgba(148, 163, 184, 0.32);
          box-shadow: 0 2px 4px rgba(15, 23, 42, 0.06), 0 16px 40px rgba(15, 23, 42, 0.09);
        }
        @media (max-width: 900px) {
          .homeTrendGrid {
            grid-template-columns: minmax(0, 1fr);
          }
        }
      `}</style>
    </div>
  );
}
