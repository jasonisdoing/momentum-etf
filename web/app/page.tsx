"use client";

import { useEffect, useState } from "react";
import { PageFrame } from "./components/PageFrame";
import { AppLoadingState } from "./components/AppLoadingState";


type HoldingsRow = {
  account_name: string;
  currency: string;
  ticker: string;
  name: string;
  quantity: number;
  current_price: string;
  current_price_num: number;
  pnl_krw: number;
  pnl_krw_num: number;
  return_pct: number;
  daily_change_pct: number | null;
  buy_amount_krw: number;
  valuation_krw: number;
  bucket_id: number;
  bucket: string;
  memo: string;
};

type AggregatedHoldingRow = HoldingsRow;

type ViewMode = "price" | "valuation";

export default function HomePage() {
  const [holdings, setHoldings] = useState<HoldingsRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>("price");

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch("/api/assets");
        if (res.ok) {
          const data = await res.json();
          // 티커가 있고 수량이 0보다 큰 실질적인 종목만 표시
          const rows = (data.rows || []).filter((r: any) => r.ticker && r.quantity > 0);
          setHoldings(rows);
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const accountNames = [...new Set(holdings.map((h) => h.account_name))];

  const aggregatedHoldings = Array.from(
    holdings.reduce((acc, row) => {
      const key = `${row.currency}:${row.ticker}`;
      const existing = acc.get(key);

      if (!existing) {
        acc.set(key, { ...row });
        return acc;
      }

      const nextBuyAmount = existing.buy_amount_krw + row.buy_amount_krw;
      const nextValuation = existing.valuation_krw + row.valuation_krw;
      const nextPnl = existing.pnl_krw + row.pnl_krw;
      const weightedDailyChange =
        existing.daily_change_pct !== null && row.daily_change_pct !== null
          ? (
              ((existing.daily_change_pct * existing.valuation_krw) + (row.daily_change_pct * row.valuation_krw)) /
              Math.max(nextValuation, 1)
            )
          : (existing.daily_change_pct ?? row.daily_change_pct);

      // 동일 티커가 여러 계좌에 있으면 현재가는 하나만 유지하고 평가금/손익만 합산한다.
      acc.set(key, {
        ...existing,
        quantity: existing.quantity + row.quantity,
        buy_amount_krw: nextBuyAmount,
        valuation_krw: nextValuation,
        pnl_krw: nextPnl,
        pnl_krw_num: nextPnl,
        return_pct: nextBuyAmount > 0 ? Number(((nextPnl / nextBuyAmount) * 100).toFixed(2)) : 0,
        daily_change_pct:
          weightedDailyChange === null ? null : Number(weightedDailyChange.toFixed(2)),
        bucket_id: row.valuation_krw > existing.valuation_krw ? row.bucket_id : existing.bucket_id,
        bucket: row.valuation_krw > existing.valuation_krw ? row.bucket : existing.bucket,
        memo: existing.memo || row.memo,
        account_name: accountNames.join(", "),
      });
      return acc;
    }, new Map<string, AggregatedHoldingRow>()).values(),
  );

  const risingHoldings = aggregatedHoldings
    .filter((h) => {
      const pct = viewMode === "price" ? (h.daily_change_pct ?? 0) : h.return_pct;
      return pct >= 0;
    })
    .sort((a, b) => b.valuation_krw - a.valuation_krw);

  const fallingHoldings = aggregatedHoldings
    .filter((h) => {
      const pct = viewMode === "price" ? (h.daily_change_pct ?? 0) : h.return_pct;
      return pct < 0;
    })
    .sort((a, b) => b.valuation_krw - a.valuation_krw);

  const bucketOrder = [1, 2, 3, 4];
  const holdingsByBucket = bucketOrder
    .map((bucketId) => {
      const bucketRising = risingHoldings.filter((h) => h.bucket_id === bucketId);
      const bucketFalling = fallingHoldings.filter((h) => h.bucket_id === bucketId);
      const bucketName = aggregatedHoldings.find((h) => h.bucket_id === bucketId)?.bucket ?? `${bucketId}. 기타`;
      return {
        bucketId,
        bucketName,
        rising: bucketRising,
        falling: bucketFalling,
      };
    })
    .filter((bucket) => bucket.rising.length > 0 || bucket.falling.length > 0);

  if (loading) {
    return (
      <PageFrame title="Home">
        <div className="container-fluid py-5">
          <AppLoadingState label="보유 종목 데이터를 불러오는 중..." />
        </div>
      </PageFrame>
    );
  }

  return (
    <PageFrame title="Home">
      <div className="container-fluid pt-1 pb-2">
        <div className="d-flex justify-content-between align-items-center mb-2 flex-wrap gap-2">
          <div className="d-flex align-items-center flex-wrap gap-3">
            <div className="text-secondary small fw-bold">
              총 {accountNames.length}개 계좌, {aggregatedHoldings.length}개 종목
            </div>
            <div className="d-flex align-items-center gap-2">
              {[
                { id: 1, name: "모멘텀", color: "#1e6bb8", sub: "#e7f1ff" },
                { id: 2, name: "시장지수", color: "#2fb344", sub: "#eaf8ed" },
                { id: 3, name: "미국배당", color: "#d63384", sub: "#fbebf3" },
                { id: 4, name: "대체헷지", color: "#f76707", sub: "#fef0e7" },
              ].map((b) => (
                <div 
                  key={b.id} 
                  className="bucket-legend-badge" 
                  style={{ backgroundColor: b.sub, color: b.color }}
                >
                  {b.id}. {b.name}
                </div>
              ))}
            </div>
          </div>
          <div className="btn-group shadow-sm">
            <button
              type="button"
              className={`btn btn-sm ${viewMode === "price" ? "btn-primary active" : "bg-white text-dark border"}`}
              onClick={() => setViewMode("price")}
            >
              현재가
            </button>
            <button
              type="button"
              className={`btn btn-sm ${viewMode === "valuation" ? "btn-primary active" : "bg-white text-dark border"}`}
              onClick={() => setViewMode("valuation")}
            >
              평가금
            </button>
          </div>
        </div>

        {loading ? (
          <div className="d-flex justify-content-center py-5 mt-5">
            <div className="spinner-border text-primary" role="status"></div>
          </div>
        ) : aggregatedHoldings.length === 0 ? (
          <div className="card shadow-sm border-0 rounded-4">
            <div className="card-body py-5 text-center">
              <div className="empty">
                <p className="empty-title">보유 종목이 없습니다.</p>
                <p className="empty-subtitle text-secondary">자산 관리 메뉴에서 종목을 추가해 보세요.</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="row g-3">
            <div className="col-12 col-lg-6">
              <div className="d-flex align-items-center mb-3">
                <div className="badge bg-danger-lt p-1 me-2" style={{ width: "12px", height: "12px", borderRadius: "50%" }}></div>
                <h3 className="h4 fw-bold mb-0 text-danger">상승 ({risingHoldings.length})</h3>
              </div>
              <div className="account-groups">
                {holdingsByBucket.map((bucket) => (
                  <div key={`rising-${bucket.bucketId}`} className="account-section mb-4">
                    <h2 className="account-header h4 mb-3 d-flex align-items-center">
                      <span className="account-dot me-2" style={{ backgroundColor: getTheme(bucket.bucketId).main }}></span>
                      {bucket.bucketName}
                    </h2>
                    <div className="row g-2">
                      {bucket.rising.length > 0 ? (
                        bucket.rising.map((h) => (
                          <div key={h.ticker} className="col-12 col-md-6">
                            <HoldingCard row={h} viewMode={viewMode} />
                          </div>
                        ))
                      ) : (
                        <div className="col-12"><div className="text-secondary small py-2 px-1">해당하는 종목이 없습니다.</div></div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="col-12 col-lg-6">
              <div className="d-flex align-items-center mb-3">
                <div className="badge bg-primary-lt p-1 me-2" style={{ width: "12px", height: "12px", borderRadius: "50%" }}></div>
                <h3 className="h4 fw-bold mb-0 text-primary">하락 중 ({fallingHoldings.length})</h3>
              </div>
              <div className="account-groups">
                {holdingsByBucket.map((bucket) => (
                  <div key={`falling-${bucket.bucketId}`} className="account-section mb-4">
                    <h2 className="account-header h4 mb-3 d-flex align-items-center">
                      <span className="account-dot me-2" style={{ backgroundColor: getTheme(bucket.bucketId).main }}></span>
                      {bucket.bucketName}
                    </h2>
                    <div className="row g-2">
                      {bucket.falling.length > 0 ? (
                        bucket.falling.map((h) => (
                          <div key={h.ticker} className="col-12 col-md-6">
                            <HoldingCard row={h} viewMode={viewMode} />
                          </div>
                        ))
                      ) : (
                        <div className="col-12"><div className="text-secondary small py-2 px-1">하락 중인 종목이 없습니다.</div></div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <style jsx global>{`
        .appContent {
          background-color: #f6f8fb !important;
        }
      `}</style>

      <style jsx>{`
        .account-header {
          color: #1d273b;
          font-weight: 800;
          letter-spacing: -0.02em;
          font-size: 1.1rem;
        }
        .account-dot {
          display: inline-block;
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }
        .bucket-legend-badge {
          display: inline-flex;
          align-items: center;
          padding: 2px 8px;
          border-radius: 999px;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: -0.01em;
        }
      `}</style>
    </PageFrame>
  );
}

function HoldingCard({ row, viewMode }: { row: HoldingsRow; viewMode: ViewMode }) {
  const displayPct = viewMode === "price" ? row.daily_change_pct : row.return_pct;
  const pct = displayPct ?? 0;
  const colorClass = pct > 0 ? "text-danger" : pct < 0 ? "text-primary" : "text-dark";
  const sign = pct > 0 ? "+" : "";

  // 호주 종목은 티커로, 한국 종목은 이름으로 표시
  const displayTitle = (row.currency === "AUD" && row.ticker !== "IS") ? row.ticker : row.name;

  const formatPrice = (val: number, currency: string) => {
    if (currency === "AUD") {
      return `A$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }
    if (currency === "USD") {
      return `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }
    return `${Math.floor(val).toLocaleString()}원`;
  };

  const formatValuation = (val: number) => {
    if (val >= 100000000) {
      return `${(val / 100000000).toFixed(1)}억원`;
    }
    if (val >= 10000) {
      return `${Math.floor(val / 10000).toLocaleString()}만원`;
    }
    return `${val.toLocaleString()}원`;
  };

  const theme = getTheme(row.bucket_id);

  return (
    <div className="stock-card">
      <div className="stock-icon-area">
        <div className="stock-icon" style={{ backgroundColor: theme.main }}>
          <div className="stock-icon-inner">
            <div className="stock-icon-dot" style={{ backgroundColor: theme.main }}></div>
          </div>
        </div>
      </div>
      <div className="stock-content">
        <div className="stock-name" title={row.name}>{displayTitle}</div>
      </div>
      <div className="stock-stats text-end">
        <div className={`stock-price ${colorClass}`} style={{ fontWeight: 700 }}>
          {viewMode === "price" ? formatPrice(row.current_price_num, row.currency) : formatValuation(row.valuation_krw)}
        </div>
        <div className={`stock-change ${colorClass}`}>
          {displayPct !== null ? `${sign}${displayPct.toFixed(2)}%` : "-"}
        </div>
      </div>

      <style jsx>{`
        .stock-card {
          background-color: ${theme.sub};
          border-radius: 12px;
          padding: 12px 16px;
          display: flex;
          align-items: center;
          gap: 12px;
          height: 100%;
          border: 1px solid rgba(0,0,0,0.02);
          transition: all 0.2s ease-in-out;
        }
        .stock-card:hover {
          filter: brightness(0.96);
          transform: translateY(-1px);
        }
        .stock-icon-area {
          flex-shrink: 0;
        }
        .stock-icon {
          width: 40px;
          height: 40px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .stock-icon-inner {
          width: 24px;
          height: 24px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          background-color: white !important;
        }
        .stock-icon-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
        }
        .stock-content {
          flex-grow: 1;
          min-width: 0;
        }
        .stock-name {
          font-size: 0.95rem;
          font-weight: 700;
          color: #1d273b;
          line-height: 1.25;
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
          word-break: keep-all;
        }
        .stock-stats {
          flex-shrink: 0;
        }
        .stock-price {
          font-size: 1.1rem;
          font-weight: 800;
          color: #1d273b;
          letter-spacing: -0.02em;
          line-height: 1.1;
        }
        .stock-change {
          font-size: 0.85rem;
          font-weight: 600;
          margin-top: 2px;
          line-height: 1.1;
        }

        @media (max-width: 576px) {
          .stock-card {
            padding: 10px 12px;
            gap: 8px;
          }
          .stock-icon {
            width: 32px;
            height: 32px;
          }
          .stock-icon-inner {
            width: 18px;
            height: 18px;
          }
          .stock-icon-dot {
            width: 8px;
            height: 8px;
          }
          .stock-name {
            font-size: 0.82rem;
          }
          .stock-price {
            font-size: 0.95rem;
          }
          .stock-change {
            font-size: 0.75rem;
          }
        }
      `}</style>
    </div>
  );
}

function getTheme(bucketId: number) {
  switch (bucketId) {
    case 1: return { main: "#1e6bb8", sub: "#e7f1ff" };
    case 2: return { main: "#2fb344", sub: "#eaf8ed" };
    case 3: return { main: "#d63384", sub: "#fbebf3" };
    case 4: return { main: "#f76707", sub: "#fef0e7" };
    default: return { main: "#616876", sub: "#f1f3f5" };
  }
}
