"use client";

import { useEffect, useState } from "react";
import { PageFrame } from "./components/PageFrame";

type HoldingsRow = {
  account_name: string;
  currency: string;
  ticker: string;
  name: string;
  current_price: string;
  current_price_num: number;
  pnl_krw: number;
  pnl_krw_num: number;
  return_pct: number;
  daily_change_pct: number | null;
  valuation_krw: number;
  bucket_id: number;
  bucket: string;
};

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
          // 티커가 있는 실질적인 종목만 표시 (예: 현금성 항목 제외, 단 IS는 포함)
          const rows = (data.rows || []).filter((r: any) => r.ticker);
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

  return (
    <PageFrame title="Home">
      <div className="container-fluid py-3">
        <div className="d-flex justify-content-between align-items-center mb-3">
          <div className="text-secondary small">
            총 {holdings.length}개 종목
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
        ) : holdings.length === 0 ? (
          <div className="card shadow-sm border-0 rounded-4">
            <div className="card-body py-5 text-center">
              <div className="empty">
                <p className="empty-title">보유 종목이 없습니다.</p>
                <p className="empty-subtitle text-secondary">자산 관리 메뉴에서 종목을 추가해 보세요.</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="row g-2 g-md-3">
            {holdings.map((h, i) => (
              <div key={`${h.ticker}-${i}`} className="col-6 col-lg-3">
                <HoldingCard row={h} viewMode={viewMode} />
              </div>
            ))}
          </div>
        )}
      </div>

      <style jsx global>{`
        .appContent {
          background-color: #f6f8fb !important;
        }
      `}</style>
    </PageFrame>
  );
}

function HoldingCard({ row, viewMode }: { row: HoldingsRow; viewMode: ViewMode }) {
  const displayPct = viewMode === "price" ? row.daily_change_pct : row.return_pct;
  const isPositive = (displayPct ?? 0) >= 0;
  const colorClass = isPositive ? "text-danger" : "text-primary";
  const sign = (displayPct ?? 0) > 0 ? "+" : "";

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
        <div className="stock-name" title={row.name}>{row.name}</div>
      </div>
      <div className="stock-stats text-end">
        <div className="stock-price">
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
