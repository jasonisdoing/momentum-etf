"""계좌 내 종목 간 가격 상관관계 분석 스크립트.

사용법:
    python scripts/analyze_correlation.py kor_kr
    python scripts/analyze_correlation.py us --threshold 0.90 --days 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.account_registry import get_account_settings, list_available_accounts
from utils.data_loader import get_latest_trading_day, prepare_price_data
from utils.stock_list_io import get_etfs


def load_close_prices(
    account_id: str,
    lookback_days: int = 120,
) -> pd.DataFrame:
    """계좌의 전체 종목 종가 데이터를 DataFrame으로 로드합니다."""
    account_settings = get_account_settings(account_id)
    country_code = (account_settings.get("country_code") or account_id).strip().lower()

    etfs = get_etfs(account_id)
    tickers = sorted({etf["ticker"] for etf in etfs if etf.get("ticker")})
    ticker_names = {etf["ticker"]: etf.get("name", etf["ticker"]) for etf in etfs if etf.get("ticker")}

    end_date = get_latest_trading_day(country_code)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.DateOffset(days=int(lookback_days * 1.5))

    prefetched_map, missing = prepare_price_data(
        tickers=tickers,
        country=country_code,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        warmup_days=0,
        account_id=account_id,
    )

    # 종가 추출
    close_dict: dict[str, pd.Series] = {}
    for ticker in tickers:
        df = prefetched_map.get(ticker)
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        col = "unadjusted_close" if "unadjusted_close" in df.columns else "Close"
        if col in df.columns:
            series = df[col].astype(float).tail(lookback_days)
            if len(series) >= 20:  # 최소 20일 데이터
                label = f"{ticker} ({ticker_names.get(ticker, '')})"
                close_dict[label] = series

    if not close_dict:
        print(f"[오류] {account_id}: 유효한 가격 데이터가 없습니다.")
        sys.exit(1)

    prices_df = pd.DataFrame(close_dict)
    prices_df = prices_df.dropna(how="all")
    return prices_df


def find_correlated_pairs(
    prices_df: pd.DataFrame,
    threshold: float = 0.95,
) -> list[tuple[str, str, float]]:
    """상관계수가 threshold 이상인 종목 쌍을 찾습니다."""
    # 일별 수익률 기반 상관관계 (가격 수준 차이 제거)
    returns_df = prices_df.pct_change().dropna()
    corr_matrix = returns_df.corr()

    pairs: list[tuple[str, str, float]] = []
    columns = corr_matrix.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                pairs.append((columns[i], columns[j], round(corr_val, 4)))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def print_report(
    account_id: str,
    pairs: list[tuple[str, str, float]],
    threshold: float,
    total_tickers: int,
    lookback_days: int,
) -> None:
    """상관관계 분석 결과를 출력합니다."""
    print()
    print(f"{'=' * 70}")
    print(f"  📊 상관관계 분석: {account_id.upper()}")
    print(f"  분석 기간: 최근 {lookback_days} 거래일 | 종목 수: {total_tickers}개")
    print(f"  기준: 상관계수 ≥ {threshold}")
    print(f"{'=' * 70}")

    if not pairs:
        print(f"\n  ✅ 상관계수 {threshold} 이상인 종목 쌍이 없습니다.\n")
        return

    print(f"\n  ⚠️  높은 상관관계 종목 쌍: {len(pairs)}건\n")
    print(f"  {'상관계수':>8}  {'종목 A':<30}  {'종목 B':<30}")
    print(f"  {'─' * 8}  {'─' * 30}  {'─' * 30}")

    for ticker_a, ticker_b, corr_val in pairs:
        emoji = "🔴" if abs(corr_val) >= 0.98 else "🟡"
        print(f"  {emoji} {corr_val:>6.4f}  {ticker_a:<30}  {ticker_b:<30}")

    # 중복 제거 후보 표시
    print(f"\n  {'─' * 70}")
    redundant = set()
    for _, ticker_b, _ in pairs:
        redundant.add(ticker_b)

    if redundant:
        print("\n  💡 제거 후보 (각 쌍에서 후순위 종목):")
        for t in sorted(redundant):
            print(f"     - {t}")

    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="계좌 내 종목 간 가격 상관관계 분석",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "account",
        choices=list_available_accounts(),
        help="분석할 계좌 ID",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="상관계수 기준값 (이 이상이면 높은 상관관계로 판단)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=120,
        help="분석할 최근 거래일 수",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    account_id = args.account.lower()
    threshold = args.threshold
    lookback_days = args.days

    print(f"\n[{account_id.upper()}] 가격 데이터 로딩 중...")
    prices_df = load_close_prices(account_id, lookback_days=lookback_days)

    print(f"[{account_id.upper()}] 상관관계 계산 중... ({len(prices_df.columns)}개 종목)")
    pairs = find_correlated_pairs(prices_df, threshold=threshold)

    print_report(
        account_id=account_id,
        pairs=pairs,
        threshold=threshold,
        total_tickers=len(prices_df.columns),
        lookback_days=lookback_days,
    )


if __name__ == "__main__":
    main()
