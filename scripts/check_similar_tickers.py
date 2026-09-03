"""종목풀 내 종목 간 가격 상관관계 분석 스크립트.

상관계수가 높은 종목을 묶어, 그룹에서 거래량이 가장 많은 종목을 [대장]으로 세우고
나머지를 중복 후보로 보여준다 — 종목풀에서 뺄 종목을 고를 때 쓴다.

사용법:
    python scripts/check_similar_tickers.py kor_kr
    python scripts/check_similar_tickers.py us_stock

기본값은 아래 상수로 조정한다.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import get_latest_trading_day, prepare_price_data
from utils.rankings import get_ticker_type_ma_rules
from utils.settings_loader import get_ticker_type_settings, list_available_ticker_types
from utils.stock_list_io import get_etfs

# ── 조정하는 값 ───────────────────────────────────────────────────────────
# 상관계수 기준 — 이 값 이상인 쌍을 같은 그룹으로 묶는다. 낮출수록 그룹이 커진다.
SIMILAR_THRESHOLD = 0.95
# 「매우 유사」로 붉게 표시하는 기준. 이 사이(기준~강조)는 노랑으로 '검토 필요'다.
STRONG_THRESHOLD = 0.97
# 이 일수보다 짧은 종목은 분석에서 뺀다 — 신규 상장은 표본이 적어 상관계수가 튄다.
MIN_TRADING_DAYS = 20
# ─────────────────────────────────────────────────────────────────────────


class StockStats(NamedTuple):
    ticker: str
    name: str
    return_pct: float
    avg_volume: float


def load_market_data(
    ticker_type: str,
) -> tuple[pd.DataFrame, dict[str, StockStats], int, list[str], list[str]]:
    """종목풀의 전체 종목 종가 데이터와 통계를 로드합니다."""
    ticker_settings = get_ticker_type_settings(ticker_type)
    country_code = str(ticker_settings.get("country_code") or "").strip().lower()
    if not country_code:
        raise ValueError(f"종목풀 '{ticker_type}'의 country_code가 비어 있습니다.")

    ma_rules = get_ticker_type_ma_rules(ticker_type)
    lookback_days = max(int(rule["long_ma_days"]) for rule in ma_rules)

    etfs = get_etfs(ticker_type)
    tickers = sorted({str(etf["ticker"]).strip().upper() for etf in etfs if etf.get("ticker")})
    ticker_names = {
        str(etf["ticker"]).strip().upper(): etf.get("name", str(etf["ticker"])) for etf in etfs if etf.get("ticker")
    }

    end_date = get_latest_trading_day(country_code)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp.now().normalize()
    # 웜업 데이터 포함하여 넉넉히 로드
    start_date = end_date - pd.DateOffset(days=int(lookback_days * 1.5))

    prefetched_map, missing = prepare_price_data(
        tickers=tickers,
        country=country_code,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        warmup_days=30,
        ticker_type=ticker_type,
    )

    close_dict: dict[str, pd.Series] = {}
    stats_dict: dict[str, StockStats] = {}
    missing_tickers: list[str] = []
    short_tickers: list[str] = []

    for ticker in tickers:
        df = prefetched_map.get(ticker)
        if df is None or df.empty:
            missing_tickers.append(f"{ticker} (데이터 없음)")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)

        close_col = "unadjusted_close" if "unadjusted_close" in df.columns else "Close"
        if close_col not in df.columns:
            missing_tickers.append(f"{ticker} (종가 컬럼 없음)")
            continue

        df_cut = df.tail(lookback_days)
        if len(df_cut) < MIN_TRADING_DAYS:
            short_tickers.append(f"{ticker} (데이터 부족: {len(df_cut)}일 < {MIN_TRADING_DAYS}일)")
            continue

        series = df_cut[close_col].astype(float)
        close_dict[ticker] = series

        start_price = series.iloc[0]
        end_price = series.iloc[-1]
        return_pct = ((end_price - start_price) / start_price * 100.0) if start_price > 0 else 0.0

        avg_vol = 0.0
        if "Volume" in df_cut.columns:
            avg_vol = float(df_cut["Volume"].mean())

        stats_dict[ticker] = StockStats(
            ticker=ticker,
            name=ticker_names.get(ticker, ticker),
            return_pct=return_pct,
            avg_volume=avg_vol,
        )

    if not close_dict:
        print(f"[오류] {ticker_type}: 유효한 가격 데이터가 없습니다.")
        sys.exit(1)

    prices_df = pd.DataFrame(close_dict)
    prices_df = prices_df.dropna(how="all")
    return prices_df, stats_dict, lookback_days, missing_tickers, short_tickers


def build_similarity_groups(
    prices_df: pd.DataFrame,
    stats: dict[str, StockStats],
    threshold: float = SIMILAR_THRESHOLD,
) -> list[tuple[str, list[tuple[str, float]]]]:
    """
    종목들을 유사 그룹으로 묶고, 각 그룹의 대장주(수익률 1위)를 선정합니다.
    Returns:
        List of (LeaderTicker, List of (MemberTicker, CorrelationWithLeader))
    """
    returns_df = prices_df.pct_change().dropna()
    corr_matrix = returns_df.corr()
    columns = corr_matrix.columns.tolist()

    # 1. 그래프 생성 (Node: Ticker, Edge: Correlation >= Threshold)
    adj = defaultdict(set)
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            t_i, t_j = columns[i], columns[j]
            if float(corr_matrix.loc[t_i, t_j]) >= threshold:
                adj[t_i].add(t_j)
                adj[t_j].add(t_i)

    # 2. 연결 요소 찾기
    visited = set()
    groups = []

    for ticker in columns:
        if ticker in visited:
            continue

        if ticker not in adj:
            continue

        component = {ticker}
        queue = [ticker]
        visited.add(ticker)

        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)

        if len(component) > 1:
            groups.append(list(component))

    # 3. 각 그룹별 대장주 선정 및 정렬
    sorted_groups = []

    for group in groups:
        group_stats = [stats[ticker] for ticker in group if ticker in stats]
        if not group_stats:
            continue

        group_stats.sort(key=lambda item: item.return_pct, reverse=True)
        leader = group_stats[0]

        members = []
        for member in group_stats[1:]:
            corr = float(corr_matrix.loc[leader.ticker, member.ticker])
            if corr >= threshold:
                members.append((member.ticker, corr))

        if members:
            sorted_groups.append((leader.ticker, members))

    sorted_groups.sort(key=lambda item: stats[item[0]].return_pct, reverse=True)
    return sorted_groups


def print_report(
    ticker_type: str,
    groups: list[tuple[str, list[tuple[str, float]]]],
    stats: dict[str, StockStats],
    threshold: float,
    total_tickers: int,
    lookback_days: int,
) -> None:
    """상관관계 및 수익률 비교 리포트를 출력합니다."""
    print()
    print(f"{'=' * 70}")
    print(f"  📊 상관관계 유사 그룹 분석: {ticker_type.upper()}")
    print(f"  분석 기간: 최근 {lookback_days} 거래일 | 대상 종목: {total_tickers}개")
    print(f"  기준: 상관계수 ≥ {threshold}")
    print(f"{'=' * 70}")

    if not groups:
        print(f"\n  ✅ 상관계수 {threshold} 이상인 유사 그룹이 없습니다.\n")
        return

    print(f"\n  총 {len(groups)}개의 유사 그룹이 발견되었습니다.\n")

    for idx, (leader_ticker, members) in enumerate(groups, 1):
        leader = stats[leader_ticker]
        print(f"  📦 그룹 {idx} (총 {len(members) + 1}종목)")
        print(f"  {'─' * 60}")
        print(f"  🥇 [대장] {leader.ticker} ({leader.name})")
        print(f"      수익률: {leader.return_pct:+.1f}% | 거래량: {leader.avg_volume:,.0f} | (기준)")
        print(f"  {'─' * 60}")

        for member_ticker, corr in members:
            member = stats[member_ticker]

            if abs(corr) >= STRONG_THRESHOLD:
                emoji = "🔴"
                action_msg = "👉 (중복 후보)"
            else:
                emoji = "🟡"
                action_msg = "👉 (검토 필요)"

            print(f"  └─ {emoji} {corr:.4f} {member.ticker} ({member.name})")
            print(f"          수익률: {member.return_pct:+.1f}% | 거래량: {member.avg_volume:,.0f}")
            print(f"          {action_msg}")
        print()

    print(f"{'=' * 70}")
    print("  💡 '유사 그룹'은 서로 상관관계가 높은 종목들의 묶음입니다.")
    print("  각 그룹 내에서 [대장] 종목의 성과가 가장 좋습니다.")
    print(f"  🔴 {STRONG_THRESHOLD} 이상: 매우 유사함 -> 중복 후보")
    print(f"  🟡 {threshold} ~ {STRONG_THRESHOLD}: 유사함 -> 검토 필요")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="종목풀 내 종목 간 가격 상관관계 그룹 분석",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ticker_type",
        choices=list_available_ticker_types(),
        help="분석할 종목풀 ID (예: kor_kr, aus)",
    )
    # 기준값은 인자로 받지 않는다 — 파일 상단 상수 하나만 보게 해서,
    # 실행할 때마다 다른 값이 섞여 결과를 비교할 수 없게 되는 것을 막는다.
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    ticker_type = args.ticker_type.lower()

    print(f"\n[{ticker_type.upper()}] 가격 데이터 로딩 중...")
    prices_df, stats, lookback_days, missing, short = load_market_data(ticker_type)

    if missing:
        print(f"\n[오류] 데이터가 없는 {len(missing)}개 종목이 발견되었습니다:")
        for item in missing:
            print(f"  - {item}")
        print("\n모든 종목의 데이터가 필요합니다. 캐시를 갱신해주세요.")
        print("실행: python scripts/stock_price_cache_updater.py")
        sys.exit(1)

    if short:
        print(f"\n[주의] 데이터 기간이 짧은 {len(short)}개 종목은 분석에서 제외됩니다:")
        for item in short:
            print(f"  - {item}")
        print()

    print(f"[{ticker_type.upper()}] 유사 그룹 분석 중... ({len(prices_df.columns)}개 종목)")
    groups = build_similarity_groups(prices_df, stats)

    print_report(
        ticker_type=ticker_type,
        groups=groups,
        stats=stats,
        threshold=SIMILAR_THRESHOLD,
        total_tickers=len(prices_df.columns),
        lookback_days=lookback_days,
    )


if __name__ == "__main__":
    main()
