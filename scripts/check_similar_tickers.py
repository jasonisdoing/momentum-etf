"""종목풀 내 종목 간 가격 상관관계 분석 스크립트.

사용법:
    python scripts/check_similar_tickers.py kor
    python scripts/check_similar_tickers.py tax --threshold 0.90
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import get_latest_trading_day, prepare_price_data
from utils.pool_registry import PoolSettingsError, get_pool_dir, list_available_pools
from utils.stock_list_io import get_etfs


class StockStats(NamedTuple):
    ticker: str
    name: str
    return_pct: float
    avg_volume: float


def load_market_data(
    pool_id: str,
) -> tuple[pd.DataFrame, dict[str, StockStats], int, int, list[str], list[str]]:
    """종목풀의 전체 종목 종가 데이터와 통계를 로드합니다."""
    country_code, ma_month = load_pool_rank_settings(pool_id)

    # 1개월 = 20거래일 기준
    lookback_days = int(ma_month * 20)

    etfs = get_etfs(pool_id)
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
        warmup_days=30,  # 웜업 충분히 확보
        account_id=pool_id,
    )

    close_dict: dict[str, pd.Series] = {}
    stats_dict: dict[str, StockStats] = {}
    missing_tickers: list[str] = []  # 데이터 아예 없음 (캐시 문제)
    short_tickers: list[str] = []  # 데이터 짧음 (신규 상장 등)

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
        if len(df_cut) < 20:
            short_tickers.append(f"{ticker} (데이터 부족: {len(df_cut)}일 < 20일)")
            continue

        series = df_cut[close_col].astype(float)
        label = ticker
        close_dict[label] = series

        start_price = series.iloc[0]
        end_price = series.iloc[-1]

        if start_price > 0:
            return_pct = (end_price - start_price) / start_price * 100
        else:
            return_pct = 0.0

        avg_vol = 0.0
        if "Volume" in df_cut.columns:
            avg_vol = df_cut["Volume"].mean()

        stats_dict[ticker] = StockStats(
            ticker=ticker,
            name=ticker_names.get(ticker, ticker),
            return_pct=return_pct,
            avg_volume=avg_vol,
        )

    if not close_dict:
        print(f"[오류] {pool_id}: 유효한 가격 데이터가 없습니다.")
        sys.exit(1)

    prices_df = pd.DataFrame(close_dict)
    prices_df = prices_df.dropna(how="all")
    return prices_df, stats_dict, lookback_days, ma_month, missing_tickers, short_tickers


def load_pool_rank_settings(pool_id: str) -> tuple[str, int]:
    """종목풀 config.json에서 분석에 필요한 rank 설정을 읽습니다."""
    pool_dir = get_pool_dir(pool_id)
    config_path = pool_dir / "config.json"
    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PoolSettingsError(f"종목풀 설정을 읽을 수 없습니다: {config_path}") from exc

    if not isinstance(config_data, dict):
        raise PoolSettingsError(f"종목풀 설정 루트는 객체여야 합니다: {config_path}")

    rank_cfg = config_data.get("rank")
    if not isinstance(rank_cfg, dict):
        raise PoolSettingsError(f"종목풀 '{pool_id}'의 rank 설정이 없습니다.")

    country_code = str(rank_cfg.get("country") or "").strip().lower()
    if not country_code:
        raise PoolSettingsError(f"종목풀 '{pool_id}'의 rank.country가 비어 있습니다.")

    months_raw = rank_cfg.get("months")
    try:
        ma_month = int(months_raw)
    except (TypeError, ValueError) as exc:
        raise PoolSettingsError(f"종목풀 '{pool_id}'의 rank.months가 올바른 정수가 아닙니다: {months_raw}") from exc

    if ma_month <= 0:
        raise PoolSettingsError(f"종목풀 '{pool_id}'의 rank.months는 1 이상이어야 합니다: {ma_month}")

    return country_code, ma_month


def build_similarity_groups(
    prices_df: pd.DataFrame,
    stats: dict[str, StockStats],
    threshold: float = 0.95,
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
            if abs(corr_matrix.loc[t_i, t_j]) >= threshold:
                adj[t_i].add(t_j)
                adj[t_j].add(t_i)

    # 2. 연결 요소(Connected Components) 찾기 - BFS/DFS
    visited = set()
    groups = []

    for t in columns:
        if t in visited:
            continue

        # 새로운 그룹 발견 (혹은 독립 노드)
        if t not in adj:  # 연결된 간선이 없는 독립 노드는 제외 (그룹 아님)
            continue

        # BFS로 그룹 탐색
        component = {t}
        queue = [t]
        visited.add(t)

        while queue:
            curr = queue.pop(0)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)

        if len(component) > 1:
            groups.append(list(component))

    # 3. 각 그룹별 대장주 선정 및 정렬
    sorted_groups = []

    for group in groups:
        # 수익률 순으로 정렬 (내림차순) --> 첫 번째가 대장주
        group_stats = [stats[t] for t in group if t in stats]
        if not group_stats:
            continue

        group_stats.sort(key=lambda s: s.return_pct, reverse=True)
        leader = group_stats[0]

        # 멤버 리스트 생성: (티커, 대장주와의 상관계수)
        members = []
        for member in group_stats[1:]:  # 대장주 제외
            corr = corr_matrix.loc[leader.ticker, member.ticker]
            # 대장주와의 상관계수가 임계값 이상인 경우만 포함
            if abs(corr) >= threshold:
                members.append((member.ticker, corr))

        if members:
            sorted_groups.append((leader.ticker, members))

    # 그룹 간 정렬 (대장주 수익률 높은 순)
    sorted_groups.sort(key=lambda x: stats[x[0]].return_pct, reverse=True)

    return sorted_groups


def print_report(
    pool_id: str,
    groups: list[tuple[str, list[tuple[str, float]]]],
    stats: dict[str, StockStats],
    threshold: float,
    total_tickers: int,
    lookback_days: int,
    ma_month: int,
) -> None:
    """상관관계 및 수익률 비교 리포트를 출력합니다."""
    print()
    print(f"{'=' * 70}")
    print(f"  📊 상관관계 유사 그룹 분석: {pool_id.upper()}")
    print(f"  분석 기간: 최근 {ma_month}개월 ({lookback_days} 거래일) | 대상 종목: {total_tickers}개")
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

            if abs(corr) >= 0.95:
                emoji = "🔴"
                action_msg = "👉 (제거 추천)"
            else:
                emoji = "🟡"
                action_msg = "👉 (제거 고려)"

            print(f"  └─ {emoji} {corr:.4f} {member.ticker} ({member.name})")
            print(f"          수익률: {member.return_pct:+.1f}% | 거래량: {member.avg_volume:,.0f}")
            print(f"          {action_msg}")
        print()

    print(f"{'=' * 70}")
    print("  💡 '유사 그룹'은 서로 상관관계가 높은 종목들의 묶음입니다.")
    print("  각 그룹 내에서 [대장] 종목의 성과가 가장 좋습니다.")
    print("  🔴 0.95 이상: 매우 유사함 -> 교체 강력 추천")
    print("  🟡 0.90 ~ 0.95: 유사함 -> 교체 고려")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="종목풀 내 종목 간 가격 상관관계 그룹 분석",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pool",
        choices=list_available_pools(),
        help="분석할 종목풀 ID",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="상관계수 기준값",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pool_id = args.pool.lower()
    threshold = args.threshold

    print(f"\n[{pool_id.upper()}] 가격 데이터 로딩 중...")
    prices_df, stats, lookback_days, ma_month, missing, short = load_market_data(pool_id)

    if missing:
        print(f"\n[오류] 데이터가 없는 {len(missing)}개 종목이 발견되었습니다:")
        for s in missing:
            print(f"  - {s}")
        print("\n모든 종목의 데이터가 필요합니다. 캐시를 갱신해주세요.")
        print(f"실행: python scripts/update_price_cache.py {pool_id}")
        sys.exit(1)

    if short:
        print(f"\n[주의] 데이터 기간이 짧은 {len(short)}개 종목은 분석에서 제외됩니다:")
        for s in short:
            print(f"  - {s}")
        print()

    print(f"[{pool_id.upper()}] 유사 그룹 분석 중... ({len(prices_df.columns)}개 종목)")
    groups = build_similarity_groups(prices_df, stats, threshold=threshold)

    print_report(
        pool_id=pool_id,
        groups=groups,
        stats=stats,
        threshold=threshold,
        total_tickers=len(prices_df.columns),
        lookback_days=lookback_days,
        ma_month=ma_month,
    )


if __name__ == "__main__":
    main()
