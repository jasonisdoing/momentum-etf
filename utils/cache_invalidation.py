"""종목풀 계산 캐시 무효화 — 종목풀을 고치는 모든 경로가 여기 하나만 부른다.

순위·모멘텀·신고가는 각자 TTL 캐시(5분)를 들고 있는데, 캐시 키는 **저장 설정**이다.
그래서 설정이 아닌 것 — 종목풀에 담긴 종목 목록, 제외(`exclude_from_ranking`)
토글, 종목풀 설정(이평선·슬리피지) — 이 바뀌면 키가 그대로라 TTL 이 끝날 때까지
옛 결과가 나온다. 화면에서는 "고쳤는데 반영이 안 된다" 로 보인다.

호출부가 순위 캐시만 지우고 나머지를 잊는 일이 반복돼서, 지울 대상을 이 함수 하나로
묶었다. 새 캐시가 생기면 여기에만 추가한다.

임포트는 함수 안에서 한다 — 이 모듈을 쓰는 저장 경로들이 무거운 계산 모듈을
로드 시점에 끌어오지 않도록.
"""

from __future__ import annotations


def invalidate_pool_caches(ticker_type: str | None = None) -> None:
    """종목풀 내용·설정이 바뀌었을 때 그 풀을 쓰는 계산 캐시를 모두 지운다.

    Args:
        ticker_type: 바뀐 종목풀. 순위 캐시는 이 풀 것만 지운다. 전략 캐시는 키가
            설정 묶음이라 풀만 뽑아낼 수 없어 통째로 지운다 — 항목이 몇 개뿐이라
            다시 계산해도 손해가 없고, 잘못된 값을 보여주는 쪽이 훨씬 비싸다.
    """
    from utils.holdings_alarm_service import _invalidate_badges_cache
    from utils.rank_service import invalidate_rank_data_cache

    invalidate_rank_data_cache(ticker_type)
    invalidate_strategy_caches()
    # 이동선 이탈 배지는 종목풀의 이평선으로 판정한다 — 풀 설정·종목이 바뀌면 다시 계산해야 한다.
    _invalidate_badges_cache()


def invalidate_strategy_caches() -> None:
    """모멘텀·신고가의 '현재 상태' 캐시를 지운다.

    전략 설정을 저장할 때도 부른다 — 설정을 바꿨다 되돌리면 옛 키에 그대로 걸려서
    그 사이 달라진 종목 목록이 반영되지 않은 결과가 다시 나오기 때문이다.
    """
    from utils.momentum_service import _PICKS_CACHE
    from utils.new_high_backtest import _POSITIONS_CACHE

    _PICKS_CACHE.invalidate()
    _POSITIONS_CACHE.invalidate()
