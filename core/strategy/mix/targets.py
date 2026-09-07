"""합성 목표 정수 주수 배분 — 계좌 조회 없이 목표 행·배정액·환율만 받는 핵심 계산."""

from __future__ import annotations

from typing import Any


def sleeve_target_shares(
    targets_by_key: dict[str, list[dict[str, Any]]],
    sleeve_amount_krw: dict[str, float],
    krw_rate: float,
) -> dict[str, int]:
    """백테스트 목표 비중을 계좌 금액으로 환산하고 정수 주수로 배분한다.

    실제 보유 수량은 입력하지 않는다. 계좌 규모를 바꿔 백테스트를 다시 실행하면
    비싼 종목의 진입 가능 여부가 바뀌므로, 엔진 결과를 비례 환산만 한다.
    """
    from math import fsum

    from utils.share_allocation import ShareTarget, allocate_integer_shares

    # 소수 목표를 **티커별로 합산**한다 — 두 슬리브가 같은 종목을 담으면 몫이 더해진다.
    # 슬리브마다 따로 정하면 뒤에 온 슬리브가 앞의 값을 덮어써, 목표비중(합산)과 목표 주수가
    # 어긋난다(kor_test 에서 125.5주 + 111.1주가 113주가 되어 115주를 팔라는 지시가 났다).
    amount_by_ticker: dict[str, float] = {}
    unit_by_ticker: dict[str, float] = {}
    for key, target_rows in targets_by_key.items():
        budget = sleeve_amount_krw.get(key, 0.0)
        if budget <= 0 or krw_rate <= 0:
            continue
        # 슬리브 안 비중 — 이미 산 종목은 흘러간 실제 비중, 진입 예정은 엔진이 배정한 비중.
        # 모든 목표 행은 엔진(또는 엔진과 같은 규칙의 장중 공통 계산)이 비중을 실어 보낸다 —
        # 비면 데이터 결함이므로 임의 값으로 메우지 않고 에러로 알린다.
        for row in target_rows:
            price = row.get("price")
            if not price or row.get("is_exiting"):
                continue
            weight = row.get("drift_pct")
            if weight is None:
                raise ValueError(f"목표 비중이 없습니다: {key} 슬리브 {row.get('ticker')}")
            weight = float(weight)
            if weight <= 0:
                continue
            ticker = str(row["ticker"]).strip()
            unit_by_ticker[ticker] = float(price) * krw_rate
            amount_by_ticker[ticker] = amount_by_ticker.get(ticker, 0.0) + budget * weight / 100.0

    return allocate_integer_shares(
        [
            ShareTarget(key=ticker, target_amount=amount, price=unit_by_ticker[ticker])
            for ticker, amount in amount_by_ticker.items()
        ],
        # 합성 유보 현금은 슬리브 배정액에서 이미 제외된다. 전략 내부 현금도 보호하려면
        # 주식 목표 금액까지만 쓸 수 있다. 추가 주수에는 내림으로 생긴 단주 잔여만 쓴다.
        # 동시에 전체 슬리브 배정액을 넘지 않도록 계좌 예산 한도도 유지한다.
        budget=min(fsum(amount_by_ticker.values()), fsum(sleeve_amount_krw.values())),
    )
