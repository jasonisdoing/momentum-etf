"""합성 목표 정수 주수 배분 — 계좌 조회 없이 목표 행·배정액·환율만 받는 핵심 계산."""

from __future__ import annotations

from typing import Any


def sleeve_target_shares(
    targets_by_key: dict[str, list[dict[str, Any]]],
    sleeve_amount_krw: dict[str, float],
    krw_rate: float,
) -> dict[str, int]:
    """고정 슬리브 배정액과 종목별 기준 비중을 합쳐 내림 수량을 만든다."""
    import math

    # 소수 목표를 **티커별로 합산**한다 — 두 슬리브가 같은 종목을 담으면 몫이 더해진다.
    # 슬리브마다 따로 정하면 뒤에 온 슬리브가 앞의 값을 덮어써, 목표비중(합산)과 목표 주수가
    # 어긋난다(kor_test 에서 125.5주 + 111.1주가 113주가 되어 115주를 팔라는 지시가 났다).
    amount_by_ticker: dict[str, float] = {}
    unit_by_ticker: dict[str, float] = {}
    for key, target_rows in targets_by_key.items():
        budget = sleeve_amount_krw.get(key, 0.0)
        if budget <= 0 or krw_rate <= 0:
            continue
        # 호출자가 엔진 선정 종목에 저장 배분 또는 1/N을 적용한 값이다.
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

    return {ticker: math.floor(amount / unit_by_ticker[ticker]) for ticker, amount in amount_by_ticker.items()}


def dated_target_shares(
    targets_by_key: dict[str, list[dict[str, Any]]],
    sleeve_amount_krw: dict[str, float],
    krw_rate: float,
    total_assets_krw: float,
    next_trading_day: str,
    *,
    adjustment_day: str | None = None,
) -> dict[str, dict[str, Any]]:
    """엔진의 날짜별 목표를 같은 정수 배분으로 환산한다. 실제 보유는 입력하지 않는다.

    ``adjustment_day`` 는 이벤트 없는 조정(목표·보유 차이)의 기준일 — 마감 전이면 오늘이라
    다음 거래일보다 앞설 수 있다. 미지정 이벤트의 체결일 기본값은 여전히 다음 거래일이다.
    """
    if not next_trading_day:
        raise ValueError("목표 수량을 배정할 다음 거래일이 없습니다.")
    dates = {adjustment_day or next_trading_day, next_trading_day}
    for targets in targets_by_key.values():
        dates.update(row["fill_date"] for row in targets if row.get("fill_date"))
    schedule = {}
    for day in sorted(dates):
        effective = {}
        weights: dict[str, float] = {}
        previous_amounts: dict[str, float] = {}
        for key, targets in targets_by_key.items():
            rows = []
            for target in targets:
                fill_date = target.get("fill_date") or next_trading_day
                # 해당 날짜의 진입·청산 직전 합산액. 첫 액션 날짜에도 실제 보유로 추정하지 않는다.
                existed_before = not (target.get("plan") == "buy" and fill_date >= day)
                exited_before = bool(target.get("is_exiting")) and fill_date < day
                if existed_before and not exited_before:
                    weight = target.get("drift_pct")
                    if weight is None:
                        raise ValueError(f"목표 비중이 없습니다: {key} 슬리브 {target.get('ticker')}")
                    ticker = target["ticker"]
                    previous_amounts[ticker] = previous_amounts.get(ticker, 0.0) + (
                        sleeve_amount_krw[key] * float(weight) / 100.0 / krw_rate
                    )
                if target.get("plan") == "buy" and fill_date > day:
                    continue
                row = dict(target)
                # 아직 청산일이 오지 않은 보유는 이 날짜의 목표에 남는다.
                row["is_exiting"] = bool(target.get("is_exiting")) and fill_date <= day
                rows.append(row)
                if not row["is_exiting"]:
                    weight = row.get("drift_pct")
                    if weight is None:
                        raise ValueError(f"목표 비중이 없습니다: {key} 슬리브 {row.get('ticker')}")
                    ticker = row["ticker"]
                    weights[ticker] = weights.get(ticker, 0.0) + (
                        sleeve_amount_krw[key] * float(weight) / total_assets_krw if total_assets_krw > 0 else 0.0
                    )
            effective[key] = rows
        schedule[day] = {
            "previous_amounts": previous_amounts,
            "quantities": sleeve_target_shares(effective, sleeve_amount_krw, krw_rate),
            "weights": weights,
            "amounts": {ticker: total_assets_krw * weight / 100.0 / krw_rate for ticker, weight in weights.items()},
        }
    return schedule
