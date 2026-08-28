"""티커 → 종목풀 메타 해석 — 화면·서비스 공용.

입력 티커가 **어느 종목풀의 어느 종목인지** 찾아 표준 표기·이름·국가를 돌려준다.
종목풀에 없으면 에러다 — 형식(6자리 숫자면 한국, 알파벳이면 미국)으로 추측하지 않는다.
그렇게 찍으면 호주 종목이 미국으로 넘어가 엉뚱한 시세를 붙인다.

`/assets` 는 예전에 종목풀을 **하나씩 돌며 원천 조회**(`validate_stock_candidate`)를 해서
티커 하나 확인에 19초가 걸렸다 — 한국 종목인데 미국·호주 풀 차례에 yfinance 로 찾다가
404 를 반복했다. 여기 있는 목록 대조는 외부 호출이 없어 2초 안에 끝난다.
"""

from __future__ import annotations

from utils.asx_ticker import ensure_asx_prefix
from utils.cash_model import currency_for_country
from utils.stock_list_io import get_etfs
from utils.ticker_registry import load_ticker_type_configs

# 시장 명시 접두사 → 국가 코드. 값은 종목풀 설정의 `country_code` 와 같은 어휘다.
_PREFIX_COUNTRIES: dict[str, str] = {"ASX:": "au", "US:": "us", "KOR:": "kor"}


def resolve_ticker_meta(
    ticker: str,
    allowed_ticker_types: set[str] | None = None,
    account_id: str | None = None,
) -> dict[str, object]:
    ticker_key = str(ticker or "").strip().upper()
    if not ticker_key:
        raise ValueError("ticker 파라미터가 필요합니다.")

    # 계좌에 연결된 종목풀로 검색 범위를 제한할 때 사용 (None 이면 전체 검색).
    def _allowed(ticker_type: str) -> bool:
        return allowed_ticker_types is None or ticker_type in allowed_ticker_types

    # 시장 명시 접두사가 있으면 그 **국가**로 범위를 좁힌다. 동일 심볼이 여러 시장에 있을 때
    # 구분하려는 것이라, 접두사가 가리키는 것은 국가지 종목풀 이름이 아니다.
    # (예전에는 "ASX:" 를 풀 이름 "aus" 로 못 박아 뒀는데, 풀 이름을 `aus_etf` 로 바꾸자
    #  호주 종목이 한 건도 안 잡혔다 — 설정에서 바뀌는 값을 코드에 적어 두면 이렇게 깨진다.)
    forced_country: str | None = None
    for prefix, country in _PREFIX_COUNTRIES.items():
        if ticker_key.startswith(prefix):
            ticker_key = ticker_key[len(prefix) :]
            forced_country = country
            if not ticker_key:
                raise ValueError(f"{prefix} 뒤에 티커가 필요합니다.")
            break

    # 호주 종목은 stock_meta 에 `ASX:MVR` 로 저장된다(시스템 표준 표기, utils/asx_ticker).
    # 접두사를 뗀 `MVR` 로 비교하면 호주 풀에서 한 건도 매칭되지 않고, 아래 폴백에서
    # 미국 티커로 넘어가 yfinance 가 `$MVR` 를 조회하다 실패한다.
    def _pool_ticker(country_code: str) -> str:
        return ensure_asx_prefix(ticker_key) if str(country_code).strip().lower() == "au" else ticker_key

    configs = load_ticker_type_configs()
    matches: list[dict[str, object]] = []
    for config in configs:
        ticker_type = config["ticker_type"]
        try:
            pool_order = int(config["order"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"종목풀 {ticker_type}의 우선순위(order)가 없습니다.") from exc
        if forced_country is not None and str(config.get("country_code") or "").strip().lower() != forced_country:
            continue
        if not _allowed(str(ticker_type)):
            continue
        country_code = config.get("country_code", "")
        for item in get_etfs(ticker_type):
            item_ticker = str(item.get("ticker") or "").strip().upper()
            if item_ticker != _pool_ticker(country_code):
                continue
            matches.append(
                {
                    # 저장 표기를 그대로 돌려준다 — 이 값이 계좌 보유·실험 포트폴리오에 그대로 쓰인다.
                    "ticker": item_ticker,
                    "name": str(item.get("name") or "").strip() or ticker_key,
                    "ticker_type": ticker_type,
                    "country_code": country_code,
                    "is_etf": bool(item.get("is_etf", False)),
                    "bucket": int(item.get("bucket") or 1),
                    "_pool_order": pool_order,
                }
            )

    def _public_match(match: dict[str, object]) -> dict[str, object]:
        return {key: value for key, value in match.items() if not str(key).startswith("_")}

    # 같은 티커가 여러 시장에 있으면(예: 미국·호주 IOO) 계좌가 보유하는 현금 통화로 후보를 좁힌다.
    # (계좌 담기 검증 validate_ticker_for_account 와 동일한 규칙 — 낙관적 표시와 실제 저장의 시장 일치)
    if len(matches) > 1 and account_id:
        try:
            from utils.cash_model import resolve_cash_currencies
            from utils.settings_loader import get_account_settings

            acct = get_account_settings(account_id)
            acct_currencies = set(resolve_cash_currencies(acct.get("settings") or acct))
            filtered = [
                match
                for match in matches
                if currency_for_country(str(match.get("country_code") or "")) in acct_currencies
            ]
            if filtered:
                matches = filtered
        except Exception:
            pass

    if len(matches) == 1:
        return _public_match(matches[0])
    if len(matches) > 1:
        sorted_matches = sorted(matches, key=lambda match: (int(match["_pool_order"]), str(match["ticker_type"])))
        return _public_match(sorted_matches[0])

    # 종목풀에 없으면 여기서 끝난다 — 티커 **형식**으로 국가·종목풀을 추측하지 않는다.
    # (6자리 숫자면 한국, 알파벳이면 미국 … 식의 폴백이 있었는데, 호주 종목이 전부 미국으로
    #  넘어가 yfinance 가 `$MVR` 를 조회하다 죽었다. 잘못 찍은 시장으로 저장되는 쪽이
    #  "못 찾았다" 보다 훨씬 비싸다.)
    if allowed_ticker_types is not None:
        joined = ", ".join(sorted(allowed_ticker_types)) or "없음"
        raise RuntimeError(f"{ticker_key} 티커를 연결된 종목풀({joined})에서 찾지 못했습니다.")
    raise RuntimeError(f"{ticker_key} 티커를 찾지 못했습니다.")
