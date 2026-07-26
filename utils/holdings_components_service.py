"""보유종목 상세 — 보유 ETF의 구성종목을 통합 합산하여 비중 순으로 반환한다."""

from __future__ import annotations

from typing import Any

from services.component_price_service import enrich_component_prices
from services.stock_cache_service import get_stock_cache_meta_map
from utils.account_registry import load_account_configs
from utils.asx_ticker import ensure_asx_prefix
from utils.logger import get_app_logger
from utils.ticker_registry import load_ticker_type_configs

logger = get_app_logger()
_MAX_VISIBLE_COMPONENTS = 100


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _estimate_daily_profit(current_value_krw: float, change_pct: Any) -> float | None:
    rate = _safe_float(change_pct)
    if current_value_krw == 0.0:
        return 0.0
    denominator = 1.0 + (rate / 100.0)
    if denominator <= 0.0:
        return None
    return current_value_krw - (current_value_krw / denominator)


def _infer_price_country_code(ticker: str) -> str:
    ticker_norm = _normalize_ticker(ticker)
    if len(ticker_norm) == 6:
        return "kor"
    if ticker_norm.endswith(".AX"):
        return "au"
    return "us"


# 종목 국가 분류 (보유종목 상세 화면의 필터 셀렉터용).
# 전체 / 미국 / 한국 / 호주 / 기타국가 5개 그룹 (전체는 분류 결과가 아니라 "필터 없음").
_HOLDING_COUNTRY_ALL = "all"
_HOLDING_COUNTRY_US = "us"
_HOLDING_COUNTRY_KOR = "kor"
_HOLDING_COUNTRY_AU = "au"
_HOLDING_COUNTRY_OTHER = "other"
_HOLDING_COUNTRY_LABELS: dict[str, str] = {
    _HOLDING_COUNTRY_ALL: "전체",
    _HOLDING_COUNTRY_US: "미국",
    _HOLDING_COUNTRY_KOR: "한국",
    _HOLDING_COUNTRY_AU: "호주",
    _HOLDING_COUNTRY_OTHER: "기타국가",
}
_HOLDING_COUNTRY_ORDER: list[str] = [
    _HOLDING_COUNTRY_ALL,
    _HOLDING_COUNTRY_US,
    _HOLDING_COUNTRY_KOR,
    _HOLDING_COUNTRY_AU,
    _HOLDING_COUNTRY_OTHER,
]


def _classify_ticker_country(ticker: str) -> str:
    """티커 패턴만으로 종목 국가를 미국/한국/호주/기타국가 4개로 분류한다.

    호주 종목은 시스템 전 구간에서 `ASX:` 접두사가 강제 부착되어 유통된다
    (docs/developer_guide.md "호주 티커 식별 규칙" 참조). 분류도 그 접두사를
    그대로 식별 키로 사용한다.
    """
    ticker_norm = _normalize_ticker(ticker)
    if not ticker_norm or ticker_norm in {"-", "IS"}:
        return _HOLDING_COUNTRY_OTHER
    # 호주: ASX: 접두사 또는 .AX 접미사
    if ticker_norm.startswith("ASX:") or ticker_norm.endswith(".AX"):
        return _HOLDING_COUNTRY_AU
    if len(ticker_norm) == 6 and ticker_norm.isdigit():
        return _HOLDING_COUNTRY_KOR
    # 점이 없는 영문 티커는 미국으로 간주.
    if "." not in ticker_norm and ticker_norm.isascii() and ticker_norm.isalpha():
        return _HOLDING_COUNTRY_US
    return _HOLDING_COUNTRY_OTHER


# 상장 통화 → 종목 국가. 표에 없는 통화는 기타국가로 본다(임의 추정하지 않는다).
_HOLDING_COUNTRY_BY_LISTING_CURRENCY: dict[str, str] = {
    "AUD": _HOLDING_COUNTRY_AU,
    "KRW": _HOLDING_COUNTRY_KOR,
    "USD": _HOLDING_COUNTRY_US,
}


def _classify_holding_country(component: dict[str, Any]) -> str:
    """구성종목의 상장 국가를 미국/한국/호주/기타국가 4개로 분류한다.

    구성종목 수집 소스가 알려준 원본 심볼(`yahoo_symbol`, 예: `NAB.AX`)과 상장 통화
    (`listing_currency`)가 티커 패턴보다 정확하므로 먼저 본다. 둘 다 없을 때만
    티커 패턴으로 판정한다. (`NAB` 처럼 점 없는 영문 티커는 패턴만으로는 미국과
    구분되지 않는다.)
    """
    yahoo_symbol = _normalize_ticker(component.get("yahoo_symbol"))
    if yahoo_symbol.endswith(".AX"):
        return _HOLDING_COUNTRY_AU

    listing_currency = _normalize_ticker(component.get("listing_currency"))
    if listing_currency:
        return _HOLDING_COUNTRY_BY_LISTING_CURRENCY.get(listing_currency, _HOLDING_COUNTRY_OTHER)

    return _classify_ticker_country(component.get("ticker"))


def _resolve_row_ticker(row: Any) -> tuple[str, str, str]:
    """보유 행에서 (표준 티커, 환종, 가격 조회 국가코드) 를 뽑는다.

    호주 ETF/종목은 시스템 전 구간에서 ASX: 접두사를 강제 부착한다.
    (docs/developer_guide.md "호주 티커 식별 규칙" 참조)
    """
    raw_ticker = _normalize_ticker(row.get("티커", row.get("ticker", "")))
    currency = str(row.get("환종") or row.get("currency") or "").strip().upper() or "KRW"
    country_code = str(row.get("country_code") or "").strip().lower() or _infer_price_country_code(raw_ticker)
    ticker = ensure_asx_prefix(raw_ticker) if (country_code == "au" or currency == "AUD") else raw_ticker
    return ticker, currency, country_code


def list_holding_country_options() -> list[dict[str, str]]:
    """종목 국가 셀렉터에 사용할 코드/라벨 목록 (전체, 미국, 한국, 호주, 기타국가 순).

    첫 항목이 화면의 기본 선택값이 된다.
    """
    return [
        {"code": code, "label": _HOLDING_COUNTRY_LABELS[code]}
        for code in _HOLDING_COUNTRY_ORDER
    ]


def _is_hidden_component_ticker(ticker: Any) -> bool:
    return _normalize_ticker(str(ticker or "")) == "IS"


def _load_account_holdings_frame(account_id: str) -> Any:
    """계좌 보유 테이블을 1회 로드한다 (시세 조회를 포함하므로 계좌당 한 번만 부른다).

    평가금액 합산과 구성종목 통합이 같은 테이블을 쓰므로 호출자가 결과를 공유한다.
    """
    from utils.portfolio_io import load_real_holdings_table

    try:
        return load_real_holdings_table(account_id)
    except Exception as exc:
        logger.warning("포트폴리오 조회를 실패했습니다 (%s): %s", account_id, exc)
        return None


def _account_valuation_krw(df: Any) -> float:
    if df is None or df.empty:
        return 0.0
    return float(df["평가금액(KRW)"].sum())


def _load_account_cash_balance_krw(account_id: str) -> float:
    from utils.portfolio_io import load_portfolio_master

    master = load_portfolio_master(account_id) or {}
    return _safe_float(master.get("cash_balance"))


def _append_account_cash_component(
    *,
    account_id: str,
    account_name: str,
    merged: dict[str, dict[str, Any]],
    cash_balance_krw: float,
    total_assets_krw: float,
) -> None:
    if cash_balance_krw <= 0.0 or total_assets_krw <= 0.0:
        return

    cash_weight = cash_balance_krw / total_assets_krw * 100.0
    source = {
        "etf_ticker": "-",
        "etf_name": "현금",
        "weight": cash_weight,
        "current_price": None,
        "change_pct": None,
        "currency": "KRW",
        "price_country_code": "kor",
        "buy_amount_krw": 0.0,
        "current_value_krw": cash_balance_krw,
        "cumulative_profit_krw": 0.0,
        "return_pct": None,
        "account_id": account_id,
        "account_name": account_name,
    }

    if "-" in merged:
        merged["-"]["total_weight"] += cash_weight
        merged["-"]["current_value_krw"] += cash_balance_krw
        merged["-"]["sources"].append(source)
        return

    merged["-"] = {
        "ticker": "-",
        "name": "현금",
        "has_components": False,
        "total_weight": cash_weight,
        "current_price": None,
        "change_pct": None,
        "currency": "KRW",
        "price_country_code": "kor",
        "buy_amount_krw": 0.0,
        "current_value_krw": cash_balance_krw,
        "cumulative_profit_krw": 0.0,
        "sources": [source],
    }


def _append_account_components(
    *,
    account_id: str,
    account_name: str,
    df: Any,
    merged: dict[str, dict[str, Any]],
    etf_details: list[dict[str, Any]],
    total_valuation_krw: float | None = None,
    ticker_type_filter: set[str] | None = None,
) -> None:
    """단일 계좌의 보유 ETF 구성종목을 누적 병합한다.

    df 는 호출자가 `_load_account_holdings_frame` 으로 미리 로드한 보유 테이블이다.
    ticker_type_filter 가 주어지면 해당 ticker_type 의 ETF 만 통합 대상으로 한다
    (노출국가 필터에 사용).
    """
    if df is None or df.empty:
        return

    ticker_types = [str(config["ticker_type"]).strip().lower() for config in load_ticker_type_configs()]
    if not ticker_types:
        raise RuntimeError("사용 가능한 종목풀이 없습니다.")

    total_valuation = float(total_valuation_krw) if total_valuation_krw is not None else float(df["평가금액(KRW)"].sum())
    if total_valuation <= 0:
        total_valuation = 1.0

    def _is_target_row(target_row: Any) -> bool:
        """통합 대상 보유 행인지 — 수량이 있고 ticker_type 필터를 통과하는 행."""
        if int(target_row.get("수량", target_row.get("quantity", 0))) <= 0:
            return False
        if ticker_type_filter is None:
            return True
        return str(target_row.get("ticker_type") or "").strip().lower() in ticker_type_filter

    # 구성종목 캐시를 ETF×종목풀 조합마다 개별 조회하면 수백 회 DB 왕복이 된다.
    # 종목풀당 1회 배치 조회로 모아 두고 아래 루프에서는 맵 조회만 한다.
    # 캐시도 시스템 표준 표기(ASX:SYI)로 저장되므로 접두사를 그대로 두고 조회한다.
    target_tickers = [_resolve_row_ticker(r)[0] for _, r in df.iterrows() if _is_target_row(r)]
    cache_maps = {t_type: get_stock_cache_meta_map(t_type, target_tickers) for t_type in ticker_types}

    for _, row in df.iterrows():
        if not _is_target_row(row):
            continue

        quantity = int(row.get("수량", row.get("quantity", 0)))

        valuation = float(row.get("평가금액(KRW)") or 0.0)
        buy_amount = float(row.get("매입금액(KRW)") or 0.0)
        etf_profit = float(row.get("평가손익(KRW)") or 0.0)
        etf_return_pct = float(row.get("수익률(%)") or 0.0)
        etf_daily_pct = row.get("일간(%)")
        etf_current_price = row.get("현재가")
        ticker, etf_currency, etf_price_country_code = _resolve_row_ticker(row)
        portfolio_weight = valuation / total_valuation

        cache_doc = None
        for t_type in ticker_types:
            cache_doc = cache_maps[t_type].get(ticker)
            if cache_doc and cache_doc.get("holdings_cache", {}).get("items"):
                break

        holdings_cache = dict((cache_doc or {}).get("holdings_cache") or {}) if isinstance(cache_doc, dict) else {}
        items = list(holdings_cache.get("items") or [])

        etf_detail: dict[str, Any] = {
            "ticker": ticker,
            "name": str(row.get("종목명", row.get("name", ticker))),
            "quantity": quantity,
            "component_count": len(items),
            "has_components": bool(items),
            "account_id": account_id,
            "account_name": account_name,
        }
        etf_details.append(etf_detail)

        if not items:
            comp_key = ticker
            comp_name = etf_detail["name"]

            if "현금" in comp_name:
                comp_key = "-"
                comp_name = "현금"

            comp_weight = 100.0 * portfolio_weight
            if comp_key in merged:
                merged[comp_key]["total_weight"] += comp_weight
                merged[comp_key]["buy_amount_krw"] += buy_amount
                merged[comp_key]["current_value_krw"] += valuation
                merged[comp_key]["cumulative_profit_krw"] += etf_profit
                merged[comp_key]["sources"].append(
                    {
                        "etf_ticker": ticker,
                        "etf_name": etf_detail["name"],
                        "weight": comp_weight,
                        "current_price": etf_current_price,
                        "change_pct": etf_daily_pct,
                        "currency": etf_currency,
                        "price_country_code": etf_price_country_code,
                        "buy_amount_krw": buy_amount,
                        "current_value_krw": valuation,
                        "cumulative_profit_krw": etf_profit,
                        "return_pct": etf_return_pct,
                        "account_id": account_id,
                        "account_name": account_name,
                    }
                )
            else:
                merged[comp_key] = {
                    "ticker": comp_key,
                    "name": comp_name,
                    "has_components": False,
                    "total_weight": comp_weight,
                    "current_price": etf_current_price,
                    "change_pct": etf_daily_pct,
                    "currency": etf_currency,
                    "price_country_code": etf_price_country_code,
                    "buy_amount_krw": buy_amount,
                    "current_value_krw": valuation,
                    "cumulative_profit_krw": etf_profit,
                    "sources": [
                        {
                            "etf_ticker": ticker,
                            "etf_name": etf_detail["name"],
                            "weight": comp_weight,
                            "current_price": etf_current_price,
                            "change_pct": etf_daily_pct,
                            "currency": etf_currency,
                            "price_country_code": etf_price_country_code,
                            "buy_amount_krw": buy_amount,
                            "current_value_krw": valuation,
                            "cumulative_profit_krw": etf_profit,
                            "return_pct": etf_return_pct,
                            "account_id": account_id,
                            "account_name": account_name,
                        }
                    ],
                }
            continue

        for item in items:
            comp_ticker = _normalize_ticker(item.get("ticker", ""))
            if not comp_ticker:
                continue
            comp_name = str(item.get("name") or item.get("raw_name") or "").strip()

            if "현금" in comp_name:
                comp_ticker = "-"
                comp_name = "현금"

            raw_weight = float(item.get("weight") or 0.0)
            weight = raw_weight * portfolio_weight
            component_ratio = raw_weight / 100.0
            source_buy_amount = buy_amount * component_ratio
            source_current_value = valuation * component_ratio
            source_cumulative_profit = etf_profit * component_ratio
            source_return_pct = (source_cumulative_profit / source_buy_amount * 100.0) if source_buy_amount > 0 else None
            # 거래소 접미사가 붙은 원본 심볼(NAB.AX)이 티커(NAB)보다 상장 시장을 정확히 알려준다.
            component_yahoo_symbol = str(item.get("yahoo_symbol") or "").strip().upper()
            component_listing_currency = str(item.get("listing_currency") or "").strip().upper()
            component_price_country_code = _infer_price_country_code(component_yahoo_symbol or comp_ticker)
            component_currency = component_listing_currency or (
                "KRW" if component_price_country_code == "kor" else "AUD" if component_price_country_code == "au" else "USD"
            )
            component_price_fields = {
                "raw_code": item.get("raw_code"),
                "reuters_code": item.get("reuters_code"),
                "yahoo_symbol": item.get("yahoo_symbol"),
                "listing_currency": item.get("listing_currency"),
            }

            if comp_ticker in merged:
                merged[comp_ticker]["total_weight"] += weight
                merged[comp_ticker]["buy_amount_krw"] += source_buy_amount
                merged[comp_ticker]["current_value_krw"] += source_current_value
                merged[comp_ticker]["cumulative_profit_krw"] += source_cumulative_profit
                merged[comp_ticker]["has_components"] = bool(merged[comp_ticker].get("has_components")) or True
                for key, value in component_price_fields.items():
                    if value and not merged[comp_ticker].get(key):
                        merged[comp_ticker][key] = value
                merged[comp_ticker]["sources"].append(
                    {
                        "etf_ticker": ticker,
                        "etf_name": etf_detail["name"],
                        "weight": weight,
                        "price_country_code": component_price_country_code,
                        "currency": component_currency,
                        "buy_amount_krw": source_buy_amount,
                        "current_value_krw": source_current_value,
                        "cumulative_profit_krw": source_cumulative_profit,
                        "return_pct": source_return_pct,
                        "account_id": account_id,
                        "account_name": account_name,
                    }
                )
            else:
                merged[comp_ticker] = {
                    "ticker": comp_ticker,
                    "name": comp_name,
                    "has_components": True,
                    "total_weight": weight,
                    "currency": component_currency,
                    "price_country_code": component_price_country_code,
                    **component_price_fields,
                    "buy_amount_krw": source_buy_amount,
                    "current_value_krw": source_current_value,
                    "cumulative_profit_krw": source_cumulative_profit,
                    "sources": [
                        {
                            "etf_ticker": ticker,
                            "etf_name": etf_detail["name"],
                            "weight": weight,
                            "price_country_code": component_price_country_code,
                            "currency": component_currency,
                            "buy_amount_krw": source_buy_amount,
                            "current_value_krw": source_current_value,
                            "cumulative_profit_krw": source_cumulative_profit,
                            "return_pct": source_return_pct,
                            "account_id": account_id,
                            "account_name": account_name,
                        }
                    ],
                }


def load_account_holdings_components(
    account_id: str,
    *,
    ticker_type_filter: set[str] | None = None,
    include_cash: bool = True,
    max_components: int | None = None,
) -> dict[str, Any]:
    """특정 계좌의 보유 ETF 구성종목을 통합 합산하여 비중 순으로 반환한다.

    구성종목 캐시가 없는 ETF는 자기 자신을 100%로 취급한다.

    Args:
        account_id: 계좌 ID 또는 "TOTAL" (전체 계좌 통합).
        ticker_type_filter: 지정 시 해당 ticker_type 의 ETF 만 통합 (노출국가 필터에 사용).
        include_cash: False 면 현금 항목을 합산 결과에서 제외 (노출국가 필터에 사용).
    """
    all_accounts = load_account_configs()
    account_id_norm = str(account_id or "").strip()
    is_total = account_id_norm.upper() == "TOTAL"
    if not is_total:
        account_config = next((a for a in all_accounts if str(a["account_id"]) == account_id_norm), None)
        if not account_config:
            raise ValueError(f"존재하지 않는 계좌입니다: {account_id_norm}")

    # 구성종목 통합 합산용 딕셔너리
    merged: dict[str, dict[str, Any]] = {}
    etf_details: list[dict[str, Any]] = []

    if is_total:
        account_totals: list[dict[str, Any]] = []
        total_assets_krw = 0.0
        for account in all_accounts:
            curr_account_id = str(account["account_id"])
            # 보유 테이블은 시세 조회를 포함해 비싸므로 계좌당 1회만 로드해 아래 통합 단계와 공유한다.
            curr_df = _load_account_holdings_frame(curr_account_id)
            curr_valuation_krw = _account_valuation_krw(curr_df)
            curr_cash_krw = _load_account_cash_balance_krw(curr_account_id)
            account_totals.append(
                {
                    "account_id": curr_account_id,
                    "account_name": str(account.get("name", curr_account_id)),
                    "df": curr_df,
                    "valuation_krw": curr_valuation_krw,
                    "cash_krw": curr_cash_krw,
                }
            )
            total_assets_krw += curr_valuation_krw + curr_cash_krw

        for account_total in account_totals:
            curr_account_id = str(account_total["account_id"])
            curr_account_name = str(account_total["account_name"])
            _append_account_components(
                account_id=curr_account_id,
                account_name=curr_account_name,
                df=account_total["df"],
                merged=merged,
                etf_details=etf_details,
                total_valuation_krw=total_assets_krw,
                ticker_type_filter=ticker_type_filter,
            )
            if include_cash:
                _append_account_cash_component(
                    account_id=curr_account_id,
                    account_name=curr_account_name,
                    merged=merged,
                    cash_balance_krw=float(account_total["cash_krw"]),
                    total_assets_krw=total_assets_krw,
                )
    else:
        account_name = str(account_config.get("name", account_id_norm))
        account_df = _load_account_holdings_frame(account_id_norm)
        account_valuation_krw = _account_valuation_krw(account_df)
        account_cash_krw = _load_account_cash_balance_krw(account_id_norm)
        account_total_assets_krw = account_valuation_krw + account_cash_krw
        _append_account_components(
            account_id=account_id_norm,
            account_name=account_name,
            df=account_df,
            merged=merged,
            etf_details=etf_details,
            total_valuation_krw=account_total_assets_krw,
            ticker_type_filter=ticker_type_filter,
        )
        if include_cash:
            _append_account_cash_component(
                account_id=account_id_norm,
                account_name=account_name,
                merged=merged,
                cash_balance_krw=account_cash_krw,
                total_assets_krw=account_total_assets_krw,
            )

    filtered_etf_details = [detail for detail in etf_details if not _is_hidden_component_ticker(detail.get("ticker"))]

    if not filtered_etf_details and not merged:
        return {
            "account_id": "TOTAL" if is_total else account_id_norm,
            "account_name": "전체" if is_total else account_name,
            "held_etf_count": 0,
            "components": [],
            "etf_details": [],
        }

    visible_components = [
        component for component in merged.values() if not _is_hidden_component_ticker(component.get("ticker"))
    ]

    # 비중 순 정렬 후 화면에는 상위 구성종목만 반환한다.
    # 임계값은 0 초과(전체 자산 대비 비중이 양수). ETF 안 작은 비중 종목도 누락되지 않도록.
    all_sorted_components = sorted(
        (component for component in visible_components if float(component.get("total_weight") or 0.0) > 0.0),
        key=lambda x: x["total_weight"],
        reverse=True,
    )
    total_component_count = len(all_sorted_components)
    cap = max_components if max_components is not None else _MAX_VISIBLE_COMPONENTS
    sorted_components = all_sorted_components[:cap] if cap and cap > 0 else all_sorted_components

    sorted_components, _ = enrich_component_prices(
        sorted_components,
        price_fetch_limit=None,  # 보유 종목 전체에 가격 채움 (작은 비중 종목 누락 방지).
        preserve_existing=True,
    )

    from services.price_service import get_realtime_snapshot

    source_tickers_by_country: dict[str, set[str]] = {"kor": set(), "au": set(), "us": set()}
    for component in sorted_components:
        for source in component["sources"]:
            source_ticker = _normalize_ticker(source.get("etf_ticker"))
            if not source_ticker or source_ticker in {"-", "IS"}:
                continue
            country = str(source.get("price_country_code") or "").strip().lower() or _infer_price_country_code(source_ticker)
            if country in source_tickers_by_country:
                source_tickers_by_country[country].add(source_ticker)

    source_price_map: dict[str, dict[str, Any]] = {}
    for country, tickers in source_tickers_by_country.items():
        if not tickers:
            continue
        try:
            source_price_map.update(get_realtime_snapshot(country, list(tickers)))
        except Exception as exc:
            logger.warning("보유종목 상세 소스 ETF 가격 조회 실패 (%s): %s", country, exc)

    # 수치 반올림 및 가격 정보 병합
    for comp in sorted_components:
        comp["total_weight"] = round(comp["total_weight"], 2)

        ticker = comp["ticker"]
        change_val = comp.get("change_pct")
        if comp.get("price_currency"):
            comp["currency"] = comp.get("price_currency")
        comp["daily_profit_krw"] = _estimate_daily_profit(_safe_float(comp.get("current_value_krw")), change_val)
        comp["valuation_krw"] = _safe_float(comp.get("current_value_krw"))
        comp["return_pct"] = (
            (_safe_float(comp.get("cumulative_profit_krw")) / _safe_float(comp.get("buy_amount_krw")) * 100.0)
            if _safe_float(comp.get("buy_amount_krw")) > 0
            else None
        )

        if ticker == "-":
            comp["currency"] = "KRW"
            comp["daily_profit_krw"] = 0.0
            comp["return_pct"] = None

        # 소스 ETF 가격 정보 삽입
        for src in comp["sources"]:
            src["weight"] = round(src["weight"], 2)
            source_price = source_price_map.get(_normalize_ticker(src.get("etf_ticker")), {})
            if source_price:
                src["current_price"] = (
                    source_price.get("nowVal") if source_price.get("nowVal") is not None else source_price.get("price")
                )
                s_change_val = source_price.get("changeRate")
                if s_change_val is None:
                    s_change_val = source_price.get("change_pct")
                src["change_pct"] = s_change_val
            else:
                s_change_val = src.get("change_pct")
            src["daily_profit_krw"] = _estimate_daily_profit(_safe_float(src.get("current_value_krw")), s_change_val)
            src["valuation_krw"] = _safe_float(src.get("current_value_krw"))

    return {
        "account_id": "TOTAL" if is_total else account_id_norm,
        "account_name": "전체" if is_total else account_name,
        "held_etf_count": len(filtered_etf_details),
        "components_total_count": total_component_count,
        "components_visible_limit": _MAX_VISIBLE_COMPONENTS,
        "components": sorted_components,
        "etf_details": sorted(filtered_etf_details, key=lambda x: (str(x.get("account_name") or ""), x["ticker"])),
    }


def load_holding_country_components(country_code: str) -> dict[str, Any]:
    """종목 국가(all/us/kor/au/other) 기준 보유 ETF 구성종목 통합.

    모든 계좌의 보유 ETF 의 구성종목을 통합한 뒤, 각 종목의 티커 패턴으로 분류된
    국가가 인자와 일치하는 종목만 남긴다. `all` 은 이 필터를 건너뛴다(추가 계산 없음
    — 어차피 아래 통합 단계에서 전 국가를 모두 계산하고 있다). 현금은 항상 제외하며
    비중은 원본(전체 자산 대비) 그대로 유지한다.
    """
    code = str(country_code or "").strip().lower()
    if code not in _HOLDING_COUNTRY_LABELS:
        raise ValueError(f"지원하지 않는 종목 국가 코드: {country_code}")

    # 국가 필터 후 박스 뷰에서 작은 비중 종목까지 보여야 하므로, 통합 시 cap 을 적용하지 않는다.
    # (TOTAL cap=100 을 그대로 두면 한국·호주의 작은 종목들이 미국 큰 종목 100개에 밀려 응답에서 통째 사라짐.)
    base = load_account_holdings_components("TOTAL", include_cash=False, max_components=0)
    all_components: list[dict[str, Any]] = base.get("components") or []
    if code == _HOLDING_COUNTRY_ALL:
        filtered_components = all_components
    else:
        filtered_components = [comp for comp in all_components if _classify_holding_country(comp) == code]

    return {
        "account_id": f"HOLDING_COUNTRY:{code}",
        "account_name": _HOLDING_COUNTRY_LABELS[code],
        "held_etf_count": base.get("held_etf_count", 0),
        "components_total_count": len(filtered_components),
        # 종목 국가별 통합은 cap 을 두지 않으므로 visible_limit 도 total 과 동일.
        "components_visible_limit": len(filtered_components),
        "components": filtered_components,
        "etf_details": base.get("etf_details") or [],
    }
