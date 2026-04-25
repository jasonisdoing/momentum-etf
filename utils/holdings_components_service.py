"""보유종목 상세 — 보유 ETF의 구성종목을 통합 합산하여 비중 순으로 반환한다."""

from __future__ import annotations

from typing import Any

from services.component_price_service import enrich_component_prices
from services.stock_cache_service import get_stock_cache_meta
from utils.account_registry import load_account_configs
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


def _is_hidden_component_ticker(ticker: Any) -> bool:
    return _normalize_ticker(str(ticker or "")) == "IS"


def _renormalize_component_weights(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_weight_sum = sum(_safe_float(component.get("total_weight")) for component in components)
    if total_weight_sum <= 0.0:
        return components

    scale = 100.0 / total_weight_sum
    for component in components:
        component["total_weight"] = _safe_float(component.get("total_weight")) * scale
        for source in component.get("sources", []):
            source["weight"] = _safe_float(source.get("weight")) * scale
    return components


def _append_account_components(
    *,
    account_id: str,
    account_name: str,
    merged: dict[str, dict[str, Any]],
    etf_details: list[dict[str, Any]],
    total_valuation_krw: float | None = None,
) -> None:
    """단일 계좌의 보유 ETF 구성종목을 누적 병합한다."""
    from utils.portfolio_io import load_real_holdings_table

    try:
        df = load_real_holdings_table(account_id)
    except Exception as exc:
        logger.warning("포트폴리오 조회를 실패했습니다 (%s): %s", account_id, exc)
        return

    if df is None or df.empty:
        return

    ticker_types = [str(config["ticker_type"]).strip().lower() for config in load_ticker_type_configs()]
    if not ticker_types:
        raise RuntimeError("사용 가능한 종목풀이 없습니다.")

    total_valuation = float(total_valuation_krw) if total_valuation_krw is not None else float(df["평가금액(KRW)"].sum())
    if total_valuation <= 0:
        total_valuation = 1.0

    for _, row in df.iterrows():
        ticker = _normalize_ticker(row.get("티커", row.get("ticker", "")))
        quantity = int(row.get("수량", row.get("quantity", 0)))
        if quantity <= 0:
            continue

        valuation = float(row.get("평가금액(KRW)") or 0.0)
        buy_amount = float(row.get("매입금액(KRW)") or 0.0)
        etf_profit = float(row.get("평가손익(KRW)") or 0.0)
        etf_return_pct = float(row.get("수익률(%)") or 0.0)
        etf_daily_pct = row.get("일간(%)")
        etf_current_price = row.get("현재가")
        etf_currency = str(row.get("환종") or row.get("currency") or "").strip().upper() or "KRW"
        etf_price_country_code = str(row.get("country_code") or "").strip().lower() or _infer_price_country_code(ticker)
        portfolio_weight = valuation / total_valuation

        cache_doc = None
        for t_type in ticker_types:
            cache_doc = get_stock_cache_meta(t_type, ticker)
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
            component_price_country_code = _infer_price_country_code(comp_ticker)
            component_currency = "KRW" if component_price_country_code == "kor" else "AUD" if component_price_country_code == "au" else "USD"
            component_price_fields = {
                "raw_code": item.get("raw_code"),
                "reuters_code": item.get("reuters_code"),
                "yahoo_symbol": item.get("yahoo_symbol"),
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


def load_account_holdings_components(account_id: str) -> dict[str, Any]:
    """특정 계좌의 보유 ETF 구성종목을 통합 합산하여 비중 순으로 반환한다.

    구성종목 캐시가 없는 ETF는 자기 자신을 100%로 취급한다.
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
        total_valuation_krw = 0.0
        for account in all_accounts:
            curr_account_id = str(account["account_id"])
            try:
                from utils.portfolio_io import load_real_holdings_table

                curr_df = load_real_holdings_table(curr_account_id)
            except Exception as exc:
                logger.warning("포트폴리오 조회를 실패했습니다 (%s): %s", curr_account_id, exc)
                continue
            if curr_df is None or curr_df.empty:
                continue
            total_valuation_krw += float(curr_df["평가금액(KRW)"].sum())

        for account in all_accounts:
            curr_account_id = str(account["account_id"])
            curr_account_name = str(account.get("name", curr_account_id))
            _append_account_components(
                account_id=curr_account_id,
                account_name=curr_account_name,
                merged=merged,
                etf_details=etf_details,
                total_valuation_krw=total_valuation_krw,
            )
    else:
        account_name = str(account_config.get("name", account_id_norm))
        _append_account_components(
            account_id=account_id_norm,
            account_name=account_name,
            merged=merged,
            etf_details=etf_details,
        )

    filtered_etf_details = [detail for detail in etf_details if not _is_hidden_component_ticker(detail.get("ticker"))]

    if not filtered_etf_details:
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
    visible_components = _renormalize_component_weights(visible_components)

    # 비중 순 정렬 후 화면에는 상위 구성종목만 반환한다.
    all_sorted_components = sorted(
        (component for component in visible_components if float(component.get("total_weight") or 0.0) >= 0.01),
        key=lambda x: x["total_weight"],
        reverse=True,
    )
    total_component_count = len(all_sorted_components)
    sorted_components = all_sorted_components[:_MAX_VISIBLE_COMPONENTS]

    sorted_components, _ = enrich_component_prices(
        sorted_components,
        price_fetch_limit=100,
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
