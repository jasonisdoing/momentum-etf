"""보유종목 상세 — 보유 ETF의 구성종목을 통합 합산하여 비중 순으로 반환한다."""

from __future__ import annotations

from typing import Any

from services.stock_cache_service import get_stock_cache_meta
from utils.account_registry import load_account_configs
from utils.logger import get_app_logger
from utils.portfolio_io import load_portfolio_master
from utils.stock_list_io import get_etfs
from utils.ticker_registry import load_ticker_type_configs

logger = get_app_logger()


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()




def load_account_holdings_components(account_id: str) -> dict[str, Any]:
    """특정 계좌의 보유 ETF 구성종목을 통합 합산하여 비중 순으로 반환한다.

    구성종목 캐시가 없는 ETF는 자기 자신을 100%로 취급한다.
    """
    from utils.portfolio_io import load_real_holdings_table
    from utils.account_registry import load_account_configs

    # 계좌 정보 확인
    all_accounts = load_account_configs()
    account_config = next((a for a in all_accounts if str(a["account_id"]) == account_id), None)
    if not account_config:
        raise ValueError(f"존재하지 않는 계좌입니다: {account_id}")

    try:
        df = load_real_holdings_table(account_id)
    except Exception as e:
        logger.warning(f"포트폴리오 조회를 실패했습니다 ({account_id}): {e}")
        df = None

    if df is None or df.empty:
        return {
            "account_id": account_id,
            "account_name": str(account_config.get("name", account_id)),
            "held_etf_count": 0,
            "components": [],
            "etf_details": [],
        }

    # 해당 계좌에서 사용하는 종목풀(ticker_type) 목록
    ticker_codes = account_config.get("settings", {}).get("ticker_codes", [])
    if isinstance(ticker_codes, str):
        ticker_codes = [ticker_codes]
    if not ticker_codes:
        ticker_codes = ["1_kor_kr"]

    # 구성종목 통합 합산용 딕셔너리
    merged: dict[str, dict[str, Any]] = {}
    etf_details: list[dict[str, Any]] = []

    # 전체 평가금 통합 (비율 계산용)
    total_valuation = df["평가금액(KRW)"].sum()
    if total_valuation <= 0:
        total_valuation = 1.0

    for _, row in df.iterrows():
        ticker = _normalize_ticker(row.get("티커", row.get("ticker", "")))
        quantity = int(row.get("수량", row.get("quantity", 0)))
        if quantity <= 0:
            continue

        valuation = float(row.get("평가금액(KRW)") or 0.0)
        portfolio_weight = valuation / total_valuation

        # 캐시된 구성종목 정보 조회 (계좌에 설정된 모든 ticker_type을 순회하며 찾는다)
        cache_doc = None
        for t_type in ticker_codes:
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
        }
        etf_details.append(etf_detail)

        if not items:
            # 구성종목 정보가 없으면 자기 자신을 100% 비중으로 계산
            comp_key = ticker
            comp_name = etf_detail["name"]

            # "현금" 키워드 합산 처리
            if "현금" in comp_name:
                comp_key = "-"
                comp_name = "현금"

            comp_weight = 100.0 * portfolio_weight
            
            if comp_key in merged:
                merged[comp_key]["total_weight"] += comp_weight
                merged[comp_key]["sources"].append({
                    "etf_ticker": ticker,
                    "etf_name": etf_detail["name"],
                    "weight": comp_weight,
                })
            else:
                merged[comp_key] = {
                    "ticker": comp_key,
                    "name": comp_name,
                    "total_weight": comp_weight,
                    "sources": [{
                        "etf_ticker": ticker,
                        "etf_name": etf_detail["name"],
                        "weight": comp_weight,
                    }],
                }
            continue

        for item in items:
            comp_ticker = _normalize_ticker(item.get("ticker", ""))
            if not comp_ticker:
                continue
            comp_name = str(item.get("name") or item.get("raw_name") or "").strip()
            
            # "현금" 키워드 합산 처리
            if "현금" in comp_name:
                comp_ticker = "-"
                comp_name = "현금"

            # (ETF 내 비중) * (ETF가 전체 계좌에서 차지하는 비율)
            raw_weight = float(item.get("weight") or 0.0)
            weight = raw_weight * portfolio_weight

            if comp_ticker in merged:
                merged[comp_ticker]["total_weight"] += weight
                merged[comp_ticker]["sources"].append({
                    "etf_ticker": ticker,
                    "etf_name": etf_detail["name"],
                    "weight": weight,
                })
            else:
                merged[comp_ticker] = {
                    "ticker": comp_ticker,
                    "name": comp_name,
                    "total_weight": weight,
                    "sources": [{
                        "etf_ticker": ticker,
                        "etf_name": etf_detail["name"],
                        "weight": weight,
                    }],
                }

    # 비중 순 정렬
    sorted_components = sorted(
        merged.values(),
        key=lambda x: x["total_weight"],
        reverse=True,
    )

    # 실시간 가격 정보 추가
    from services.price_service import get_realtime_snapshot
    
    # 국가별 조회가 필요한 티커 목록 분류 (구성종목 + 소스 ETF)
    kor_tickers = set()
    us_tickers = set()
    
    def classify_ticker(t: str):
        if not t or t == "-": return
        if len(t) == 6 and t.isdigit():
            kor_tickers.add(t)
        else:
            us_tickers.add(t)

    for c in sorted_components:
        classify_ticker(c["ticker"])
        for src in c["sources"]:
            classify_ticker(src["etf_ticker"])
    
    # 실시간 스냅샷 조회
    price_map: dict[str, dict[str, Any]] = {}
    
    if kor_tickers:
        try:
            kor_results = get_realtime_snapshot("kor", list(kor_tickers))
            price_map.update(kor_results)
        except Exception as e:
            logger.warning(f"보유종목 상세 가격 조회 실패 (kor): {e}")
            
    if us_tickers:
        try:
            us_results = get_realtime_snapshot("us", list(us_tickers))
            price_map.update(us_results)
        except Exception as e:
            logger.warning(f"보유종목 상세 가격 조회 실패 (us): {e}")

    # 수치 반올림 및 가격 정보 병합
    for comp in sorted_components:
        comp["total_weight"] = round(comp["total_weight"], 2)
        
        # 구성종목 가격 정보 삽입
        ticker = comp["ticker"]
        p_data = price_map.get(ticker, {})
        comp["current_price"] = p_data.get("nowVal") or p_data.get("price")
        comp["change_pct"] = p_data.get("changeRate") or p_data.get("change_pct")
        
        if ticker != "-":
            comp["currency"] = "KRW" if len(ticker) == 6 and ticker.isdigit() else "USD"
        else:
            comp["currency"] = "KRW"

        # 소스 ETF 가격 정보 삽입
        for src in comp["sources"]:
            src["weight"] = round(src["weight"], 2)
            s_ticker = src["etf_ticker"]
            s_p_data = price_map.get(s_ticker, {})
            src["current_price"] = s_p_data.get("nowVal") or s_p_data.get("price")
            src["change_pct"] = s_p_data.get("changeRate") or s_p_data.get("change_pct")
            src["currency"] = "KRW" if len(s_ticker) == 6 and s_ticker.isdigit() else "USD"

    return {
        "account_id": account_id,
        "account_name": str(account_config.get("name", account_id)),
        "held_etf_count": len(etf_details),
        "components": sorted_components,
        "etf_details": sorted(etf_details, key=lambda x: x["ticker"]),
    }
