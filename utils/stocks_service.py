from __future__ import annotations

from datetime import datetime
from typing import Any

from services.price_service import get_realtime_snapshot
from services.stock_cache_service import delete_stock_cache
from utils.cache_invalidation import invalidate_pool_caches
from utils.db_manager import get_db_connection
from utils.logger import get_app_logger
from utils.normalization import normalize_nullable_number, normalize_text
from utils.stock_list_io import add_stock, hard_remove_stock, invalidate_ticker_type_cache
from utils.stock_meta_updater import fetch_stock_info
from utils.ticker_registry import load_ticker_type_configs as load_account_configs

BUCKETS: dict[int, str] = {
    1: "1. 모멘텀",
    2: "2. 시장지수",
    3: "3. 배당방어",
    4: "4. 대체헷지",
}


def _format_deleted_date(value: Any) -> str:
    if not value:
        return "-"
    if isinstance(value, datetime):
        return value.isoformat()[:10]
    return str(value).strip()[:10] or "-"


def _load_ticker_types_payload() -> list[dict[str, Any]]:
    configs = load_account_configs()
    if not configs:
        raise RuntimeError("종목풀 설정이 없습니다.")
    return [
        {
            "ticker_type": config["ticker_type"],
            "order": config["order"],
            "name": config["name"],
            "icon": config["icon"],
            "country_code": config.get("country_code", ""),
        }
        for config in configs
    ]


def _pick_ticker_type(ticker_types: list[dict[str, Any]], ticker_type: str | None) -> str:
    target = str(ticker_type or "").strip().lower()
    available_ids = [str(t["ticker_type"]).lower() for t in ticker_types]

    if target and target in available_ids:
        return target

    return available_ids[0] if available_ids else ""


def _normalize_candidate_ticker(ticker: str, country_code: str) -> str:
    text = str(ticker or "").strip().upper()
    if not text:
        raise RuntimeError("티커를 입력하세요.")
    if ":" in text:
        text = text.split(":")[-1].strip().upper()
    if (country_code or "").strip().lower() == "au" and text.endswith(".AX"):
        text = text[:-3]
    if (country_code or "").strip().lower() == "us" and "." in text:
        text = text.replace(".", "-")
    return text


def _load_account_config_map() -> dict[str, dict[str, Any]]:
    return {config["ticker_type"]: config for config in load_account_configs()}


def _require_ticker_type_config(ticker_type: str) -> dict[str, Any]:
    type_norm = str(ticker_type or "").strip().lower()
    configs = _load_account_config_map()
    config = configs.get(type_norm)
    if not config:
        raise RuntimeError("종목풀을 찾을 수 없습니다.")
    return config


def _load_stock_meta_doc(ticker_type: str, ticker: str) -> dict[str, Any] | None:
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    return db.stock_meta.find_one(
        {
            "ticker_type": str(ticker_type or "").strip().lower(),
            "ticker": str(ticker or "").strip().upper(),
        },
        {
            "ticker": 1,
            "name": 1,
            "listing_date": 1,
            "bucket": 1,
            "is_deleted": 1,
        },
    )


def _find_other_pool_with_ticker(ticker_type: str, ticker: str) -> str | None:
    """같은 티커를 이미 담고 있는 **다른** 종목풀의 ticker_type. 없으면 None.

    한 티커가 여러 풀에 있으면 계좌 보유 종목의 소속 풀을 특정할 수 없고
    (``portfolio_io._pick_meta_doc`` 이 통화로만 골라 같은 통화의 풀끼리는 구분이 안 된다),
    그러면 그 풀의 이평선으로 이탈을 판정할 수 없다. 그래서 추가 단계에서 막는다.

    호주는 ``ASX:`` 접두사를 붙여 저장하므로(예: ``ASX:IOO`` vs ``IOO``) 미국에 같은 티커가
    있어도 문자열이 달라 자연히 구분된다 — 국가를 따로 볼 필요가 없다.
    """
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    doc = db.stock_meta.find_one(
        {
            "ticker": str(ticker or "").strip().upper(),
            "ticker_type": {"$ne": str(ticker_type or "").strip().lower()},
            "is_deleted": {"$ne": True},
        },
        {"ticker_type": 1},
    )
    return str(doc["ticker_type"]) if doc else None


def _pool_label(ticker_type: str) -> str:
    """오류 문구용 종목풀 표기 — '🇰🇷 국내상장 국내 ETF (kor_kr)'."""
    from utils.settings_loader import get_ticker_type_settings

    try:
        config = get_ticker_type_settings(ticker_type) or {}
    except Exception:
        return ticker_type
    icon = str(config.get("icon") or "").strip()
    name = str(config.get("name") or "").strip()
    if not name:
        return ticker_type
    return f"{icon} {name} ({ticker_type})".strip()


def load_active_stocks_table(ticker_type: str | None = None) -> dict[str, Any]:
    ticker_types = _load_ticker_types_payload()
    target_ticker_type = _pick_ticker_type(ticker_types, ticker_type)

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    docs = list(
        db.stock_meta.find(
            {
                "ticker_type": target_ticker_type,
                "is_deleted": {"$ne": True},
            },
            {
                "ticker": 1,
                "name": 1,
                "bucket": 1,
                "added_date": 1,
                "listing_date": 1,
                "1_week_avg_volume": 1,
                "1_day_earn_rate": 1,
                "1_week_earn_rate": 1,
                "2_week_earn_rate": 1,
                "1_month_earn_rate": 1,
                "3_month_earn_rate": 1,
                "6_month_earn_rate": 1,
                "12_month_earn_rate": 1,
                "exclude_from_ranking": 1,
            },
        )
    )

    # 실시간 스냅샷 가져오기
    tickers = [doc.get("ticker", "") for doc in docs if doc.get("ticker")]
    config = _require_ticker_type_config(target_ticker_type)
    country_code = config.get("country_code", "kor")
    realtime_snapshot = {}
    try:
        realtime_snapshot = get_realtime_snapshot(country_code, tickers)
    except Exception:
        pass

    rows = sorted(
        [
            {
                "ticker": normalize_text(doc.get("ticker"), ""),
                "name": normalize_text(doc.get("name"), ""),
                "bucket_id": int(doc.get("bucket") or 1),
                "bucket_name": BUCKETS.get(int(doc.get("bucket") or 1), BUCKETS[1]),
                "added_date": normalize_text(doc.get("added_date"), "-"),
                "listing_date": normalize_text(doc.get("listing_date"), "-"),
                "week_volume": normalize_nullable_number(doc.get("1_week_avg_volume")),
                "return_1d": normalize_nullable_number(
                    realtime_snapshot.get(doc.get("ticker", ""), {}).get("changeRate")
                ),
                "괴리율": normalize_nullable_number(realtime_snapshot.get(doc.get("ticker", ""), {}).get("deviation")),
                "return_1w": normalize_nullable_number(doc.get("1_week_earn_rate")),
                "return_2w": normalize_nullable_number(doc.get("2_week_earn_rate")),
                "return_1m": normalize_nullable_number(doc.get("1_month_earn_rate")),
                "return_3m": normalize_nullable_number(doc.get("3_month_earn_rate")),
                "return_6m": normalize_nullable_number(doc.get("6_month_earn_rate")),
                "return_12m": normalize_nullable_number(doc.get("12_month_earn_rate")),
                "exclude_from_ranking": bool(doc.get("exclude_from_ranking")),
            }
            for doc in docs
        ],
        key=lambda row: (
            row["bucket_id"],
            -(row["return_1w"] if row["return_1w"] is not None else float("-inf")),
        ),
    )

    return {
        "ticker_types": ticker_types,
        "rows": rows,
        "ticker_type": target_ticker_type,
    }


def validate_stock_candidate(ticker_type: str, ticker: str) -> dict[str, Any]:
    config = _require_ticker_type_config(ticker_type)
    ticker_type_norm = str(config["ticker_type"]).strip().lower()
    country_code = str(config.get("country_code") or "").strip().lower()
    ticker_norm = _normalize_candidate_ticker(ticker, country_code)

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    # 호주(aus) 종목풀은 stock_meta 에 "ASX:TICKER" 접두어 형태로 저장돼 있어(예: ASX:ASIA),
    # 접두어를 뗀 정규화 티커로만 조회하면 매칭이 안 된다. 두 형태를 모두 매칭한다.
    ticker_candidates = [ticker_norm]
    if country_code == "au":
        ticker_candidates.append(f"ASX:{ticker_norm}")

    existing = db.stock_meta.find_one(
        {
            "ticker_type": ticker_type_norm,
            "ticker": {"$in": ticker_candidates},
        },
        {
            "name": 1,
            "listing_date": 1,
            "is_deleted": 1,
            "deleted_reason": 1,
            "bucket": 1,
        },
    )

    is_deleted = bool(existing and existing.get("is_deleted") is True)
    is_active = bool(existing and existing.get("is_deleted") is not True)

    # 이미 등록돼 있는 종목은 외부 조회를 건너뛴다 — 어차피 "이미 등록된 종목입니다" 로
    # 끝나서 받아온 값을 쓰지도 않는다. 여러 종목을 한꺼번에 추가할 때 대부분이 중복인데,
    # 이 호출 때문에 종목당 수백 ms~수 초씩 걸렸다.
    # 저장된 이름이 비어 있으면(옛 문서) 건너뛸 근거가 없으므로 그대로 조회한다.
    existing_name = normalize_text((existing or {}).get("name"), "") if is_active else ""
    if is_active and existing_name:
        stock_info = {"name": existing_name, "listing_date": (existing or {}).get("listing_date")}
    else:
        stock_info = None
        # 미국은 ETF 마켓 캐시(KIS 마스터 이름)를 먼저 본다 — yfinance info 는 레이트리밋에
        # 자주 걸려서(배치 직후 등) 캐시에 있는 종목까지 추가 실패로 끝났다.
        if country_code == "us":
            from utils.us_etf_market_service import us_etf_name_of

            cached_name = us_etf_name_of(ticker_norm)
            if cached_name:
                stock_info = {"name": cached_name, "listing_date": None}
        if stock_info is None:
            stock_info = fetch_stock_info(ticker_norm, country_code)
        if not stock_info or not str(stock_info.get("name") or "").strip():
            raise RuntimeError("유효한 티커를 찾지 못했습니다.")

    deleted_reason = normalize_text(existing.get("deleted_reason"), "") if existing else ""
    listing_date = normalize_text(stock_info.get("listing_date") or (existing or {}).get("listing_date"), "-")
    bucket_id = int((existing or {}).get("bucket") or 1)

    # 반환 티커는 시스템 표준 표기 — 호주는 ASX: 접두사를 붙인다.
    # (이 값이 stock_meta 등록·계좌 보유 저장에 그대로 쓰이므로 여기서 표준화해야
    #  이후 경로 전부가 표준 표기로 저장된다. docs/developer_guide.md "호주 티커 식별 규칙")
    from utils.asx_ticker import ensure_asx_prefix

    return {
        "ticker": ensure_asx_prefix(ticker_norm) if country_code == "au" else ticker_norm,
        "name": normalize_text(stock_info.get("name"), ""),
        "listing_date": listing_date,
        "status": "active" if is_active else "deleted" if is_deleted else "new",
        "is_deleted": is_deleted,
        "deleted_reason": deleted_reason,
        "bucket_id": bucket_id,
        "ticker_type": ticker_type_norm,
        "country_code": country_code,
    }


def refresh_single_stock(ticker_type: str, ticker: str) -> dict[str, str]:
    """단일 종목의 메타데이터와 가격 캐시를 갱신합니다."""
    logger = get_app_logger()
    type_norm = str(ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    if not type_norm or not ticker_norm:
        raise RuntimeError("계좌와 티커를 지정하세요.")

    config = _require_ticker_type_config(type_norm)
    country_code = str(config.get("country_code") or "kor").strip().lower()

    # 1) 가격 캐시 업데이트 — 먼저 채워야 2)의 메타(3달 백테스트 MDD 등)가 캐시에서 데이터를 읽어 계산된다.
    #    (순서가 바뀌면 신규 종목은 캐시가 빈 상태로 메타가 계산돼 MDD 등이 비어버린다.)
    from utils.data_loader import fetch_ohlcv

    try:
        from utils.settings_loader import load_common_settings

        settings = load_common_settings() or {}
        start_date = settings.get("CACHE_START_DATE", "2024-01-01")

        fetch_ohlcv(
            ticker_norm,
            country=country_code,
            months_back=None,
            date_range=[start_date, None],
            update_listing_meta=False,
            force_refresh=True,
            ticker_type=type_norm,
        )
    except Exception as e:
        logger.error(f"[{type_norm.upper()}/{ticker_norm}] 가격 캐시 갱신 실패: {e}")

    # 2) 메타데이터 업데이트 — 이름/상장일 등 식별 메타를 채운다.
    #    (MDD·소르티노는 순위 조회 시 실시간 계산이라 여기서 만들지 않는다.)
    from utils.stock_meta_updater import update_single_ticker_metadata

    try:
        update_single_ticker_metadata(type_norm, ticker_norm)
    except Exception as e:
        logger.error(f"[{type_norm.upper()}/{ticker_norm}] 메타데이터 갱신 실패: {e}")

    invalidate_pool_caches(type_norm)
    return {"ticker": ticker_norm, "ticker_type": type_norm}


def add_active_stock(ticker_type: str, ticker: str, bucket_id: int) -> dict[str, Any]:
    validated = validate_stock_candidate(ticker_type, ticker)
    ticker_type_norm = str(validated["ticker_type"]).strip().lower()
    ticker_norm = str(validated["ticker"]).strip().upper()
    bucket_value = int(bucket_id or 0)
    if bucket_value not in BUCKETS:
        raise RuntimeError("버킷을 선택하세요.")

    # 한 티커는 한 종목풀에만 — 계좌 보유 종목의 소속 풀이 유일해야 그 풀의 이평선으로 판정할 수 있다.
    other_pool = _find_other_pool_with_ticker(ticker_type_norm, ticker_norm)
    if other_pool:
        raise RuntimeError(f"이미 다른 종목풀에 있습니다: {_pool_label(other_pool)}")

    created = add_stock(
        ticker_type_norm,
        ticker_norm,
        name=str(validated["name"]),
        listing_date=None if validated["listing_date"] == "-" else validated["listing_date"],
        bucket=bucket_value,
    )
    if not created:
        current = _load_stock_meta_doc(ticker_type_norm, ticker_norm)
        if current and current.get("is_deleted") is not True:
            if validated["status"] == "active":
                raise RuntimeError("이미 등록된 종목입니다.")

            return {
                "ticker": ticker_norm,
                "name": normalize_text(current.get("name"), str(validated["name"])),
                "listing_date": normalize_text(
                    current.get("listing_date"),
                    str(validated["listing_date"]),
                ),
                "bucket_id": int(current.get("bucket") or bucket_value),
                "bucket_name": BUCKETS.get(int(current.get("bucket") or bucket_value), BUCKETS[bucket_value]),
                "status": "active",
            }

        if validated["status"] == "active":
            raise RuntimeError("이미 등록된 종목입니다.")
        raise RuntimeError(f"종목 추가에 실패했습니다: {ticker_norm}")

    try:
        # 추가 성공은 메타/가격 캐시 준비 완료까지 포함한다.
        refresh_single_stock(ticker_type_norm, ticker_norm)
    except Exception as refresh_error:
        db = get_db_connection()
        if db is not None:
            db.stock_meta.delete_one(
                {
                    "ticker_type": ticker_type_norm,
                    "ticker": ticker_norm,
                    "is_deleted": {"$ne": True},
                }
            )
            invalidate_ticker_type_cache(ticker_type_norm)
        raise RuntimeError(f"종목 캐시 갱신에 실패해 추가를 취소했습니다: {refresh_error}") from refresh_error

    invalidate_pool_caches(ticker_type_norm)
    return {
        "ticker": ticker_norm,
        "name": str(validated["name"]),
        "listing_date": str(validated["listing_date"]),
        "bucket_id": bucket_value,
        "bucket_name": BUCKETS[bucket_value],
        "status": validated["status"],
    }


def update_stock_bucket(ticker_type: str, ticker: str, bucket_id: int) -> None:
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    type_norm = str(ticker_type or "").strip().lower()
    result = db.stock_meta.update_one(
        {
            "ticker_type": type_norm,
            "ticker": str(ticker or "").strip().upper(),
            "is_deleted": {"$ne": True},
        },
        {
            "$set": {
                "bucket": int(bucket_id),
                "updated_at": datetime.now(),
            }
        },
    )

    if result.matched_count == 0:
        raise RuntimeError("수정할 종목을 찾을 수 없습니다.")

    # stock_list_io 의 TTL 캐시가 60초간 이전 값을 반환해 버려서 UI 에 반영되지 않는 것을 방지한다.
    invalidate_ticker_type_cache(type_norm)
    invalidate_pool_caches(type_norm)


def delete_active_stock(ticker_type: str, ticker: str) -> None:
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    type_norm = str(ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()

    if not hard_remove_stock(type_norm, ticker_norm):
        raise RuntimeError("삭제할 종목을 찾을 수 없습니다.")

    from utils.cache_utils import delete_cached_frame
    from utils.stock_list_io import get_active_holding_tickers

    try:
        is_currently_held = ticker_norm in get_active_holding_tickers()
    except Exception:
        is_currently_held = False

    if not is_currently_held:
        try:
            delete_cached_frame(type_norm, ticker_norm)
        except Exception:
            pass

        try:
            delete_stock_cache(type_norm, ticker_norm)
        except Exception:
            pass

    invalidate_pool_caches(type_norm)


def movable_pools(ticker_type: str) -> list[dict[str, Any]]:
    """그 종목풀에서 종목을 옮길 수 있는 대상 풀 목록.

    **같은 국가 + 같은 구분(`pool_kind`)** 만 허용한다. 국가가 다르면 거래 달력·통화가 갈리고,
    구분이 다르면(개별주 ↔ ETF) 업종 상한 같은 설정의 의미가 달라진다. 자기 자신은 뺀다.
    구분이 미설정인 풀은 무엇과도 같다고 볼 수 없어 대상에서 제외한다(추정하지 않는다).
    """
    from utils.momentum_service import pool_options
    from utils.settings_loader import get_ticker_type_settings

    source = get_ticker_type_settings(str(ticker_type or "").strip().lower()) or {}
    country = str(source.get("country_code") or "").strip().lower()
    kind = str(source.get("pool_kind") or "").strip().lower()
    if not country or not kind:
        return []
    return [
        option
        for option in pool_options()
        if option["ticker_type"] != str(ticker_type or "").strip().lower()
        and option.get("country_code") == country
        and option.get("pool_kind") == kind
    ]


def move_active_stock(from_pool: str, to_pool: str, ticker: str) -> dict[str, Any]:
    """종목 하나를 다른 종목풀로 옮긴다 — 옛 풀에서 빼고 새 풀에 담는다.

    한 티커는 한 종목풀에만 있어야 하므로(계좌 보유 종목의 소속 풀이 유일해야 그 풀의
    이평선·손절 기준으로 판정할 수 있다) '양쪽에 두기' 가 아니라 이동이다.

    **다시 받는 것 없이 옮기기만 한다.** 종목 메타(이름·상장일·업종·메모)·가격 캐시·배치
    계산값이 전부 같은 값이라 원천 조회가 필요 없다. 예전에는 삭제 후 신규 추가로 처리해
    종목당 10초(메타 9초 + 시세 1.6초)가 들었다.
    """
    source = str(from_pool or "").strip().lower()
    target = str(to_pool or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    if not source or not target or not ticker_norm:
        raise RuntimeError("출발 종목풀·대상 종목풀·티커가 모두 필요합니다.")
    if source == target:
        raise RuntimeError("출발 종목풀과 대상 종목풀이 같습니다.")
    if target not in {option["ticker_type"] for option in movable_pools(source)}:
        raise RuntimeError(
            f"'{_pool_label(source)}' 에서 '{_pool_label(target)}' 로는 옮길 수 없습니다 — "
            "국가와 구분(개별주/ETF)이 같은 종목풀로만 옮길 수 있습니다."
        )

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    current = db.stock_meta.find_one({"ticker_type": source, "ticker": ticker_norm, "is_deleted": {"$ne": True}})
    if not current:
        raise RuntimeError(f"'{_pool_label(source)}' 에 없는 종목입니다: {ticker_norm}")
    bucket_value = int(current.get("bucket") or 1)

    # 대상 풀에 이미 활성으로 있으면 여기서 막는다. 아래 이동은 '먼저 빼고 담는' 순서라
    # 담기에서 걸리면 되돌리기까지 돌아야 하고, 화면에는 원인 없이 '이동 실패'만 남는다.
    if db.stock_meta.find_one({"ticker_type": target, "ticker": ticker_norm, "is_deleted": {"$ne": True}}):
        raise RuntimeError(
            f"'{_pool_label(target)}' 에 {ticker_norm} 가 이미 있습니다 — 한 티커는 한 종목풀에만 둡니다. "
            "대상 종목풀에서 먼저 지운 뒤 옮겨주세요."
        )
    # 대상 풀에 **삭제 상태로** 남아 있던 문서 — add_stock 이 이걸 되살린다. 이동이 실패하면
    # 되살아난 문서를 원래대로 돌려놔야 한다(안 그러면 출발·대상 양쪽에 종목이 남는다).
    target_before = db.stock_meta.find_one({"ticker_type": target, "ticker": ticker_norm})

    # 이름·상장일·마켓·업종·메모는 종목풀이 바뀐다고 달라지지 않는다 — 옛 문서 값을 그대로
    # 가져간다. 원천에서 다시 받으면 종목당 9초가 더 든다.
    carried = {
        key: value
        for key, value in current.items()
        # name 은 add_stock 이 별도 인자로 받으므로 여기서 뺀다(중복 전달 방지).
        if key
        not in ("_id", "ticker", "ticker_type", "name", "is_deleted", "deleted_at", "deleted_reason", "created_at")
    }
    carried_name = str(current.get("name") or ticker_norm)

    from utils.cache_utils import move_cached_frame
    from utils.stock_list_io import add_stock

    # 옛 풀에서 먼저 뺀다 — 남겨 두면 '이미 다른 종목풀에 있습니다' 로 막힌다.
    # 가격 캐시·배치 계산값은 지우지 않고 새 풀로 옮긴다(같은 시세라 다시 받을 이유가 없다).
    if not hard_remove_stock(source, ticker_norm):
        raise RuntimeError(f"'{_pool_label(source)}' 에서 빼지 못했습니다: {ticker_norm}")
    try:
        if not add_stock(target, ticker_norm, name=carried_name, **carried):
            raise RuntimeError("대상 종목풀에 담지 못했습니다.")
        move_cached_frame(source, target, ticker_norm)
        for meta_coll in ("stock_cache_meta", "previous_stock_cache_meta"):
            # 대상 풀에 남아 있던 옛 계산값은 먼저 지운다 — (ticker_type, ticker) 유일 인덱스가
            # 걸려 있어 그대로 두면 이동이 중복 키로 실패한다. 옮겨오는 쪽이 실제 종목의 값이다.
            db[meta_coll].delete_many({"ticker_type": target, "ticker": ticker_norm})
            db[meta_coll].update_many({"ticker_type": source, "ticker": ticker_norm}, {"$set": {"ticker_type": target}})
    except Exception as exc:
        # 새 풀에 담지 못했으면 옛 풀로 되돌린다 — 이미 뺀 뒤라 되돌리지 않으면 종목이 사라진다.
        # 대상 풀 문서도 이동 전 상태로 돌린다. add_stock 이 성공한 뒤 그 다음 단계에서
        # 실패하면 종목이 출발·대상 양쪽에 활성으로 남는다(실제로 그렇게 됐다).
        try:
            if target_before is None:
                hard_remove_stock(target, ticker_norm)
            else:
                db.stock_meta.replace_one({"_id": target_before["_id"]}, target_before, upsert=True)
        except Exception as revert_error:
            get_app_logger().error("[이동] %s 대상 풀 되돌리기 실패: %s", ticker_norm, revert_error)
        try:
            restored = add_stock(source, ticker_norm, name=carried_name, **carried)
        except Exception as restore_error:
            restored = False
            get_app_logger().error("[이동] %s 복구 중 오류: %s", ticker_norm, restore_error)
        if not restored:
            # 조용히 넘기면 안 된다 — 종목이 어느 풀에도 없는 상태로 남는다.
            get_app_logger().error(
                "[이동] %s 복구 실패 — '%s' 에서 빠진 채로 남았습니다. 수동 재등록이 필요합니다.",
                ticker_norm,
                source,
            )
            raise RuntimeError(
                f"{ticker_norm} 이동에 실패했고 '{_pool_label(source)}' 로 되돌리지도 못했습니다. "
                f"이 종목은 지금 어느 종목풀에도 없습니다 — 다시 추가해 주세요. (원인: {exc})"
            ) from exc
        raise RuntimeError(f"{ticker_norm} 이동 실패: {exc}") from exc

    invalidate_pool_caches(source)
    invalidate_pool_caches(target)
    return {"ticker": ticker_norm, "from": source, "to": target, "bucket_id": bucket_value}


def load_deleted_stocks_table(ticker_type: str | None = None) -> dict[str, Any]:
    ticker_types = _load_ticker_types_payload()
    target_ticker_type = _pick_ticker_type(ticker_types, ticker_type)

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    docs = list(
        db.stock_meta.find(
            {
                "ticker_type": target_ticker_type,
                "is_deleted": True,
            },
            {
                "ticker": 1,
                "name": 1,
                "bucket": 1,
                "added_date": 1,
                "listing_date": 1,
                "1_week_avg_volume": 1,
                "1_day_earn_rate": 1,
                "1_week_earn_rate": 1,
                "2_week_earn_rate": 1,
                "1_month_earn_rate": 1,
                "3_month_earn_rate": 1,
                "6_month_earn_rate": 1,
                "12_month_earn_rate": 1,
                "deleted_at": 1,
                "deleted_reason": 1,
            },
        )
    )

    # 실시간 스냅샷 가져오기
    tickers = [doc.get("ticker", "") for doc in docs if doc.get("ticker")]
    config = _require_ticker_type_config(target_ticker_type)
    country_code = config.get("country_code", "kor")
    realtime_snapshot = {}
    try:
        realtime_snapshot = get_realtime_snapshot(country_code, tickers)
    except Exception:
        pass

    rows = sorted(
        [
            {
                "ticker": normalize_text(doc.get("ticker"), ""),
                "name": normalize_text(doc.get("name"), ""),
                "bucket_id": int(doc.get("bucket") or 1),
                "bucket_name": BUCKETS.get(int(doc.get("bucket") or 1), BUCKETS[1]),
                "added_date": normalize_text(doc.get("added_date"), "-"),
                "listing_date": normalize_text(doc.get("listing_date"), "-"),
                "week_volume": normalize_nullable_number(doc.get("1_week_avg_volume")),
                "return_1d": normalize_nullable_number(
                    realtime_snapshot.get(doc.get("ticker", ""), {}).get("changeRate")
                ),
                "괴리율": normalize_nullable_number(realtime_snapshot.get(doc.get("ticker", ""), {}).get("deviation")),
                "return_1w": normalize_nullable_number(doc.get("1_week_earn_rate")),
                "return_2w": normalize_nullable_number(doc.get("2_week_earn_rate")),
                "return_1m": normalize_nullable_number(doc.get("1_month_earn_rate")),
                "return_3m": normalize_nullable_number(doc.get("3_month_earn_rate")),
                "return_6m": normalize_nullable_number(doc.get("6_month_earn_rate")),
                "return_12m": normalize_nullable_number(doc.get("12_month_earn_rate")),
                "deleted_date": _format_deleted_date(doc.get("deleted_at")),
                "deleted_reason": normalize_text(doc.get("deleted_reason"), "-"),
            }
            for doc in docs
        ],
        key=lambda row: (row["bucket_id"], row["deleted_date"]),
        reverse=True,
    )

    return {
        "ticker_types": ticker_types,
        "rows": rows,
        "ticker_type": target_ticker_type,
    }


def restore_deleted_stocks(ticker_type: str, tickers: list[str]) -> int:
    type_norm = str(ticker_type or "").strip().lower()
    ticker_list = [str(ticker or "").strip().upper() for ticker in tickers if str(ticker or "").strip()]
    if not type_norm or not ticker_list:
        raise RuntimeError("복구할 종목을 선택하세요.")

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    now = datetime.now()
    result = db.stock_meta.update_many(
        {
            "ticker_type": type_norm,
            "ticker": {"$in": ticker_list},
            "is_deleted": True,
        },
        {
            "$set": {
                "is_deleted": False,
                "deleted_at": None,
                "deleted_reason": None,
                "added_date": now.date().isoformat(),
                "updated_at": now,
            }
        },
    )
    if result.modified_count > 0:
        invalidate_ticker_type_cache(type_norm)
        invalidate_pool_caches(type_norm)
    return int(result.modified_count)


def hard_delete_stocks(ticker_type: str, tickers: list[str]) -> int:
    type_norm = str(ticker_type or "").strip().lower()
    ticker_list = [str(ticker or "").strip().upper() for ticker in tickers if str(ticker or "").strip()]
    if not type_norm or not ticker_list:
        raise RuntimeError("삭제할 종목을 선택하세요.")

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    result = db.stock_meta.delete_many(
        {
            "ticker_type": type_norm,
            "ticker": {"$in": ticker_list},
            "is_deleted": True,
        }
    )

    # 캐시 도큐먼트도 함께 제거
    from utils.cache_utils import delete_cached_frame

    for ticker in ticker_list:
        try:
            delete_cached_frame(type_norm, ticker)
        except Exception:
            pass

    if result.deleted_count > 0:
        invalidate_ticker_type_cache(type_norm)
        invalidate_pool_caches(type_norm)
    return int(result.deleted_count)


def toggle_exclude_from_ranking(ticker_type: str, ticker: str, exclude: bool) -> None:
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    type_norm = str(ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()

    result = db.stock_meta.update_one(
        {
            "ticker_type": type_norm,
            "ticker": ticker_norm,
            "is_deleted": {"$ne": True},
        },
        {
            "$set": {
                "exclude_from_ranking": bool(exclude),
                "updated_at": datetime.now(),
            }
        },
    )

    if result.matched_count == 0:
        raise RuntimeError("수정할 종목을 찾을 수 없습니다.")

    # 캐시 무효화 (종목풀 리스트 캐시 재생성 및 랭킹 캐시 무효화)
    invalidate_ticker_type_cache(type_norm)
    invalidate_pool_caches(type_norm)
