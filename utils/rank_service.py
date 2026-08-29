from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from config import CACHE_TTL_COMPUTE, TOP_N_HOLD
from services.stock_cache_service import get_stock_cache_meta_map
from utils.data_loader import get_trading_days
from utils.ma_options import ma_options_payload


def _pool_country(ticker_type: str) -> str:
    """풀 설정의 국가 — 이평선 선택지가 국가별이라 응답마다 실어 보낸다."""
    from utils.settings_loader import get_ticker_type_settings

    return str((get_ticker_type_settings(ticker_type) or {}).get("country_code") or "").strip().lower()


from utils.market_cap_rank import market_cap_rank_of
from utils.rankings import (
    MONTHLY_RETURN_LABEL_COUNT,
    build_effective_ma_rules,
    build_ticker_type_rankings,
    get_recent_monthly_return_labels,
)
from utils.stock_list_io import get_etfs
from utils.ticker_registry import load_ticker_type_configs, pick_default_ticker_type
from utils.ttl_cache import TtlCache

_RankCacheKey = tuple[str, tuple[tuple[int, int], ...]]
_RANK_DATA_CACHE = TtlCache(CACHE_TTL_COMPUTE, name="rank_data")


def _build_rank_cache_key(
    ticker_type: str,
    ma_rules: list[dict[str, Any]],
) -> _RankCacheKey:
    ma_rule_key = tuple(
        (
            int(rule.get("short_ma_days") or 0),
            int(rule.get("long_ma_days") or 0),
        )
        for rule in ma_rules
    )
    return (ticker_type, ma_rule_key)


def invalidate_rank_data_cache(ticker_type: str | None = None) -> None:
    """랭킹 응답 메모리 캐시를 무효화한다."""

    if ticker_type is None:
        _RANK_DATA_CACHE.invalidate()
        return

    target = str(ticker_type or "").strip().lower()
    if not target:
        return

    _RANK_DATA_CACHE.invalidate(lambda cache_key: cache_key[0] == target)


def _serialize_datetime(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return str(value)

    if pd.isna(timestamp):
        return None
    return timestamp.isoformat()


def _serialize_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if pd.isna(value):
            return None
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, list):
        return [_serialize_value(item) for item in value]

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return _serialize_datetime(value)

    return value


def _serialize_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        row = {str(key): _serialize_value(value) for key, value in record.items()}
        rows.append(row)
    return rows


def _format_listed_date(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if len(normalized) == 10 and normalized[4] == "-" and normalized[7] == "-":
        return normalized
    if len(normalized) == 8 and normalized.isdigit():
        return f"{normalized[:4]}-{normalized[4:6]}-{normalized[6:8]}"
    return normalized


def _apply_industry_labels(dataframe: pd.DataFrame, ticker_type: str) -> pd.DataFrame:
    """업종을 붙인다 — 공용 맵(utils/industry_map)이 단일 소스.

    한국 종목은 네이버 분류(한국어 원본), 미국은 지수 구성종목의 yfinance 분류다.
    번역하지 않는다 — 각 시장에서 쓰는 용어 그대로 보여주는 게 정확하다.
    분류가 없는 종목(ETF 등)은 빈 값이며, 화면은 값이 있는 풀에서만 컬럼을 노출한다.
    """
    if dataframe.empty or "티커" not in dataframe.columns:
        return dataframe

    from utils.industry_map import industry_map

    industry_by = {str(k).strip().upper(): v for k, v in industry_map(ticker_type).items()}

    upper = dataframe["티커"].astype(str).str.strip().str.upper()
    dataframe["업종"] = upper.map(lambda t: industry_by.get(t, ""))
    return dataframe


def _load_trade_value_mult(ticker_type: str, tickers: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    """티커별 거래대금 배수(20일 평균 대비) — `(판정 기준, 실시간 합산)` 두 벌.

    **판정 기준**은 가격 캐시 배치가 저장해 둔 값이다(KRX 정규시장 확정). 장중에는 오늘
    확정값이 아직 없으므로 실시간 값으로 채운다 — 마감 후 배치가 돌면 확정값으로 바뀐다.

    **실시간 합산**은 토스 스냅샷 기준이다. 토스는 KRX 에 더해 대체거래소(NXT) 거래분까지
    합산해서 준다(응답의 `nxtSinglePrice`). 그래서 같은 날이라도 KRX 확정값보다 크다
    (씨젠 2026-08-28: KRX 393억 → 3.20배, 합산 711억 → 4.57배). 어느 쪽도 틀린 값이
    아니라 **재는 범위가 다르다** — 화면이 둘을 나란히 보여주고 판정은 확정값으로 한다.

    배치가 안 돌았거나 20일치가 없는 종목은 키가 없다 — 화면은 '-' 로 둔다.
    """
    if not tickers:
        return {}, {}
    try:
        from utils.db_manager import get_db_connection

        db = get_db_connection()
        if db is None:
            return {}
        docs = list(
            db.stock_meta.find(
                {"ticker_type": ticker_type, "ticker": {"$in": tickers}},
                {"_id": 0, "ticker": 1, "trade_value_mult": 1, "trade_value_sum19": 1},
            )
        )
    except Exception:
        return {}, {}

    result = {
        str(doc["ticker"]).strip().upper(): float(doc["trade_value_mult"])
        for doc in docs
        if doc.get("trade_value_mult") is not None
    }
    sum19 = {
        str(doc["ticker"]).strip().upper(): float(doc["trade_value_sum19"])
        for doc in docs
        if doc.get("trade_value_sum19") is not None
    }
    live = _live_trade_value_mult(ticker_type, sum19)
    # 배치 값이 없는 종목(오늘 상장 등)은 실시간이라도 보여준다.
    for ticker, value in live.items():
        result.setdefault(ticker, value)
    return result, live


def _live_trade_value_mult(ticker_type: str, sum19: dict[str, float]) -> dict[str, float]:
    """오늘 누적 거래대금으로 다시 계산한 배수. 국내 상장 종목만 해당한다.

    분모는 배치가 넘겨준 직전 19거래일 합에 오늘을 더한 20일 평균이다 — 확정된 날의
    계산식과 같다. 조회에 실패하거나 오늘 값이 없는 종목은 비워 두고 배치 값을 쓴다.
    """
    from utils.settings_loader import get_ticker_type_settings

    settings = get_ticker_type_settings(ticker_type) or {}
    if str(settings.get("country_code") or "").strip().lower() != "kor" or not sum19:
        return {}
    try:
        from utils.data_loader import fetch_toss_kr_stock_snapshot

        snapshot = fetch_toss_kr_stock_snapshot(list(sum19))
    except Exception:
        return {}

    out: dict[str, float] = {}
    for ticker, base_sum in sum19.items():
        today = (snapshot.get(ticker) or {}).get("tradeValue")
        if today is None or float(today) <= 0:
            continue
        base = (base_sum + float(today)) / 20
        if base > 0:
            out[ticker] = float(today) / base
    return out


def _apply_rank_info_cache(dataframe: pd.DataFrame, ticker_type: str) -> pd.DataFrame:
    """정보 컬럼(배당률·보수·순자산·상장일)을 메타 캐시에서 붙인다.

    backtest_stats(MDD·소르티노·is_partial)는 순위 계산(rankings)이 장기 이평선 파생
    기간으로 실시간 계산해 행에 이미 담겨 있으므로 여기서 건드리지 않는다.
    """
    if dataframe.empty:
        return dataframe

    tickers = [
        str(record.get("티커") or "").strip().upper()
        for record in dataframe.to_dict(orient="records")
        if str(record.get("티커") or "").strip()
    ]
    # 거래대금 배수는 가격 캐시 배치가 stock_meta 에 미리 넣어둔 값을 읽는다.
    # 여기서 직접 계산하려면 거래량이 든 큰 blob 을 받아야 해서 순위 계산이 3초 더 걸린다.
    mult_map, live_mult_map = _load_trade_value_mult(ticker_type, tickers)

    cache_map = get_stock_cache_meta_map(ticker_type, tickers)
    if not cache_map:
        enriched = dataframe.copy()
        enriched["배당률"] = None
        enriched["보수"] = None
        enriched["순자산총액"] = None
        enriched["상장일"] = None
        enriched["시총순위"] = None
        if "티커" in enriched.columns:
            keys = enriched["티커"].map(lambda t: str(t or "").strip().upper())
            enriched["거래대금"] = keys.map(mult_map.get)
            enriched["거래대금(실시간)"] = keys.map(live_mult_map.get)
        else:
            enriched["거래대금"] = None
            enriched["거래대금(실시간)"] = None
        enriched.attrs.update(dict(dataframe.attrs))
        return enriched

    rows: list[dict[str, Any]] = []
    for row in dataframe.to_dict(orient="records"):
        ticker = str(row.get("티커") or "").strip().upper()
        doc = cache_map.get(ticker, {})
        meta_cache = doc.get("meta_cache") if isinstance(doc, dict) else {}
        meta_cache = meta_cache if isinstance(meta_cache, dict) else {}
        row["배당률"] = meta_cache.get("dividend_yield_ttm")
        row["보수"] = meta_cache.get("expense_ratio")
        row["순자산총액"] = meta_cache.get("total_net_assets")
        row["시총순위"] = market_cap_rank_of(meta_cache)  # 배치 B 가 적어 둔 시장 전체 시총 순위(개별주 풀만 값 있음)
        row["상장일"] = _format_listed_date(meta_cache.get("listed_date") or row.get("상장일"))
        row["거래대금"] = mult_map.get(ticker)
        # 대체거래소(NXT) 합산 실시간 배수 — 화면이 확정값 옆에 괄호로 보여준다.
        row["거래대금(실시간)"] = live_mult_map.get(ticker)

        rows.append(row)

    enriched = pd.DataFrame(rows)
    enriched.attrs.update(dict(dataframe.attrs))
    return enriched


def _build_configs_payload() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configs = load_ticker_type_configs()
    if not configs:
        raise ValueError("사용 가능한 종목풀이 없습니다.")

    default_config = pick_default_ticker_type(configs)
    payload = [
        {
            "ticker_type": str(cfg["ticker_type"]),
            "order": int(cfg["order"]),
            "name": str(cfg["name"]),
            "icon": str(cfg.get("icon") or ""),
            "country_code": str(cfg.get("country_code") or ""),
            # 풀 성격(stock/etf) — 미설정이면 빈 값(화면이 행 값으로 추정).
            "pool_kind": str(cfg.get("pool_kind") or ""),
            # 시스템 공통 보유 종목 수(config) — 풀별 설정은 폐기했다.
            "top_n_hold": TOP_N_HOLD,
            "currency": str(cfg["settings"].get("currency") or ""),
        }
        for cfg in configs
    ]
    return payload, default_config


def _build_missing_ticker_labels(ticker_type: str, missing_tickers: list[str]) -> list[str]:
    if not missing_tickers:
        return []

    ticker_name_map = {
        str(item.get("ticker") or "").strip().upper(): str(item.get("name") or "").strip()
        for item in get_etfs(ticker_type)
        if str(item.get("ticker") or "").strip()
    }

    labels: list[str] = []
    for ticker in missing_tickers:
        normalized_ticker = str(ticker or "").strip().upper()
        name = ticker_name_map.get(normalized_ticker, "")
        if name:
            labels.append(f"{name}({normalized_ticker})")
        else:
            labels.append(normalized_ticker)
    return labels


def _build_missing_ticker_rows(
    selected_ticker_type: str,
    missing_tickers: list[str],
    is_all: bool,
) -> list[dict[str, Any]]:
    """캐시 누락 ETF를 그리드에서 선택·삭제할 수 있도록 placeholder 행으로 만든다.

    추세/순위 등 지표는 None(그리드에 "-")으로 두고, 식별·삭제에 필요한
    티커/종목명/source_ticker_type 만 채운다. (캐시 누락으로 순위 계산이 막혀도
    사용자가 신규 ETF를 제거할 수 있게 한다 — 경고 배너는 그대로 표시됨.)
    """
    if not missing_tickers:
        return []

    name_maps: dict[str, dict[str, str]] = {}

    def _name_of(src: str, ticker: str) -> str:
        cached = name_maps.get(src)
        if cached is None:
            cached = {
                str(item.get("ticker") or "").strip().upper(): str(item.get("name") or "").strip()
                for item in get_etfs(src)
                if str(item.get("ticker") or "").strip()
            }
            name_maps[src] = cached
        return cached.get(ticker, "")

    rows: list[dict[str, Any]] = []
    for entry in missing_tickers:
        text = str(entry or "").strip()
        if is_all and ":" in text:
            source, _, ticker = text.partition(":")
            source = source.strip().lower() or selected_ticker_type
            ticker = ticker.strip().upper()
        else:
            source = selected_ticker_type
            ticker = text.upper()
        if not ticker:
            continue
        rows.append(
            {
                "티커": ticker,
                "종목명": _name_of(source, ticker),
                "메모": "",
                "source_ticker_type": source,
                "순번": "-",
                "순위": None,
                "이전순위": None,
                "1주순위": None,
                "버킷": "",
                "bucket": None,
                "상장일": "-",
                "추세": None,
                "점수": None,
                "보유": "",
                "보유대상": False,
                "현재가": None,
                "exclude_from_ranking": False,
                "cache_missing": True,
            }
        )
    return rows


def _rows_with_missing_placeholders(
    dataframe: pd.DataFrame,
    selected_ticker_type: str,
    is_all: bool,
) -> list[dict[str, Any]]:
    """직렬화된 행 + 캐시 누락 ETF placeholder 행. 누락 종목을 그리드에서 제거할 수 있게 한다."""
    rows = _serialize_rows(dataframe)
    missing = [str(item) for item in (dataframe.attrs.get("missing_tickers") or [])]
    if not missing:
        return rows
    existing = {str(row.get("티커") or "").strip().upper() for row in rows}
    placeholders = [
        row
        for row in _build_missing_ticker_rows(selected_ticker_type, missing, is_all)
        if str(row.get("티커") or "").strip().upper() not in existing
    ]
    return rows + placeholders


def _get_nth_previous_trading_day(
    country_code: str,
    reference_date: pd.Timestamp | None,
    offset: int,
) -> pd.Timestamp | None:
    if offset < 1:
        raise ValueError("이전 거래일 offset은 1 이상이어야 합니다.")
    if reference_date is None:
        return None

    try:
        reference = pd.Timestamp(reference_date)
    except Exception:
        return None
    if pd.isna(reference):
        return None
    reference = reference.normalize()
    search_start = reference - pd.DateOffset(days=max(15, offset * 4 + 10))
    trading_days = get_trading_days(
        search_start.strftime("%Y-%m-%d"),
        reference.strftime("%Y-%m-%d"),
        country_code,
    )
    previous_days = sorted(
        {pd.Timestamp(day).normalize() for day in trading_days if pd.Timestamp(day).normalize() < reference}
    )
    if len(previous_days) < offset:
        return None
    return previous_days[-offset]


def _get_previous_trading_day(country_code: str, reference_date: pd.Timestamp | None) -> pd.Timestamp | None:
    return _get_nth_previous_trading_day(country_code, reference_date, 1)


def _build_rank_map(dataframe: pd.DataFrame) -> dict[str, int]:
    rank_map: dict[str, int] = {}
    if dataframe.empty:
        return rank_map

    for index, row in enumerate(dataframe.to_dict(orient="records"), start=1):
        ticker = str(row.get("티커") or "").strip().upper()
        if ticker:
            rank_map[ticker] = index
    return rank_map


def _normalize_trend_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(score):
        return None
    return score


def _build_score_ranked_rows(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """순위 번호(「순위」 컬럼)를 매긴다 — 기준은 순위 점수(`core.strategy.scoring.rank_score`).

    표시용 「장기」(이격)·「단기」(단기이격)는 원천 값 그대로 두고, 줄 세우기만 점수로 한다.
    벤치마크 대비 표시(`is_below_benchmark`)도 같은 점수로 비교해야 순위와 어긋나지 않는다.
    """
    rows_with_index: list[dict[str, Any]] = []
    for index, row in enumerate(dataframe.to_dict(orient="records")):
        rows_with_index.append(
            {
                **row,
                "추세": _normalize_trend_value(row.get("추세")),
                "점수": _normalize_trend_value(row.get("점수")),
                "__base_index": index,
            }
        )

    rows_with_index.sort(
        key=lambda row: (
            1 if row.get("점수") is None else 0,
            -(float(row["점수"]) if row.get("점수") is not None else 0.0),
            int(row["__base_index"]),
        )
    )

    bm_score = None
    for row in rows_with_index:
        if row.get("is_benchmark") and row.get("점수") is not None:
            bm_score = float(row["점수"])
            break

    ranked_rows: list[dict[str, Any]] = []
    current_rank = 1
    for row in rows_with_index:
        normalized = dict(row)
        normalized.pop("__base_index", None)
        score = normalized.get("점수")

        is_below_bm = False
        if bm_score is not None and score is not None:
            is_below_bm = float(score) < bm_score

        normalized["is_below_benchmark"] = is_below_bm

        normalized["순위"] = current_rank
        current_rank += 1
        ranked_rows.append(normalized)
    return ranked_rows


def _build_rank_map_from_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    rank_map: dict[str, int] = {}
    for row in rows:
        ticker = _build_rank_row_key(row)
        rank = row.get("순위")
        if ticker and isinstance(rank, int):
            rank_map[ticker] = rank
    return rank_map


def _build_rank_row_key(row: dict[str, Any] | pd.Series) -> str:
    ticker = str(row.get("티커") or "").strip().upper()
    source_ticker_type = str(row.get("source_ticker_type") or "").strip().lower()
    if source_ticker_type:
        return f"{source_ticker_type}:{ticker}"
    return ticker


def _strip_pool_order_prefix(name: str) -> str:
    text = str(name or "").strip()
    if ". " in text:
        prefix, _, suffix = text.partition(". ")
        if prefix.isdigit() and suffix.strip():
            return suffix.strip()
    return text


def load_rank_toolbar_data(ticker_type: str | None = None) -> dict[str, Any]:
    configs_payload, default_config = _build_configs_payload()
    target = str(ticker_type or "").strip().lower()
    available_ids = [str(cfg["ticker_type"]).lower() for cfg in configs_payload]

    if target and target in available_ids:
        selected_ticker_type = target
    else:
        selected_ticker_type = str(default_config["ticker_type"]).strip().lower()

    selected_config = next(
        (cfg for cfg in configs_payload if str(cfg["ticker_type"]).lower() == selected_ticker_type), None
    )
    if selected_config is None:
        raise ValueError("선택된 종목풀 설정을 찾을 수 없습니다.")

    ma_rules = build_effective_ma_rules(selected_ticker_type, None)

    return {
        "ticker_types": configs_payload,
        "ticker_type": selected_ticker_type,
        "ma_rules": ma_rules,
        # 이평선 일수 선택지 — 백엔드 상수가 단일 소스(풀 국가별).
        **ma_options_payload(_pool_country(selected_ticker_type)),
        # 종목 수·업종 상한 선택지도 같은 단일 소스(`config`)다. 화면에 복사본을 두면
        # config 를 고쳐도 이 화면만 옛 목록이 남는다(실제로 업종 상한 4 가 빠져 있었다).
        # 업종 상한의 None(제한 없음)은 쿼리로 넘길 수 있게 -1 로 바꿔 보낸다.
    }


def _compute_rank_data_payload(
    *,
    configs_payload: list[dict[str, Any]],
    selected_ticker_type: str,
    country_code: str,
    ma_rules: list[dict[str, Any]],
) -> dict[str, Any]:
    # 기준일은 항상 오늘이다 — 아래에서 데이터가 실제로 어느 날짜인지(effective) 다시 읽는다.
    dataframe = build_ticker_type_rankings(selected_ticker_type, ma_rules=ma_rules)
    effective_as_of_date: pd.Timestamp | None = None
    raw_as_of_date = dataframe.attrs.get("as_of_date")
    if raw_as_of_date is not None:
        try:
            parsed_as_of_date = pd.Timestamp(raw_as_of_date)
        except Exception:
            parsed_as_of_date = None
        if parsed_as_of_date is not None and not pd.isna(parsed_as_of_date):
            effective_as_of_date = parsed_as_of_date.normalize()
    current_rows = _build_score_ranked_rows(dataframe)
    current_rank_map = _build_rank_map_from_rows(current_rows)
    previous_rank_map: dict[str, int] = {}
    weekly_rank_map: dict[str, int] = {}
    raw_latest_trading_day = dataframe.attrs.get("latest_trading_day")
    previous_trading_day = _get_previous_trading_day(country_code, raw_latest_trading_day)
    weekly_rank_trading_day = _get_nth_previous_trading_day(country_code, raw_latest_trading_day, 5)
    if previous_trading_day is not None:
        previous_dataframe = build_ticker_type_rankings(
            selected_ticker_type,
            ma_rules=ma_rules,
            as_of_date=previous_trading_day,
        )
        previous_rows = _build_score_ranked_rows(previous_dataframe)
        previous_rank_map = _build_rank_map_from_rows(previous_rows)
    if weekly_rank_trading_day is not None:
        weekly_dataframe = build_ticker_type_rankings(
            selected_ticker_type,
            ma_rules=ma_rules,
            as_of_date=weekly_rank_trading_day,
        )
        weekly_rows = _build_score_ranked_rows(weekly_dataframe)
        weekly_rank_map = _build_rank_map_from_rows(weekly_rows)

    if current_rows:
        dataframe_attrs = dict(dataframe.attrs)
        enriched_rows: list[dict[str, Any]] = []
        for row in current_rows:
            row_key = _build_rank_row_key(row)
            row["순위"] = current_rank_map.get(row_key)
            row["이전순위"] = previous_rank_map.get(row_key)
            row["1주순위"] = weekly_rank_map.get(row_key)
            enriched_rows.append(row)
        dataframe = pd.DataFrame(enriched_rows)
        dataframe.attrs.update(dataframe_attrs)

    dataframe = _apply_rank_info_cache(dataframe, selected_ticker_type)
    dataframe = _apply_industry_labels(dataframe, selected_ticker_type)

    return {
        "ticker_types": configs_payload,
        "ticker_type": selected_ticker_type,
        "ma_rules": ma_rules,
        **ma_options_payload(_pool_country(selected_ticker_type)),
        "as_of_date": _serialize_datetime(effective_as_of_date),
        "monthly_return_labels": get_recent_monthly_return_labels(
            MONTHLY_RETURN_LABEL_COUNT,
            reference_date=effective_as_of_date,
        ),
        "rows": _rows_with_missing_placeholders(dataframe, selected_ticker_type, False),
        "cache_blocked": bool(dataframe.attrs.get("cache_blocked", False)),
        "latest_trading_day": _serialize_datetime(dataframe.attrs.get("latest_trading_day")),
        "cache_updated_at": _serialize_datetime(dataframe.attrs.get("cache_updated_at")),
        "ranking_computed_at": _serialize_datetime(dataframe.attrs.get("ranking_computed_at")),
        "realtime_fetched_at": _serialize_datetime(dataframe.attrs.get("realtime_fetched_at")),
        "previous_trading_day": _serialize_datetime(previous_trading_day),
        "weekly_rank_trading_day": _serialize_datetime(weekly_rank_trading_day),
        "missing_tickers": [str(item) for item in (dataframe.attrs.get("missing_tickers") or [])],
        "missing_ticker_labels": _build_missing_ticker_labels(
            selected_ticker_type,
            [str(item) for item in (dataframe.attrs.get("missing_tickers") or [])],
        ),
        "stale_tickers": [str(item) for item in (dataframe.attrs.get("stale_tickers") or [])],
    }


def load_rank_data(
    *,
    ticker_type: str | None = None,
    ma_rule_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configs_payload, default_config = _build_configs_payload()

    # 요청받은 ticker_type이 현재 유효한 목록 내에 있는지 검사 (없으면 기본값 사용)
    target = str(ticker_type or "").strip().lower()
    available_ids = [str(cfg["ticker_type"]).lower() for cfg in configs_payload]

    if target and target in available_ids:
        selected_ticker_type = target
    else:
        selected_ticker_type = str(default_config["ticker_type"]).strip().lower()
    selected_config = next(
        (cfg for cfg in configs_payload if str(cfg["ticker_type"]).lower() == selected_ticker_type), None
    )
    country_code = str(selected_config.get("country_code") or "") if selected_config else ""

    ma_rules = build_effective_ma_rules(selected_ticker_type, ma_rule_override)
    if selected_config is None:
        raise ValueError("선택된 종목풀 설정을 찾을 수 없습니다.")

    cache_key = _build_rank_cache_key(selected_ticker_type, ma_rules)
    return _RANK_DATA_CACHE.get_or_compute(
        cache_key,
        lambda: _compute_rank_data_payload(
            configs_payload=configs_payload,
            selected_ticker_type=selected_ticker_type,
            country_code=country_code,
            ma_rules=ma_rules,
        ),
    )
