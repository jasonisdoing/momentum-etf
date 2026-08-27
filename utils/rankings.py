from __future__ import annotations

import re
from datetime import datetime
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from config import (
    BUCKET_MAPPING,
    MARKET_SCHEDULES,
    METRIC_WINDOW_MONTHS,
)
from core.strategy.scoring import (
    build_composite_rank_scores,
    compute_trend_frame,
    drawdown_from_high_pct,
    hold_eligible,
    rank_score,
    select_holdings,
)
from services.price_service import get_realtime_snapshot, get_realtime_snapshot_meta
from utils.asx_ticker import ensure_asx_prefix
from utils.cache_utils import (
    load_cached_close_series_bulk_with_fallback,
    load_cached_updated_at_bulk_with_fallback,
)
from utils.data_loader import get_latest_trading_day, get_trading_days
from utils.logger import get_app_logger
from utils.moving_averages import get_moving_average_type
from utils.perf_metrics import single_stock_backtest_stats
from utils.pool_settings_store import get_pool_benchmark_ticker
from utils.settings_loader import AccountSettingsError, get_ticker_type_settings
from utils.stock_list_io import get_etfs

logger = get_app_logger()
MONTHLY_RETURN_LABEL_COUNT = 13
RSI_PERIOD = 14


def _build_ma_rule_score_column() -> str:
    return "추세(기본)"


def _normalize_ma_rule(ticker_type: str, ma_rule_raw: Any) -> dict[str, Any]:
    if not isinstance(ma_rule_raw, dict):
        raise AccountSettingsError(f"'{ticker_type}' 설정의 MA 규칙 항목은 객체여야 합니다.")

    short_raw = ma_rule_raw.get("SHORT_MA_DAYS")
    long_raw = ma_rule_raw.get("LONG_MA_DAYS")
    if short_raw is None:
        raise AccountSettingsError(f"'{ticker_type}' 설정의 'SHORT_MA_DAYS'가 누락되었습니다.")
    if long_raw is None:
        raise AccountSettingsError(f"'{ticker_type}' 설정의 'LONG_MA_DAYS'가 누락되었습니다.")
    try:
        short_days = int(short_raw)
        long_days = int(long_raw)
    except (TypeError, ValueError) as exc:
        raise AccountSettingsError(
            f"'{ticker_type}' 설정의 MA 일수는 정수여야 합니다: SHORT={short_raw}, LONG={long_raw}"
        ) from exc
    # 선택지 포함 여부는 종목풀 설정 **저장** 때만 검사한다(pool_settings_store). 읽기에서 막으면
    # 선택지가 바뀐 직후 옛 값이 남은 풀 하나 때문에 순위·배치가 전부 죽는다.
    if short_days < 1 or long_days < 1:
        raise AccountSettingsError(
            f"'{ticker_type}' 설정의 MA 일수는 1 이상이어야 합니다: SHORT={short_days}, LONG={long_days}"
        )

    return {
        "order": 1,
        "short_ma_days": short_days,
        "long_ma_days": long_days,
        "score_column": _build_ma_rule_score_column(),
        "ma_type": get_moving_average_type(),
    }


def get_ticker_type_ma_rules(ticker_type: str) -> list[dict[str, Any]]:
    """종목풀 설정의 단일 MA 파라미터를 내부 규칙 리스트로 변환한다."""
    settings = get_ticker_type_settings(ticker_type)
    for key in ("SHORT_MA_DAYS", "LONG_MA_DAYS"):
        if key not in settings:
            raise AccountSettingsError(f"'{ticker_type}' 설정에 필수 항목 '{key}'가 누락되었습니다.")
    return [
        _normalize_ma_rule(
            ticker_type,
            {
                "SHORT_MA_DAYS": settings["SHORT_MA_DAYS"],
                "LONG_MA_DAYS": settings["LONG_MA_DAYS"],
            },
        )
    ]


def build_effective_ma_rules(
    ticker_type: str,
    override: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    base_rule = get_ticker_type_ma_rules(ticker_type)[0]
    if not override:
        return [base_rule]
    return [
        _normalize_ma_rule(
            ticker_type,
            {
                "SHORT_MA_DAYS": override.get("short_ma_days") or base_rule["short_ma_days"],
                "LONG_MA_DAYS": override.get("long_ma_days") or base_rule["long_ma_days"],
            },
        )
    ]


def _calculate_rsi(close_series: pd.Series, period: int) -> float | None:
    series = pd.to_numeric(close_series, errors="coerce").dropna()
    if len(series) < period + 1:
        return None

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    if avg_loss.empty or pd.isna(avg_loss.iloc[-1]):
        return None
    if float(avg_loss.iloc[-1]) == 0.0:
        return 100.0
    rs = float(avg_gain.iloc[-1]) / float(avg_loss.iloc[-1])
    return 100.0 - (100.0 / (1.0 + rs))


def _get_market_timezone(country_code: str) -> ZoneInfo:
    schedule = MARKET_SCHEDULES.get(str(country_code or "").strip().lower())
    timezone_name = str((schedule or {}).get("timezone") or "").strip()
    if not timezone_name:
        return ZoneInfo("Asia/Seoul")
    return ZoneInfo(timezone_name)


def _normalize_market_timestamp(
    value: datetime | pd.Timestamp | None,
    country_code: str,
    *,
    assume_utc: bool,
) -> pd.Timestamp | None:
    if value is None:
        return None

    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None

    market_tz = _get_market_timezone(country_code)
    if ts.tzinfo is None:
        if assume_utc:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_localize(market_tz)

    return ts.tz_convert(market_tz).tz_localize(None)


def _build_blocked_rankings_result(
    *,
    latest_trading_day: pd.Timestamp,
    cache_updated_at: pd.Timestamp | None,
    missing_tickers: list[str],
    stale_tickers: list[str],
) -> pd.DataFrame:
    blocked = pd.DataFrame()
    blocked.attrs["cache_blocked"] = True
    blocked.attrs["latest_trading_day"] = latest_trading_day
    blocked.attrs["cache_updated_at"] = cache_updated_at
    blocked.attrs["missing_tickers"] = missing_tickers
    blocked.attrs["stale_tickers"] = stale_tickers
    return blocked


def _get_latest_trading_day_for_reference(country_code: str, reference_date: pd.Timestamp) -> pd.Timestamp:
    reference = pd.Timestamp(reference_date).normalize()
    today_korea = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    if reference >= today_korea:
        return get_latest_trading_day(country_code).normalize()

    start_date = (reference - pd.DateOffset(days=10)).strftime("%Y-%m-%d")
    end_date = reference.strftime("%Y-%m-%d")
    trading_days = get_trading_days(start_date, end_date, country_code)
    if trading_days:
        return max(trading_days).normalize()
    return reference


# 마감 후 가격 캐시가 채워지기까지 기다려 주는 시간. 캐시 배치는 매시 20분에 돌고 실행에
# 7분 안팎이 걸려서, 마감 직후 최대 한 시간 남짓은 아직 당일 봉이 없는 게 정상이다.
_CACHE_REFRESH_GRACE = pd.Timedelta(minutes=90)


def _expected_cache_trading_day(country_code: str, latest_trading_day: pd.Timestamp) -> pd.Timestamp:
    """가격 캐시가 **이미 갖고 있어야 마땅한** 거래일.

    최신 거래일은 시장 마감(+여유) 시각에 바뀌는데 캐시는 그다음 배치에서야 채워진다.
    그 사이를 '오래된 캐시' 로 판정하면 매일 마감 직후 순위 화면 전체가 막힌다
    (한국은 16:00~16:27). 배치가 한 번 돌 시간을 준 뒤에도 없으면 그때 진짜 문제다.
    """
    schedule = (MARKET_SCHEDULES or {}).get((country_code or "").strip().lower())
    if not isinstance(schedule, dict):
        return latest_trading_day
    tz_name = str(schedule.get("timezone") or "").strip()
    close_time = schedule.get("close")
    if not tz_name or close_time is None:
        return latest_trading_day
    try:
        closed_at = pd.Timestamp(
            f"{latest_trading_day.date()} {close_time.hour:02d}:{close_time.minute:02d}", tz=tz_name
        ) + pd.Timedelta(minutes=int(schedule.get("close_offset_minutes") or 0))
        if pd.Timestamp.now(tz=tz_name) >= closed_at + _CACHE_REFRESH_GRACE:
            return latest_trading_day
        previous = get_trading_days(
            (latest_trading_day - pd.DateOffset(days=10)).strftime("%Y-%m-%d"),
            (latest_trading_day - pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
            country_code,
        )
    except Exception:
        return latest_trading_day
    return max(previous).normalize() if previous else latest_trading_day


def _slice_close_series_to_date(close_series: pd.Series | None, cutoff_date: pd.Timestamp) -> pd.Series | None:
    if close_series is None or close_series.empty:
        return close_series

    normalized = close_series.copy()
    normalized.index = pd.to_datetime(normalized.index)
    sliced = normalized.loc[normalized.index.normalize() <= pd.Timestamp(cutoff_date).normalize()]
    if sliced.empty:
        return None
    return sliced.sort_index()


def _calc_period_return(close_series: pd.Series, days: int) -> float | None:
    series = pd.to_numeric(close_series, errors="coerce").dropna()
    if series.empty:
        return None

    current = float(series.iloc[-1])
    if current <= 0:
        return None

    if len(series) > days:
        previous = float(series.iloc[-(days + 1)])
        if previous > 0:
            return (current / previous - 1.0) * 100.0

    if days == 252 and len(series) >= 240:
        previous = float(series.iloc[0])
        if previous > 0:
            return (current / previous - 1.0) * 100.0

    return None


def get_recent_monthly_return_labels(
    count: int,
    reference_date: pd.Timestamp | None = None,
) -> list[str]:
    base_month = (reference_date or pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)).to_period("M")
    return [f"{(base_month - offset).strftime('%Y-%m')}(%)" for offset in range(count)]


def build_recent_monthly_return_metrics(
    close_series: pd.Series | None,
    *,
    reference_date: pd.Timestamp | None = None,
    labels: list[str] | None = None,
) -> dict[str, float | None]:
    return _build_monthly_return_metrics(
        close_series,
        reference_date=reference_date,
        labels=labels,
    )


def _build_monthly_return_metrics(
    close_series: pd.Series | None,
    *,
    reference_date: pd.Timestamp | None = None,
    labels: list[str] | None = None,
) -> dict[str, float | None]:
    labels = labels or get_recent_monthly_return_labels(
        MONTHLY_RETURN_LABEL_COUNT,
        reference_date=reference_date,
    )
    empty_metrics = {label: None for label in labels}
    if close_series is None:
        return empty_metrics

    series = pd.to_numeric(close_series, errors="coerce").dropna()
    if series.empty:
        return empty_metrics

    normalized = series.copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized = normalized.sort_index()
    month_end_series = normalized.groupby(normalized.index.to_period("M")).last()
    current_month = (reference_date or pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)).to_period("M")
    metrics: dict[str, float | None] = {}

    for label in labels:
        match = re.fullmatch(r"(\d{4}-\d{2})\(%\)", label)
        if not match:
            metrics[label] = None
            continue

        month_period = pd.Period(match.group(1), freq="M")
        prev_month_close = month_end_series.get(month_period - 1)
        if prev_month_close is None or pd.isna(prev_month_close):
            metrics[label] = None
            continue

        target_close = normalized.iloc[-1] if month_period == current_month else month_end_series.get(month_period)
        if target_close is None or pd.isna(target_close):
            metrics[label] = None
            continue

        prev_value = float(prev_month_close)
        target_value = float(target_close)
        metrics[label] = None if prev_value <= 0 else ((target_value / prev_value) - 1.0) * 100.0

    return metrics


def _extract_price_metrics_from_close_series(
    close_series: pd.Series | None,
    *,
    reference_date: pd.Timestamp | None = None,
    monthly_labels: list[str] | None = None,
) -> dict[str, Any]:
    monthly_return_metrics = _build_monthly_return_metrics(
        close_series,
        reference_date=reference_date,
        labels=monthly_labels,
    )
    empty_result = {
        "현재가": None,
        "괴리율": None,
        "일간(%)": None,
        "1주(%)": None,
        "2주(%)": None,
        "3주(%)": None,
        "4주(%)": None,
        "1달(%)": None,
        "2달(%)": None,
        "3달(%)": None,
        "4달(%)": None,
        "5달(%)": None,
        "6달(%)": None,
        "7달(%)": None,
        "8달(%)": None,
        "9달(%)": None,
        "10달(%)": None,
        "11달(%)": None,
        "12달(%)": None,
        "24달(%)": None,
        "36달(%)": None,
        "고점": None,
        "추세(3달)": [],
        "RSI": None,
        **monthly_return_metrics,
    }
    if close_series is None:
        return empty_result

    series = pd.to_numeric(close_series, errors="coerce").dropna()
    if series.empty:
        return empty_result

    current_price = float(series.iloc[-1])
    daily_pct = None
    if len(series) > 1:
        prev_close = float(series.iloc[-2])
        if prev_close > 0:
            daily_pct = ((current_price / prev_close) - 1.0) * 100.0

    # 고점 대비(%) — 모멘텀 전략과 **같은 함수**(core.strategy.scoring.drawdown_from_high_pct).
    drawdown = drawdown_from_high_pct(series, current_price)

    return {
        "현재가": current_price,
        "일간(%)": daily_pct,
        "1주(%)": _calc_period_return(series, 5),
        "2주(%)": _calc_period_return(series, 10),
        "3주(%)": _calc_period_return(series, 15),
        "4주(%)": _calc_period_return(series, 20),
        "1달(%)": _calc_period_return(series, 20),
        "2달(%)": _calc_period_return(series, 40),
        "3달(%)": _calc_period_return(series, 60),
        "4달(%)": _calc_period_return(series, 80),
        "5달(%)": _calc_period_return(series, 100),
        "6달(%)": _calc_period_return(series, 126),
        "7달(%)": _calc_period_return(series, 147),
        "8달(%)": _calc_period_return(series, 168),
        "9달(%)": _calc_period_return(series, 189),
        "10달(%)": _calc_period_return(series, 210),
        "11달(%)": _calc_period_return(series, 231),
        "12달(%)": _calc_period_return(series, 252),
        "24달(%)": _calc_period_return(series, 504),
        "36달(%)": _calc_period_return(series, 756),
        "고점": drawdown,
        "추세(3달)": series.iloc[-60:].astype(float).tolist(),
        "RSI": _calculate_rsi(series, RSI_PERIOD),
        **monthly_return_metrics,
    }


def _load_realtime_snapshot(country_code: str, tickers: list[str]) -> dict[str, dict[str, float]]:
    """국가별 실시간 현재가/등락률 스냅샷을 로드합니다."""
    if not tickers:
        return {}

    try:
        return get_realtime_snapshot(country_code, tickers)
    except Exception as exc:
        logger.warning("순위용 실시간 스냅샷 조회 실패: %s", exc)
        return {}


def _apply_realtime_overlay(
    price_metrics: dict[str, Any],
    realtime_entry: dict[str, float] | None,
) -> dict[str, Any]:
    """실시간 현재가/일간 등락률이 있으면 캐시 값을 덮어씁니다."""
    if not isinstance(realtime_entry, dict) or not realtime_entry:
        return price_metrics

    updated = dict(price_metrics)

    now_val = realtime_entry.get("nowVal")
    if now_val is not None:
        try:
            updated["현재가"] = float(now_val)
        except (TypeError, ValueError):
            pass

    change_rate = realtime_entry.get("changeRate")
    if change_rate is not None:
        try:
            updated["일간(%)"] = float(change_rate)
        except (TypeError, ValueError):
            pass

    deviation = realtime_entry.get("deviation")
    if deviation is not None:
        try:
            updated["괴리율"] = float(deviation)
        except (TypeError, ValueError):
            pass

    return updated


def build_effective_close_series(
    cached_close_series: pd.Series | None,
    realtime_entry: dict[str, float] | None,
) -> pd.Series | None:
    """실시간 가격을 반영한 종가 시리즈를 생성합니다."""
    if cached_close_series is None or cached_close_series.empty:
        return None
    if not isinstance(realtime_entry, dict) or not realtime_entry:
        return cached_close_series

    now_val = realtime_entry.get("nowVal")
    if now_val is None:
        return cached_close_series

    try:
        realtime_price = float(now_val)
    except (TypeError, ValueError):
        return cached_close_series

    adjusted = cached_close_series.copy()
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    last_index = pd.Timestamp(adjusted.index[-1])
    if last_index.tzinfo is not None:
        last_index = last_index.tz_localize(None)
    last_index = last_index.normalize()

    if last_index < today:
        adjusted.loc[today] = realtime_price
    else:
        adjusted.iloc[-1] = realtime_price
    return adjusted.sort_index()


def _normalize_ranking_values(
    df: pd.DataFrame,
    country_code: str,
    *,
    monthly_labels: list[str] | None = None,
) -> pd.DataFrame:
    normalized = df.copy()

    country_norm = str(country_code or "").strip().lower()
    price_digits = 2 if country_norm in ("au", "us") else 0
    percent_columns = [
        "괴리율",
        "일간(%)",
        "1주(%)",
        "2주(%)",
        "3주(%)",
        "4주(%)",
        "1달(%)",
        "2달(%)",
        "3달(%)",
        "4달(%)",
        "5달(%)",
        "6달(%)",
        "7달(%)",
        "8달(%)",
        "9달(%)",
        "10달(%)",
        "11달(%)",
        "12달(%)",
        "24달(%)",
        "36달(%)",
        "고점",
        *(monthly_labels or []),
    ]
    one_decimal_columns = ["RSI"]
    score_columns = ["추세", "이격", "단기이격", "점수"]
    score_columns.extend(
        str(column) for column in normalized.columns if str(column).startswith("추세(") and str(column).endswith(")")
    )

    def _round_if_present(column: str, digits: int) -> None:
        if column not in normalized.columns:
            return
        series = pd.to_numeric(normalized[column], errors="coerce")
        normalized[column] = series.round(digits)

    _round_if_present("현재가", price_digits)
    for column in percent_columns:
        _round_if_present(column, 2)

    for column in one_decimal_columns:
        _round_if_present(column, 1)

    for column in score_columns:
        _round_if_present(column, 1)

    return normalized


def hold_eligible_mask(disparity: pd.Series, short_disparity: pd.Series) -> pd.Series:
    """보유 가능 조건 — 규칙은 공용 함수(`core.strategy.scoring.hold_eligible`)다.

    호출부(모멘텀 후보 선별 등)가 쓰던 이름을 유지하기 위한 얇은 래퍼다.
    """
    return hold_eligible(disparity, short_disparity)


def _mark_hold_targets(
    df: pd.DataFrame,
    top_n: int,
    max_per_industry: int | None = None,
    industry_by: dict[str, str] | None = None,
) -> pd.DataFrame:
    """규칙상 보유 대상인 종목에 ``보유대상`` 을 표시한다. 화면의 추천(✅) 기준.

    선정 자체는 **모멘텀 전략과 같은 공용 함수**(`core.strategy.scoring.select_holdings`)가
    한다 — 자격(장기 이격 > 0, 단기 이격 >= 0) → 순위 점수 → 업종 상한 순이다.
    예전에는 여기서 자격·점수만 보고 업종 상한을 빼먹어, 상한이 걸린 풀에서 표의 ✅ 와
    실제 전략 선정이 어긋났다.

    여기서 따로 거는 건 **제외**(`exclude_from_ranking`)뿐이다 — 투자 후보가 아니다.
    벤치마크는 빼지 않는다(종목풀에 있으면 매수 후보다. 모멘텀·신고가도 같다).
    조건을 만족하는 종목이 ``top_n`` 보다 적으면 그만큼만 표시한다(억지로 채우지 않는다).
    """
    df["보유대상"] = False
    if "이격" not in df.columns or "단기이격" not in df.columns:
        return df

    candidates = [
        {
            "ticker": str(row.get("티커") or "").strip().upper(),
            "long_disparity_pct": row.get("이격"),
            "short_disparity_pct": row.get("단기이격"),
        }
        for _, row in df.iterrows()
        if not bool(row.get("exclude_from_ranking"))
    ]
    picked = set(
        select_holdings(
            candidates,
            top_n=int(top_n),
            max_per_industry=max_per_industry,
            industry_by=industry_by or {},
        )
    )
    if picked:
        df.loc[df["티커"].astype(str).str.strip().str.upper().isin(picked), "보유대상"] = True
    return df


def _apply_common_rank_scores(
    df: pd.DataFrame,
    effective_close_series_map: dict[str, pd.Series],
    ma_rules: list[dict[str, Any]],
) -> pd.DataFrame:
    """공통 랭킹 엔진으로 추세(%) 컬럼을 일괄 주입한다.

    - 자격기준은 공통 엔진의 composite 결손 여부를 사용한다.
    - 화면/정렬/백테스트 기준값은 signed-percentile 이 아니라 원천 MA 이격률(%)이다.
    - 평가 시점은 각 티커의 ``effective_close_series`` 최신 일자들의 최댓값.
    - ETF 풀에 있으나 종가 시리즈가 없는 티커는 NaN 유지.
    """
    score_columns = [str(rule["score_column"]) for rule in ma_rules]

    if df.empty:
        return df

    if not effective_close_series_map or not ma_rules:
        df["추세"] = pd.NA
        df["이격"] = pd.NA
        df["단기이격"] = pd.NA
        df["점수"] = pd.NA
        for column in score_columns:
            if column not in df.columns:
                df[column] = pd.NA
        return df

    # [일자 × 티커] 종가 프레임 구성
    series_frames: dict[str, pd.Series] = {}
    for ticker, series in effective_close_series_map.items():
        if series is None or series.empty:
            continue
        normalized = pd.to_numeric(series, errors="coerce").copy()
        normalized.index = pd.to_datetime(normalized.index).normalize()
        normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
        series_frames[ticker] = normalized

    if not series_frames:
        df["추세"] = pd.NA
        df["이격"] = pd.NA
        df["단기이격"] = pd.NA
        df["점수"] = pd.NA
        for column in score_columns:
            df[column] = pd.NA
        return df

    union_index = sorted({ts for s in series_frames.values() for ts in s.index})
    close_frame = pd.DataFrame(
        {t: s.reindex(union_index) for t, s in series_frames.items()},
        index=pd.DatetimeIndex(union_index),
    )

    composite_frame, trend_by_order, _ = build_composite_rank_scores(close_frame, ma_rules)
    eval_date = close_frame.index.max()

    # 티커별 값 매핑. composite 는 signed-percentile 이지만 여기서는 자격 마스크 용도로만 쓴다.
    composite_row = composite_frame.loc[eval_date]
    trend_maps: dict[str, dict[str, float | None]] = {}
    for rule in ma_rules:
        column = str(rule["score_column"])
        trend_frame = trend_by_order[int(rule["order"])]
        if eval_date in trend_frame.index:
            trend_row = trend_frame.loc[eval_date]
            trend_maps[column] = {
                ticker: (None if pd.isna(trend_row.get(ticker)) else float(trend_row.get(ticker)))
                for ticker in trend_row.index
            }
        else:
            trend_maps[column] = {}

    tickers_col = df["티커"].astype(str)
    trend_values_by_rule = [tickers_col.map(trend_map).astype(float) for trend_map in trend_maps.values()]
    if trend_values_by_rule:
        trend_sum = trend_values_by_rule[0].copy()
        for values in trend_values_by_rule[1:]:
            trend_sum = trend_sum + values
        df["추세"] = trend_sum / len(trend_values_by_rule)
    else:
        df["추세"] = pd.NA
    df["이격"] = df["추세"]

    composite_missing = tickers_col.map(
        {
            ticker: (None if pd.isna(composite_row.get(ticker)) else float(composite_row.get(ticker)))
            for ticker in composite_row.index
        }
    ).isna()

    # 자격 미달 종목(결손 행) 일관성 마스킹 처리
    df.loc[composite_missing, "추세"] = None
    df.loc[composite_missing, "이격"] = None

    for column, trend_map in trend_maps.items():
        df[column] = tickers_col.map(trend_map).astype("object")
        df.loc[composite_missing, column] = None

    main_rule = ma_rules[0]
    short_days = int(main_rule["short_ma_days"])

    # 단기이격 = 종가와 단기 이평선의 이격률(%). 이격(장기 기준)과 같은 함수를 써서
    # 두 값이 항상 동일한 지표의 기간만 다른 버전이 되도록 한다.
    # 종목 선택은 이격(장기), 손절/익절 판단은 단기이격이 담당한다.
    short_trend_frame = compute_trend_frame(close_frame, short_days)
    short_trend_row = short_trend_frame.loc[eval_date]
    df["단기이격"] = tickers_col.map(
        {
            ticker: (None if pd.isna(short_trend_row.get(ticker)) else float(short_trend_row.get(ticker)))
            for ticker in short_trend_row.index
        }
    ).astype("object")
    df.loc[composite_missing, "단기이격"] = None

    # 순위 점수 — 정렬과 추천(✅)이 쓰는 단일 기준. 모멘텀 전략과 **같은 함수**다.
    # 표시용 「장기」(이격)·「단기」(단기이격)는 원천 값 그대로 두고, 점수만 따로 붙인다.
    df["점수"] = rank_score(pd.to_numeric(df["이격"], errors="coerce"), pd.to_numeric(df["단기이격"], errors="coerce"))
    df.loc[composite_missing, "점수"] = None

    return df


def build_ticker_type_rankings(
    ticker_type: str,
    *,
    ma_rules: list[dict[str, Any]] | None = None,
    as_of_date: pd.Timestamp | None = None,
    realtime_snapshot_override: dict[str, dict[str, float]] | None = None,
    # 화면 상단에서 임시로 바꿔 보는 업종 상한. None 이면 종목풀 저장값을 쓴다.
    # `-1` 은 '제한 없음' 이다 — None(미지정)과 구분해야 저장값으로 되돌아가지 않는다.
    max_per_industry_override: int | None = None,
    status_callback: Any | None = None,
) -> pd.DataFrame:
    if callable(status_callback):
        status_callback("최신 거래일 기준 캐시 상태 확인")
    started_at = perf_counter()
    settings = get_ticker_type_settings(ticker_type)
    benchmark_ticker = get_pool_benchmark_ticker(settings)
    country_code = str(settings.get("country_code") or "").strip().lower()

    etfs = get_etfs(ticker_type)
    if not etfs:
        return pd.DataFrame()

    fetch_started_at = perf_counter()
    tickers = [str(item.get("ticker") or "").strip().upper() for item in etfs if str(item.get("ticker") or "").strip()]
    selected_as_of_date = (as_of_date or pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)).normalize()
    cache_updated_map_raw = load_cached_updated_at_bulk_with_fallback(ticker_type, tickers)
    latest_trading_day = _get_latest_trading_day_for_reference(country_code, selected_as_of_date)
    monthly_labels = get_recent_monthly_return_labels(MONTHLY_RETURN_LABEL_COUNT, reference_date=selected_as_of_date)
    missing_tickers = sorted({ticker for ticker in tickers if ticker not in cache_updated_map_raw})
    normalized_cache_updated = {
        ticker: normalized
        for ticker, updated_at in cache_updated_map_raw.items()
        if (normalized := _normalize_market_timestamp(updated_at, country_code, assume_utc=True)) is not None
    }
    # 판정 기준은 최신 거래일이 아니라 '캐시가 이미 갖고 있어야 할 거래일' 이다.
    # 마감 직후 배치가 아직 안 돈 구간까지 막으면 매일 그 시간에 화면을 못 쓴다.
    expected_day = _expected_cache_trading_day(country_code, latest_trading_day)
    stale_tickers = sorted(
        ticker for ticker, updated_at in normalized_cache_updated.items() if updated_at.normalize() < expected_day
    )
    latest_cache_updated_at = max(normalized_cache_updated.values()) if normalized_cache_updated else None

    if missing_tickers or stale_tickers:
        fetch_elapsed = perf_counter() - fetch_started_at
        total_elapsed = perf_counter() - started_at
        logger.warning(
            "[rankings] type=%s blocked latest_trading_day=%s missing=%d stale=%d total=%.3fs fetch=%.3fs",
            ticker_type,
            latest_trading_day.date(),
            len(missing_tickers),
            len(stale_tickers),
            total_elapsed,
            fetch_elapsed,
        )
        return _build_blocked_rankings_result(
            latest_trading_day=latest_trading_day,
            cache_updated_at=latest_cache_updated_at,
            missing_tickers=missing_tickers,
            stale_tickers=stale_tickers,
        )

    if callable(status_callback):
        status_callback("기준 종가 캐시 로드")
    cached_close_series_map = load_cached_close_series_bulk_with_fallback(ticker_type, tickers)
    if callable(status_callback):
        status_callback("실시간 가격 조회")
    today_korea = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    realtime_allowed = selected_as_of_date == today_korea
    realtime_snapshot = (
        realtime_snapshot_override
        if realtime_snapshot_override is not None
        else _load_realtime_snapshot(country_code, tickers)
        if realtime_allowed
        else {}
    )
    fetch_elapsed = perf_counter() - fetch_started_at
    realtime_meta = None
    if realtime_allowed and realtime_snapshot_override is None:
        realtime_meta = get_realtime_snapshot_meta(country_code, tickers)

    effective_ma_rules = ma_rules or get_ticker_type_ma_rules(ticker_type)
    # MDD·소르티노·is_partial(신규상장) 기준 기간 — 고점과 같은 공용 창(METRIC_WINDOW_MONTHS).
    # 배치 저장값 대신 기준일·실시간 가격과 항상 같은 기준으로 계산된다.
    backtest_months = METRIC_WINDOW_MONTHS
    rows: list[dict[str, Any]] = []
    effective_close_series_map: dict[str, pd.Series] = {}
    preprocess_elapsed = 0.0
    metric_elapsed = 0.0
    process_elapsed = 0.0

    from utils.portfolio_io import load_holding_accounts_by_ticker

    holding_accounts = load_holding_accounts_by_ticker()

    if callable(status_callback):
        status_callback("순위 계산")

    for etf in etfs:
        ticker = str(etf.get("ticker") or "").strip().upper()
        if not ticker:
            continue

        cached_close_series = cached_close_series_map.get(ticker)
        realtime_entry = realtime_snapshot.get(ticker)
        preprocess_started_at = perf_counter()
        base_close_series = _slice_close_series_to_date(cached_close_series, latest_trading_day)
        effective_close_series = build_effective_close_series(base_close_series, realtime_entry)
        if effective_close_series is not None and not effective_close_series.empty:
            effective_close_series_map[ticker] = effective_close_series
        preprocess_elapsed += perf_counter() - preprocess_started_at

        metric_started_at = perf_counter()
        price_metrics = _extract_price_metrics_from_close_series(
            effective_close_series,
            reference_date=selected_as_of_date,
            monthly_labels=monthly_labels,
        )
        price_metrics = _apply_realtime_overlay(price_metrics, realtime_entry)
        metric_elapsed += perf_counter() - metric_started_at

        # 추세(%)는 아래 공통 엔진에서 한 번에 주입된다.
        ma_rule_scores = {str(rule["score_column"]): None for rule in effective_ma_rules}

        row = {
            "버킷": BUCKET_MAPPING.get(int(etf.get("bucket") or 0), str(etf.get("bucket") or "")),
            "bucket": int(etf.get("bucket") or 0),
            "티커": ticker,
            "종목명": etf.get("name", ""),
            # 종목 메모 — 자산 관리 화면과 같은 값(stock_meta.memo, utils/stock_memo_store).
            "메모": str(etf.get("memo") or ""),
            "마켓": etf.get("market", ""),
            "country_code": country_code,
            "currency": str(settings.get("currency") or ""),
            "source_ticker_type": ticker_type,
            "is_benchmark": ticker == benchmark_ticker,
            "상장일": etf.get("listing_date", "-"),
            # 보유: 이 종목을 실제로 들고 있는 계좌명 목록(쉼표 구분). 없으면 빈 문자열.
            "보유": ", ".join(holding_accounts.get(ensure_asx_prefix(ticker) if country_code == "au" else ticker, [])),
            "exclude_from_ranking": bool(etf.get("exclude_from_ranking")),
            **ma_rule_scores,
            **price_metrics,
            "거래량": float(etf.get("volume", 0)) if etf.get("volume") is not None else None,
            "backtest_stats": (
                single_stock_backtest_stats(effective_close_series, backtest_months)
                if effective_close_series is not None and not effective_close_series.empty
                else None
            ),
        }

        rows.append(row)

    dataframe_started_at = perf_counter()
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 공통 엔진 호출: rankings 와 backtest 가 동일하게 MA 이격률(%) 기준 추세를 쓰도록 강제.
    process_started_at = perf_counter()
    df = _apply_common_rank_scores(
        df,
        effective_close_series_map,
        effective_ma_rules,
    )
    process_elapsed += perf_counter() - process_started_at

    realtime_active = bool(realtime_snapshot)
    ranking_computed_at = datetime.now()

    def _to_sortable_score(value: Any) -> float:
        if value is None or pd.isna(value):
            return float("-inf")
        return float(value)

    def _sort_key(row: pd.Series) -> tuple[int, float, str]:
        # 순위 점수(장기·단기 이격률 평균) 내림차순. 값이 없는 종목은 뒤로.
        score = row.get("점수")
        return (
            1 if score is None or pd.isna(score) else 0,
            _to_sortable_score(score),
            str(row.get("티커", "")),
        )

    sort_values = df.apply(_sort_key, axis=1, result_type="expand")
    sort_values.columns = [
        "_missing_trend",
        "_trend_value",
        "_ticker_sort",
    ]
    df = pd.concat([df, sort_values], axis=1)
    df = df.sort_values(
        by=[
            "_missing_trend",
            "_trend_value",
            "_ticker_sort",
        ],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)

    df = df.drop(
        columns=[
            "_missing_trend",
            "_trend_value",
            "_ticker_sort",
        ]
    )
    # 업종 상한 — 화면에서 바꿔 보는 값이 있으면 그것, 없으면 종목풀 저장값(모멘텀과 같은 값).
    if max_per_industry_override is None:
        max_per_industry = settings.get("MAX_PER_INDUSTRY")
    else:
        max_per_industry = None if int(max_per_industry_override) < 0 else int(max_per_industry_override)
    # 업종 맵은 공용 소스(`utils/industry_map`)에서 직접 읽는다 — 화면의 「업종」 컬럼은
    # 이 함수보다 뒤(`rank_service._apply_industry_labels`)에서 붙어 여기서는 아직 없다.
    from utils.industry_map import industry_map

    industry_by = {str(key).strip().upper(): value for key, value in industry_map(ticker_type).items()}
    df = _mark_hold_targets(df, int(settings["TOP_N_HOLD"]), max_per_industry, industry_by)
    df = _normalize_ranking_values(df, country_code, monthly_labels=monthly_labels)
    df.attrs["realtime_active"] = realtime_active
    df.attrs["ranking_computed_at"] = ranking_computed_at
    if realtime_meta:
        df.attrs["realtime_fetched_at"] = realtime_meta.get("fetched_at")
        df.attrs["realtime_expires_at"] = realtime_meta.get("expires_at")
        df.attrs["realtime_is_stale"] = bool(realtime_meta.get("is_stale", False))
        df.attrs["realtime_source"] = realtime_meta.get("source")
    if latest_cache_updated_at is not None:
        df.attrs["cache_updated_at"] = latest_cache_updated_at
    df.attrs["latest_trading_day"] = latest_trading_day
    df.attrs["as_of_date"] = selected_as_of_date
    df.attrs["ma_rules"] = effective_ma_rules
    dataframe_elapsed = perf_counter() - dataframe_started_at
    total_elapsed = perf_counter() - started_at
    logger.info(
        "[rankings] type=%s tickers=%d total=%.3fs fetch=%.3fs preprocess=%.3fs metrics=%.3fs process=%.3fs dataframe=%.3fs",
        ticker_type,
        len(tickers),
        total_elapsed,
        fetch_elapsed,
        preprocess_elapsed,
        metric_elapsed,
        process_elapsed,
        dataframe_elapsed,
    )
    return df


__all__ = [
    "build_recent_monthly_return_metrics",
    "build_effective_ma_rules",
    "build_ticker_type_rankings",
    "get_recent_monthly_return_labels",
    "get_ticker_type_ma_rules",
]
