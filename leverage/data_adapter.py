"""leverage 엔진 ↔ momentum-etf 데이터 계층 어댑터.

기존 logic/backtest/data.py 가 제공하던 인터페이스(compute_bounds / download_prices /
download_opens / download_fx / _extract_field)를 그대로 노출하되, 실제 시세 조회는
momentum-etf 의 `utils.data_loader.fetch_ohlcv`(캐시 + 실시간 보강)를 사용한다.

fetch_ohlcv 반환 DataFrame 컬럼: Open/High/Low/Close/Volume (날짜 인덱스).
"""

import pandas as pd

from utils.data_loader import fetch_naver_realtime_price, fetch_ohlcv, get_latest_trading_day, is_trading_day


def current_trading_day(market: str) -> pd.Timestamp:
    """momentum-etf 거래일 달력 기준 '현재(최신) 거래일'. 장중에는 오늘, 휴장일엔 마지막 거래일."""
    return pd.Timestamp(get_latest_trading_day(market)).normalize()


def realtime_price(ticker: str, market: str) -> float | None:
    """실시간 현재가 (한국만 지원). 데이터 계층이 막혀 있으면 None."""
    if market != "kor" or ticker == "CASH":
        return None
    return fetch_naver_realtime_price(ticker)

# 신호 계산용 워밍업(영업일). 기존 엔진과 동일하게 12개월(252영업일) 사용.
_WARMUP_BDAYS = 252

# momentum-etf 캐시 컬렉션 키. leverage 전략 대상은 모두 한국 ETF 이므로 "etf".
_TICKER_TYPE = "etf"


def compute_bounds(settings: dict, end_bound: pd.Timestamp | None = None):
    """백테스트/튜닝/추천이 동일한 기간 산정 로직을 쓰도록 범위를 계산한다."""
    end = end_bound if end_bound is not None else pd.Timestamp.today().normalize()

    if "start_date" in settings:
        start = pd.Timestamp(settings["start_date"])
    elif "months_range" in settings:
        start = end - pd.DateOffset(months=int(settings["months_range"]))
    else:
        raise ValueError("settings 에 'start_date' 또는 'months_range' 가 필요합니다.")

    warmup_start = start - pd.offsets.BDay(_WARMUP_BDAYS)
    return start, warmup_start, end


def _requested_tickers(settings: dict) -> list[str]:
    return [
        settings["offense_ticker"],
        settings["signal_ticker"],
        settings["defense_ticker"],
    ]


def _market_tickers(settings: dict) -> list[str]:
    """현금(CASH)을 제외한 실제 시세 조회 대상 (입력 순서 보존, 중복 제거)."""
    seen: dict[str, None] = {}
    for ticker in _requested_tickers(settings):
        if ticker != "CASH" and ticker not in seen:
            seen[ticker] = None
    return list(seen.keys())


def _fetch_field(settings: dict, start, field: str) -> pd.DataFrame:
    """대상 티커들의 특정 필드(Open/Close)를 momentum-etf 데이터로 조회해 DataFrame 으로 만든다."""
    country = settings.get("market", "kor")
    start_str = pd.Timestamp(start).strftime("%Y-%m-%d")

    series: dict[str, pd.Series] = {}
    for ticker in _market_tickers(settings):
        df = fetch_ohlcv(
            ticker, country, months_back=None, date_range=[start_str, None], ticker_type=_TICKER_TYPE
        )
        if df is None or df.empty:
            raise ValueError(f"가격 데이터를 받아오지 못했습니다: {ticker} ({country})")

        # 장중 실시간 데이터 보강 (오늘 가격이 캐시에 없을 경우 실시간 스냅샷을 덧붙임)
        today_ts = pd.Timestamp.today().normalize()
        if is_trading_day(country, today_ts):
            latest_day = today_ts
        else:
            latest_day = get_latest_trading_day(country).normalize()

        df_last_day = pd.Timestamp(df.index[-1]).normalize()
        if latest_day > df_last_day:
            rt_val = realtime_price(ticker, country)
            if rt_val is not None and rt_val > 0:
                today_row = pd.DataFrame(
                    {"Open": [rt_val], "High": [rt_val], "Low": [rt_val], "Close": [rt_val], "Volume": [0.0]},
                    index=[latest_day]
                )
                df = pd.concat([df, today_row])
                df = df[~df.index.duplicated(keep="last")].sort_index()

        if field not in df.columns:
            raise ValueError(f"{ticker} 데이터에 '{field}' 컬럼이 없습니다. 사용 가능: {list(df.columns)}")
        series[ticker] = df[field].astype(float)

    if not series:
        raise ValueError(f"조회할 시세 티커가 없습니다: {_requested_tickers(settings)}")

    out = pd.DataFrame(series)
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    # 방어 자산이 현금이면 1.0 컬럼 주입
    if settings.get("defense_ticker") == "CASH":
        out["CASH"] = 1.0

    # 요청 티커가 모두 있는 날짜만 사용
    needed = [t for t in _requested_tickers(settings) if t in out.columns]
    out = out.dropna(subset=needed)
    if out.empty:
        raise ValueError(f"{field} 데이터가 비어 있습니다: {_requested_tickers(settings)}")
    return out


def download_prices(settings: dict, start) -> pd.DataFrame:
    """종가(Close) 시계열."""
    return _fetch_field(settings, start, "Close")


def download_opens(settings: dict, start) -> pd.DataFrame:
    """시가(Open) 시계열."""
    return _fetch_field(settings, start, "Open")


def download_fx(settings: dict, start) -> pd.Series:
    """원/달러 환율. 한국 시장은 원화 기준이라 1.0 고정."""
    market = settings.get("market", "kor")
    if market == "kor":
        idx = pd.date_range(start, pd.Timestamp.today(), freq="B")
        return pd.Series(1.0, index=idx, name="KRW")

    fx_df = fetch_ohlcv("USDKRW=X", "us", months_back=None, date_range=[pd.Timestamp(start).strftime("%Y-%m-%d"), None])
    if fx_df is None or fx_df.empty or "Close" not in fx_df.columns:
        raise ValueError("환율(USDKRW) 데이터를 받아오지 못했습니다.")
    fx = fx_df["Close"].astype(float)
    fx.index = pd.to_datetime(fx.index)
    fx.name = "USDKRW"
    return fx


def _extract_field(data: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """yfinance 멀티인덱스 결과에서 특정 필드를 추출 (tune 의 us 프리패치 호환용)."""
    key = field.lower()
    if isinstance(data.columns, pd.MultiIndex):
        candidates = [key, f"adj {key}"]
        level_idx = None
        field_key = None
        for level in range(data.columns.nlevels):
            level_values = data.columns.get_level_values(level)
            for cand in candidates:
                matches = [v for v in level_values if str(v).lower() == cand]
                if matches:
                    level_idx = level
                    field_key = matches[0]
                    break
            if level_idx is not None:
                break
        if level_idx is None:
            raise ValueError(f"{field} 컬럼을 찾지 못했습니다. 사용 가능 컬럼: {list(data.columns)}")
        out = data.xs(field_key, axis=1, level=level_idx)
    else:
        candidates = [c for c in [field, field.capitalize()] if c in data.columns]
        field_col = candidates[0] if candidates else data.columns[0]
        out = data[[field_col]].rename(columns={field_col: tickers[0]})

    return out.dropna(how="all")
