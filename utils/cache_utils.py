"""OHLCV 데이터를 디스크에 캐싱하고 관리하기 위한 헬퍼 함수 모음입니다."""

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

CACHE_ROOT = Path(__file__).resolve().parents[1] / "data" / "stocks" / "cache"


def _get_cache_start_date() -> Optional[pd.Timestamp]:
    """config.py에서 CACHE_START_DATE를 로드하여 Timestamp로 반환합니다."""
    try:
        from utils.settings_loader import load_common_settings

        common_settings = load_common_settings()
        raw = common_settings.get("CACHE_START_DATE")
    except Exception:
        return None

    if not raw:
        return None

    try:
        ts = pd.to_datetime(raw).normalize()
        if isinstance(ts, pd.DatetimeIndex):
            ts = ts[0]
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            return ts.normalize()
    except Exception:
        return None

    return None


def _sanitize_ticker(ticker: str) -> str:
    """파일 시스템에 안전하게 저장할 수 있는 티커 문자열을 생성합니다."""
    if not ticker:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9._-]", "_", ticker.upper())


def get_cache_path(country: str, ticker: str) -> Path:
    """캐시 파일 경로를 반환합니다. (존재하지 않으면 생성하지 않습니다.)"""
    safe_country = (country or "global").lower()
    safe_ticker = _sanitize_ticker(ticker)
    return CACHE_ROOT / safe_country / f"{safe_ticker}.pkl"


def load_cached_frame(country: str, ticker: str) -> Optional[pd.DataFrame]:
    """저장된 캐시 DataFrame을 로드하고, CACHE_START_DATE 이전 데이터를 필터링합니다."""
    path = get_cache_path(country, ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_pickle(path)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return None
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # CACHE_START_DATE 이전 데이터 필터링
    cache_start = _get_cache_start_date()
    if cache_start is not None:
        df = df[df.index >= cache_start]

    if df.empty:
        return None

    return df


def save_cached_frame(country: str, ticker: str, df: pd.DataFrame) -> None:
    """캐시 DataFrame을 저장합니다. CACHE_START_DATE 이전 데이터는 제외합니다."""
    if df is None or df.empty:
        return

    df_to_save = df.copy()
    df_to_save.sort_index(inplace=True)
    df_to_save = df_to_save[~df_to_save.index.duplicated(keep="first")]

    # CACHE_START_DATE 이전 데이터 필터링
    cache_start = _get_cache_start_date()
    if cache_start is not None:
        df_to_save = df_to_save[df_to_save.index >= cache_start]

    if df_to_save.empty:
        return

    path = get_cache_path(country, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_to_save.to_pickle(path)


def get_cached_date_range(country: str, ticker: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    df = load_cached_frame(country, ticker)
    if df is None or df.empty:
        return None
    return df.index.min(), df.index.max()
