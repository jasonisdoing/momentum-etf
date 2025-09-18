"""Utility helpers for OHLCV caching on disk."""

import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

CACHE_ROOT = Path(__file__).resolve().parents[1] / "data" / "cache"


def _sanitize_ticker(ticker: str) -> str:
    """파일 시스템에 안전한 티커 문자열을 반환합니다."""
    if not ticker:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9._-]", "_", ticker.upper())


def get_cache_path(country: str, ticker: str) -> Path:
    """캐시 파일 경로를 반환합니다 (없으면 생성하지는 않음)."""
    safe_country = (country or "global").lower()
    safe_ticker = _sanitize_ticker(ticker)
    return CACHE_ROOT / safe_country / f"{safe_ticker}.pkl"


def load_cached_frame(country: str, ticker: str) -> Optional[pd.DataFrame]:
    """저장된 캐시 DataFrame을 로드합니다. 없으면 None."""
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
    return df


def save_cached_frame(country: str, ticker: str, df: pd.DataFrame) -> None:
    """캐시 DataFrame을 저장합니다."""
    if df is None or df.empty:
        return
    path = get_cache_path(country, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_to_save = df.copy()
    df_to_save.sort_index(inplace=True)
    df_to_save = df_to_save[~df_to_save.index.duplicated(keep="first")]
    df_to_save.to_pickle(path)


def get_cached_date_range(country: str, ticker: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    df = load_cached_frame(country, ticker)
    if df is None or df.empty:
        return None
    return df.index.min(), df.index.max()
