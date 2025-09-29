"""OHLCV 데이터를 디스크에 캐싱하고 관리하기 위한 헬퍼 함수 모음입니다."""

import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

CACHE_ROOT = Path(__file__).resolve().parents[1] / "data" / "stocks" / "cache"
_DEPLOY_SENTINEL_NAME = "__coin_cache_deploy_id.txt"


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
    """저장된 캐시 DataFrame을 로드하고, 없으면 None을 반환합니다."""
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


def _detect_current_deploy_id() -> Optional[str]:
    """Return a string that changes when a new deploy is detected."""

    env_keys = (
        "APP_DEPLOY_ID",
        "DEPLOY_RELEASE_ID",
        "RENDER_GIT_COMMIT",
        "GIT_COMMIT",
        "SOURCE_VERSION",
    )
    for key in env_keys:
        value = os.environ.get(key)
        if value:
            return value.strip()

    try:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        return result.decode().strip()
    except Exception:
        return None


def reset_coin_cache_for_new_deploy() -> bool:
    """Delete cached coin OHLCV when a new deploy is detected.

    Returns True if the cache directory was cleared during this call.
    """

    deploy_id = _detect_current_deploy_id()
    if not deploy_id:
        deploy_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    sentinel_path = CACHE_ROOT / _DEPLOY_SENTINEL_NAME
    previous_id: Optional[str] = None
    if sentinel_path.exists():
        try:
            previous_id = sentinel_path.read_text(encoding="utf-8").strip()
        except Exception:
            previous_id = None

    if previous_id == deploy_id:
        return False

    coin_cache_dir = CACHE_ROOT / "coin"
    cache_cleared = False

    if coin_cache_dir.exists():
        for child in coin_cache_dir.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                cache_cleared = True
            except Exception:
                continue
    else:
        coin_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_path.write_text(deploy_id, encoding="utf-8")
    except Exception:
        pass

    return cache_cleared or previous_id != deploy_id
