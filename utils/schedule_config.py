"""Central access helpers for signal & notification schedules."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_BASE_DIR, "data", "settings", "schedule_config.json")


class ScheduleConfigError(RuntimeError):
    """Raised when the schedule configuration cannot be loaded."""


@lru_cache(maxsize=1)
def _load_config() -> Dict[str, Any]:
    if not os.path.exists(_CONFIG_PATH):
        raise ScheduleConfigError(f"Schedule config 파일을 찾을 수 없습니다: {_CONFIG_PATH}")
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except json.JSONDecodeError as exc:  # pragma: no cover - configuration error
        raise ScheduleConfigError(f"Schedule config JSON 파싱 오류: {exc}") from exc


def get_global_schedule_settings() -> Dict[str, Any]:
    return _load_config().get("global", {})


def get_country_schedule(country: str) -> Dict[str, Any]:
    countries = _load_config().get("countries", {})
    return countries.get(str(country).lower(), {})


def get_cache_schedule() -> Dict[str, Any]:
    return _load_config().get("cache", {})


def get_all_country_schedules() -> Dict[str, Dict[str, Any]]:
    countries = _load_config().get("countries", {})
    return {key: value for key, value in countries.items()}


def refresh_cache() -> None:
    """Clear cached config (mainly for tests)."""
    _load_config.cache_clear()


__all__ = [
    "ScheduleConfigError",
    "get_global_schedule_settings",
    "get_country_schedule",
    "get_all_country_schedules",
    "get_cache_schedule",
    "refresh_cache",
]
