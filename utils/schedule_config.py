"""Scheduling helpers built dynamically from account configuration files."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from utils.account_registry import list_available_accounts
from utils.settings_loader import AccountSettingsError, get_account_settings


_DEFAULT_TIMEZONES: Dict[str, str] = {
    "kor": "Asia/Seoul",
    "aus": "Australia/Sydney",
}


class ScheduleConfigError(RuntimeError):
    """Raised when schedule information cannot be derived from account settings."""


def _normalize_country(value: Any, fallback: str) -> str:
    text = str(value or "").strip().lower()
    return text or fallback


def _default_timezone(country_code: str) -> str:
    return _DEFAULT_TIMEZONES.get(country_code, "UTC")


def _build_schedule_entry(account_id: str) -> Dict[str, Any] | None:
    try:
        settings = get_account_settings(account_id)
    except AccountSettingsError as exc:  # pragma: no cover - 설정 오류 방어
        raise ScheduleConfigError(str(exc)) from exc

    schedule = settings.get("schedule")
    if not isinstance(schedule, dict) or not schedule:
        return None

    country_code = _normalize_country(settings.get("country_code"), account_id)
    enabled_raw = schedule.get("enabled")
    enabled_flag = True if enabled_raw is None else bool(enabled_raw)

    timezone_value = (
        schedule.get("timezone")
        or schedule.get("recommendation_timezone")
        or _default_timezone(country_code)
    )
    notify_timezone_value = (
        schedule.get("notify_timezone")
        or schedule.get("timezone")
        or _default_timezone(country_code)
    )

    entry: Dict[str, Any] = {
        "account_id": account_id,
        "country_code": country_code,
        "enabled": enabled_flag,
        "recommendation_cron": schedule.get("recommendation_cron"),
        "notify_cron": schedule.get("notify_cron"),
        "timezone": timezone_value,
        "notify_timezone": notify_timezone_value,
        "run_immediately_on_start": schedule.get("run_immediately_on_start"),
    }

    label = settings.get("name")
    if isinstance(label, str) and label.strip():
        entry["label"] = label.strip()

    return entry


@lru_cache(maxsize=1)
def _load_account_schedules() -> Dict[str, Dict[str, Any]]:
    schedules: Dict[str, Dict[str, Any]] = {}

    for account_id in list_available_accounts():
        try:
            entry = _build_schedule_entry(account_id)
        except ScheduleConfigError:
            continue

        if entry and entry.get("recommendation_cron"):
            schedules[account_id] = entry

    return schedules


def get_global_schedule_settings() -> Dict[str, Any]:
    """Return global schedule metadata derived from account configs."""

    values = [
        entry.get("run_immediately_on_start")
        for entry in _load_account_schedules().values()
        if entry.get("run_immediately_on_start") is not None
    ]
    if not values:
        return {}

    return {"run_immediately_on_start": bool(values[0])}


def get_country_schedule(country: str) -> Dict[str, Any]:
    country_norm = _normalize_country(country, "")
    if not country_norm:
        return {}

    for entry in _load_account_schedules().values():
        if entry.get("country_code") == country_norm:
            return dict(entry)

    return {}


def get_cache_schedule() -> Dict[str, Any]:
    """Return cache schedule configuration (disabled by default)."""

    return {"enabled": False}


def get_all_country_schedules() -> Dict[str, Dict[str, Any]]:
    return {key: dict(value) for key, value in _load_account_schedules().items()}


def refresh_cache() -> None:
    """Clear cached schedule data (primarily for tests)."""

    _load_account_schedules.cache_clear()


__all__ = [
    "ScheduleConfigError",
    "get_global_schedule_settings",
    "get_country_schedule",
    "get_all_country_schedules",
    "get_cache_schedule",
    "refresh_cache",
]
