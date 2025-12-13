"""Helpers for reconciling cron expressions across different schedulers."""

from __future__ import annotations

from typing import Literal

TargetType = Literal["apscheduler", "croniter"]


def normalize_cron_weekdays(cron_expr: str, *, target: TargetType = "apscheduler") -> str:
    """
    Adjust the day-of-week field of a cron expression to match the expectation of a scheduler.

    - crontab style: 0/7=Sunday, 1=Monday.
    - APScheduler CronTrigger: 0=Monday ... 6=Sunday.
    - croniter (for Slack notifications): follows the crontab convention (0/7=Sunday).
    """

    parts = cron_expr.split()
    if len(parts) not in (5, 6):
        return cron_expr

    dow_index = 4 if len(parts) == 5 else 5
    dow_expr = parts[dow_index]

    def _convert_atom(token: str) -> str:
        if token in ("*", "?"):
            return token
        if "/" in token:
            base, step = token.split("/", 1)
            converted_base = _convert_atom(base)
            return f"{converted_base}/{step}"
        if "-" in token:
            start, end = token.split("-", 1)
            if start.isdigit() and end.isdigit():
                return f"{_convert_atom(start)}-{_convert_atom(end)}"
            return token
        if token.isdigit():
            value = int(token)
            if 0 <= value <= 7:
                if target == "apscheduler":
                    return str((value - 1) % 7)
                return str(value % 7)
        return token

    converted_tokens = [_convert_atom(t) for t in dow_expr.split(",")]
    converted_expr = ",".join(converted_tokens)
    if converted_expr == dow_expr:
        return cron_expr

    parts[dow_index] = converted_expr
    return " ".join(parts)


__all__ = ["normalize_cron_weekdays"]
