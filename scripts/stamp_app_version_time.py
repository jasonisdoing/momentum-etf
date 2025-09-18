#!/usr/bin/env python3
"""Stamp settings.APP_VERSION_TIME with the current datetime."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _now(fmt_tz: str | None) -> datetime:
    if fmt_tz and ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(fmt_tz))
        except Exception:
            pass
    return datetime.now(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update settings.APP_VERSION_TIME to current time")
    parser.add_argument(
        "--tz", default=os.environ.get("APP_VERSION_TZ"), help="Timezone name (optional)"
    )
    parser.add_argument(
        "--settings",
        default="settings.py",
        help="Path to settings.py (default: settings.py)",
    )
    args = parser.parse_args()

    stamp = _now("Asia/Seoul").strftime("%Y-%m-%d-%H")
    settings_path = Path(args.settings).resolve()
    if not settings_path.exists():
        raise SystemExit(f"cannot find settings file: {settings_path}")

    original = settings_path.read_text(encoding="utf-8")
    pattern = r'APP_VERSION_TIME\s*=\s*os\.environ\.get\("APP_VERSION_TIME",\s*".*?"\)'
    replacement = f'APP_VERSION_TIME = os.environ.get("APP_VERSION_TIME", "{stamp}")'
    updated, count = re.subn(pattern, replacement, original, count=1)
    if count == 0:
        raise SystemExit("APP_VERSION_TIME assignment not found; aborting")

    if updated == original:
        return

    settings_path.write_text(updated, encoding="utf-8")
    print(f"Stamped APP_VERSION_TIME -> {stamp}")


if __name__ == "__main__":
    main()
