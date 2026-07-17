#!/usr/bin/env python3
"""국가별 거래일 JSON 파일을 갱신한다."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CACHE_START_DATE

COUNTRY_DIR = PROJECT_ROOT / "data" / "country"
HOLIDAY_OVERRIDE_PATH = PROJECT_ROOT / "config" / "market_holiday_overrides.json"
KST = ZoneInfo("Asia/Seoul")

MARKET_CALENDARS: dict[str, str] = {
    "kor": "XKRX",
    "au": "XASX",
    "us": "NYSE",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="국가별 거래일 캘린더 JSON 파일을 갱신합니다.")
    parser.add_argument(
        "--start-date",
        default=CACHE_START_DATE,
        help="생성 시작일(YYYY-MM-DD). 기본값은 config.CACHE_START_DATE.",
    )
    parser.add_argument(
        "--end-date",
        default=f"{date.today().year + 1}-12-31",
        help="생성 종료일(YYYY-MM-DD). 기본값은 내년 말.",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        choices=sorted(MARKET_CALENDARS),
        default=sorted(MARKET_CALENDARS),
        help="갱신할 국가 코드 목록.",
    )
    return parser.parse_args()


def _load_holiday_overrides() -> dict[str, list[str]]:
    if not HOLIDAY_OVERRIDE_PATH.exists():
        return {country: [] for country in MARKET_CALENDARS}

    payload = json.loads(HOLIDAY_OVERRIDE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"휴장일 오버라이드 파일 형식이 올바르지 않습니다: {HOLIDAY_OVERRIDE_PATH}")

    result: dict[str, list[str]] = {}
    for country in MARKET_CALENDARS:
        raw_dates = payload.get(country, [])
        if not isinstance(raw_dates, list):
            raise RuntimeError(f"휴장일 오버라이드 목록이 아닙니다: country={country}")
        result[country] = sorted({str(raw_date).strip() for raw_date in raw_dates if str(raw_date).strip()})
    return result


def _build_trading_days(country: str, calendar_name: str, start_date: str, end_date: str) -> list[str]:
    calendar = mcal.get_calendar(calendar_name)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    if schedule.empty:
        raise RuntimeError(f"거래일 캘린더가 비어 있습니다: country={country}, calendar={calendar_name}")
    return [str(index.date()) for index in schedule.index]


def _write_calendar(country: str, start_date: str, end_date: str, holiday_overrides: list[str]) -> int:
    calendar_name = MARKET_CALENDARS[country]
    raw_days = _build_trading_days(country, calendar_name, start_date, end_date)
    filtered_days = sorted(set(raw_days) - set(holiday_overrides))

    output_path = COUNTRY_DIR / country / "market_calendars.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "country_code": country,
        "calendar": calendar_name,
        "source": "pandas_market_calendars",
        "start_date": start_date,
        "end_date": end_date,
        "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "holiday_overrides": holiday_overrides,
        "trading_days": filtered_days,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    removed_count = len(set(raw_days) & set(holiday_overrides))
    print(
        f"{country}: {calendar_name} {start_date}~{end_date} "
        f"거래일 {len(filtered_days)}개 저장, 오버라이드 제외 {removed_count}개"
    )
    return removed_count


def main() -> int:
    args = _parse_args()
    start_date = str(args.start_date).strip()
    end_date = str(args.end_date).strip()
    if not start_date or not end_date:
        raise RuntimeError("거래일 캘린더 시작일과 종료일이 필요합니다.")

    holiday_overrides = _load_holiday_overrides()
    for country in args.countries:
        _write_calendar(country, start_date, end_date, holiday_overrides.get(country, []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
