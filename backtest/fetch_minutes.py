"""사고팔기 간격 실험용 1분봉 데이터 준비 — 종목마다 `SINCE` 이후 토스 1분봉을 월별 파일로 저장한다.

실행: ``python backtest/fetch_minutes.py`` (지난달까지의 파일은 있으면 건너뛰고, 이번 달 파일은 매번 다시 받는다)

- 1분봉은 토스 c-chart(`from` 커서로 과거를 이어 받음). 미장은 프리·정규·애프터·데이장이 모두 들어 있다.
- 저장: ``backtest/data/{ticker}/{YYYY-MM}.json`` — ``[[utc_ms, 시가, 고가, 저가, 종가], …]`` (과거→최신).
  달의 경계는 그 시장 현지 날짜(자정)다. 현지 시각·세션 날짜는 시뮬레이션이 utc_ms 로 다시 계산한다.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from services.toss_market_service import TOSS_INVEST_API_BASE_URL, toss_request  # noqa: E402
from utils.realtime_quotes import resolve_toss_us_product_codes  # noqa: E402

#: 실험 대상 종목 → 국가(kor·us).
TICKERS: dict[str, str] = {
    "069500": "kor",
    "229200": "kor",
    "122630": "kor",
    "233740": "kor",
    "114800": "kor",
    "QQQM": "us",
    "QLD": "us",
    "TQQQ": "us",
    "SOXL": "us",
    "PSQ": "us",
}
SINCE = date(2026, 1, 1)  # 시뮬레이션 구간(`ladder_sim.SIM_SINCE`)과 같다 — MDD·변동성은 일봉 캐시로 계산한다
DATA_DIR = Path(__file__).resolve().parent / "data"
PAGE_SIZE = 400  # 토스 c-chart 한 번에 받는 최대 개수(500 이상은 400 Bad Request)
ZONES = {"kor": ZoneInfo("Asia/Seoul"), "us": ZoneInfo("America/New_York")}


def _chart_url(ticker: str, country: str) -> str:
    if country == "kor":
        return f"{TOSS_INVEST_API_BASE_URL}/api/v1/c-chart/kr-s/A{ticker}/min:1"
    code = resolve_toss_us_product_codes([ticker]).get(ticker)
    if not code:
        raise RuntimeError(f"{ticker} 토스 종목 코드를 찾지 못했습니다.")
    return f"{TOSS_INVEST_API_BASE_URL}/api/v1/c-chart/us-s/{code}/min:1"


def fetch_bars(url: str, start: datetime, end: datetime) -> list[list[float]]:
    """``end`` 직전부터 과거로 이어 받아 [start, end) 의 1분봉을 과거→최신 순으로 돌려준다."""
    rows: dict[int, list[float]] = {}
    cursor = end.isoformat()
    while True:
        resp = toss_request("GET", url, params={"count": PAGE_SIZE, "from": cursor}, timeout=10)
        resp.raise_for_status()
        result = resp.json()["result"]
        candles = result.get("candles") or []
        for candle in candles:
            ms = int(datetime.fromisoformat(candle["dt"]).timestamp() * 1000)
            rows[ms] = [float(candle[key]) for key in ("open", "high", "low", "close")]
        oldest = min((datetime.fromisoformat(c["dt"]) for c in candles), default=None)
        next_cursor = result.get("nextDateTime")
        if oldest is None or oldest < start or next_cursor in (None, cursor):
            break
        cursor = next_cursor
        time.sleep(0.2)
    lo, hi = start.timestamp() * 1000, end.timestamp() * 1000
    return [[ms, *rows[ms]] for ms in sorted(rows) if lo <= ms < hi]


def _next_month(month: date) -> date:
    return month.replace(year=month.year + (month.month == 12), month=month.month % 12 + 1)


def main() -> int:
    for ticker, country in TICKERS.items():
        zone = ZONES[country]
        now = datetime.now(zone)
        url = _chart_url(ticker, country)
        month = SINCE.replace(day=1)
        while month <= now.date():
            path = DATA_DIR / ticker / f"{month:%Y-%m}.json"
            current_month = (month.year, month.month) == (now.year, now.month)
            if not path.exists() or current_month:
                start = datetime.combine(month, datetime.min.time(), zone)
                end = min(datetime.combine(_next_month(month), datetime.min.time(), zone), now)
                bars = fetch_bars(url, start, end)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(bars))
                print(f"{ticker} {month:%Y-%m}: 분봉 {len(bars):,}개", flush=True)
            month = _next_month(month)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
