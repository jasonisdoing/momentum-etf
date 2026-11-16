"""사고팔기 간격 실험용 1분봉 데이터 준비 — 종목마다 상승·횡보·하락 3개월 구간을 골라 토스 1분봉을 저장한다.

실행: ``python backtest/fetch_minutes.py`` (이미 받은 구간 파일은 건너뛴다. 다시 받으려면 파일을 지운다)

- 구간은 최근 2년 일봉(가격 캐시)에서 63거래일 구간을 모두 훑어 고른다 — 상승은 수익률 최고, 하락은 최저,
  횡보는 수익률 ±3% 이내 중 고점~저점 폭이 가장 좁은 구간.
- 1분봉은 토스 c-chart(`from` 커서로 과거를 이어 받음). 미장은 프리·정규·애프터·데이장이 모두 들어 있다.
- 저장: ``backtest/data/{ticker}/{regime}.json`` — 구간 정보와 ``bars``([utc_ms, 현지 시각 ISO, 시가, 고가, 저가, 종가]).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from services.toss_market_service import TOSS_INVEST_API_BASE_URL, toss_request  # noqa: E402
from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types  # noqa: E402
from utils.realtime_quotes import resolve_toss_us_product_codes  # noqa: E402

#: 실험 대상 종목 → 국가(kor·us).
TICKERS: dict[str, str] = {
    "069500": "kor",
    "229200": "kor",
    "122630": "kor",
    "233740": "kor",
    "QQQM": "us",
    "QLD": "us",
    "TQQQ": "us",
    "SOXL": "us",
}
DATA_DIR = Path(__file__).resolve().parent / "data"
WINDOW_DAYS = 63  # 3개월(거래일)
LOOKBACK_YEARS = 2
SIDEWAYS_MAX_RETURN_PCT = 3.0
PAGE_SIZE = 400  # 토스 c-chart 한 번에 받는 최대 개수(500 이상은 400 Bad Request)
REGIMES = ("up", "side", "down")
_ZONES = {"kor": ZoneInfo("Asia/Seoul"), "us": ZoneInfo("America/New_York")}


def select_windows(ticker: str) -> dict[str, dict]:
    """최근 2년 일봉에서 상승·횡보·하락 63거래일 구간을 고른다 — {regime: {start, end, return_pct, range_pct}}."""
    frame = load_cached_frames_bulk_from_all_ticker_types([ticker]).get(ticker)
    if frame is None or frame.empty:
        raise RuntimeError(f"{ticker} 일봉 캐시가 없습니다.")
    since = date.today() - timedelta(days=365 * LOOKBACK_YEARS)
    frame = frame.loc[str(since) :]
    windows = []
    for i in range(len(frame) - WINDOW_DAYS + 1):
        part = frame.iloc[i : i + WINDOW_DAYS]
        start_price = float(part["Open"].iloc[0])
        windows.append(
            {
                "start": part.index[0].date().isoformat(),
                "end": part.index[-1].date().isoformat(),
                "return_pct": round((float(part["Close"].iloc[-1]) / start_price - 1) * 100, 2),
                "range_pct": round((float(part["High"].max()) - float(part["Low"].min())) / start_price * 100, 2),
            }
        )
    sideways = [w for w in windows if abs(w["return_pct"]) <= SIDEWAYS_MAX_RETURN_PCT]
    if not sideways:
        raise RuntimeError(f"{ticker} 에 수익률 ±{SIDEWAYS_MAX_RETURN_PCT}% 이내 구간이 없습니다.")
    return {
        "up": max(windows, key=lambda w: w["return_pct"]),
        "side": min(sideways, key=lambda w: w["range_pct"]),
        "down": min(windows, key=lambda w: w["return_pct"]),
    }


def _bounds(country: str, start: str, end: str) -> tuple[datetime, datetime]:
    """구간의 분봉 시각 범위 — 미장은 첫 거래일의 데이장(전날 20:00 ET)부터 마지막 거래일 애프터 끝(20:00 ET)까지."""
    zone = _ZONES[country]
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if country == "us":
        return (
            datetime.combine(first - timedelta(days=1), datetime.min.time(), zone).replace(hour=20),
            datetime.combine(last, datetime.min.time(), zone).replace(hour=20),
        )
    return datetime.combine(first, datetime.min.time(), zone), datetime.combine(
        last, datetime.min.time(), zone
    ).replace(hour=15, minute=30)


def _chart_url(ticker: str, country: str) -> str:
    if country == "kor":
        return f"{TOSS_INVEST_API_BASE_URL}/api/v1/c-chart/kr-s/A{ticker}/min:1"
    code = resolve_toss_us_product_codes([ticker]).get(ticker)
    if not code:
        raise RuntimeError(f"{ticker} 토스 종목 코드를 찾지 못했습니다.")
    return f"{TOSS_INVEST_API_BASE_URL}/api/v1/c-chart/us-s/{code}/min:1"


def fetch_bars(ticker: str, country: str, start: datetime, end: datetime) -> list[list]:
    """``end`` 부터 과거로 이어 받아 ``start`` 이후 1분봉을 과거→최신 순으로 돌려준다."""
    url = _chart_url(ticker, country)
    rows: dict[int, list] = {}
    cursor = end.isoformat()
    while True:
        resp = toss_request("GET", url, params={"count": PAGE_SIZE, "from": cursor}, timeout=10)
        resp.raise_for_status()
        result = resp.json()["result"]
        candles = result.get("candles") or []
        for candle in candles:
            at = datetime.fromisoformat(candle["dt"])
            rows[int(at.timestamp() * 1000)] = [
                candle["dt"],
                float(candle["open"]),
                float(candle["high"]),
                float(candle["low"]),
                float(candle["close"]),
            ]
        oldest = min((datetime.fromisoformat(c["dt"]) for c in candles), default=None)
        next_cursor = result.get("nextDateTime")
        if oldest is None or oldest < start or next_cursor in (None, cursor):
            break
        cursor = next_cursor
        time.sleep(0.2)
    lo, hi = start.timestamp() * 1000, end.timestamp() * 1000
    return [[ms, *rows[ms]] for ms in sorted(rows) if lo <= ms <= hi]


def main() -> int:
    for ticker, country in TICKERS.items():
        windows = select_windows(ticker)
        for regime in REGIMES:
            path = DATA_DIR / ticker / f"{regime}.json"
            window = windows[regime]
            if path.exists():
                print(f"{ticker} {regime}: 이미 있음 — 건너뜀 ({path})")
                continue
            bars = fetch_bars(ticker, country, *_bounds(country, window["start"], window["end"]))
            if not bars:
                raise RuntimeError(f"{ticker} {regime} 분봉을 받지 못했습니다.")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"ticker": ticker, "country": country, "regime": regime, **window, "bars": bars})
            )
            print(
                f"{ticker} {regime}: {window['start']}~{window['end']} 수익률 {window['return_pct']:+.1f}% "
                f"폭 {window['range_pct']:.1f}% · 분봉 {len(bars):,}개"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
