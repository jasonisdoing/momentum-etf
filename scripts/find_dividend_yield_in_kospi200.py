"""kospi200 종목풀의 **최근 3년 배당수익률**을 나열한다 (올해는 컨센서스).

배당금 자체가 아니라 **배당률**을 연도별로 나란히 놓아, 배당이 늘어도 주가가 더 올라
실질 수익률은 떨어졌는지 같은 것을 한눈에 보게 한다. 정렬은 올해 컨센서스 배당률 순.

배당률의 분모 — **그 해에 실제로 받았을 수익률**
  · 지난 해들 : 그 해 **연말 종가**  (2025년 배당률 = 2025 DPS ÷ 2025-12-30 종가)
  · 올해      : **현재가**           (아직 연말 종가가 없다. 컨센서스 DPS 를 쓴다)
  분모가 해마다 달라 배당금 추세와 배당률 추세는 다르게 움직인다 — 그게 이 표의 요점이다.

PER·PBR 은 네이버 연간 재무의 해당 연도 값이다. 올해 값은 컨센서스라 `*` 를 붙인다.

데이터 소스
  - 유니버스: kospi200 종목풀 — 종목풀 화면이 단일 소스
  - 현재가: 네이버 시가총액 순위 API (/kor-market-stock 화면과 동일 함수)
  - 연말 종가: 종목풀 가격 캐시 (배치가 채우는 OHLCV)
  - 주당배당금·PER·PBR: 네이버 연간 재무 API — 컨센서스 연도(isConsensus=Y) 포함

사용 예
    python scripts/find_dividend_yield_in_kospi200.py
    python scripts/find_dividend_yield_in_kospi200.py --top 50 --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import urllib.request
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from utils.cache_utils import load_cached_frames_bulk_from_ticker_types  # noqa: E402
from utils.kor_stock_market_service import load_kor_stock_market  # noqa: E402
from utils.logger import get_app_logger  # noqa: E402
from utils.report import render_table_eaw  # noqa: E402
from utils.stock_list_io import get_etfs  # noqa: E402

logger = get_app_logger()

# ── 조정 가능한 기준 ────────────────────────────────────────────────────────
# 유니버스로 쓸 종목풀들 — 합집합으로 쓴다. 종목풀 화면에서 관리하는 목록이 그대로 대상이 된다.
UNIVERSE_POOLS = ("kospi200",)

# 표에 보여줄 상위 종목 수 (--top 으로 덮어쓸 수 있다).
DEFAULT_TOP = 20

# 배당률을 보여줄 연도 수 — 올해(컨센서스) 포함. 3이면 `2026(예) 2025 2024`.
YEARS_TO_SHOW = 3

# 함께 보여줄 주가 수익률 구간(개월). 배당률이 높아진 게 배당 증가인지 주가 하락인지 구분하는 데 쓴다.
RETURN_MONTHS = (1, 3, 6, 12)

# 네이버 시총 순위에서 가져올 상위 종목 수 — 종목풀 종목의 현재가를 여기서 찾는다.
MARKET_CAP_FETCH_LIMIT = 300

# 동시 조회 워커 수 (--workers 로 덮어쓸 수 있다). 네이버 API 를 종목당 1회 호출한다.
DEFAULT_WORKERS = 6

_NAVER_ANNUAL_URL = "https://m.stock.naver.com/api/stock/{ticker}/finance/annual"
_NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
# 연도별로 통째로 보존할 행 — 표의 각 컬럼이 된다.
_ROWS_BY_YEAR = {"주당배당금": "dps", "PER": "per", "PBR": "pbr"}


def _parse_number(value: Any) -> float | None:
    text = str(value or "").replace(",", "").strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fetch_naver_annual(ticker: str) -> dict[str, Any] | None:
    """네이버 연간 재무에서 연도별 주당배당금·PER·PBR 과 컨센서스 연도를 가져온다.

    반환: {"consensus_year": "2026", "dps": {"2024": 995.0, ...}, "per": {...}, "pbr": {...}}
    컨센서스 연도가 없으면(추정치 미제공 종목) None — 올해 값이 없으면 표의 정렬 기준이 없다.
    """
    try:
        request = urllib.request.Request(_NAVER_ANNUAL_URL.format(ticker=ticker), headers=_NAVER_HEADERS)
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    except Exception as exc:
        logger.debug("%s 네이버 연간 재무 조회 실패: %s", ticker, exc)
        return None

    finance_info = (payload or {}).get("financeInfo") or {}
    consensus_keys = [
        str(title.get("key"))
        for title in finance_info.get("trTitleList") or []
        if str(title.get("isConsensus") or "") == "Y"
    ]
    if not consensus_keys:
        return None

    result: dict[str, Any] = {"consensus_year": consensus_keys[-1][:4]}
    for row in finance_info.get("rowList") or []:
        field = _ROWS_BY_YEAR.get(str(row.get("title") or "").strip())
        if not field:
            continue
        result[field] = {
            str(key)[:4]: parsed
            for key, cell in (row.get("columns") or {}).items()
            if (parsed := _parse_number((cell or {}).get("value"))) is not None
        }
    return result if result.get("dps") else None


def _load_universe() -> list[dict[str, Any]]:
    """kospi200 종목풀 종목에 네이버 현재가를 붙인 목록.

    종목풀 문서에는 시세가 없어(가격지표 배치가 담당하지 않는다) 네이버 시총 순위에서
    가져와 티커로 맞춘다. 시총 순위 밖의 종목은 현재가 없음으로 남아 올해 배당률을 못 낸다.
    """
    pool_items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pool in UNIVERSE_POOLS:
        for item in get_etfs(pool):
            ticker = str(item.get("ticker") or "").strip()
            if item.get("is_etf") or not ticker or ticker in seen:
                continue
            seen.add(ticker)
            pool_items.append(item)
    if not pool_items:
        raise SystemExit(f"{UNIVERSE_POOLS} 종목풀에 종목이 없습니다.")

    quote_by_ticker: dict[str, dict[str, Any]] = {}
    payload = load_kor_stock_market("KOSPI", limit=MARKET_CAP_FETCH_LIMIT, min_market_cap_jo=0)
    for row in payload.get("rows") or []:
        key = str(row.get("ticker") or "").strip()
        if key:
            quote_by_ticker[key] = row

    return [
        {
            "ticker": str(item["ticker"]).strip(),
            "name": item.get("name") or item["ticker"],
            "current_price": (quote_by_ticker.get(str(item["ticker"]).strip()) or {}).get("current_price"),
        }
        for item in pool_items
    ]


def _price_history(tickers: list[str], years: list[str]) -> dict[str, dict[str, Any]]:
    """티커별 {"year_end": {연도: 종가}, "past": {개월: 그 시점 종가}}.

    캐시 조회는 한 번만 한다 — 연말 종가(배당률 분모)와 기간 수익률이 같은 시계열을 쓴다.
    """
    frames = load_cached_frames_bulk_from_ticker_types(UNIVERSE_POOLS, tickers)
    wanted_years = {int(year) for year in years}
    today = pd.Timestamp.today().normalize()

    result: dict[str, dict[str, Any]] = {}
    for ticker, frame in frames.items():
        if frame is None or frame.empty or "Close" not in frame.columns:
            continue
        close = frame["Close"].astype(float).dropna()
        close.index = pd.to_datetime(close.index)
        if close.empty:
            continue

        year_end = {
            str(year): float(series.iloc[-1])
            for year, series in close.groupby(close.index.year)
            if int(year) in wanted_years and len(series)
        }
        # N개월 전 종가 — 그 날이 휴장이면 직전 거래일을 쓴다(asof).
        past: dict[int, float] = {}
        for months in RETURN_MONTHS:
            target = today - pd.DateOffset(months=months)
            earlier = close[close.index <= target]
            if len(earlier):
                past[months] = float(earlier.iloc[-1])
        result[ticker] = {"year_end": year_end, "past": past}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="kospi200 종목풀의 최근 3년 배당수익률 (올해는 컨센서스)")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help=f"표에 보여줄 상위 종목 수 (기본 {DEFAULT_TOP})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"동시 조회 수 (기본 {DEFAULT_WORKERS})")
    args = parser.parse_args()

    print(f"{'+'.join(UNIVERSE_POOLS)} 종목풀 명단과 현재가를 불러옵니다...")
    universe = _load_universe()
    print(f"  {len(universe)}종목 확보 — 네이버 연간 재무 조회 시작 (워커 {args.workers}개)")

    annual_by_ticker: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_fetch_naver_annual, item["ticker"]): item["ticker"] for item in universe}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            if index % 25 == 0:
                print(f"    ... {index}/{len(universe)} 조회")
            annual = future.result()
            if annual:
                annual_by_ticker[futures[future]] = annual

    if not annual_by_ticker:
        raise SystemExit("컨센서스가 있는 종목을 찾지 못했습니다. 네이버 응답을 확인하세요.")

    # 올해(컨센서스 연도)는 종목마다 같다고 보고 가장 많이 나온 값을 기준 연도로 삼는다.
    this_year = max(
        {annual["consensus_year"] for annual in annual_by_ticker.values()},
        key=lambda year: sum(1 for a in annual_by_ticker.values() if a["consensus_year"] == year),
    )
    years = [str(int(this_year) - offset) for offset in range(YEARS_TO_SHOW)]  # 최신 → 과거
    past_years = years[1:]

    print(f"  가격 이력을 불러옵니다 (연말 종가 {', '.join(past_years)} · 기간 수익률)...")
    history_by_ticker = _price_history([item["ticker"] for item in universe], past_years)

    results: list[dict[str, Any]] = []
    for item in universe:
        ticker = item["ticker"]
        annual = annual_by_ticker.get(ticker)
        if not annual or annual["consensus_year"] != this_year:
            continue
        price = item.get("current_price")
        if not price:
            continue  # 현재가가 없으면 올해 배당률을 낼 수 없다 — 정렬 기준이 사라진다

        dps_by_year = annual.get("dps") or {}
        history = history_by_ticker.get(ticker) or {}
        year_end = history.get("year_end") or {}
        # 분모: 올해는 현재가, 지난 해들은 그 해 연말 종가.
        yields_by_year: dict[str, float | None] = {}
        for year in years:
            dps = dps_by_year.get(year)
            base = float(price) if year == this_year else year_end.get(year)
            yields_by_year[year] = (dps / base * 100.0) if (dps is not None and base) else None

        if yields_by_year[this_year] is None:
            continue  # 올해 예상 배당이 없으면(무배당 예상) 이 표의 대상이 아니다

        # 기간 수익률 — 분자는 현재가(배당률 올해 분모와 같은 값)라 표 안에서 기준이 일관된다.
        past = history.get("past") or {}
        returns = {
            months: (float(price) / base - 1.0) * 100.0 if (base := past.get(months)) else None
            for months in RETURN_MONTHS
        }

        results.append(
            {
                "ticker": ticker,
                "name": item["name"],
                "price": float(price),
                "returns": returns,
                "yields": yields_by_year,
                "per": (annual.get("per") or {}).get(this_year),
                "pbr": (annual.get("pbr") or {}).get(this_year),
            }
        )

    results.sort(key=lambda row: -row["yields"][this_year])
    print(f"\n조사 완료 — 배당률 산출 {len(results)}개 / 전체 {len(universe)}개")

    # ── 표 설명은 표 **위에** 둔다 — 숫자를 읽기 전에 분모가 무엇인지 알아야 한다.
    print(
        f"\n{'=' * 72}"
        f"\n배당률 = 주당배당금 ÷ 주가. **그 해에 실제로 받았을 수익률**이라 분모가 해마다 다르다."
        f"\n  · {', '.join(past_years)} : 그 해 **연말 종가** 기준 (확정 배당금)"
        f"\n  · {this_year}       : **현재가** 기준 (연말 종가가 아직 없다 · 컨센서스 배당금)"
        f"\n기간 수익률(1개월~1년)은 현재가 대비 주가 등락이다 — 배당률이 오른 게 배당 증가 때문인지"
        f"\n  주가 하락 때문인지 구분하는 데 쓴다."
        f"\nPER·PBR 은 {this_year} 컨센서스 값이다. 정렬은 {this_year} 배당률 높은 순."
        f"\n{'=' * 72}"
    )

    return_labels = [f"{months // 12}년" if months % 12 == 0 else f"{months}개월" for months in RETURN_MONTHS]
    headers = ["순위", "티커", "종목명", "현재가", *return_labels, f"{this_year}(예)", *past_years, "PER", "PBR"]
    aligns = ["right", "left", "left", *["right"] * (1 + len(RETURN_MONTHS) + YEARS_TO_SHOW + 2)]
    table_rows: list[list[str]] = []
    for rank, row in enumerate(results[: args.top], 1):
        table_rows.append(
            [
                str(rank),
                row["ticker"],
                row["name"],
                f"{row['price']:,.0f}",
                *[
                    f"{row['returns'][months]:+.1f}%" if row["returns"][months] is not None else "-"
                    for months in RETURN_MONTHS
                ],
                *[f"{row['yields'][year]:.2f}%" if row["yields"][year] is not None else "-" for year in years],
                f"{row['per']:.1f}" if row["per"] is not None else "-",
                f"{row['pbr']:.2f}" if row["pbr"] is not None else "-",
            ]
        )
    for line in render_table_eaw(headers, table_rows, aligns):
        print(line)

    print(f"\n대상: {'+'.join(UNIVERSE_POOLS)} 종목풀 · 상위 {min(args.top, len(results))}개 표시")
    print("유니버스나 표시 개수 기본값을 바꾸려면 스크립트 상단의 상수를 수정하세요.")


if __name__ == "__main__":
    main()
