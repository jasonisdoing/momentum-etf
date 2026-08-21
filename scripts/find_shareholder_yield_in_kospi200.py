"""kospi200 종목풀에서 **주주환원 수익률**이 높은 종목을 뽑아 콘솔에 출력한다 (한국판 DIVB).

미국 DIVB(iShares U.S. Dividend and Buyback ETF)가 추종하는
**Morningstar US Dividend and Buyback Index** 의 계산 정의를 그대로 옮겼다.
"이익을 얼마나 냈나"가 아니라 **"내 지분 가치 대비 현금을 얼마나 돌려줬나"** 로 줄 세운다.

    총주주환원율 = 배당수익률 + 자사주(순매입)수익률

지수 원문(2020-04 Construction Rules)의 정의와 이 스크립트의 대응
  · 배당수익률   : 데이터 기준일 직전 **12개월** 주당배당금 합 ÷ 주가
                   → OpenDART 배당 공시의 **최근 확정 회계연도** 보통주 주당 현금배당금 ÷ 현재가.
                      (지급일 기준 12개월 합 대신 회계연도 기준 — 국내 연 1~2회 배당에서는
                      사실상 같고, 공식 공시 값이라 더 정확하다)
  · 자사주수익률 : 최근 **8분기** 순증자현금흐름 합의 부호를 뒤집어 **÷ 2 ÷ 시가총액**
                   → OpenDART 현금흐름표 **최근 2개 회계연도**의
                      (자기주식 취득 − 자기주식 처분 − 주식 발행) 합 ÷ 2 ÷ 시가총액.
                      **순매입**이라 자사주를 사도 그만큼 증자하면 상쇄된다(발행이 더 많으면 음수).
  · 편입 조건    : 총주주환원율 **0.1% 초과**. 배당·자사주 중 **하나만 해도** 편입된다.

지수와 일부러 다르게 한 것
  · 지수는 환원 **금액**이 큰 순으로 담아 벤치마크 총환원액의 90%를 채우고, 환원금액으로
    가중해 4.9% 상한을 둔다. 여기서는 그 선정·가중 단계를 쓰지 않고 **환원율 순으로 나열**한다
    (지수 복제가 아니라 종목을 찾는 게 목적이라, 같은 폴더의 저평가 스크립트와 같은 방식).
  · **우선주는 제외한다.** 배당은 종목별 주당배당금이라 정확하지만, 자사주는 회사 전체
    순매입액을 그 종목 시가총액으로 나누게 되어 우선주만 값이 부풀려진다(짝이 안 맞는다).

데이터 소스
  - 유니버스: kospi200 종목풀 — 종목풀 화면이 단일 소스
  - 시가총액·현재가: 네이버 시가총액 순위 API (/kor-market-stock 화면과 동일 함수)
  - 주당배당금·현금흐름표: OpenDART 사업보고서 (services/opendart_service —
    공식 API, .env 에 DART_API_KEY 필요)

사용 예
    python scripts/find_shareholder_yield_in_kospi200.py
    python scripts/find_shareholder_yield_in_kospi200.py --top 50 --workers 8
    python scripts/find_shareholder_yield_in_kospi200.py --detail 10
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from utils.kor_stock_market_service import load_kor_stock_market  # noqa: E402
from utils.logger import get_app_logger  # noqa: E402
from utils.report import render_table_eaw  # noqa: E402
from utils.stock_list_io import get_etfs  # noqa: E402

logger = get_app_logger()

# ── 조정 가능한 기준 ────────────────────────────────────────────────────────
# 유니버스로 쓸 종목풀들 — 합집합으로 쓴다. 종목풀 화면에서 관리하는 목록이 그대로 대상이 된다.
UNIVERSE_POOLS = ("kospi200",)

# 편입 하한 — 지수 원문의 "total shareholder yield above 0.1%".
MIN_SHAREHOLDER_YIELD = 0.001

# 자사주수익률 계산 창(회계연도 수) — 지수 원문의 8분기(2년)에 대응. 합을 이 값으로 나눠 연율화한다.
BUYBACK_WINDOW_YEARS = 2

# 우선주 제외 여부. 자사주는 회사 전체 순매입액이라 우선주 시총으로 나누면 과대 계산된다.
EXCLUDE_PREFERRED = True

# 네이버 시총 순위에서 가져올 시장별 상위 종목 수 — 종목풀 종목의 시가총액을 여기서 찾는다.
MARKET_CAP_FETCH_LIMIT = 200


def _year_of(label: Any) -> str:
    """컬럼 라벨(Timestamp 또는 연도 정수)에서 연도 문자열을 뽑는다."""
    if isinstance(label, (int, float)):
        return str(int(label))
    return str(pd.Timestamp(label).year)


def _format_jo(value: float) -> str:
    """원 단위 금액을 조/억 단위로 읽기 쉽게 만든다. 음수는 부호를 유지한다."""
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1e12:
        return f"{sign}{magnitude / 1e12:.1f}조"
    if magnitude >= 1e8:
        return f"{sign}{magnitude / 1e8:.0f}억"
    return f"{sign}{magnitude:.0f}"


def _is_preferred_stock(ticker: str) -> bool:
    """우선주 여부 — 국내 6자리 종목코드는 보통주가 항상 0 으로 끝난다.

    우선주는 끝자리가 5(구형)·7(신형 2우B)·K/L/M(종류주) 등으로 갈린다.
    예: 005930 삼성전자(보통주) vs 005935 삼성전자우, 00680K 미래에셋증권2우B.
    """
    code = str(ticker or "").strip()
    return len(code) == 6 and not code.endswith("0")


# OpenDART 조회에 쓰는 공유 상태 — main() 이 채운 뒤 워커(_analyze)들이 읽기만 한다.
_CORP_CODES: dict[str, str] = {}
_LATEST_YEAR: int = 0


def _latest_confirmed_dps(corp_code: str) -> tuple[float, int | None]:
    """최근 확정 회계연도의 보통주 주당 현금배당금과 그 연도. 공시가 없으면 (0, None)."""
    from services.opendart_service import dps_history

    history = dps_history(corp_code, _LATEST_YEAR)
    for year in sorted(history, reverse=True):
        return float(history[year]), year
    return 0.0, None


def _annualized_net_buyback(corp_code: str) -> tuple[float, list[str]]:
    """최근 BUYBACK_WINDOW_YEARS 개 회계연도의 **연평균 순매입액**과 사용한 연도 목록.

    지수 정의는 8분기 순증자현금흐름 합의 부호를 뒤집어 2로 나눈 값이다. 연간 값 2개년을
    쓰면 같은 결과가 된다. 순매입 = 자기주식 취득 − 자기주식 처분 − 주식 발행.
    항목이 없는 해는 그 해에 해당 활동이 없었다는 뜻이라 0 으로 본다.

    반환값이 음수면 순발행(증자)이 더 많았다는 뜻이고, 지수도 그대로 환원율을 깎는다.
    """
    from services.opendart_service import annual_financials

    total = 0.0
    years: list[str] = []
    for year in range(_LATEST_YEAR - BUYBACK_WINDOW_YEARS + 1, _LATEST_YEAR + 1):
        financials = annual_financials(corp_code, year)
        if financials is None:  # 그 연도 사업보고서 자체가 없음 (신규 상장 등)
            continue
        years.append(str(year))
        total += (
            financials.get("buyback", 0.0)
            - financials.get("buyback_disposal", 0.0)
            - financials.get("share_issuance", 0.0)
        )
    return total / BUYBACK_WINDOW_YEARS, years


def _analyze(row: dict[str, Any]) -> dict[str, Any] | None:
    """한 종목의 총주주환원율을 계산한다. 제외 대상은 사유를 담아 돌려준다."""
    ticker = str(row.get("ticker") or "").strip()
    name = str(row.get("name") or ticker)
    market_cap = row.get("market_cap")
    price = row.get("current_price")
    base = {"ticker": ticker, "name": name, "market": row.get("market") or ""}

    if EXCLUDE_PREFERRED and _is_preferred_stock(ticker):
        return {**base, "excluded_reason": "우선주(자사주 기준이 회사 전체라 짝이 안 맞음)"}
    if not market_cap or float(market_cap) <= 0 or not price or float(price) <= 0:
        return {**base, "excluded_reason": "시가총액·현재가 없음"}
    # 네이버 시총은 억원 단위라 현금흐름표(원)와 자릿수를 맞춘다.
    market_cap_won = float(market_cap) * 1e8

    corp_code = _CORP_CODES.get(ticker)
    if not corp_code:
        return {**base, "excluded_reason": "DART 고유번호 없음"}
    try:
        dps, dps_year = _latest_confirmed_dps(corp_code)
        buyback_amount, buyback_years = _annualized_net_buyback(corp_code)
    except Exception as exc:
        logger.debug("재무 조회 실패 (%s): %s", ticker, exc)
        return {**base, "excluded_reason": "재무 조회 실패"}

    if not buyback_years and dps_year is None:
        return {**base, "excluded_reason": "재무 조회 실패"}

    dividend_yield = dps / float(price)
    buyback_yield = buyback_amount / market_cap_won
    shareholder_yield = dividend_yield + buyback_yield

    if shareholder_yield <= MIN_SHAREHOLDER_YIELD:
        reason = "환원 없음" if shareholder_yield == 0 else f"환원율 {MIN_SHAREHOLDER_YIELD * 100:.1f}% 이하"
        return {**base, "excluded_reason": reason}

    return {
        **base,
        "excluded_reason": None,
        "market_cap_won": market_cap_won,
        "current_price": float(price),
        "dps": dps,
        "dps_year": dps_year,
        "buyback_amount": buyback_amount,
        "buyback_years": buyback_years,
        "dividend_yield": dividend_yield,
        "buyback_yield": buyback_yield,
        "shareholder_yield": shareholder_yield,
        # 지수의 'total shareholder payout dollars' — 환원율 × 시가총액.
        "payout_amount": shareholder_yield * market_cap_won,
    }


def _load_universe() -> list[dict[str, Any]]:
    """kospi200 종목풀 종목에 네이버 시가총액·현재가를 붙인 목록.

    종목풀 문서에는 시가총액이 없어(가격지표 배치가 담당하지 않는다) 네이버 시총 순위에서
    가져와 티커로 맞춘다. 시총 순위 밖의 종목은 값 없음으로 남겨 관문에서 걸린다.
    """
    pool_items: list[dict[str, Any]] = []
    seen_tickers: set[str] = set()
    for pool in UNIVERSE_POOLS:
        for item in get_etfs(pool):
            ticker = str(item.get("ticker") or "").strip()
            if item.get("is_etf") or not ticker or ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)
            pool_items.append(item)
    if not pool_items:
        raise SystemExit(f"{UNIVERSE_POOLS} 종목풀에 종목이 없습니다.")

    cap_by_ticker: dict[str, dict[str, Any]] = {}
    for market in ("KOSPI",):
        payload = load_kor_stock_market(market, limit=MARKET_CAP_FETCH_LIMIT, min_market_cap_jo=0)
        for row in payload.get("rows") or []:
            key = str(row.get("ticker") or "").strip()
            if key:
                cap_by_ticker[key] = row

    universe: list[dict[str, Any]] = []
    for item in pool_items:
        ticker = str(item.get("ticker") or "").strip()
        if not ticker:
            continue
        quote = cap_by_ticker.get(ticker) or {}
        universe.append(
            {
                "ticker": ticker,
                "name": item.get("name") or ticker,
                "market": item.get("market") or quote.get("market") or "",
                "market_cap": quote.get("market_cap"),
                "current_price": quote.get("current_price"),
            }
        )
    return universe


def _print_detail(rank: int, item: dict[str, Any]) -> None:
    """한 종목의 환원 내역을 배당·자사주로 갈라 보여준다."""
    years = "~".join(item["buyback_years"]) if item["buyback_years"] else "-"
    print(f"\n[{rank}위] {item['name']}({item['ticker']}) · {item['market']} · 현재가 {item['current_price']:,.0f}원")
    headers = ["항목", "기준", "금액", "수익률"]
    rows = [
        [
            "배당",
            f"{item['dps_year']}년 확정" if item.get("dps_year") else "-",
            f"주당 {item['dps']:,.0f}원",
            f"{item['dividend_yield'] * 100:.2f}%",
        ],
        [
            "자사주(순매입)",
            f"{years} 연평균",
            _format_jo(item["buyback_amount"]),
            f"{item['buyback_yield'] * 100:.2f}%",
        ],
        ["합계", "", _format_jo(item["payout_amount"]), f"{item['shareholder_yield'] * 100:.2f}%"],
    ]
    for line in render_table_eaw(headers, rows, ["left", "left", "right", "right"]):
        print(f"  {line}")
    print(f"  시가총액 {_format_jo(item['market_cap_won'])}")


def main() -> None:
    global _LATEST_YEAR

    parser = argparse.ArgumentParser(description="국내 종목풀 주주환원 수익률 상위 종목 찾기 (한국판 DIVB)")
    parser.add_argument("--top", type=int, default=30, help="출력할 상위 종목 수 (기본 30).")
    parser.add_argument("--workers", type=int, default=8, help="OpenDART 병렬 조회 워커 수 (기본 8).")
    parser.add_argument("--detail", type=int, default=5, help="환원 내역을 출력할 상위 종목 수 (기본 5).")
    args = parser.parse_args()

    from services.opendart_service import corp_code_by_stock, latest_annual_year
    from utils.env import load_env_if_present

    load_env_if_present()  # DART_API_KEY
    _CORP_CODES.update(corp_code_by_stock())
    _LATEST_YEAR = latest_annual_year()

    print(f"{'+'.join(UNIVERSE_POOLS)} 종목풀 명단과 시가총액을 불러옵니다...")
    universe = _load_universe()
    print(f"  {len(universe)}종목 확보 — 재무 데이터 조회 시작 (워커 {args.workers}개, 최근 사업연도 {_LATEST_YEAR})")

    results: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(_analyze, universe), 1):
            if index % 25 == 0:
                print(f"    ... {index}/{len(universe)} 조회")
            if not result:
                continue
            if result.get("excluded_reason"):
                excluded.append(result)
            else:
                results.append(result)

    results.sort(key=lambda r: (-r["shareholder_yield"], -r["market_cap_won"]))
    print(f"\n조사 완료 — 편입 {len(results)}개 / 제외 {len(excluded)}개")

    if excluded:
        by_reason: dict[str, list[str]] = {}
        for item in excluded:
            by_reason.setdefault(item["excluded_reason"], []).append(f"{item['name']}({item['ticker']})")
        print("\n[제외 내역]")
        for reason, names in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            sample = ", ".join(names[:8]) + (f" 외 {len(names) - 8}개" if len(names) > 8 else "")
            print(f"  · {reason} — {len(names)}개: {sample}")
    print()

    # ── 1. 전체 순위 리스트 ──────────────────────────────────────────
    headers = ["순위", "티커", "종목명", "시장", "환원률", "배당률", "자사주률", "주당배당", "자사주", "환원액", "시총"]
    aligns = ["right", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right"]
    table_rows: list[list[str]] = []
    for rank, item in enumerate(results[: args.top], 1):
        table_rows.append(
            [
                str(rank),
                item["ticker"],
                item["name"],
                item["market"],
                f"{item['shareholder_yield'] * 100:.2f}%",
                f"{item['dividend_yield'] * 100:.2f}%",
                f"{item['buyback_yield'] * 100:.2f}%",
                f"{item['dps']:,.0f}",
                _format_jo(item["buyback_amount"]),
                _format_jo(item["payout_amount"]),
                _format_jo(item["market_cap_won"]),
            ]
        )
    for line in render_table_eaw(headers, table_rows, aligns):
        print(line)

    # ── 2. 항목 설명 ────────────────────────────────────────────────
    print(
        "\n항목 설명 — 환원률: 배당률 + 자사주률. 이 값 하나로 순위를 매긴다(DIVB 지수 정의)"
        "\n           배당률: 최근 확정 회계연도 주당 현금배당금(DART 공시) / 현재가"
        f"\n           자사주률: 최근 {BUYBACK_WINDOW_YEARS}개 회계연도 **순매입**(취득 − 처분 − 발행) 연평균 / 시가총액"
        "\n                     증자가 매입보다 많으면 음수가 되어 환원률을 깎는다"
        "\n           주당배당·자사주: 위 두 값의 분자 — 배당은 1주 기준, 자사주는 회사 전체 금액"
        "\n           환원액: 환원률 × 시가총액 (지수의 total shareholder payout dollars)"
        f"\n대상: {'+'.join(UNIVERSE_POOLS)} 종목풀 중 환원률이 {MIN_SHAREHOLDER_YIELD * 100:.1f}% 를 넘는 보통주"
        "\n      (지수와 달리 90% 커버리지 선정·환원액 가중은 쓰지 않고 환원률 순으로 나열한다)"
        "\n유니버스나 기준을 바꾸려면 스크립트 상단의 상수를 수정하세요."
    )

    # ── 3. 상위 N개 환원 내역 ────────────────────────────────────────
    detail_count = min(args.detail, len(results))
    if detail_count > 0:
        print(f"\n{'=' * 60}\n상위 {detail_count}개 환원 내역\n{'=' * 60}")
        for rank, item in enumerate(results[:detail_count], 1):
            _print_detail(rank, item)


if __name__ == "__main__":
    main()
