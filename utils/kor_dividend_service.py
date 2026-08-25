"""한국 배당주 화면(`/kor-dividend`) 데이터 — 배치가 만들고 화면은 읽기만 한다.

종목당 OpenDART 사업보고서·배당공시와 네이버 연간 재무를 부르므로 200종목이면 외부 호출이
1,000건을 넘는다. 화면 진입마다 돌릴 수 없어 **배치가 계산해 DB 에 적재**하고, 화면은
그 문서를 읽는다 (`stock_cache_meta`·`market_breadth` 와 같은 구조).

유니버스는 **KODEX 200(069500) 의 ETF 보유종목**이다. 지수 추종 ETF 의 보유종목이 곧 지수
구성종목이라, 종목풀처럼 사람이 관리하지 않아도 운용사 리밸런싱이 그대로 반영된다.
대체 ETF 로 물러서지 않는다 — 조용히 옛 목록을 쓰면 화면이 틀린 유니버스로 계산한 값을
맞는 것처럼 보여준다. 실패하면 배치가 죽고 러너가 슬랙으로 알린다.
"""

from __future__ import annotations

import concurrent.futures
import json
import urllib.request
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from utils.index_constituents_loader import load_index_constituents, save_index_constituents
from utils.logger import get_app_logger

logger = get_app_logger()

# 유니버스로 쓸 지수와 그 지수를 추종하는 ETF. 이 ETF 의 보유종목이 곧 구성종목이다.
INDEX_KEY = "KOSPI200"
SOURCE_ETF_TICKER = "069500"  # KODEX 200
SOURCE_ETF_NAME = "KODEX 200"

# 구성종목 수가 이보다 적으면 응답이 깨진 것으로 본다 (KOSPI200 은 200종목 안팎).
MIN_CONSTITUENTS = 150

# 배치가 적재하는 컬렉션 — 종목당 문서 1개.
COLLECTION = "kor_dividend_stocks"

# 추세·연도별 표에 쓸 시작 회계연도. 이 연도부터 최근 확정연도까지 DART 사업보고서를 읽는다.
TREND_START_YEAR = 2023

# 자사주 순매입을 연율화할 창(회계연도 수) — DIVB 지수 원문의 8분기(2년)에 대응.
BUYBACK_WINDOW_YEARS = 2

# 종목당 외부 호출이 5~6건이라 병렬로 돈다. DART 는 키당 일 20,000건 한도라 여유가 크다.
FETCH_WORKERS = 8

_NAVER_ANNUAL_URL = "https://m.stock.naver.com/api/stock/{ticker}/finance/annual"
_NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
# 네이버 연간 재무에서 연도별로 통째로 보존할 행.
_NAVER_ROWS = {"주당배당금": "dps", "PER": "per", "PBR": "pbr"}


def refresh_kospi200_constituents() -> dict[str, Any]:
    """KODEX 200 보유종목을 읽어 ``index_constituents`` 에 KOSPI200 으로 저장한다.

    Returns:
        {"count": 저장 종목 수, "as_of_date": ETF 보유종목 기준일|None}

    Raises:
        RuntimeError: 보유종목을 못 가져왔거나 수가 비정상일 때. 배치는 여기서 죽어야 한다.
    """
    from services.etf_holdings_service import fetch_korean_etf_holdings_from_naver

    payload = fetch_korean_etf_holdings_from_naver(SOURCE_ETF_TICKER)
    holdings = (payload or {}).get("holdings") or []

    tickers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in holdings:
        ticker = str(item.get("ticker") or "").strip().upper()
        # 현금·선물 등 종목코드가 아닌 항목이 섞여 오므로 6자리 코드만 남긴다.
        if len(ticker) != 6 or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(
            {
                "ticker": ticker,
                "name": str(item.get("name") or ticker).strip(),
                # 지수 내 비중 — 규모·유동성의 대리 지표라 화면 필터에 쓴다.
                "weight": float(item["weight"]) if item.get("weight") is not None else None,
            }
        )

    if len(tickers) < MIN_CONSTITUENTS:
        raise RuntimeError(
            f"{SOURCE_ETF_NAME}({SOURCE_ETF_TICKER}) 보유종목이 {len(tickers)}개뿐입니다 "
            f"(최소 {MIN_CONSTITUENTS}개 기대). 네이버 응답이 바뀌었거나 조회에 실패했습니다."
        )

    as_of_date = (payload or {}).get("as_of_date")
    save_index_constituents(
        INDEX_KEY,
        tickers,
        {
            "updated_at": date.today().isoformat(),
            "source": f"{SOURCE_ETF_NAME}({SOURCE_ETF_TICKER}) ETF 보유종목",
            "as_of_date": as_of_date,
        },
    )
    logger.info("[KOR DIVIDEND] %s 구성종목 %d개 저장 (기준일 %s)", INDEX_KEY, len(tickers), as_of_date or "-")
    return {"count": len(tickers), "as_of_date": as_of_date}


def load_universe() -> list[dict[str, Any]]:
    """저장된 KOSPI200 구성종목. 배치가 아직 안 돌았으면 LookupError 가 난다."""
    return load_index_constituents(INDEX_KEY)


# ── [2] 종목별 지표 수집 ────────────────────────────────────────────────────
#
# 여기서 만드는 값은 **가격과 무관한 것만**이다. 배당금·재무·추세는 하루에 한 번이면 되지만
# 배당률·자사주률은 분모가 주가라 장중에 계속 변한다. 파생값까지 굳혀 두면 화면이 아침 8시
# 가격으로 계산한 값을 하루 종일 보여주게 되므로, 연도별 **원시값**을 남기고 가격이 필요한
# 계산은 조회 시점에 한다.


def _parse_number(value: Any) -> float | None:
    text = str(value or "").replace(",", "").strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fetch_naver_annual(ticker: str) -> dict[str, Any]:
    """네이버 연간 재무 — 연도별 주당배당금·PER·PBR 과 컨센서스 연도.

    반환: {"consensus_year": "2026"|None, "dps": {연도: 원}, "per": {...}, "pbr": {...}}
    조회에 실패하면 빈 dict (컨센서스가 없는 종목도 있어 이것만으로 제외하지 않는다).
    """
    try:
        request = urllib.request.Request(_NAVER_ANNUAL_URL.format(ticker=ticker), headers=_NAVER_HEADERS)
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    except Exception as exc:
        logger.debug("%s 네이버 연간 재무 조회 실패: %s", ticker, exc)
        return {}

    finance_info = (payload or {}).get("financeInfo") or {}
    consensus_keys = [
        str(title.get("key"))
        for title in finance_info.get("trTitleList") or []
        if str(title.get("isConsensus") or "") == "Y"
    ]
    result: dict[str, Any] = {"consensus_year": consensus_keys[-1][:4] if consensus_keys else None}
    for row in finance_info.get("rowList") or []:
        field = _NAVER_ROWS.get(str(row.get("title") or "").strip())
        if not field:
            continue
        result[field] = {
            str(key)[:4]: parsed
            for key, cell in (row.get("columns") or {}).items()
            if (parsed := _parse_number((cell or {}).get("value"))) is not None
        }
    return result


_NAVER_INTEGRATION_URL = "https://m.stock.naver.com/api/stock/{ticker}/integration"


def _parse_korean_amount(text: Any) -> float | None:
    """`1,502조 4,936억` · `8,460억` → 원 단위 숫자. 형식이 다르면 None."""
    raw = str(text or "").replace(",", "").replace(" ", "").strip()
    if not raw or raw == "N/A":
        return None
    total = 0.0
    matched = False
    for unit, scale in (("조", 1e12), ("억", 1e8), ("만", 1e4)):
        if unit in raw:
            head, raw = raw.split(unit, 1)
            try:
                total += float(head) * scale
                matched = True
            except ValueError:
                return None
    if raw:  # 단위 없이 남은 꼬리(원 단위)
        try:
            total += float(raw)
            matched = True
        except ValueError:
            return None
    return total if matched else None


def _fetch_share_count(ticker: str) -> float | None:
    """상장주식수 = 시가총액 ÷ 종가 (네이버 종목 통합 정보).

    시가총액을 그대로 저장하지 않는 이유는, 그러면 배치가 돈 시점의 시총이 하루 종일 박히기
    때문이다. 주식수는 증자·소각이 없는 한 변하지 않으므로 화면이 현재가와 곱해 쓴다.
    (`load_kor_stock_market` 은 시총 상위 200개로 잘려 KOSPI200 의 중형주가 빠진다.)
    """
    try:
        request = urllib.request.Request(_NAVER_INTEGRATION_URL.format(ticker=ticker), headers=_NAVER_HEADERS)
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    except Exception as exc:
        logger.debug("%s 네이버 통합 정보 조회 실패: %s", ticker, exc)
        return None

    info = {str(item.get("code")): item.get("value") for item in (payload or {}).get("totalInfos") or []}
    market_value = _parse_korean_amount(info.get("marketValue"))
    close_price = _parse_number(info.get("lastClosePrice"))
    if not market_value or not close_price:
        return None
    return market_value / close_price


def _corp_code_for(ticker: str, corp_codes: dict[str, str]) -> str | None:
    """종목코드 → DART 고유번호. 우선주는 회사 재무가 없어 보통주 코드로 찾는다."""
    code = corp_codes.get(ticker)
    if code:
        return code
    if len(ticker) == 6 and not ticker.endswith("0"):
        return corp_codes.get(ticker[:5] + "0")
    return None


def _trend_label(values: list[float]) -> tuple[str | None, float | None]:
    """오래된 → 최신 순 값의 우상향 정도. 반환: ("3/3", 1.0). 값이 2개 미만이면 (None, None)."""
    if len(values) < 2:
        return None, None
    increases = sum(1 for prev, curr in zip(values, values[1:], strict=False) if curr > prev)
    comparisons = len(values) - 1
    return f"{increases}/{comparisons}", increases / comparisons


def _collect_ticker(
    item: dict[str, Any],
    corp_codes: dict[str, str],
    years: list[int],
) -> dict[str, Any]:
    """종목 1개의 연도별 원시값을 모은다. 실패한 소스는 값이 비고 사유가 남는다."""
    from services.opendart_service import annual_financials, dps_history

    ticker = str(item["ticker"])
    notes: list[str] = []

    naver = _fetch_naver_annual(ticker)
    if not naver:
        notes.append("네이버 연간 재무 없음")

    share_count = _fetch_share_count(ticker)
    if share_count is None:
        notes.append("상장주식수 없음")

    corp_code = _corp_code_for(ticker, corp_codes)
    dart_by_year: dict[int, dict[str, float]] = {}
    dart_dps: dict[int, float] = {}
    if not corp_code:
        notes.append("DART 고유번호 없음")
    else:
        for year in years:
            try:
                financials = annual_financials(corp_code, year)
            except Exception as exc:
                logger.debug("%s(%d) DART 재무 조회 실패: %s", ticker, year, exc)
                notes.append(f"{year} DART 조회 실패")
                continue
            if financials:
                dart_by_year[year] = financials
        # 네이버에 주당배당금이 없는 종목만 DART 배당공시로 보완한다(호출 1건 절약).
        if not (naver.get("dps") or {}):
            try:
                dart_dps = dps_history(corp_code, max(years))
            except Exception as exc:
                logger.debug("%s DART 배당공시 조회 실패: %s", ticker, exc)

    return {
        "ticker": ticker,
        "name": str(item.get("name") or ticker),
        "weight": item.get("weight"),
        "share_count": share_count,
        "naver": naver,
        "dart_by_year": dart_by_year,
        "dart_dps": dart_dps,
        "notes": notes,
    }


def _year_end_closes(tickers: list[str], years: list[int]) -> dict[str, dict[str, float]]:
    """티커별 {연도: 그 해 마지막 거래일 종가}. 지난 해 배당률의 분모다."""
    from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types

    frames = load_cached_frames_bulk_from_all_ticker_types(tickers)
    wanted = set(years)
    result: dict[str, dict[str, float]] = {}
    for ticker, frame in frames.items():
        if frame is None or frame.empty or "Close" not in frame.columns:
            continue
        close = frame["Close"].astype(float).dropna()
        close.index = pd.to_datetime(close.index)
        result[ticker] = {
            str(year): float(series.iloc[-1])
            for year, series in close.groupby(close.index.year)
            if int(year) in wanted and len(series)
        }
    return result


def _build_document(collected: dict[str, Any], year_end: dict[str, float], years: list[int]) -> dict[str, Any]:
    """수집한 원시값을 저장 문서로 조립한다. 가격이 필요한 값은 여기서 만들지 않는다."""
    naver = collected["naver"] or {}
    consensus_year = naver.get("consensus_year")
    naver_dps = naver.get("dps") or {}
    dart_dps = {str(year): value for year, value in (collected["dart_dps"] or {}).items()}
    dart_by_year = collected["dart_by_year"] or {}

    # 표에 실을 연도 — 확정 연도 + 컨센서스 연도.
    year_keys = {str(year) for year in years} | set(naver_dps) | set(dart_dps)
    if consensus_year:
        year_keys.add(consensus_year)

    by_year: dict[str, dict[str, Any]] = {}
    for key in sorted(year_keys):
        financials = dart_by_year.get(int(key)) or {}
        # DPS 는 네이버(회계연도 기준) 우선, 없으면 DART 배당공시.
        dps = naver_dps.get(key)
        if dps is None:
            dps = dart_dps.get(key)
        # 자사주는 **구성요소를 그대로** 남긴다 — 정의가 하나가 아니기 때문이다.
        #   취득만        : 회사가 그 해 사들인 금액 (find_undervalued 의 '자사주')
        #   순매입(취득−처분−발행) : 지분 희석까지 반영 (DIVB 지수 정의, find_shareholder_yield)
        # 증자가 매입보다 크면 순매입이 음수가 되어 환원을 깎는다. 어느 쪽을 쓸지는 화면이 정한다.
        acquisition = financials.get("buyback")
        disposal = financials.get("buyback_disposal")
        issuance = financials.get("share_issuance")
        buyback_net = None
        if acquisition is not None or disposal is not None or issuance is not None:
            buyback_net = float(acquisition or 0.0) - float(disposal or 0.0) - float(issuance or 0.0)
        # 배당지급만 **한 해 뒤 현금흐름표**에서 가져온다.
        #
        # 현금흐름표의 '배당금의 지급'은 그 해에 나간 현금이고, 그건 **전년도 결산배당**이다.
        # 같은 해끼리 묶으면 분자는 전년 배당, 분모는 당해 순이익이 되어 배당성향이 틀어진다.
        #   NH투자증권: CF FY2025 배당 3,283억 ÷ FY2024 순이익 6,866억 = 47.8% (공시 배당성향과 일치)
        #   같은 해로 묶으면            3,283억 ÷ FY2025 순이익 10,315억 = 31.8% (틀림)
        # 자사주 취득은 그 해에 실제로 산 금액이라 당기지 않는다.
        # 가장 최근 확정연도는 다음 사업보고서가 나와야 채워지므로 None 으로 남는다 — 추정하지 않는다.
        dividends_paid = (dart_by_year.get(int(key) + 1) or {}).get("dividends_paid")
        entry = {
            "dps": dps,
            "per": (naver.get("per") or {}).get(key),
            "pbr": (naver.get("pbr") or {}).get(key),
            "operating_income": financials.get("operating_income"),
            "net_income": financials.get("net_income"),
            "dividends_paid": dividends_paid,
            "buyback_acquisition": acquisition,
            "buyback_disposal": disposal,
            "share_issuance": issuance,
            "buyback_net": buyback_net,
            "year_end_close": year_end.get(key),
            "is_consensus": key == consensus_year,
        }
        # 값이 하나도 없는 연도는 버린다 (is_consensus 는 값이 아니라 표시라 제외).
        if any(value is not None for field, value in entry.items() if field != "is_consensus"):
            by_year[key] = entry

    confirmed_years = [str(y) for y in years if str(y) in by_year]

    def _series(field: str) -> list[float]:
        return [float(by_year[key][field]) for key in confirmed_years if by_year[key].get(field) is not None]

    # 추세는 확정 연도만으로 낸다. 컨센서스를 섞은 판정은 화면이 필요할 때 따로 만든다.
    trends: dict[str, Any] = {}
    for field, name in (("operating_income", "operating"), ("net_income", "net"), ("dps", "dividend")):
        label, ratio = _trend_label(_series(field))
        trends[name] = {"label": label, "ratio": ratio}

    # 주주환원율 = (배당지급 + 자사주) / 순이익. 순이익이 있는 가장 최근 확정연도 기준.
    # 자사주 정의가 둘이라 비율도 둘을 함께 낸다 — 화면이 고르게 하고 여기서 하나로 굳히지 않는다.
    payout: dict[str, Any] = {
        "base_year": None,
        "dividends_paid": None,
        "buyback_acquisition": None,
        "buyback_net": None,
        "ratio_gross": None,  # 취득 기준
        "ratio_net": None,  # 순매입(취득−처분−발행) 기준
    }
    for key in sorted(confirmed_years, reverse=True):
        net_income = by_year[key].get("net_income")
        # 배당지급은 이듬해 현금흐름표에서 온다 — 아직 없는 최근 연도는 건너뛴다.
        # 0 으로 채우면 배당을 안 준 회사처럼 보여 환원율이 조용히 낮게 나온다.
        if net_income is None or by_year[key].get("dividends_paid") is None:
            continue
        dividends_paid = float(by_year[key]["dividends_paid"])
        acquisition = float(by_year[key].get("buyback_acquisition") or 0.0)
        buyback_net = float(by_year[key].get("buyback_net") or 0.0)
        payout = {
            "base_year": key,
            "dividends_paid": dividends_paid,
            "buyback_acquisition": acquisition,
            "buyback_net": buyback_net,
            "ratio_gross": (dividends_paid + acquisition) / net_income if net_income > 0 else None,
            "ratio_net": (dividends_paid + buyback_net) / net_income if net_income > 0 else None,
        }
        break

    # 자사주 연평균 — 화면이 시가총액으로 나눠 자사주률을 낸다(분모가 가격이라 여기선 못 낸다).
    recent = sorted(confirmed_years, reverse=True)[:BUYBACK_WINDOW_YEARS]

    def _annual_average(field: str) -> float | None:
        values = [by_year[key][field] for key in recent if by_year[key].get(field) is not None]
        return sum(values) / BUYBACK_WINDOW_YEARS if values else None

    buyback_net_annual = _annual_average("buyback_net")
    buyback_acquisition_annual = _annual_average("buyback_acquisition")

    return {
        "_id": collected["ticker"],
        "ticker": collected["ticker"],
        "name": collected["name"],
        "index_weight": collected.get("weight"),
        # 상장주식수 — 화면이 현재가와 곱해 시가총액을 낸다(자사주률의 분모).
        "share_count": collected.get("share_count"),
        "consensus_year": consensus_year,
        "by_year": by_year,
        "trends": trends,
        "payout": payout,
        "buyback_net_annual": buyback_net_annual,
        "buyback_acquisition_annual": buyback_acquisition_annual,
        "notes": collected["notes"],
        "updated_at": datetime.now(timezone.utc),
    }


def refresh_kor_dividend_stocks() -> dict[str, Any]:
    """KOSPI200 각 종목의 연도별 재무·배당 원시값을 모아 ``kor_dividend_stocks`` 에 적재한다.

    걸러내지 않는다 — 유니버스 전 종목을 저장하고, 배당주로 볼지 말지는 화면 필터가 정한다.
    """
    from services.opendart_service import corp_code_by_stock, latest_annual_year
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패해 배당주 지표를 저장할 수 없습니다.")

    universe = load_universe()
    latest_year = latest_annual_year()
    years = list(range(TREND_START_YEAR, latest_year + 1))
    corp_codes = corp_code_by_stock()
    logger.info("[KOR DIVIDEND] 지표 수집 시작 — %d종목 · 회계연도 %d~%d", len(universe), years[0], years[-1])

    collected: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        futures = [executor.submit(_collect_ticker, item, corp_codes, years) for item in universe]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            if index % 25 == 0:
                logger.info("[KOR DIVIDEND]   ... %d/%d 조회", index, len(universe))
            collected.append(future.result())

    year_ends = _year_end_closes([row["ticker"] for row in collected], years)

    collection = db[COLLECTION]
    saved = 0
    for row in collected:
        document = _build_document(row, year_ends.get(row["ticker"]) or {}, years)
        collection.replace_one({"_id": document["_id"]}, document, upsert=True)
        saved += 1

    # 지수에서 빠진 종목은 남겨두지 않는다 — 화면이 유니버스 밖 종목을 섞어 보여주게 된다.
    current_ids = [row["ticker"] for row in collected]
    removed = collection.delete_many({"_id": {"$nin": current_ids}}).deleted_count

    with_notes = sum(1 for row in collected if row["notes"])
    logger.info("[KOR DIVIDEND] 적재 완료 — 저장 %d · 제거 %d · 결측 있음 %d", saved, removed, with_notes)
    return {"saved": saved, "removed": removed, "with_notes": with_notes, "years": years}


# ── [3] 화면 조회 — 저장된 원시값에 **지금 가격**을 입혀 파생값을 만든다 ──────────
#
# 배당률·자사주률·총주주환원율·점수는 분모가 주가라 배치가 굳혀 두면 아침 8시 값이 하루
# 종일 남는다. 여기서 조회 시점의 가격으로 계산한다.

# 기간 수익률 구간(개월).
RETURN_MONTHS = (1, 3, 6, 12)

# 점수 배점 (합계 100) — 품질(실적·배당·환원) 60 + 가격(저평가) 40.
SCORE_WEIGHTS = {
    "operating_income": 15,
    "net_income": 15,
    "dividend": 15,
    "payout": 15,
    "per": 20,
    "dividend_yield": 20,
}
# 주주환원율 만점 기준 — 이 값 이상이면 만점.
PAYOUT_FULL_SCORE_RATIO = 0.8
# PER: FULL 이하 만점, ZERO 이상 0점 (사이는 선형). 적자(PER 없음)는 0점.
PER_FULL_SCORE = 8.0
PER_ZERO_SCORE = 20.0
# 배당수익률 만점 기준(%) — 이 값 이상이면 만점.
DIVIDEND_YIELD_FULL_SCORE = 3.0


def _linear_score(value: float | None, full: float, zero: float, weight: int) -> float:
    """full 쪽이 만점, zero 쪽이 0점인 선형 배점. 값이 없으면 0점."""
    if value is None:
        return 0.0
    if full < zero:  # 낮을수록 좋다 (PER)
        if value <= full:
            return float(weight)
        if value >= zero:
            return 0.0
        return weight * (zero - value) / (zero - full)
    # 높을수록 좋다 (배당수익률·환원율)
    return weight * min(max(value, 0.0) / full, 1.0)


def _live_prices(tickers: list[str]) -> dict[str, float]:
    """현재가 — 종목풀 순위·알림과 같은 실시간 스냅샷 소스. 실패한 종목은 키가 빠진다."""
    from services.price_service import get_realtime_snapshot

    try:
        snapshot = get_realtime_snapshot("kor", tickers)
    except Exception as exc:
        logger.warning("[KOR DIVIDEND] 실시간 시세 조회 실패: %s", exc)
        return {}
    result: dict[str, float] = {}
    for ticker, entry in (snapshot or {}).items():
        price = (entry or {}).get("nowVal")
        if price:
            result[str(ticker).strip().upper()] = float(price)
    return result


def _past_closes(tickers: list[str]) -> dict[str, dict[int, float]]:
    """티커별 {개월: N개월 전 종가}. 그 날이 휴장이면 직전 거래일을 쓴다."""
    from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types

    frames = load_cached_frames_bulk_from_all_ticker_types(tickers)
    today = pd.Timestamp.today().normalize()
    result: dict[str, dict[int, float]] = {}
    for ticker, frame in frames.items():
        if frame is None or frame.empty or "Close" not in frame.columns:
            continue
        close = frame["Close"].astype(float).dropna()
        close.index = pd.to_datetime(close.index)
        by_months: dict[int, float] = {}
        for months in RETURN_MONTHS:
            earlier = close[close.index <= today - pd.DateOffset(months=months)]
            if len(earlier):
                by_months[months] = float(earlier.iloc[-1])
        result[ticker] = by_months
    return result


def _build_row(doc: dict[str, Any], price: float | None, past: dict[int, float]) -> dict[str, Any]:
    """저장 문서 + 현재가 → 화면 한 행."""
    by_year = doc.get("by_year") or {}
    consensus_year = doc.get("consensus_year")

    # 연도별 배당률 — 그 해에 실제로 받았을 수익률이라 분모가 해마다 다르다.
    #   지난 해 : 그 해 연말 종가 / 올해(컨센서스) : 현재가
    dividend_yield_by_year: dict[str, float | None] = {}
    for year, entry in by_year.items():
        dps = entry.get("dps")
        base = price if year == consensus_year else entry.get("year_end_close")
        dividend_yield_by_year[year] = (dps / base * 100.0) if (dps is not None and base) else None

    # 지금 사면 받을 배당수익률 — 컨센서스 DPS 우선, 없으면 최근 확정 DPS.
    confirmed = sorted((y for y, e in by_year.items() if not e.get("is_consensus")), reverse=True)
    forward_dps = (by_year.get(consensus_year) or {}).get("dps") if consensus_year else None
    latest_dps = next((by_year[y]["dps"] for y in confirmed if by_year[y].get("dps") is not None), None)
    current_dps = forward_dps if forward_dps is not None else latest_dps
    dividend_yield = (current_dps / price * 100.0) if (current_dps is not None and price) else None

    # 자사주률 = 연평균 자사주 ÷ 시가총액. 시총은 상장주식수 × 현재가.
    share_count = doc.get("share_count")
    market_cap = share_count * price if (share_count and price) else None
    buyback_yield = None
    buyback_annual = doc.get("buyback_net_annual")
    if buyback_annual is not None and market_cap:
        buyback_yield = buyback_annual / market_cap * 100.0
    # 총주주환원율 = 배당률 + 자사주률 (DIVB 지수 정의).
    shareholder_yield = None
    if dividend_yield is not None or buyback_yield is not None:
        shareholder_yield = (dividend_yield or 0.0) + (buyback_yield or 0.0)

    returns = {
        months: ((price / base - 1.0) * 100.0 if (price and (base := past.get(months))) else None)
        for months in RETURN_MONTHS
    }

    trends = doc.get("trends") or {}
    payout = doc.get("payout") or {}
    # PER·PBR 은 컨센서스 연도 값, 없으면 최근 확정 연도.
    reference_year = consensus_year if consensus_year in by_year else (confirmed[0] if confirmed else None)
    reference = by_year.get(reference_year) or {}

    score = (
        SCORE_WEIGHTS["operating_income"] * (trends.get("operating", {}).get("ratio") or 0.0)
        + SCORE_WEIGHTS["net_income"] * (trends.get("net", {}).get("ratio") or 0.0)
        + SCORE_WEIGHTS["dividend"] * (trends.get("dividend", {}).get("ratio") or 0.0)
        + _linear_score(payout.get("ratio_gross"), PAYOUT_FULL_SCORE_RATIO, 0.0, SCORE_WEIGHTS["payout"])
        + _linear_score(reference.get("per"), PER_FULL_SCORE, PER_ZERO_SCORE, SCORE_WEIGHTS["per"])
        + _linear_score(dividend_yield, DIVIDEND_YIELD_FULL_SCORE, 0.0, SCORE_WEIGHTS["dividend_yield"])
    )

    return {
        "ticker": doc["ticker"],
        "name": doc.get("name"),
        "index_weight": doc.get("index_weight"),
        "current_price": price,
        "market_cap": market_cap,
        "returns": {str(months): value for months, value in returns.items()},
        "dividend_yield_by_year": dividend_yield_by_year,
        "dividend_yield": dividend_yield,
        "dividend_yield_is_forward": forward_dps is not None,
        "buyback_yield": buyback_yield,
        "shareholder_yield": shareholder_yield,
        "payout_ratio_gross": payout.get("ratio_gross"),
        "payout_ratio_net": payout.get("ratio_net"),
        "payout_base_year": payout.get("base_year"),
        "trend_operating": trends.get("operating", {}).get("label"),
        "trend_net": trends.get("net", {}).get("label"),
        "trend_dividend": trends.get("dividend", {}).get("label"),
        # 화면 필터가 "우상향만" 을 판정하려면 라벨(3/3)이 아니라 비율이 필요하다.
        "trend_operating_ratio": trends.get("operating", {}).get("ratio"),
        "trend_net_ratio": trends.get("net", {}).get("ratio"),
        "trend_dividend_ratio": trends.get("dividend", {}).get("ratio"),
        "per": reference.get("per"),
        "pbr": reference.get("pbr"),
        "score": round(score, 1),
        "by_year": by_year,
        "notes": doc.get("notes") or [],
    }


def load_kor_dividend_rows() -> dict[str, Any]:
    """화면용 전체 행. 필터·정렬은 화면이 한다 — 여기서 걸러내지 않는다."""
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패해 배당주 데이터를 읽을 수 없습니다.")

    documents = list(db[COLLECTION].find({}, {"_id": 0}))
    if not documents:
        raise LookupError(
            f"{COLLECTION} 컬렉션이 비어 있습니다. scripts/update_kor_dividend_stocks.py 를 먼저 실행하세요."
        )

    tickers = [doc["ticker"] for doc in documents]
    prices = _live_prices(tickers)
    past_closes = _past_closes(tickers)

    rows = [_build_row(doc, prices.get(doc["ticker"]), past_closes.get(doc["ticker"]) or {}) for doc in documents]
    rows.sort(key=lambda row: (row["dividend_yield"] is None, -(row["dividend_yield"] or 0.0)))

    years = sorted({year for doc in documents for year in (doc.get("by_year") or {})}, reverse=True)
    consensus_years = [doc.get("consensus_year") for doc in documents if doc.get("consensus_year")]
    return {
        "rows": rows,
        "years": years,
        "consensus_year": max(set(consensus_years), key=consensus_years.count) if consensus_years else None,
        "return_months": [str(months) for months in RETURN_MONTHS],
        "updated_at": max((doc.get("updated_at") for doc in documents if doc.get("updated_at")), default=None),
    }
