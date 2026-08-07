"""코스피 시가총액 상위 200 중 "품질 좋고 아직 싼"(저평가) 종목을 뽑아 콘솔에 출력한다.

판정 구조 — "관문(필터) 통과 후 점수화" 2단계
  [1차 관문 — 확정 실적 기준, 하나라도 탈락하면 순위 제외]
  · FILTER_REQUIRE_LATEST_INCREASE: 최신 확정연도가 전년 대비 증가해야 하는 지표
    (기본: 영업이익·순이익·배당 — 예: 2025년 영업이익이 감소한 강원랜드는 여기서 제외.
     배당은 회계연도 기준 주당배당금으로 판정)
  · FILTER_MIN_TREND_RATIO: '증가한 해의 비율' 최소치 (기본 비활성)
  · FILTER_MIN_DIVIDEND_YIELD: 배당수익률 하한 (기본 4% — '배당주 아님' 제외)
    포워드(컨센서스 예상 DPS/현재가) 우선, 컨센서스 없으면 확정(최근 확정 DPS/현재가)
  · 배당·자사주가 모두 없는 종목도 제외 (주주환원 대상 아님)
  · 제외 종목은 결과 하단에 사유별로 요약 출력한다

  [2차 점수화 — 관문 통과 종목만, 합계 100 = 품질 60 + 가격 40]
  1. 영업이익 우상향 (15) — TREND_START_YEAR(2023)년 이후 추세, 2026 컨센서스 있으면 포함
  2. 순이익 우상향   (15) — 위와 동일
  3. 배당금 우상향   (15) — 주당배당금(DPS) 추세, 2026 예상 DPS 있으면 포함
  4. 주주환원율      (15) — (배당지급 + 자사주매입) / 순이익, 확정 현금흐름표 기준
  5. PER            (20) — 2026 예상 PER 우선, 없으면 시가총액/최근 순이익. 낮을수록 만점
  6. 배당수익률      (20) — 2026 예상 DPS/현재가 우선, 없으면 최근 완결연도 기준. 높을수록 만점

데이터 소스
  - 종목 명단·시가총액·현재가: 네이버 시가총액 순위 API (/kor-market-stock 화면과 동일 함수)
  - 확정 재무(연도별): yfinance income_stmt / cashflow
  - 주당배당금(DPS): 네이버 연간 재무의 회계연도 기준 값 (빈 연도는 무배당 0원으로 간주,
    네이버에 없으면 yfinance 배당 이력을 지급일 기준으로 합산해 폴백)
  - 2026 컨센서스: 네이버 연간 재무 API (m.stock.naver.com/api/stock/{티커}/finance/annual)
    · isConsensus=Y 연도의 영업이익·당기순이익·주당배당금·PER 을 쓴다 (금액 단위: 억원)
    · 자사주매입·배당지급총액은 컨센서스가 존재하지 않는 항목이라 확정치만 쓴다
  - pykrx 의 지수 구성종목·펀더멘털 API 는 현재 KRX 응답 변경으로 빈 값이라 쓰지 않는다

출력
  1) 전체 순위 표 → 2) 항목 설명 → 3) 상위 N개 연도별 상세 (--detail, 기본 5)
  컨센서스가 반영된 값에는 `*` 를 붙인다. 상세 표의 연도 컬럼은 `2026(예) 2025 2024 2023` 순이며
  하단에 연도별 PER(참고, 네이버) 행을 함께 보여준다.

사용 예
    python scripts/find_undervalued_company_in_kospi200.py
    python scripts/find_undervalued_company_in_kospi200.py --top 50 --workers 8
    python scripts/find_undervalued_company_in_kospi200.py --limit 30 --detail 3   # 빠른 확인

⚠️ 점수 기준은 임시다. SCORE_WEIGHTS / 만점 기준 상수를 보고 자유롭게 조정하면 된다.
(이 파일 상단 설명은 코드 동작이 바뀔 때마다 현재 상태에 맞게 함께 갱신한다.)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from utils.kor_stock_market_service import load_kor_stock_market  # noqa: E402
from utils.logger import get_app_logger  # noqa: E402
from utils.report import render_table_eaw  # noqa: E402

logger = get_app_logger()

# ── 조정 가능한 기준 ────────────────────────────────────────────────────────
# 1차 관문(필터) — 점수화 이전에 적용한다. 하나라도 통과 못 하면 순위에서 제외.
# 확정 실적만 본다(컨센서스로 관문을 통과시키지 않는다).
#
# FILTER_REQUIRE_LATEST_INCREASE: 최신 확정연도가 전년 대비 증가해야 하는 지표.
#   예: 강원랜드(035250)는 2025년 영업이익·순이익이 감소 → 여기서 제외된다.
#   배당(dividend)은 회계연도 기준 주당배당금(DPS)으로 판정한다.
#   데이터가 2개 연도 미만이라 판정 불가인 종목도 제외한다(통과 근거 없음).
FILTER_REQUIRE_LATEST_INCREASE: tuple[str, ...] = ("operating_income", "net_income", "dividend")
# FILTER_MIN_TREND_RATIO: '전년 대비 증가한 해의 비율' 최소치 (0~1). 항목을 넣으면 활성화.
#   예: {"dividend": 0.5} → 배당이 절반 이상의 해에서 증가하지 않으면 제외.
FILTER_MIN_TREND_RATIO: dict[str, float] = {}
# FILTER_MIN_DIVIDEND_YIELD: 배당수익률 하한 — 미만이면 "배당주 아님"으로 제외.
#   컨센서스 예상 DPS 가 있으면 포워드(예상 DPS ÷ 현재가), 없으면 확정(최근 확정 DPS ÷ 현재가).
#   None 이면 비활성.
FILTER_MIN_DIVIDEND_YIELD: float | None = 0.04

# 항목별 배점 (합계 100). "품질"(실적·배당·환원)과 "가격"(저평가) 두 축으로 나뉜다.
# 품질만 보면 이미 주가가 오른 좋은 회사가 상위를 차지하므로,
# PER·배당수익률로 같은 품질이면 싼 종목이 위로 오게 한다.
SCORE_WEIGHTS = {
    # 품질 축 (60)
    "operating_income": 15,  # 영업이익 우상향
    "net_income": 15,  # 순이익 우상향
    "dividend": 15,  # 배당금 우상향
    "payout": 15,  # 주주환원율
    # 가격 축 (40) — 저평가 판정
    "per": 20,  # PER 낮을수록 (시가총액 / 최근 순이익)
    "dividend_yield": 20,  # 배당수익률 높을수록 (최근 완결연도 주당배당금 / 현재가)
}
# 추세 판정·상세 표에 쓸 시작 연도 — 이 연도 이후(포함)의 확정 실적만 본다.
TREND_START_YEAR = 2023
# 주주환원율 만점 기준 — 이 값 이상이면 배점 만점.
PAYOUT_FULL_SCORE_RATIO = 0.8
# PER 점수: FULL 이하면 만점, ZERO 이상이면 0점 (사이는 선형). 적자(PER 없음)는 0점.
PER_FULL_SCORE = 8.0
PER_ZERO_SCORE = 20.0
# 배당수익률 만점 기준 — 이 값 이상이면 배점 만점 (0% 는 0점, 사이는 선형).
DIVIDEND_YIELD_FULL_SCORE = 0.03
# 현금흐름표에서 자사주 매입액을 담는 항목명 (yfinance 표기).
BUYBACK_KEYS = ("Repurchase Of Capital Stock", "Common Stock Payments")
DIVIDEND_PAID_KEYS = ("Cash Dividends Paid", "Common Stock Dividend Paid")
NET_INCOME_KEYS = ("Net Income", "Net Income Common Stockholders")
OPERATING_INCOME_KEYS = ("Operating Income", "EBIT")


def _pick_row(frame: pd.DataFrame | None, keys: tuple[str, ...]) -> pd.Series | None:
    """여러 후보 항목명 중 프레임에 존재하는 첫 행을 돌려준다."""
    if frame is None or frame.empty:
        return None
    for key in keys:
        if key in frame.index:
            series = pd.to_numeric(frame.loc[key], errors="coerce").dropna()
            if not series.empty:
                return series
    return None


def _values_from_start_year(series: pd.Series | None) -> list[float]:
    """TREND_START_YEAR 이후(포함)의 확정 값을 오래된 → 최신 순으로 돌려준다."""
    if series is None or series.empty:
        return []
    by_year = {int(_year_of(label)): float(value) for label, value in series.items()}
    return [by_year[year] for year in sorted(by_year) if year >= TREND_START_YEAR]


def _trend_score(
    series: pd.Series | None, max_score: int, consensus_value: float | None = None
) -> tuple[float, str]:
    """TREND_START_YEAR 이후 값의 우상향 정도를 0~max_score 로 점수화한다.

    '전년 대비 증가한 해의 비율' 을 그대로 배점에 곱한다.
    consensus_value 가 주어지면 최신 연도 다음의 예상치로 추세에 포함하고 라벨에 `*` 를 붙인다.
    """
    values = _values_from_start_year(series)
    if consensus_value is not None:
        values = values + [float(consensus_value)]
    if len(values) < 2:
        return 0.0, "-"
    increases = sum(1 for prev, curr in zip(values, values[1:]) if curr > prev)
    comparisons = len(values) - 1
    ratio = increases / comparisons
    label = f"{increases}/{comparisons}" + ("*" if consensus_value is not None else "")
    return max_score * ratio, label


def _confirmed_trend_ratio(series: pd.Series | None) -> tuple[float | None, bool | None]:
    """확정 연도만으로 (증가 비율, 최신 연도 증가 여부) 를 계산한다. 데이터 부족이면 (None, None)."""
    values = _values_from_start_year(series)
    if len(values) < 2:
        return None, None
    increases = sum(1 for prev, curr in zip(values, values[1:]) if curr > prev)
    ratio = increases / (len(values) - 1)
    latest_increased = values[-1] > values[-2]
    return ratio, latest_increased


def _apply_filters(
    metric_series: dict[str, pd.Series | None],
) -> str | None:
    """1차 관문. 탈락 사유 문자열을 돌려주고, 통과하면 None."""
    for metric in FILTER_REQUIRE_LATEST_INCREASE:
        _, latest_increased = _confirmed_trend_ratio(metric_series.get(metric))
        if latest_increased is None:
            return f"{metric}: 확정 연도 2개 미만(판정 불가)"
        if not latest_increased:
            return f"{metric}: 최신 확정연도 감소"
    for metric, min_ratio in FILTER_MIN_TREND_RATIO.items():
        ratio, _ = _confirmed_trend_ratio(metric_series.get(metric))
        if ratio is None:
            return f"{metric}: 확정 연도 2개 미만(판정 불가)"
        if ratio < min_ratio:
            return f"{metric}: 증가 비율 {ratio:.0%} < {min_ratio:.0%}"
    return None


def _value_at_year(frame: pd.DataFrame | None, keys: tuple[str, ...], year: str) -> float:
    """특정 회계연도의 값. 해당 연도에 값이 없으면 0.

    `_pick_row` 는 NaN 을 제거하므로 `iloc[0]` 이 종목마다 다른 연도를 가리킬 수 있다.
    배당·자사주·순이익을 같은 연도로 맞추기 위해 연도를 명시해 조회한다.
    """
    if frame is None or frame.empty:
        return 0.0
    for key in keys:
        if key not in frame.index:
            continue
        for label, value in frame.loc[key].items():
            if _year_of(label) == year and pd.notna(value):
                return float(value)
    return 0.0


def _payout_ratio(
    cashflow: pd.DataFrame | None, income: pd.DataFrame | None
) -> tuple[float | None, float, float, str | None]:
    """최근 회계연도 주주환원율 = (배당 + 자사주매입) / 순이익.

    기준 연도는 '순이익이 있는 가장 최근 회계연도' 하나로 고정하고, 배당·자사주도
    같은 연도 값만 쓴다. 현금흐름표의 지출은 음수로 들어오므로 절댓값으로 바꿔 더한다.
    """
    net_income_row = _pick_row(income, NET_INCOME_KEYS)
    if net_income_row is None or net_income_row.empty:
        return None, 0.0, 0.0, None

    base_year = _year_of(net_income_row.index[0])
    dividend_amount = abs(_value_at_year(cashflow, DIVIDEND_PAID_KEYS, base_year))
    buyback_amount = abs(_value_at_year(cashflow, BUYBACK_KEYS, base_year))

    net_income = float(net_income_row.iloc[0])
    if net_income <= 0:
        return None, dividend_amount, buyback_amount, base_year
    ratio = (dividend_amount + buyback_amount) / net_income
    return ratio, dividend_amount, buyback_amount, base_year


def _yearly_dividend_series(ticker_obj: Any) -> pd.Series | None:
    """[폴백] yfinance 배당 이력을 지급일 기준으로 연도 합산한다 (최신 연도가 앞).

    지급일이 회계연도와 어긋나는 회사(배당기준일을 12월 → 이듬해 3월로 옮긴 경우)는
    특정 연도가 비어 보이므로, 기본 소스는 `_naver_dps_series`(회계연도 기준)를 쓴다.
    """
    dividends = getattr(ticker_obj, "dividends", None)
    if dividends is None or dividends.empty:
        return None
    yearly = dividends.groupby(dividends.index.year).sum().sort_index(ascending=False)
    if yearly.empty:
        return None
    current_year = pd.Timestamp.now().year
    return yearly[yearly.index < current_year]


def _naver_dps_series(consensus: dict[str, Any]) -> pd.Series | None:
    """네이버 연간 재무의 주당배당금(회계연도 기준)으로 확정 DPS 시리즈를 만든다 (최신이 앞).

    컨센서스 연도는 제외한다(예상 DPS 는 추세에 consensus_value 로 따로 반영).
    TREND_START_YEAR 이후의 빈 연도는 무배당(0원)으로 채운다 — 네이버 표에서 '-' 는
    해당 회계연도에 배당이 없었다는 뜻이므로, 건너뛴 해가 추세에서 사라지지 않게 한다.
    """
    dps_by_year = consensus.get("dps_by_year") or {}
    consensus_year = str(consensus.get("year") or "")
    confirmed = {int(year): value for year, value in dps_by_year.items() if year != consensus_year}
    if not confirmed:
        return None
    latest_year = max(confirmed)
    filled = {
        year: confirmed.get(year, 0.0)
        for year in range(min(TREND_START_YEAR, min(confirmed)), latest_year + 1)
    }
    return pd.Series(filled).sort_index(ascending=False)


_NAVER_ANNUAL_URL = "https://m.stock.naver.com/api/stock/{ticker}/finance/annual"
_NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
# 네이버 연간 재무 API 의 행 제목 → 우리가 쓸 키. 금액 항목은 억원 단위다.
_CONSENSUS_ROW_TITLES = {"영업이익": "operating", "당기순이익": "net", "주당배당금": "dps", "PER": "per"}


def _parse_consensus_number(value: Any) -> float | None:
    text = str(value or "").replace(",", "").strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fetch_naver_consensus(ticker: str) -> dict[str, Any] | None:
    """네이버 연간 재무 API 에서 컨센서스(isConsensus=Y) 연도의 추정치를 가져온다.

    반환: {"year": "2026", "operating": 원, "net": 원, "dps": 원, "per": float}
    (없는 항목은 키 자체가 빠진다. 컨센서스 연도가 없으면 None.)
    """
    import urllib.request

    try:
        request = urllib.request.Request(_NAVER_ANNUAL_URL.format(ticker=ticker), headers=_NAVER_HEADERS)
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    except Exception as exc:
        logger.debug("%s 네이버 컨센서스 조회 실패: %s", ticker, exc)
        return None

    finance_info = (payload or {}).get("financeInfo") or {}
    consensus_keys = [
        str(title.get("key"))
        for title in finance_info.get("trTitleList") or []
        if str(title.get("isConsensus") or "") == "Y"
    ]
    if not consensus_keys:
        return None
    consensus_key = consensus_keys[-1]  # 가장 미래 연도 (보통 1개)

    result: dict[str, Any] = {"year": consensus_key[:4]}
    for row in finance_info.get("rowList") or []:
        title = str(row.get("title") or "").strip()
        columns = row.get("columns") or {}
        # PER·주당배당금은 연도별 값 전체를 보존한다.
        # PER 는 상세 표의 참고 행, 주당배당금은 회계연도 기준 DPS 소스로 쓴다.
        if title == "PER":
            result["per_by_year"] = {
                str(key)[:4]: parsed
                for key, cell in columns.items()
                if (parsed := _parse_consensus_number((cell or {}).get("value"))) is not None
            }
        if title == "주당배당금":
            result["dps_by_year"] = {
                str(key)[:4]: parsed
                for key, cell in columns.items()
                if (parsed := _parse_consensus_number((cell or {}).get("value"))) is not None
            }
        field = _CONSENSUS_ROW_TITLES.get(title)
        if not field:
            continue
        value = _parse_consensus_number((columns.get(consensus_key) or {}).get("value"))
        if value is None:
            continue
        # 영업이익·순이익은 억원 단위 → 원으로 통일 (DPS·PER 는 그대로)
        result[field] = value * 1e8 if field in ("operating", "net") else value
    return result if len(result) > 1 else None


def _analyze(row: dict[str, Any]) -> dict[str, Any] | None:
    """종목 1개의 재무 지표를 조회해 점수를 계산한다."""
    import yfinance as yf

    ticker = str(row.get("ticker") or "").strip()
    if not ticker:
        return None
    try:
        ticker_obj = yf.Ticker(f"{ticker}.KS")
        income = ticker_obj.income_stmt
        cashflow = ticker_obj.cashflow
    except Exception as exc:
        logger.debug("%s 재무 조회 실패: %s", ticker, exc)
        return None

    # 네이버 연간 재무 — 컨센서스 + 회계연도 기준 DPS. 관문 판정에 DPS 가 필요해 먼저 조회한다.
    consensus = _fetch_naver_consensus(ticker) or {}
    # DPS 소스: 네이버(회계연도 기준) 우선, 없으면 yfinance 지급일 합산으로 폴백.
    dividend_series = _naver_dps_series(consensus)
    if dividend_series is None:
        dividend_series = _yearly_dividend_series(ticker_obj)

    # ── 1차 관문(필터): 확정 실적 기준. 탈락하면 사유만 담아 돌려준다. ──
    metric_series: dict[str, pd.Series | None] = {
        "operating_income": _pick_row(income, OPERATING_INCOME_KEYS),
        "net_income": _pick_row(income, NET_INCOME_KEYS),
        "dividend": dividend_series,
    }
    excluded_reason = _apply_filters(metric_series)
    if excluded_reason:
        return {"ticker": ticker, "name": str(row.get("name") or "-"), "excluded_reason": excluded_reason}

    # 배당수익률 관문 — 포워드(컨센서스 예상 DPS) 우선, 없으면 확정(최근 확정 DPS). ÷ 현재가.
    if FILTER_MIN_DIVIDEND_YIELD is not None:
        current_price = row.get("current_price")
        gate_dps = consensus.get("dps")
        gate_basis = "포워드"
        if gate_dps is None:
            gate_basis = "확정"
            if dividend_series is not None and not dividend_series.empty:
                gate_dps = float(dividend_series.iloc[0])
        if not current_price or gate_dps is None:
            return {
                "ticker": ticker,
                "name": str(row.get("name") or "-"),
                "excluded_reason": "배당수익률: 판정 불가(현재가·DPS 없음)",
            }
        gate_yield = float(gate_dps) / float(current_price)
        if gate_yield < FILTER_MIN_DIVIDEND_YIELD:
            return {
                "ticker": ticker,
                "name": str(row.get("name") or "-"),
                "excluded_reason": (
                    f"배당수익률({gate_basis}) {gate_yield:.2%} < {FILTER_MIN_DIVIDEND_YIELD:.0%}"
                ),
            }

    operating_score, operating_label = _trend_score(
        metric_series["operating_income"],
        SCORE_WEIGHTS["operating_income"],
        consensus_value=consensus.get("operating"),
    )
    net_score, net_label = _trend_score(
        metric_series["net_income"], SCORE_WEIGHTS["net_income"], consensus_value=consensus.get("net")
    )
    dividend_score, dividend_label = _trend_score(
        dividend_series, SCORE_WEIGHTS["dividend"], consensus_value=consensus.get("dps")
    )

    payout, dividend_amount, buyback_amount, payout_year = _payout_ratio(cashflow, income)
    if payout is None:
        payout_score = 0.0
    else:
        payout_score = SCORE_WEIGHTS["payout"] * min(payout / PAYOUT_FULL_SCORE_RATIO, 1.0)

    # 배당도 자사주도 없으면 주주환원 종목이 아니므로 제외한다.
    if dividend_amount <= 0 and buyback_amount <= 0:
        return {"ticker": ticker, "name": str(row.get("name") or "-"), "excluded_reason": "배당·자사주 모두 없음"}

    # ── 가격 축 (저평가 판정) ──
    # PER: 컨센서스 예상 PER 우선. 없으면 시가총액(네이버, 억 단위) / 최근 순이익. 적자면 0점.
    per = consensus.get("per")
    per_is_consensus = per is not None
    if per is None:
        net_income_row = _pick_row(income, NET_INCOME_KEYS)
        recent_net_income = float(net_income_row.iloc[0]) if net_income_row is not None else 0.0
        market_cap_eok = row.get("market_cap")
        if market_cap_eok and recent_net_income > 0:
            per = float(market_cap_eok) * 1e8 / recent_net_income
    if per is None:
        per_score = 0.0
    elif per <= PER_FULL_SCORE:
        per_score = float(SCORE_WEIGHTS["per"])
    elif per >= PER_ZERO_SCORE:
        per_score = 0.0
    else:
        per_score = SCORE_WEIGHTS["per"] * (PER_ZERO_SCORE - per) / (PER_ZERO_SCORE - PER_FULL_SCORE)

    # 배당수익률: 예상 DPS 우선, 없으면 최근 완결연도 주당배당금. / 현재가.
    current_price = row.get("current_price")
    dividend_yield = None
    yield_is_consensus = False
    if current_price:
        if consensus.get("dps") is not None:
            dividend_yield = float(consensus["dps"]) / float(current_price)
            yield_is_consensus = True
        elif dividend_series is not None and not dividend_series.empty:
            dividend_yield = float(dividend_series.iloc[0]) / float(current_price)
    if dividend_yield is None:
        yield_score = 0.0
    else:
        yield_score = SCORE_WEIGHTS["dividend_yield"] * min(dividend_yield / DIVIDEND_YIELD_FULL_SCORE, 1.0)

    return {
        "ticker": ticker,
        "name": str(row.get("name") or "-"),
        "score": operating_score + net_score + dividend_score + payout_score + per_score + yield_score,
        "operating_label": operating_label,
        "net_label": net_label,
        "dividend_label": dividend_label,
        "payout": payout,
        "payout_year": payout_year,
        "dividend_amount": dividend_amount,
        "buyback_amount": buyback_amount,
        "market_cap_eok": row.get("market_cap"),
        "per": per,
        "per_is_consensus": per_is_consensus,
        "dividend_yield": dividend_yield,
        "yield_is_consensus": yield_is_consensus,
        "consensus": consensus or None,
        # 상세 표(상위 N개)용 — 연도별 원본 시리즈를 그대로 들고 간다.
        "scores": {
            "operating_income": operating_score,
            "net_income": net_score,
            "dividend": dividend_score,
            "payout": payout_score,
            "per": per_score,
            "dividend_yield": yield_score,
        },
        "series": {
            "영업이익": _pick_row(income, OPERATING_INCOME_KEYS),
            "순이익": _pick_row(income, NET_INCOME_KEYS),
            "배당지급액": _pick_row(cashflow, DIVIDEND_PAID_KEYS),
            "자사주매입": _pick_row(cashflow, BUYBACK_KEYS),
        },
        "dividend_series": dividend_series,
    }


def _format_jo(value: float) -> str:
    """원 단위 금액을 조/억 단위로 읽기 쉽게 만든다. 음수는 부호를 유지한다."""
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1e12:
        return f"{sign}{magnitude / 1e12:.1f}조"
    if magnitude >= 1e8:
        return f"{sign}{magnitude / 1e8:.0f}억"
    return f"{sign}{magnitude:.0f}"


def _year_of(label: Any) -> str:
    """컬럼 라벨(Timestamp 또는 연도 정수)에서 연도 문자열을 뽑는다."""
    if isinstance(label, (int, float)):
        return str(int(label))
    return str(pd.Timestamp(label).year)


def _print_detail(rank: int, item: dict[str, Any]) -> None:
    """상위 종목 1개의 연도별 재무 상세를 표로 출력한다."""
    market_cap_eok = item.get("market_cap_eok")
    cap_text = f" · 시가총액 {float(market_cap_eok) / 10000:.1f}조" if market_cap_eok else ""
    payout_year = f", {item['payout_year']}년 기준" if item.get("payout_year") else ""
    payout_text = f"{item['payout'] * 100:.0f}%{payout_year}" if item["payout"] is not None else "-"
    per_text = (
        f"{item['per']:.1f}{'*' if item.get('per_is_consensus') else ''}" if item.get("per") is not None else "-"
    )
    yield_text = (
        f"{item['dividend_yield'] * 100:.1f}%{'*' if item.get('yield_is_consensus') else ''}"
        if item.get("dividend_yield") is not None
        else "-"
    )
    scores = item["scores"]
    print(
        f"\n[{rank}위] {item['ticker']} {item['name']}{cap_text} · PER {per_text} · 배당수익률 {yield_text}"
        f"\n  점수 {item['score']:.1f} = 영업익 {scores['operating_income']:.1f}"
        f" + 순이익 {scores['net_income']:.1f}"
        f" + 배당 {scores['dividend']:.1f}"
        f" + 환원율 {scores['payout']:.1f}"
        f" + PER {scores['per']:.1f}"
        f" + 배당률 {scores['dividend_yield']:.1f}  (환원율 {payout_text})"
    )

    # 연도 축 = 재무제표 연도 ∪ 배당 연도 (최신 → 과거, TREND_START_YEAR 이후만)
    years: set[str] = set()
    for series in item["series"].values():
        if series is not None:
            years.update(_year_of(label) for label in series.index)
    dividend_series = item.get("dividend_series")
    if dividend_series is not None:
        years.update(_year_of(label) for label in dividend_series.index)
    ordered_years = [y for y in sorted(years, reverse=True) if int(y) >= TREND_START_YEAR]
    if not ordered_years:
        print("  (연도별 데이터 없음)")
        return

    # 컨센서스가 있으면 맨 앞에 `YYYY(예)` 컬럼을 추가한다.
    consensus = item.get("consensus") or {}
    consensus_year_label = f"{consensus['year']}(예)" if consensus.get("year") else None
    consensus_by_row = {
        "영업이익": consensus.get("operating"),
        "순이익": consensus.get("net"),
        "주당배당금": consensus.get("dps"),
    }

    rows: list[tuple[str, list[str]]] = []
    for name, series in item["series"].items():
        by_year = (
            {_year_of(label): float(value) for label, value in series.items()} if series is not None else {}
        )
        values = [_format_jo(by_year[y]) if y in by_year else "-" for y in ordered_years]
        if consensus_year_label:
            estimate = consensus_by_row.get(name)
            values = [_format_jo(estimate) if estimate is not None else "-", *values]
        rows.append((name, values))
    if dividend_series is not None:
        by_year = {_year_of(label): float(value) for label, value in dividend_series.items()}
        values = [f"{by_year[y]:,.0f}원" if y in by_year else "-" for y in ordered_years]
        if consensus_year_label:
            dps = consensus_by_row.get("주당배당금")
            values = [f"{dps:,.0f}원" if dps is not None else "-", *values]
        rows.append(("주당배당금", values))

    # PER 참고 행 — 네이버 연간 재무의 연도별 PER (점수와 무관한 참고 데이터).
    per_by_year = consensus.get("per_by_year") or {}
    if per_by_year:
        values = [f"{per_by_year[y]:.1f}" if y in per_by_year else "-" for y in ordered_years]
        if consensus_year_label:
            consensus_per = per_by_year.get(consensus.get("year"))
            values = [f"{consensus_per:.1f}" if consensus_per is not None else "-", *values]
        rows.append(("PER(참고)", values))

    year_headers = [consensus_year_label, *ordered_years] if consensus_year_label else ordered_years
    headers = ["항목", *year_headers]
    aligns = ["left"] + ["right"] * len(year_headers)
    for line in render_table_eaw(headers, [[name, *values] for name, values in rows], aligns):
        print("  " + line)


def main() -> None:
    parser = argparse.ArgumentParser(description="코스피200 주주환원 우수 종목 찾기")
    parser.add_argument("--top", type=int, default=30, help="출력할 상위 종목 수 (기본 30).")
    parser.add_argument("--limit", type=int, default=200, help="조사할 시가총액 상위 종목 수 (기본 200).")
    parser.add_argument("--workers", type=int, default=8, help="yfinance 병렬 조회 워커 수 (기본 8).")
    parser.add_argument("--detail", type=int, default=5, help="연도별 상세를 출력할 상위 종목 수 (기본 5).")
    args = parser.parse_args()

    print(f"코스피 시가총액 상위 {args.limit}종목 명단을 불러옵니다...")
    market = load_kor_stock_market("KOSPI", limit=args.limit, min_market_cap_jo=0)
    rows = market.get("rows") or market.get("items") or []
    if not rows:
        raise SystemExit("코스피 종목 목록을 가져오지 못했습니다.")
    print(f"  {len(rows)}종목 확보 — 재무 데이터 조회 시작 (워커 {args.workers}개)")

    results: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(_analyze, rows), 1):
            if index % 25 == 0:
                print(f"    ... {index}/{len(rows)} 조회")
            if not result:
                continue
            if result.get("excluded_reason"):
                excluded.append(result)
            else:
                results.append(result)

    results.sort(key=lambda r: -r["score"])
    print(f"\n조사 완료 — 관문 통과 {len(results)}개 / 필터 제외 {len(excluded)}개")

    # 제외 요약 — 사유별 묶음 (기준 조정에 참고).
    if excluded:
        by_reason: dict[str, list[str]] = {}
        for item in excluded:
            by_reason.setdefault(item["excluded_reason"], []).append(f"{item['name']}({item['ticker']})")
        print("\n[필터 제외 내역]")
        for reason, names in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            sample = ", ".join(names[:8]) + (f" 외 {len(names) - 8}개" if len(names) > 8 else "")
            print(f"  · {reason} — {len(names)}개: {sample}")
    print()

    # ── 1. 전체 순위 리스트 ──────────────────────────────────────────
    headers = ["순위", "티커", "종목명", "점수", "영업익", "순이익", "배당", "환원율", "PER", "배당률", "배당액", "자사주"]
    aligns = ["right", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right"]
    table_rows: list[list[str]] = []
    for rank, item in enumerate(results[: args.top], 1):
        table_rows.append(
            [
                str(rank),
                item["ticker"],
                item["name"],
                f"{item['score']:.1f}",
                item["operating_label"],
                item["net_label"],
                item["dividend_label"],
                f"{item['payout'] * 100:.0f}%" if item["payout"] is not None else "-",
                (
                    f"{item['per']:.1f}{'*' if item.get('per_is_consensus') else ''}"
                    if item.get("per") is not None
                    else "-"
                ),
                (
                    f"{item['dividend_yield'] * 100:.1f}%{'*' if item.get('yield_is_consensus') else ''}"
                    if item.get("dividend_yield") is not None
                    else "-"
                ),
                _format_jo(item["dividend_amount"]),
                _format_jo(item["buyback_amount"]),
            ]
        )
    for line in render_table_eaw(headers, table_rows, aligns):
        print(line)

    # ── 2. 항목 설명 ────────────────────────────────────────────────
    print(
        "\n항목 설명 — 영업익·순이익·배당: 전년 대비 증가한 해 / 비교한 해 (예: 3/3 = 3년 연속 증가)"
        "\n           `*`: 2026 컨센서스(네이버 추정치)가 반영된 값"
        f"\n           환원율: (배당 + 자사주매입) / 순이익, {PAYOUT_FULL_SCORE_RATIO * 100:.0f}% 이상이면 배점 만점"
        f"\n           PER: 시가총액 / 최근 순이익 — {PER_FULL_SCORE:.0f} 이하 만점, {PER_ZERO_SCORE:.0f} 이상 0점 (적자 0점)"
        f"\n           배당률: 최근 완결연도 주당배당금 / 현재가 — {DIVIDEND_YIELD_FULL_SCORE * 100:.0f}% 이상 만점"
        "\n           배당액·자사주: 최근 회계연도 현금흐름표 기준 실제 지출액"
        f"\n           점수 배분: {SCORE_WEIGHTS}"
        "\n기준을 바꾸려면 스크립트 상단의 SCORE_WEIGHTS 와 만점 기준 상수들을 수정하세요."
    )

    # ── 3. 상위 N개 연도별 상세 ──────────────────────────────────────
    detail_count = min(args.detail, len(results))
    if detail_count > 0:
        print(f"\n{'=' * 60}\n상위 {detail_count}개 상세 (연도별)\n{'=' * 60}")
        for rank, item in enumerate(results[:detail_count], 1):
            _print_detail(rank, item)


if __name__ == "__main__":
    main()
