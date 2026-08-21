"""OpenDART(전자공시) API 공용 서비스 — 상장사 확정 재무 데이터의 공식 소스.

find_shareholder_yield_in_kor.py / find_undervalued_company_in_kospi200.py 가
yfinance 대신 쓰는 확정 재무 소스다. 컨센서스(예상치)는 DART 에 없으므로 네이버를 유지한다.

필요 환경변수: DART_API_KEY (.env — https://opendart.fss.or.kr 에서 무료 발급)
호출 한도: 키당 일 20,000건 — 200종목 × 연 3~4콜 수준이라 여유가 크다.

제공 함수
  - corp_code_by_stock(): 종목코드(6자리) → DART 고유번호(8자리) 매핑. 30일 파일 캐시.
  - annual_financials(corp_code, year): 그 회계연도 사업보고서의 핵심 계정
    (영업이익·당기순이익·배당금지급·자기주식 취득/처분·주식 발행) — 연결(CFS) 우선,
    없으면 별도(OFS) 폴백. 해당 연도 보고서가 없으면 None.
  - dps_history(corp_code, base_year): 보통주 주당 현금배당금 {연도: 원} —
    사업보고서 1건에 3개년(당기/전기/전전기)이 담겨 온다.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from utils.logger import get_app_logger

logger = get_app_logger()

_BASE_URL = "https://opendart.fss.or.kr/api"
_CORP_CODE_CACHE = Path(__file__).resolve().parents[1] / "data" / "opendart_corp_codes.json"
_CORP_CODE_TTL_DAYS = 30

# 사업보고서 코드. 분기·반기는 이 모듈의 용도(연간 확정 실적)에 없다.
_ANNUAL_REPORT = "11011"


class OpenDartError(RuntimeError):
    """OpenDART 호출 실패 — 키 문제·한도 초과 등. '데이터 없음'(013)은 여기 해당하지 않는다."""


def _api_key() -> str:
    key = os.environ.get("DART_API_KEY")
    if not key:
        raise OpenDartError(".env 에 DART_API_KEY 가 필요합니다 (opendart.fss.or.kr 에서 발급).")
    return key


def _request(api: str, **params: str) -> dict[str, Any] | None:
    """API 호출. 정상(000)이면 payload, 데이터 없음(013)이면 None, 그 외는 에러."""
    query = urllib.parse.urlencode({"crtfc_key": _api_key(), **params})
    url = f"{_BASE_URL}/{api}?{query}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise OpenDartError(f"OpenDART 호출 실패 ({api}): {exc}") from exc
    status = str(payload.get("status") or "")
    if status == "013":  # 조회된 데이터가 없습니다
        return None
    if status != "000":
        raise OpenDartError(f"OpenDART 오류 ({api}, status={status}): {payload.get('message')}")
    return payload


def corp_code_by_stock() -> dict[str, str]:
    """종목코드(6자리) → corp_code(8자리). 30일 파일 캐시 — 원본 zip 이 28MB 라 매번 받지 않는다."""
    if _CORP_CODE_CACHE.exists():
        cached = json.loads(_CORP_CODE_CACHE.read_text(encoding="utf-8"))
        if time.time() - float(cached.get("fetched_at") or 0) < _CORP_CODE_TTL_DAYS * 86400:
            return cached["by_stock"]

    query = urllib.parse.urlencode({"crtfc_key": _api_key()})
    with urllib.request.urlopen(f"{_BASE_URL}/corpCode.xml?{query}", timeout=60) as response:
        blob = response.read()
    try:
        archive = zipfile.ZipFile(BytesIO(blob))
        xml_text = archive.read(archive.namelist()[0]).decode("utf-8")
    except zipfile.BadZipFile as exc:  # 키가 틀리면 zip 대신 에러 XML 이 온다
        raise OpenDartError(f"corpCode 응답이 zip 이 아닙니다 — API 키를 확인하세요: {blob[:200]!r}") from exc

    by_stock: dict[str, str] = {}
    for element in ElementTree.fromstring(xml_text).iter("list"):
        stock_code = (element.findtext("stock_code") or "").strip()
        corp_code = (element.findtext("corp_code") or "").strip()
        if len(stock_code) == 6 and corp_code:
            by_stock[stock_code] = corp_code

    _CORP_CODE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CORP_CODE_CACHE.write_text(
        json.dumps({"fetched_at": time.time(), "by_stock": by_stock}, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("[OPENDART] corp_code 매핑 갱신 — 상장 %d종목", len(by_stock))
    return by_stock


# 계정 매칭 — account_id 를 우선 보고, 없으면 account_nm 으로 폴백한다.
# (표준계정(ifrs-full/dart)이 아닌 회사도 있어 이름 폴백이 필요하다)
_ACCOUNT_RULES: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    # 필드: (재무제표 구분들, account_id 후보, account_nm 키워드)
    "operating_income": (("IS", "CIS"), ("dart_OperatingIncomeLoss",), ("영업이익",)),
    "net_income": (("IS", "CIS"), ("ifrs-full_ProfitLoss", "ifrs_ProfitLoss"), ("당기순이익",)),
    "dividends_paid": (
        ("CF",),
        ("ifrs-full_DividendsPaidClassifiedAsFinancingActivities", "ifrs_DividendsPaid"),
        ("배당금의 지급", "배당금지급"),
    ),
    "buyback": (
        ("CF",),
        ("dart_AcquisitionOfTreasuryShares",),
        ("자기주식의 취득", "자기주식 취득", "자사주의 취득", "자사주 취득"),
    ),
    "buyback_disposal": (
        ("CF",),
        ("dart_DisposalOfTreasuryShares",),
        ("자기주식의 처분", "자기주식 처분", "자사주의 처분"),
    ),
    "share_issuance": (
        ("CF",),
        ("ifrs-full_ProceedsFromIssuingShares", "dart_ProceedsFromIssuanceOfCommonStock"),
        ("유상증자", "주식의 발행", "신주의 발행"),
    ),
}


def _match_account(row: dict[str, Any]) -> str | None:
    sj_div = str(row.get("sj_div") or "")
    account_id = str(row.get("account_id") or "")
    account_nm = str(row.get("account_nm") or "").strip()
    for field, (sj_divs, ids, name_keys) in _ACCOUNT_RULES.items():
        if sj_div not in sj_divs:
            continue
        if account_id in ids or any(key in account_nm for key in name_keys):
            return field
    return None


def _parse_amount(value: Any) -> float | None:
    text = str(value or "").replace(",", "").strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def annual_financials(corp_code: str, year: int) -> dict[str, float] | None:
    """``year`` 회계연도 사업보고서의 핵심 계정 (당기 금액, 원 단위).

    연결(CFS) 우선, 연결이 없는 회사는 별도(OFS) 폴백. 보고서 자체가 없으면 None.
    현금흐름표 지출 계정(배당지급·자기주식취득)은 회사에 따라 부호가 달라 절댓값으로 통일한다.
    """
    payload = None
    for fs_div in ("CFS", "OFS"):
        payload = _request(
            "fnlttSinglAcntAll.json",
            corp_code=corp_code,
            bsns_year=str(year),
            reprt_code=_ANNUAL_REPORT,
            fs_div=fs_div,
        )
        if payload is not None:
            break
    if payload is None:
        return None

    result: dict[str, float] = {}
    for row in payload.get("list") or []:
        field = _match_account(row)
        if field is None or field in result:  # 같은 계정이 여러 구분(IS·CIS)에 오면 첫 값 사용
            continue
        amount = _parse_amount(row.get("thstrm_amount"))
        if amount is None:
            continue
        result[field] = abs(amount) if field != "operating_income" and field != "net_income" else amount
    return result or None


def dps_history(corp_code: str, base_year: int) -> dict[int, float]:
    """보통주 주당 현금배당금 {연도: 원}. ``base_year`` 사업보고서의 3개년(당기/전기/전전기).

    보고서가 없으면 빈 dict. 무배당 연도는 값이 '-' 로 와서 0 으로 채운다.
    """
    payload = _request("alotMatter.json", corp_code=corp_code, bsns_year=str(base_year), reprt_code=_ANNUAL_REPORT)
    if payload is None:
        return {}
    for row in payload.get("list") or []:
        se = str(row.get("se") or "")
        stock_kind = str(row.get("stock_knd") or "")
        if "주당 현금배당금" in se and (not stock_kind or "보통주" in stock_kind):
            return {
                year: _parse_amount(row.get(key)) or 0.0
                for key, year in (("thstrm", base_year), ("frmtrm", base_year - 1), ("lwfr", base_year - 2))
            }
    return {}


def latest_annual_year() -> int:
    """가장 최근에 확정됐을 사업연도 — 사업보고서는 이듬해 3월에 나온다.

    4월 이후면 전년도, 1~3월이면 전전년도를 기본으로 본다 (없으면 호출부에서 한 해 더 내려간다).
    """
    from datetime import date

    today = date.today()
    return today.year - 1 if today.month >= 4 else today.year - 2
