"""이평선 일수 선택지를 국가별로 골라 주는 곳 — 시스템 전체가 여기로 받는다.

**목록 자체는 `config.SHORT_MA_DAYS_BY_COUNTRY` / `LONG_MA_DAYS_BY_COUNTRY` 가 단일
소스다** — 값을 늘리거나 줄일 때는 config 만 고친다. 이 모듈은 국가 판정과 응답 형태만 맡는다.

종목풀 설정·순위·종목풀 백테스트·모멘텀·신고가(이탈선)·보유종목 알림이 모두 여기서 받는다.
국가는 종목풀/계좌 설정의 ``country_code`` 를 그대로 쓰고, 모르는 국가면 에러(임의 기본값 없음).
화면 셀렉트는 API 응답으로 받은 목록만 렌더한다(프론트에 별도 목록을 두지 않는다).
"""

from __future__ import annotations

from config import LONG_MA_DAYS_BY_COUNTRY as _LONG_BY_COUNTRY
from config import SHORT_MA_DAYS_BY_COUNTRY as _SHORT_BY_COUNTRY


def _norm(country_code: str | None) -> str:
    country = str(country_code or "").strip().lower()
    if country not in _LONG_BY_COUNTRY:
        raise ValueError(f"이평선 선택지를 지원하지 않는 국가입니다: {country_code!r}")
    return country


def short_ma_options(country_code: str | None) -> tuple[int, ...]:
    return _SHORT_BY_COUNTRY[_norm(country_code)]


def long_ma_options(country_code: str | None) -> tuple[int, ...]:
    return _LONG_BY_COUNTRY[_norm(country_code)]


def ma_options_payload(country_code: str | None) -> dict[str, list[int]]:
    """단일 풀/계좌 화면용 — 그 국가의 목록."""
    return {
        "short_ma_options": list(short_ma_options(country_code)),
        "long_ma_options": list(long_ma_options(country_code)),
    }


def ma_options_by_country() -> dict[str, dict[str, list[int]]]:
    """여러 풀/계좌를 한 화면에 두는 곳(종목풀 설정·알림)용 — 화면이 행의 국가로 고른다."""
    return {country: ma_options_payload(country) for country in _LONG_BY_COUNTRY}


# 전환기 검증용 — 어느 국가든 허용되는 값의 합집합. 국가별 엄격 검증은 설정을 모두 새 목록으로
# 저장한 뒤(3단계)에 전환한다. 그 전에 엄격히 막으면 옛 저장값(미국 120 등)으로 순위 화면이 깨진다.
SHORT_MA_OPTIONS: tuple[int, ...] = tuple(sorted({d for days in _SHORT_BY_COUNTRY.values() for d in days}))
LONG_MA_OPTIONS: tuple[int, ...] = tuple(sorted({d for days in _LONG_BY_COUNTRY.values() for d in days}))
