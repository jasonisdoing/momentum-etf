"""이평선 일수 선택지 — 시스템 전체의 단일 소스. **국가별**로 다르다.

한국은 거래일 기준 3·4.5·6·9개월(60·90·120·180), 미국·호주는 관례(50·100일선)에 맞춘 50·75·100·150 —
한국:미국 = 1.2 배로 칸마다 짝이 맞아 두 시장 결과를 같은 자리끼리 비교할 수 있다. 단기는 공통.

종목풀 설정·순위·종목풀 백테스트·모멘텀·신고가(이탈선)·보유종목 알림이 모두 여기서 받는다.
국가는 종목풀/계좌 설정의 ``country_code`` 를 그대로 쓰고, 모르는 국가면 에러(임의 기본값 없음).
화면 셀렉트는 API 응답으로 받은 목록만 렌더한다(프론트에 별도 목록을 두지 않는다).
"""

from __future__ import annotations

_SHORT_BY_COUNTRY: dict[str, tuple[int, ...]] = {
    "kor": (20, 30, 40),
    "us": (20, 30, 40),
    "au": (20, 30, 40),
}
_LONG_BY_COUNTRY: dict[str, tuple[int, ...]] = {
    "kor": (60, 90, 120, 180),
    "us": (50, 75, 100, 150),
    "au": (50, 75, 100, 150),
}


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
