"""모멘텀 진입 문턱 선택지를 국가별로 골라 주는 곳 — 이평선 선택지(`utils/ma_options`)와 같은 방식.

**목록 자체는 `config.ENTRY_VOL_MULT_OPTIONS_BY_COUNTRY` 가 단일 소스다** — 값을 늘리거나 줄일 때는
config 만 고친다. 이 모듈은 국가 판정과 응답 형태만 맡는다. 국가는 종목풀 설정의 ``country_code`` 를
그대로 쓰고, 모르는 국가면 에러(임의 기본값 없음). 화면 셀렉트는 API 응답으로 받은 목록만 렌더한다.
"""

from __future__ import annotations

from config import ENTRY_VOL_MULT_OPTIONS_BY_COUNTRY as _BY_COUNTRY


def entry_vol_mult_options(country_code: str | None) -> tuple[float | None, ...]:
    country = str(country_code or "").strip().lower()
    if country not in _BY_COUNTRY:
        raise ValueError(f"진입 문턱 선택지를 지원하지 않는 국가입니다: {country_code!r}")
    return _BY_COUNTRY[country]


def entry_vol_mult_options_by_country() -> dict[str, list[float | None]]:
    """여러 풀을 한 화면에 두는 곳(종목풀 설정·모멘텀)용 — 화면이 행·풀의 국가로 고른다."""
    return {country: list(options) for country, options in _BY_COUNTRY.items()}


# 전환기 검증용 — 어느 국가든 허용되는 값의 합집합(없음이 첫 칸). 국가별 엄격 검증은 풀을 아는 곳
# (모멘텀 설정·순위·튜닝)이 `entry_vol_mult_options` 로 한다. 풀 설정 저장은 국가를 몰라 이 합집합만 본다.
ENTRY_VOL_MULT_ANY_COUNTRY: tuple[float | None, ...] = (
    None,
    *sorted({value for options in _BY_COUNTRY.values() for value in options if value is not None}),
)
