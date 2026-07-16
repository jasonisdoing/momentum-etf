"""설정 로딩 유틸리티 (기본값/자동 보정 없음)."""

import json
from pathlib import Path

# 새 형식: signal, offense, defense는 {ticker, name} 객체
# 기존 형식: signal_ticker, offense_ticker, defense_ticker는 문자열
REQUIRED_KEYS_NEW: list[str] = [
    "signal",
    "defense",
    "drawdown_buy_cutoff",
    "drawdown_sell_cutoff",
    "benchmarks",
    # "months_range",  <-- 제거됨
    # "start_date",    <-- 추가됨 (필수)
    "slippage",
]
# 참고: 공격 자산(offense)은 더 이상 설정 키가 아니다 — 진입 시점마다
# tuning.offense_candidates 중 SMA 20일 이격도 1위를 동적으로 선택한다.

REQUIRED_KEYS_OLD: list[str] = [
    "signal_ticker",
    "offense_ticker",
    "defense_ticker",
    "drawdown_buy_cutoff",
    "drawdown_sell_cutoff",
    "benchmarks",
    # "months_range",
    # "start_date",
    "slippage",
]


def normalize_settings(settings: dict) -> dict:
    """raw 설정 dict 를 검증·정규화한다(파일/DB 공용)."""
    # 전략 구분 (기본값: switch - 하위 호환)
    settings["strategy"] = settings.get("strategy", "switch")

    # start_date 또는 months_range 중 하나는 있어야 함
    if "start_date" not in settings and "months_range" not in settings:
        raise ValueError("설정에 'start_date' 또는 'months_range'가 필요합니다.")

    return _normalize_switch_settings(settings)


def load_settings(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return normalize_settings(json.load(f))


def _normalize_offense_candidates(settings: dict) -> None:
    """공격 후보군을 정규화한다.

    tuning.offense_candidates 가 있으면 그것이 공격 후보군(동적 선택 대상)이고,
    없으면 단일 offense 로 동작한다(하위 호환). CASH 는 공격 후보가 될 수 없다.
    """
    tuning = settings.get("tuning") or {}
    raw = tuning.get("offense_candidates") or []
    candidates: list[dict] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        ticker = str(entry.get("ticker") or "").strip()
        if not ticker or ticker.upper() == "CASH" or ticker in seen:
            continue
        seen.add(ticker)
        candidates.append({"ticker": ticker, "name": entry.get("name") or ticker})
    if not candidates and settings.get("offense_ticker"):
        # 하위 호환: 옛 설정(단일 offense)만 있으면 그것을 후보 1개로 사용
        candidates = [{"ticker": settings["offense_ticker"], "name": settings.get("offense_name", settings["offense_ticker"])}]
    if not candidates:
        raise ValueError("공격 후보(tuning.offense_candidates)가 1개 이상 필요합니다. 레버리지-설정 화면에서 저장하세요.")
    settings["offense_candidates"] = candidates


def _normalize_switch_settings(settings: dict) -> dict:
    """스위칭 전략 설정 검증 및 정규화."""
    # 새 형식인지 확인
    is_new_format = "signal" in settings and isinstance(settings.get("signal"), dict)

    if is_new_format:
        missing = [k for k in REQUIRED_KEYS_NEW if k not in settings and k != "start_date" and k != "months_range"]
        if missing:
            raise ValueError(f"설정 파일에 필수 키가 없습니다: {missing}")

        # 새 형식을 내부적으로 사용할 수 있도록 정규화
        # ticker/name을 별도 필드로 추출 (offense 는 옵션 — 있으면 후보 폴백으로만 사용)
        settings["signal_ticker"] = settings["signal"]["ticker"]
        settings["signal_name"] = settings["signal"].get("name", settings["signal"]["ticker"])
        offense = settings.get("offense")
        if isinstance(offense, dict) and offense.get("ticker"):
            settings["offense_ticker"] = offense["ticker"]
            settings["offense_name"] = offense.get("name", offense["ticker"])
        settings["defense_ticker"] = settings["defense"]["ticker"]
        settings["defense_name"] = settings["defense"].get("name", settings["defense"]["ticker"])
    else:
        # 기존 형식
        missing = [k for k in REQUIRED_KEYS_OLD if k not in settings and k != "start_date" and k != "months_range"]
        if missing:
            raise ValueError(f"설정 파일에 필수 키가 없습니다: {missing}")

        # 이름이 없으면 티커를 이름으로 사용
        settings["signal_name"] = settings.get("signal_name", settings["signal_ticker"])
        settings["offense_name"] = settings.get("offense_name", settings["offense_ticker"])
        settings["defense_name"] = settings.get("defense_name", settings["defense_ticker"])

    _normalize_offense_candidates(settings)
    return settings
