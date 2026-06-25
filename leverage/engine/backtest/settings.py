"""설정 로딩 유틸리티 (기본값/자동 보정 없음)."""

import json
from pathlib import Path

# 새 형식: signal, offense, defense는 {ticker, name} 객체
# 기존 형식: signal_ticker, offense_ticker, defense_ticker는 문자열
REQUIRED_KEYS_NEW: list[str] = [
    "signal",
    "offense",
    "defense",
    "drawdown_buy_cutoff",
    "drawdown_sell_cutoff",
    "benchmarks",
    # "months_range",  <-- 제거됨
    # "start_date",    <-- 추가됨 (필수)
    "slippage",
]

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

def load_settings(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        settings = json.load(f)

    # 전략 구분 (기본값: switch - 하위 호환)
    strategy = settings.get("strategy", "switch")
    settings["strategy"] = strategy

    # start_date 또는 months_range 중 하나는 있어야 함
    if "start_date" not in settings and "months_range" not in settings:
        raise ValueError("설정 파일에 'start_date' 또는 'months_range'가 필요합니다.")

    return _normalize_switch_settings(settings)


def _normalize_switch_settings(settings: dict) -> dict:
    """스위칭 전략 설정 검증 및 정규화."""
    # 새 형식인지 확인
    is_new_format = "signal" in settings and isinstance(settings.get("signal"), dict)

    if is_new_format:
        missing = [k for k in REQUIRED_KEYS_NEW if k not in settings and k != "start_date" and k != "months_range"]
        if missing:
            raise ValueError(f"설정 파일에 필수 키가 없습니다: {missing}")

        # 새 형식을 내부적으로 사용할 수 있도록 정규화
        # ticker/name을 별도 필드로 추출
        settings["signal_ticker"] = settings["signal"]["ticker"]
        settings["signal_name"] = settings["signal"].get("name", settings["signal"]["ticker"])
        settings["offense_ticker"] = settings["offense"]["ticker"]
        settings["offense_name"] = settings["offense"].get("name", settings["offense"]["ticker"])
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

    return settings
