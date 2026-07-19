"""설정 로딩 유틸리티 (기본값/자동 보정 없음)."""

import json
from pathlib import Path

FIXED_INDEX_BY_MARKET = {
    "kor": {"ticker": "^KS11", "name": "코스피"},
    "us": {"ticker": "^NDX", "name": "나스닥 100"},
}

# switch 새 형식: signal, offense, defense는 {ticker, name} 객체
# switch 기존 형식: signal_ticker, offense_ticker, defense_ticker는 문자열
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
# 참고: 레버리지 자산(offense legacy)은 더 이상 단일 설정 키가 아니다.
# 진입 시점마다 tuning.offense_candidates 중 SMA 20일 이격도 1위를 동적으로 선택한다.

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

    if settings["strategy"] == "sma_cross":
        return _normalize_sma_cross_settings(settings)

    # start_date 또는 months_range 중 하나는 있어야 함
    if "start_date" not in settings and "months_range" not in settings:
        raise ValueError("설정에 'start_date' 또는 'months_range'가 필요합니다.")

    return _normalize_switch_settings(settings)


def _normalize_sma_cross_settings(settings: dict) -> dict:
    """SMA 크로스 전략 설정 검증·정규화.

    필수: index/leverage/defense({ticker,name}), sma_days, peak_drawdown_pct, slippage, market.
    방어는 현금 가능(ticker='CASH'). 매수/매도 컷·후보군은 이 전략에 없다.
    """
    for key in ("index", "leverage", "defense"):
        value = settings.get(key)
        if not isinstance(value, dict) or not str(value.get("ticker") or "").strip():
            raise ValueError(f"'{key}' 는 {{ticker, name}} 객체여야 합니다. 레버리지-설정 화면에서 저장하세요.")
        ticker = str(value["ticker"]).strip()
        settings[f"{key}_ticker"] = ticker
        settings[f"{key}_name"] = str(value.get("name") or ticker).strip()

    if str(settings["leverage_ticker"]).upper() == "CASH":
        raise ValueError("레버리지 티커는 현금(CASH)일 수 없습니다.")

    sma_days = settings.get("sma_days")
    if not isinstance(sma_days, int) or sma_days < 2:
        raise ValueError(f"sma_days 는 2 이상 정수여야 합니다: {sma_days}")

    peak_drawdown_pct = settings.get("peak_drawdown_pct")
    if not isinstance(peak_drawdown_pct, (int, float)) or isinstance(peak_drawdown_pct, bool) or peak_drawdown_pct < 0:
        raise ValueError(f"peak_drawdown_pct 는 0 이상 숫자여야 합니다: {peak_drawdown_pct}")
    settings["peak_drawdown_pct"] = float(peak_drawdown_pct)

    slippage = settings.get("slippage")
    if not isinstance(slippage, (int, float)) or slippage < 0:
        raise ValueError(f"slippage 는 0 이상 숫자여야 합니다: {slippage}")

    market = str(settings.get("market") or "").strip().lower()
    if market not in {"kor", "us"}:
        raise ValueError(f"market 은 kor/us 중 하나여야 합니다: {settings.get('market')}")
    settings["market"] = market

    # SMA 크로스 전략의 지수는 시장별 고정값이다. DB에 과거 값이 남아도 계산 기준은 여기서 통일한다.
    fixed_index = FIXED_INDEX_BY_MARKET[market]
    settings["index"] = dict(fixed_index)
    settings["index_ticker"] = fixed_index["ticker"]
    settings["index_name"] = fixed_index["name"]

    # 슬랙 알람 On/Off (기본 꺼짐). 켜져 있으면 추천 배치가 장 마감 직후 알림을 보낸다.
    settings["slack_enabled"] = bool(settings.get("slack_enabled", False))

    return settings


def load_settings(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return normalize_settings(json.load(f))


def _normalize_offense_candidates(settings: dict) -> None:
    """레버리지 후보군을 정규화한다.

    tuning.offense_candidates 가 있으면 그것이 레버리지 후보군(동적 선택 대상)이고,
    없으면 단일 offense 로 동작한다(하위 호환). CASH 는 레버리지 후보가 될 수 없다.
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
        raise ValueError("레버리지 후보(tuning.offense_candidates)가 1개 이상 필요합니다. 레버리지-설정 화면에서 저장하세요.")
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
        # ticker/name을 별도 필드로 추출 (offense 는 legacy 옵션 — 있으면 후보 폴백으로만 사용)
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
