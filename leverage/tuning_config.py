"""레버리지 튜닝 탐색 공간(config.tuning) 검증 + 엔진 입력 변환 공통 로직.

DB `leverage_config` 의 `tuning` 섹션이 튜닝 그리드서치의 단일 소스다(하드코딩 제거).
스키마:

    "tuning": {
        "offense_candidates": [{"ticker": "...", "name": "..."}, ...],  # 공격 후보군
        "defense_candidates": [{"ticker": "...", "name": "..."}, ...],  # 방어 후보군
        "buy_cutoff_range":  {"min": 1, "max": 5, "step": 1},          # 매수컷 탐색 범위(끝값 포함)
        "sell_cutoff_range": {"min": 1, "max": 5, "step": 1}            # 매도컷 탐색 범위(끝값 포함)
    }

검증 실패 시 ValueError 를 던진다(임의 기본값·보정 없음).
"""

from __future__ import annotations

_CANDIDATE_KEYS = ("offense_candidates", "defense_candidates")
_RANGE_KEYS = ("buy_cutoff_range", "sell_cutoff_range")


def _validate_range(key: str, rng: object) -> None:
    if not isinstance(rng, dict):
        raise ValueError(f"'{key}' 는 min/max/step 객체여야 합니다.")
    for field in ("min", "max", "step"):
        value = rng.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"'{key}.{field}' 는 0 이상의 숫자여야 합니다.")
    if float(rng["step"]) <= 0:
        raise ValueError(f"'{key}.step' 은 0보다 커야 합니다.")
    if float(rng["max"]) < float(rng["min"]):
        raise ValueError(f"'{key}.max' 는 min 이상이어야 합니다.")


def validate_tuning_section(tuning: object) -> None:
    """config.tuning 섹션의 형식을 검증한다(실패 시 ValueError)."""
    if not isinstance(tuning, dict):
        raise ValueError("'tuning' 형식이 올바르지 않습니다.")

    for key in _CANDIDATE_KEYS:
        candidates = tuning.get(key)
        if not isinstance(candidates, list) or len(candidates) == 0:
            raise ValueError(f"'{key}' 후보가 1개 이상 필요합니다.")
        for entry in candidates:
            if not isinstance(entry, dict) or not str(entry.get("ticker") or "").strip():
                raise ValueError(f"'{key}' 항목에 티커가 필요합니다.")

    for key in _RANGE_KEYS:
        _validate_range(key, tuning.get(key))

    # 히스테리시스 제약: 엔진은 buy_cut >= sell_cut 조합을 건너뛴다.
    # 모든 조합이 스킵되지 않도록 매수컷 최솟값 < 매도컷 최댓값 이어야 한다.
    buy_min = float(tuning["buy_cutoff_range"]["min"])
    sell_max = float(tuning["sell_cutoff_range"]["max"])
    if buy_min >= sell_max:
        raise ValueError("매수컷 범위의 최솟값이 매도컷 범위의 최댓값보다 작아야 유효한 조합이 생깁니다.")


def derive_benchmarks(config: dict) -> list[dict]:
    """벤치마크 = 공격 후보 ∪ 방어 후보 (티커 중복 제거, CASH 제외).

    후보군이 단일 소스이므로 벤치마크는 따로 입력하지 않고 여기서 파생한다
    (별도 silent default 가 아니라 후보로부터의 명시적 변환).
    """
    tuning = config.get("tuning") or {}
    seen: set[str] = set()
    result: list[dict] = []
    for key in ("offense_candidates", "defense_candidates"):
        for entry in tuning.get(key) or []:
            if not isinstance(entry, dict):
                continue
            ticker = str(entry.get("ticker") or "").strip()
            if not ticker or ticker.upper() == "CASH":
                continue
            if ticker.upper() in seen:
                continue
            seen.add(ticker.upper())
            result.append({"ticker": ticker, "name": entry.get("name") or ticker})
    return result


def _arange_inclusive(rng: dict) -> list[float]:
    """min~max(끝값 포함)를 step 간격으로 나열한다."""
    import numpy as np

    mn, mx, st = float(rng["min"]), float(rng["max"]), float(rng["step"])
    # 끝값(max)을 안정적으로 포함하기 위해 stop 에 step/2 를 더한다.
    return np.arange(mn, mx + st / 2.0, st)


def build_tuning_search_space(raw_config: dict) -> dict:
    """DB raw config 의 `tuning` 섹션을 run_tuning 입력 형식으로 변환한다.

    반환: {drawdown_buy_cutoff, drawdown_sell_cutoff, offense, defense}
    (`tuning` 누락/오류 시 ValueError — 임의 기본값 없음.)
    """
    tuning = raw_config.get("tuning")
    if tuning is None:
        raise ValueError("config 에 'tuning'(튜닝 탐색 공간) 섹션이 없습니다. 레버리지-설정 화면에서 저장하세요.")
    validate_tuning_section(tuning)

    return {
        "drawdown_buy_cutoff": _arange_inclusive(tuning["buy_cutoff_range"]),
        "drawdown_sell_cutoff": _arange_inclusive(tuning["sell_cutoff_range"]),
        "offense": [dict(x) for x in tuning["offense_candidates"]],
        "defense": [dict(x) for x in tuning["defense_candidates"]],
    }
