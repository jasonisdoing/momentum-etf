"""레버리지 단타(스캘프) 매매 신호 설정의 DB 저장/조회.

- `scalp_settings` 컬렉션에 단일 문서(_id="ema_st")로 저장한다.
- 전략 모드: both(EMA+슈퍼트렌드) / ema(EMA 단독) / st(슈퍼트렌드 단독).
- 값이 없거나 스키마가 다르면 임의 기본값으로 보정하지 않고 ``None`` 을 반환한다(silent default 금지).
  최초 표시용 기본값은 화면(프런트)이 갖고, 사용자가 '저장'을 눌러야 DB 에 들어간다.
"""

from __future__ import annotations

from typing import Any

_COLLECTION = "scalp_settings"
_DOC_ID = "ema_st"
_MODES = {"both", "ema", "st"}


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결 실패 (스캘프 설정)")
    return db


def load_scalp_settings() -> dict[str, Any] | None:
    """저장된 EMA+슈퍼트렌드 설정을 반환한다. 저장된 적이 없거나 스키마가 다르면 None."""
    doc = _db()[_COLLECTION].find_one({"_id": _DOC_ID})
    if doc is None:
        return None
    try:
        mode = str(doc["mode"])
        if mode not in _MODES:
            return None
        return {
            "mode": mode,
            "ema_period": int(doc["ema_period"]),
            "st_period": int(doc["st_period"]),
            "st_mult": float(doc["st_mult"]),
        }
    except (KeyError, TypeError, ValueError):
        return None  # 구(舊) 스키마 등 → 미설정으로 간주(프런트 기본값 사용).


def _validate(settings: dict[str, Any]) -> dict[str, Any]:
    """저장 전 검증. 실패 시 ValueError(→ 400). 정상값만 DB 에 들어가게 한다."""
    if not isinstance(settings, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")

    mode = str(settings.get("mode") or "")
    if mode not in _MODES:
        raise ValueError("모드는 both/ema/st 중 하나여야 합니다.")

    try:
        ema_period = int(settings["ema_period"])
        st_period = int(settings["st_period"])
        st_mult = float(settings["st_mult"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("EMA 기간·슈퍼트렌드 기간/배수가 모두 필요합니다.") from exc

    if ema_period < 1:
        raise ValueError("EMA 기간은 1 이상이어야 합니다.")
    if st_period < 1:
        raise ValueError("슈퍼트렌드 기간은 1 이상이어야 합니다.")
    if st_mult <= 0:
        raise ValueError("슈퍼트렌드 배수는 0보다 커야 합니다.")

    return {
        "mode": mode,
        "ema_period": ema_period,
        "st_period": st_period,
        "st_mult": st_mult,
    }


def save_scalp_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """검증 후 설정을 DB 에 upsert 하고, 저장된 값을 반환한다."""
    clean = _validate(settings)
    _db()[_COLLECTION].update_one({"_id": _DOC_ID}, {"$set": clean}, upsert=True)
    return clean
