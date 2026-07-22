"""레버리지 단타(스캘프) 매매 신호 설정의 DB 저장/조회.

- `scalp_settings` 컬렉션에 단일 문서(_id="ema_st")로 저장한다.
- 전략: 슈퍼트렌드 단독(기간·배수). 값이 없거나 스키마가 다르면 임의 기본값으로 보정하지 않고
  ``None`` 을 반환한다(silent default 금지). 최초 표시용 기본값은 화면(프런트)이 갖고,
  사용자가 '저장'을 눌러야 DB 에 들어간다. (구 문서의 mode/ema_period 필드는 무시된다.)
"""

from __future__ import annotations

from typing import Any

_COLLECTION = "scalp_settings"
_DOC_ID = "ema_st"


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결 실패 (스캘프 설정)")
    return db


def load_scalp_settings() -> dict[str, Any] | None:
    """저장된 슈퍼트렌드 설정을 반환한다. 저장된 적이 없거나 스키마가 다르면 None."""
    doc = _db()[_COLLECTION].find_one({"_id": _DOC_ID})
    if doc is None:
        return None
    try:
        return {
            "st_period": int(doc["st_period"]),
            "st_mult": float(doc["st_mult"]),
        }
    except (KeyError, TypeError, ValueError):
        return None  # 구(舊) 스키마 등 → 미설정으로 간주(프런트 기본값 사용).


def _validate(settings: dict[str, Any]) -> dict[str, Any]:
    """저장 전 검증. 실패 시 ValueError(→ 400). 정상값만 DB 에 들어가게 한다."""
    if not isinstance(settings, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")

    try:
        st_period = int(settings["st_period"])
        st_mult = float(settings["st_mult"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("슈퍼트렌드 기간/배수가 모두 필요합니다.") from exc

    if st_period < 1:
        raise ValueError("슈퍼트렌드 기간은 1 이상이어야 합니다.")
    if st_mult <= 0:
        raise ValueError("슈퍼트렌드 배수는 0보다 커야 합니다.")

    return {
        "st_period": st_period,
        "st_mult": st_mult,
    }


def save_scalp_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """검증 후 설정을 DB 에 upsert 하고, 저장된 값을 반환한다."""
    clean = _validate(settings)
    _db()[_COLLECTION].update_one({"_id": _DOC_ID}, {"$set": clean}, upsert=True)
    return clean
