"""레버리지 전략 설정·상태 조회 서비스 (UI/API 용).

설정·상태의 단일 소스는 MongoDB(`leverage/config_store.py`). 여기서는 화면 표시에
필요한 형태로 묶어 반환한다.
"""

from __future__ import annotations

from typing import Any

from leverage.config_store import (
    load_leverage_config_raw,
    load_leverage_state,
    save_leverage_config_raw,
)
from leverage.engine.backtest.settings import normalize_settings
from leverage.holding import count_holding_trading_days
from leverage.tuning_config import derive_benchmarks, validate_tuning_section


def load_leverage_settings(profile: str = "switch") -> dict[str, Any]:
    """레버리지 설정(편집 대상) + 직전 추천 상태(읽기 전용)를 함께 반환한다."""
    state = load_leverage_state(profile)
    if state and state.get("holding_start_date"):
        state["holding_days"] = count_holding_trading_days(state.get("target", ""), state["holding_start_date"])

    return {
        "profile": profile,
        "config": load_leverage_config_raw(profile),
        "state": state,
    }


def _validate_leverage_config(config: dict[str, Any]) -> None:
    """저장 전 검증 (실패 시 ValueError → 400). 정상값만 DB 에 들어가게 한다."""
    if not isinstance(config, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")

    for key in ("signal", "offense", "defense"):
        asset = config.get(key)
        if not isinstance(asset, dict) or not str(asset.get("ticker") or "").strip():
            raise ValueError(f"'{key}' 자산의 티커가 필요합니다.")

    for key in ("drawdown_buy_cutoff", "drawdown_sell_cutoff", "slippage"):
        value = config.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"'{key}' 는 0 이상의 숫자여야 합니다.")

    months_range = config.get("months_range")
    has_months = isinstance(months_range, (int, float)) and not isinstance(months_range, bool) and months_range > 0
    if not config.get("start_date") and not has_months:
        raise ValueError("'months_range'(0보다 큰 수) 또는 'start_date' 가 필요합니다.")

    # 튜닝 탐색 공간 검증 — tune.py 와 동일한 공통 검증기를 사용.
    # 벤치마크는 더 이상 직접 입력하지 않고 후보군에서 파생하므로 tuning 은 필수.
    validate_tuning_section(config.get("tuning"))

    # 엔진 정규화로 추가 검증 (사본으로 — 파생 키가 저장값에 섞이지 않게).
    # benchmarks 는 후보군에서 파생해 주입(엔진 필수 키 충족).
    check = dict(config)
    check["benchmarks"] = derive_benchmarks(config)
    normalize_settings(check)


def save_leverage_settings(profile: str, config: dict[str, Any]) -> dict[str, Any]:
    """검증 후 설정을 DB 에 저장하고, 갱신된 설정+상태를 반환한다.

    벤치마크는 후보군(tuning)에서 파생해 DB 에 함께 저장한다(단일 소스 유지).
    """
    _validate_leverage_config(config)
    config = dict(config)
    config["benchmarks"] = derive_benchmarks(config)
    save_leverage_config_raw(profile, config)
    return load_leverage_settings(profile)


_TUNE_JOB_NAME = "leverage_tune"
_TUNE_SCRIPT = "scripts/leverage_tune_switch.py"


def trigger_leverage_tune(profile: str = "switch") -> dict[str, Any]:
    """튜닝 작업을 배치 큐에 추가한다(워커가 순서대로 실행). 이미 대기/실행 중이면 무시."""
    from utils.batch_queue import enqueue

    result = enqueue(_TUNE_JOB_NAME, _TUNE_SCRIPT, triggered_by="manual")
    return {
        "enqueued": bool(result.get("enqueued")),
        "reason": result.get("reason"),
    }


def _parse_tune_log(text: str) -> dict[str, Any]:
    """튜닝 로그에서 진행률/완료 여부를 파싱한다."""
    import re

    done = "종료 시각" in text
    progress_pct: float | None = None
    completed = total = None
    m = re.search(r"진행률:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)", text)
    if m:
        completed, total = int(m.group(1)), int(m.group(2))
        progress_pct = float(m.group(3))
    if done:
        progress_pct = 100.0
    return {"done": done, "progress_pct": progress_pct, "completed": completed, "total": total}


def list_tune_log_dates(profile: str = "switch") -> list[str]:
    """저장된 튜닝 로그 날짜(YYYY-MM-DD) 목록을 최신순으로 반환한다."""
    from leverage.constants import ZRESULTS_DIR

    out_dir = ZRESULTS_DIR / profile
    if not out_dir.exists():
        return []
    dates = {p.stem[len("tune_"):] for p in out_dir.glob("tune_*.log")}
    # YYYY-MM-DD 형식만(파일명 안전), 최신순 정렬(문자열 정렬 = 날짜순)
    import re

    return sorted((d for d in dates if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)), reverse=True)


def _read_tune_log(profile: str, date: str | None = None) -> dict[str, Any]:
    """튜닝 로그(zresults/<profile>/tune_<date>.log)의 내용·진행률을 반환한다.

    date 가 없으면 가장 최근 로그를 읽는다. 잘못된 date 형식은 무시(경로 조작 방지).
    """
    import re

    from leverage.constants import ZRESULTS_DIR

    out_dir = ZRESULTS_DIR / profile
    empty = {"log_text": "", "log_file": None, "selected_date": None, "done": False, "progress_pct": None, "completed": None, "total": None}

    if date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        date = None  # 형식이 아니면 무시 (디렉터리 탈출 방지)

    if date:
        path = out_dir / f"tune_{date}.log"
        if not path.exists():
            return {**empty, "selected_date": date}
    else:
        logs = sorted(out_dir.glob("tune_*.log"), key=lambda p: p.stat().st_mtime, reverse=True) if out_dir.exists() else []
        if not logs:
            return empty
        path = logs[0]

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    parsed = _parse_tune_log(text)
    return {"log_text": text, "log_file": path.name, "selected_date": date or path.stem[len("tune_"):], **parsed}


def leverage_tune_status(profile: str = "switch", date: str | None = None) -> dict[str, Any]:
    """튜닝 실행 상태 + 선택 날짜(없으면 최신) 로그(진행도/결과) + 날짜 목록을 반환한다."""
    from utils.batch_queue import get_latest_item

    item = get_latest_item(_TUNE_JOB_NAME)
    queue_status = item.get("status") if item else None  # pending/running/done/failed/None

    log = _read_tune_log(profile, date)

    def _iso(value: Any) -> str | None:
        return value.isoformat() if hasattr(value, "isoformat") else value

    return {
        "queue_status": queue_status,  # None=이력 없음
        "running": queue_status in ("pending", "running"),
        "exit_code": item.get("exit_code") if item else None,
        "error": item.get("error") if item else None,
        "triggered_at": _iso(item.get("triggered_at")) if item else None,
        "started_at": _iso(item.get("started_at")) if item else None,
        "ended_at": _iso(item.get("ended_at")) if item else None,
        "dates": list_tune_log_dates(profile),
        **log,
    }


def resolve_pool_ticker(ticker: str) -> dict[str, Any]:
    """종목풀(stock_meta)에서 해당 티커를 가진 활성 종목을 찾아 종목명을 반환합니다."""
    from utils.db_manager import get_db_connection

    ticker_norm = str(ticker or "").strip().upper()
    if not ticker_norm:
        raise ValueError("조회할 티커가 필요합니다.")

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    # active stocks 중 일치하는 종목 조회 (is_deleted가 참이 아닌 것)
    doc = db.stock_meta.find_one(
        {
            "ticker": ticker_norm,
            "is_deleted": {"$ne": True}
        },
        {"ticker": 1, "name": 1, "ticker_type": 1}
    )

    if doc is None:
        # fallback: 삭제 상태이더라도 종목풀에 등록된 적이 있는 종목 검색
        doc = db.stock_meta.find_one(
            {"ticker": ticker_norm},
            {"ticker": 1, "name": 1, "ticker_type": 1}
        )

    if doc is None:
        raise ValueError(f"종목풀에서 티커 '{ticker_norm}'를 찾을 수 없습니다.")

    return {
        "ticker": doc["ticker"],
        "name": doc["name"],
        "ticker_type": doc["ticker_type"]
    }
