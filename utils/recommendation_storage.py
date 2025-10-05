from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:  # pragma: no cover - numpy is optional at runtime
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


_BASE_DIR = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _BASE_DIR / "data" / "results"


def _make_json_safe(obj: Any) -> Any:
    """Convert arbitrary Python objects into JSON serializable structures."""

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()

    if np is not None and isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]

    if isinstance(obj, pd.Series):
        return [_make_json_safe(v) for v in obj.tolist()]

    if isinstance(obj, pd.DataFrame):
        return [
            {k: _make_json_safe(v) for k, v in record.items()}
            for record in obj.to_dict(orient="records")
        ]

    return str(obj)


def save_recommendation_payload(
    payload: Any,
    *,
    account_id: str,
    country_code: str,
    results_dir: Path | None = None,
) -> Path:
    """Persist a payload to account and country result files."""

    account_norm = (account_id or "").strip().lower()
    country_norm = (country_code or "").strip().lower()

    if not account_norm or not country_norm:
        raise RuntimeError("Both account_id and country_code are required to save recommendations")

    target_dir = (results_dir or _RESULTS_DIR).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_payload = _make_json_safe(payload)

    account_path = target_dir / f"recommendation_{account_norm}.json"
    country_path = target_dir / f"recommendation_{country_norm}.json"

    with account_path.open("w", encoding="utf-8") as fp:
        json.dump(safe_payload, fp, ensure_ascii=False, indent=2)

    if country_path != account_path:
        with country_path.open("w", encoding="utf-8") as fp:
            json.dump(safe_payload, fp, ensure_ascii=False, indent=2)

    return account_path


def save_recommendation_report(
    report: Any,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Persist a RecommendationReport-like object and return the written account path."""

    account_id = getattr(report, "account_id", "")
    country_code = getattr(report, "country_code", "")
    recommendations = getattr(report, "recommendations", None)

    if recommendations is None:
        raise RuntimeError("Recommendation report must include recommendations data")

    return save_recommendation_payload(
        recommendations,
        account_id=str(account_id),
        country_code=str(country_code),
        results_dir=results_dir,
    )
