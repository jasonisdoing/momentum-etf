"""계좌 ID와 종목풀 ID 충돌 방지 유틸리티."""

from __future__ import annotations

from utils.account_registry import list_available_accounts
from utils.pool_registry import list_available_pools


def ensure_account_pool_id_separation() -> None:
    """계좌 ID와 종목풀 ID가 겹치면 예외를 발생시킨다."""
    account_ids = {str(v).strip().lower() for v in list_available_accounts()}
    pool_ids = {str(v).strip().lower() for v in list_available_pools()}
    overlap = sorted(account_ids & pool_ids)
    if overlap:
        ids = ", ".join(overlap)
        raise RuntimeError(
            f"계좌 ID와 종목풀 ID가 중복됩니다: {ids}. 운영/종목풀 스크립트 혼선을 방지하기 위해 ID를 분리해야 합니다."
        )
