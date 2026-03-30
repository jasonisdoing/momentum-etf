#!/usr/bin/env python
"""지정된 종목 타입들의 종목을 일괄 완전 삭제한다.

기본 동작은 dry-run이며, --execute 옵션이 있어야 실제 삭제한다.
실행 시 hard delete(hard_remove_stock)만 수행한다.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_app_logger
from utils.settings_loader import list_available_ticker_types
from utils.stock_list_io import get_etfs, hard_remove_stock

DEFAULT_TARGET_TYPES = [
    "kor_kr",
    "kor_pension",
    "kor_isa",
    "kor_us",
    "aus",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="여러 종목 타입의 활성 종목을 일괄 삭제합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ticker_types",
        nargs="*",
        help="대상 종목 타입 ID 목록 (미지정 시 기본 5개 타입)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="실제 삭제 실행 (없으면 dry-run)",
    )
    return parser


def _normalize_targets(ticker_types: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for t_id in ticker_types:
        norm = str(t_id or "").strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        normalized.append(norm)
    return normalized


def main() -> int:
    logger = get_app_logger()
    args = _build_parser().parse_args()

    targets = _normalize_targets(args.ticker_types or DEFAULT_TARGET_TYPES)
    available = set(list_available_ticker_types())
    valid_targets = [t_id for t_id in targets if t_id in available]
    invalid_targets = [t_id for t_id in targets if t_id not in available]

    if invalid_targets:
        logger.warning("설정에 없는 종목 타입은 건너뜁니다: %s", ", ".join(invalid_targets))

    if not valid_targets:
        logger.error("유효한 대상 종목 타입이 없습니다.")
        return 1

    mode = "HARD DELETE"
    logger.info("대상 종목 타입: %s", ", ".join(valid_targets))
    logger.info("실행 모드: %s | execute=%s", mode, bool(args.execute))

    total_candidates = 0
    total_success = 0
    total_failed = 0

    for t_id in valid_targets:
        stocks = get_etfs(t_id) or []
        tickers = [str(item.get("ticker") or "").strip().upper() for item in stocks if item.get("ticker")]
        tickers = [t for t in tickers if t]
        total_candidates += len(tickers)

        logger.info("[%s] 활성 종목 %d개", t_id, len(tickers))
        if not tickers:
            continue

        if not args.execute:
            preview = ", ".join(tickers[:20])
            suffix = " ..." if len(tickers) > 20 else ""
            logger.info("[%s] DRY-RUN 삭제 대상: %s%s", t_id, preview, suffix)
            continue

        for ticker in tickers:
            ok = hard_remove_stock(t_id, ticker)
            if ok:
                total_success += 1
            else:
                total_failed += 1

        logger.info("[%s] 삭제 완료: 성공=%d, 실패=%d", t_id, total_success, total_failed)

    logger.info(
        "요약: 후보=%d, 성공=%d, 실패=%d, execute=%s",
        total_candidates,
        total_success,
        total_failed,
        bool(args.execute),
    )
    return 0 if (not args.execute or total_failed == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
