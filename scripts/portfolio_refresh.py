#!/usr/bin/env python
"""포트폴리오 구성종목 가격 캐시 수동 갱신 스크립트.

모든 종목풀의 ETF에 대해 포트폴리오 변동 캐시를 갱신한다.
기본 화면 조회는 TTL 기반 실시간 갱신 경로를 사용한다.
"""

from __future__ import annotations

import logging
import os
import sys

# 프로젝트 루트 등록
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.stock_price_cache_updater import refresh_portfolio_change_for_all_targets
from utils.env import load_env_if_present

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    load_env_if_present()
    logger.info("[portfolio_refresh] 포트폴리오 구성종목 가격 갱신 시작")
    refresh_portfolio_change_for_all_targets()
    logger.info("[portfolio_refresh] 완료")


if __name__ == "__main__":
    main()
