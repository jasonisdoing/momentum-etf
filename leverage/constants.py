"""leverage 전략 패키지 자체 상수 (momentum-etf config 와 분리해 자기완결적으로 둔다)."""

from datetime import time
from pathlib import Path

# leverage 패키지 루트 (zresults 등 상대경로 기준점). 설정·상태는 DB 가 단일 소스.
LEVERAGE_DIR = Path(__file__).resolve().parent
ZRESULTS_DIR = LEVERAGE_DIR / "zresults"

# 백테스트 시뮬레이션 시작 기준일 (start_date 미지정 시 사용하지 않음; 참고용)
SIMULATION_START_DATE = "2020-01-01"

# 전략 초기 자본 (원화)
INITIAL_CAPITAL_KRW = 10_000_000

# 시장별 장 운영 시간 (장중/장마감 판정용)
MARKET_SCHEDULES = {
    "kor": {
        "open": time(9, 0),
        "close": time(15, 30),
        "timezone": "Asia/Seoul",
    },
    "us": {
        "open": time(9, 30),
        "close": time(16, 0),
        "timezone": "America/New_York",
    },
}
