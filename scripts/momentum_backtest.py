"""모멘텀 백테스트 배치 (무인자 래퍼).

워커(server_scheduler)는 `python <script.py>` 무인자 형식만 실행하므로, `backtest/run.py`
를 인자 없이(= 설정된 모든 풀 순차 실행) 호출하는 얇은 래퍼다. 결과는
`backtest/results/<prefix>-backtest_<날짜>.log` 에 풀별로 기록된다.
"""

import os
import sys

# 스크립트로 직접 실행될 때 프로젝트 루트를 import 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.run import main  # noqa: E402

if __name__ == "__main__":
    # 인자 없이 호출 → BACKTEST 설정이 있는 모든 풀을 순차 실행
    sys.exit(main(["backtest.run"]))
