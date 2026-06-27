"""leverage 스위칭 전략 튜닝 배치 (무인자 래퍼).

워커(server_scheduler)는 `python <script.py>` 형태(무인자)로만 스크립트를 실행하므로,
leverage.tune 를 switch 로 호출하는 얇은 래퍼다. 결과는 DB(leverage_config)에 반영되고
진행도/결과는 leverage/zresults/switch/tune_<날짜>.log 에 기록된다.
"""

import os
import sys

# 스크립트로 직접 실행될 때 프로젝트 루트를 import 경로에 추가 (leverage 패키지 import용)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leverage.tune import main  # noqa: E402

if __name__ == "__main__":
    # leverage.tune.main() 은 argparse 로 sys.argv 를 읽는다(--auto 없이 즉시 실행).
    sys.argv = ["leverage.tune", "switch"]
    main()
