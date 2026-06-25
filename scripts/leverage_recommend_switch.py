"""leverage 스위칭 전략 추천 배치 (무인자 래퍼).

momentum-etf 의 server_scheduler/run_batch 는 `python <script.py>` 형태(무인자)로만
스크립트를 실행하므로, leverage.recommend 를 switch + Slack 으로 호출하는 얇은 래퍼다.
"""

import os
import sys

# 스크립트로 직접 실행될 때 프로젝트 루트를 import 경로에 추가 (leverage 패키지 import용)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leverage.recommend import main  # noqa: E402

if __name__ == "__main__":
    # leverage.recommend.main() 은 argparse 로 sys.argv 를 읽는다.
    sys.argv = ["leverage.recommend", "switch", "--slack"]
    main()
