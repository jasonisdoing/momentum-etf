"""'
'kor' 국가, 'm1' 계좌에 대한 백테스트를 실행하는 스크립트입니다.

이 스크립트는 cli.py를 통하지 않고 직접 test.py를 호출하여
한국(KOR) 포트폴리오의 백테스트를 간편하게 실행할 수 있도록 합니다.

[사용법]
python scripts/test_kor_m1.py

[선택적 인자]
--tickers: 테스트에 사용할 티커 목록 (쉼표로 구분, 예: 252670,379800).
           지정하지 않으면 data/kor/etf.json의 모든 종목을 사용합니다.

결과는 콘솔에 출력되고 `logs/test_kor_m1.log` 파일에도 저장됩니다.
"""

import argparse
import os
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test import main as run_test


class Tee:
    """STDOUT과 파일에 동시에 쓰기 위한 헬퍼 클래스입니다."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def main():
    """'kor' 국가, 'm1' 계좌에 대한 백테스트를 실행합니다."""
    # 로그 파일 설정
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "test_kor_m1.log")

    original_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        try:
            parser = argparse.ArgumentParser(
                description="Run a backtest for the 'kor' portfolio on account 'm1'."
            )
            parser.add_argument(
                "--tickers",
                type=str,
                default=None,
                help="테스트에 사용할 티커 리스트 (쉼표구분, 예: 252670,379800). 미지정 시 DB 목록 사용",
            )
            args = parser.parse_args()

            country = "kor"
            account = "m1"
            override_settings = {}

            if args.tickers:
                tickers_override = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
                if tickers_override:
                    override_settings["tickers_override"] = tickers_override
                    print(f"지정된 티커로 테스트를 실행합니다: {', '.join(tickers_override)}")

            print(f"\n'{country.upper()}' 국가, '{account}' 계좌에 대한 백테스트를 시작합니다...")
            run_test(
                country=country,
                quiet=False,
                account=account,
                override_settings=override_settings or None,
            )
            print("\n백테스트가 성공적으로 완료되었습니다.")
        except Exception as e:
            print(f"\n백테스트 실행 중 오류가 발생했습니다: {e}")
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
