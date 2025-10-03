"""
MomentumEtf 프로젝트의 웹 애플리케이션 실행 파일입니다.

이 스크립트는 `streamlit run web_app.py`와 동일하게 동작하여
웹 브라우저에서 프로젝트의 메인 대시보드를 실행합니다.

[사용법]
python run.py 또는 streamlit run Main.py

추천/백테스트/튜닝 등의 CLI 작업은 각 전용 스크립트를 사용하세요.
(예: python recommend.py kor, python backtest.py kor, python tune.py kor)
"""

import os
import sys
import warnings

# pkg_resources 워닝 억제
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

# 프로젝트 루트를 Python 경로에 추가합니다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """
    Streamlit 웹 애플리케이션을 실행합니다.
    `streamlit run Main.py`와 동일하게 동작합니다.
    """
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("오류: Streamlit이 설치되어 있지 않습니다. 'pip install streamlit'으로 설치해주세요.")
        sys.exit(1)

    # app.py 파일의 절대 경로를 찾습니다.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.join(dir_path, "app.py")

    # Streamlit을 실행하기 위한 인자를 구성합니다.
    # sys.argv를 수정하여 `streamlit run app.py` 처럼 보이게 합니다.
    # 추가적인 Streamlit 인자(예: --server.port)를 전달할 수 있도록 합니다.
    args = ["run", script_path] + sys.argv[1:]
    sys.argv = ["streamlit"] + args

    # Streamlit의 메인 CLI 함수를 호출합니다.
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
