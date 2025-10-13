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

from utils.logger import get_app_logger

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
    logger = get_app_logger()
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        logger.error("Streamlit이 설치되어 있지 않습니다. 'pip install streamlit'으로 설치해주세요.")
        sys.exit(1)

    # app.py 파일의 절대 경로를 찾습니다.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.join(dir_path, "app.py")

    # Streamlit을 실행하기 위한 인자를 구성합니다.
    # sys.argv를 수정하여 `streamlit run app.py` 처럼 보이게 합니다.
    # 추가적인 Streamlit 인자(예: --server.port)를 전달할 수 있도록 합니다.
    args = ["run", script_path] + sys.argv[1:]
    sys.argv = ["streamlit"] + args

    # Streamlit 페이지 설정 및 Open Graph 메타 태그 추가
    import streamlit as st

    st.set_page_config(page_title="ETF 모멘텀 자동분석", page_icon="📈")
    st.markdown(
        """
    <head>
      <meta property="og:title" content="ETF 모멘텀 자동분석" />
      <meta property="og:description" content="ETF 자동 분석 플랫폼 – 최신 데이터를 제공합니다." />
      <meta property="og:image" content="https://etf.dojason.com/thumbnail.png" />
      <meta property="og:url" content="https://etf.dojason.com/" />
      <meta property="og:type" content="website" />
      <meta name="twitter:card" content="summary_large_image" />
    </head>
    """,
        unsafe_allow_html=True,
    )

    # Streamlit의 메인 CLI 함수를 호출합니다.
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
