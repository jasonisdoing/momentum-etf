import pandas as pd
import streamlit as st

from today import generate_action_plan


def style_returns(val: str) -> str:
    """수익률 문자열에 대해 양수는 빨간색, 음수는 파란색으로 스타일을 적용합니다."""
    color = ""
    if isinstance(val, str):
        try:
            # '+1.5%', '-10.0%' 등의 문자열에서 숫자 값을 추출합니다.
            clean_val = val.replace("%", "").replace(",", "").replace("+", "")
            num_val = float(clean_val)
            if num_val > 0:
                color = "red"
            elif num_val < 0:
                color = "blue"
        except (ValueError, TypeError):
            # 숫자 변환에 실패하면 스타일을 적용하지 않습니다.
            pass
    return f"color: {color}"


def display_strategy_plan(strategy_name: str):
    """지정된 전략의 액션 플랜을 가져와 Streamlit UI에 표시합니다."""
    with st.spinner(f"'{strategy_name}' 전략 데이터를 분석하고 있습니다..."):
        result = generate_action_plan(strategy_name=strategy_name)

    if result:
        header_line, headers, rows = result

        st.info(header_line)

        # DataFrame 생성
        df = pd.DataFrame(rows, columns=headers)
        # '#' 열을 인덱스로 사용합니다.
        if "#" in df.columns:
            df = df.set_index("#")

        # 스타일을 적용할 열 목록
        style_cols = ["일간수익률", "누적수익률"]

        # 데이터프레임 스타일러 객체 생성
        styler = df.style

        # 지정된 열에 스타일 함수를 적용합니다.
        for col in style_cols:
            if col in df.columns:
                styler = styler.map(style_returns, subset=[col])

        # 스타일이 적용된 데이터프레임을 출력합니다.
        st.dataframe(styler, use_container_width=True)

    else:
        st.error(
            f"'{strategy_name}' 전략의 액션 플랜을 생성하는 데 실패했습니다. 콘솔 로그를 확인해주세요."
        )


def main():
    """MomentumPilot 오늘의 액션 플랜 웹 UI를 렌더링합니다."""
    st.set_page_config(page_title="MomentumPilot Today", layout="wide")
    st.title("MomentumPilot - 오늘의 액션 플랜")

    # 각 전략을 탭으로 표시합니다.
    strategies = ["jason", "seykota", "donchian"]
    tabs = st.tabs(strategies)

    for tab, strategy_name in zip(tabs, strategies):
        with tab:
            display_strategy_plan(strategy_name)


if __name__ == "__main__":
    main()
