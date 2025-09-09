import glob
import os
import json

import pandas as pd
import streamlit as st

import settings as global_settings
from status import generate_action_plan


@st.cache_data(ttl=3600)
def get_cached_action_plan(strategy_name: str, country: str, portfolio_path: str):
    """캐시된 액션 플랜 데이터를 가져옵니다. 1시간 동안 캐시됩니다."""
    return generate_action_plan(
        strategy_name=strategy_name, country=country, portfolio_path=portfolio_path
    )


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


def display_strategy_plan(
    strategy_name: str, country: str, portfolio_path: str, date_str: str
):
    """지정된 전략, 국가, 포트폴리오 파일의 액션 플랜을 가져와 Streamlit UI에 표시합니다."""
    try:
        with open(portfolio_path, "r", encoding="utf-8") as f:
            portfolio_data = json.load(f)
        total_equity = portfolio_data.get("total_equity", 0.0)

        if not isinstance(total_equity, (int, float)) or total_equity <= 0:
            st.warning(
                f"평가금액(total_equity)이 0입니다. '{os.path.basename(portfolio_path)}' 파일을 열어 실제 총평가액을 입력해주세요."
            )
            st.info(
                f"`python {country}.py --convert` 실행 후 생성된 JSON 파일의 `total_equity` 값을 직접 수정해야 합니다."
            )
            return
    except (json.JSONDecodeError, FileNotFoundError):
        st.error(f"포트폴리오 파일 '{os.path.basename(portfolio_path)}'을(를) 읽는 데 실패했습니다.")
        return

    spinner_message = f"'{date_str}' 기준 전략 데이터를 분석하고 있습니다..."
    with st.spinner(spinner_message):
        result = get_cached_action_plan(
            strategy_name=strategy_name, country=country, portfolio_path=portfolio_path
        )

    if result:
        header_line, headers, rows = result

        st.info(header_line)

        df = pd.DataFrame(rows, columns=headers)
        if "#" in df.columns:
            df = df.set_index("#")

        style_cols = ["일간수익률", "누적수익률"]
        styler = df.style

        for col in style_cols:
            if col in df.columns:
                styler = styler.map(style_returns, subset=[col])

        num_rows_to_display = min(len(df), 20)
        height = (num_rows_to_display + 1) * 35 + 3  # 3px for border

        st.dataframe(
            styler,
            use_container_width=True,
            height=height,
            column_config={
                "문구": st.column_config.TextColumn(
                    "문구", width="large", help="매매 결정에 대한 상세 설명입니다."
                )
            },
        )

    else:
        st.error(
            f"'{date_str}' 기준 '{strategy_name}' ({country.upper()}) 전략의 액션 플랜을 생성하는 데 실패했습니다. 콘솔 로그를 확인해주세요."
        )


def main():
    """MomentumPilot 오늘의 현황 웹 UI를 렌더링합니다."""
    st.set_page_config(page_title="MomentumPilot Status", layout="wide")
    st.title("MomentumPilot - 김치네 화이팅")

    countries = ["kor", "aus"]
    country_options = [f"[{c.upper()}]" for c in countries]

    # 1. 국가 선택을 위한 탭 생성
    country_tabs = st.tabs(country_options)

    for i, country_code in enumerate(countries):
        with country_tabs[i]:
            # 2. 국가 코드에 맞는 전략 가져오기
            if country_code == "kor":
                strategy_name = global_settings.KOR_STRATEGY
            else:  # aus
                strategy_name = global_settings.AUS_STRATEGY

            # 3. 해당 국가의 모든 포트폴리오 파일 검색
            data_dir = f"data/{country_code}"
            portfolio_files = glob.glob(os.path.join(data_dir, "portfolio_*.json"))

            if not portfolio_files:
                st.warning(f"'{data_dir}'에서 포트폴리오 파일(portfolio_*.json)을 찾을 수 없습니다.")
                continue

            # 4. 파일명에서 날짜를 추출하고, 날짜를 기준으로 내림차순 정렬
            date_path_map = {}
            for f_path in portfolio_files:
                try:
                    fname = os.path.basename(f_path)
                    date_str = fname.replace("portfolio_", "").replace(".json", "")
                    pd.to_datetime(date_str)  # 날짜 형식 유효성 검사
                    date_path_map[date_str] = f_path
                except ValueError:
                    continue  # 유효하지 않은 날짜 형식의 파일은 건너뜀

            if not date_path_map:
                st.warning(f"'{data_dir}'에서 유효한 날짜 형식의 포트폴리오 파일을 찾을 수 없습니다.")
                continue

            sorted_dates = sorted(date_path_map.keys(), reverse=True)

            # 5. 날짜별 탭 생성
            date_tabs = st.tabs(sorted_dates)

            # 6. 각 탭에 해당 날짜의 데이터 표시
            for j, date_str in enumerate(sorted_dates):
                with date_tabs[j]:
                    portfolio_path = date_path_map[date_str]
                    display_strategy_plan(
                        strategy_name=strategy_name,
                        country=country_code,
                        portfolio_path=portfolio_path,
                        date_str=date_str,
                    )


if __name__ == "__main__":
    main()
