import os
import sys
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# .env 파일이 있다면 로드합니다.
load_dotenv()
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings(
    "ignore",
    message=r"\['break_start', 'break_end'\] are discontinued",
    category=UserWarning,
    module=r"^pandas_market_calendars\.",
)


# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import settings as global_settings
from status import (
    _maybe_notify_detailed_status,
    generate_status_report,
    get_benchmark_status_string,
    get_market_regime_status_string,
)
from utils.data_loader import (
    fetch_ohlcv_for_tickers,
    fetch_yfinance_name,
    get_trading_days,
)
from utils.db_manager import (
    delete_trade_by_id,
    get_all_daily_equities,
    get_all_trades,
    get_app_settings,
    get_available_snapshot_dates,
    get_common_settings,
    get_db_connection,
    get_portfolio_snapshot,
    get_status_report_from_db,
    save_app_settings,
    save_common_settings,
    save_daily_equity,
    save_status_report_to_db,
    save_trade,
)
from utils.stock_list_io import get_etfs

try:
    from pykrx import stock as _stock
except Exception:
    _stock = None

try:
    import pytz
except ImportError:
    pytz = None

try:
    from croniter import croniter
except ImportError:
    croniter = None

try:
    from cron_descriptor import get_description as get_cron_description
except ImportError:
    get_cron_description = None


COUNTRY_CODE_MAP = {"kor": "한국", "aus": "호주", "coin": "가상화폐"}


# --- Functions ---


def render_cron_input(label, key, default_value, country_code: str):
    """Crontab 입력을 위한 UI와 실시간 유효성 검사를 렌더링합니다."""
    col1, col2 = st.columns([2, 3])
    with col1:
        st.text_input(
            label,
            value=default_value,
            key=key,
            help="Crontab 형식 입력 (예: '0 * * * *'는 매시간 실행)",
        )
    with col2:
        # 폼이 렌더링될 때 st.session_state에서 현재 입력된 값을 가져와 유효성을 검사합니다.
        current_val = st.session_state.get(key, default_value)
        if croniter and current_val:
            try:
                if not croniter.is_valid(current_val):
                    st.warning("❌ 잘못된 Crontab 형식입니다.")
                else:
                    display_text = "✅ 유효"
                    if get_cron_description:
                        try:
                            desc_ko = ""
                            try:
                                # 최신 API (cron-descriptor >= 1.2.16)
                                desc_ko = get_cron_description(
                                    current_val,
                                    locale="ko_KR",
                                    use_24hour_time_format=True,  # type: ignore
                                )
                            except TypeError:
                                # 구버전 API 폴백 (cron-descriptor < 1.2.16)
                                from cron_descriptor import (
                                    ExpressionDescriptor,
                                    Options,
                                )

                                options = Options()
                                options.use_24hour_time_format = True
                                options.locale_code = "ko_KR"
                                desc_ko = ExpressionDescriptor(
                                    current_val, options
                                ).get_description()

                            if desc_ko:
                                display_text = f"✅ 유효. {desc_ko}"
                        except Exception as e:
                            # 설명 생성 실패 시, 콘솔에 오류를 기록하고 기본 문구만 표시합니다.
                            print(f"경고: Crontab 설명 생성 중 오류: {e}")

                    # 수직 정렬을 위해 div와 패딩을 사용합니다.
                    st.markdown(
                        f"<div style='padding-top: 32px;'><span style='color:green;'>{display_text}</span></div>",
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"오류: {e}")


def _format_korean_datetime(dt: datetime) -> str:
    """날짜-시간 객체를 'YYYY년 MM월 DD일(요일) 오전/오후 HH시 MM분' 형식으로 변환합니다."""
    weekday_map = ["월", "화", "수", "목", "금", "토", "일"]
    weekday_str = weekday_map[dt.weekday()]

    hour12 = dt.hour
    if hour12 >= 12:
        ampm_str = "오후"
        if hour12 > 12:
            hour12 -= 12
    else:
        ampm_str = "오전"
    if hour12 == 0:
        hour12 = 12

    return f"{dt.strftime('%Y년 %m월 %d일')}({weekday_str}) {ampm_str} {hour12}시 {dt.minute:02d}분"


def _is_running_in_streamlit():
    """
    Streamlit 실행 환경인지 확인합니다.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        # 어떤 경우에도 실패하면 False를 반환합니다.
        return False


if _is_running_in_streamlit():

    @st.cache_data(ttl=600)
    def get_cached_benchmark_status(country: str) -> Optional[str]:
        """벤치마크 비교 문자열을 캐시하여 반환합니다. (Streamlit용)"""
        return get_benchmark_status_string(country)

else:

    def get_cached_benchmark_status(country: str) -> Optional[str]:
        """벤치마크 비교 문자열을 반환합니다. (CLI용, 캐시 없음)"""
        return get_benchmark_status_string(country)


def get_next_schedule_time_str(country_code: str) -> str:
    """지정된 국가의 다음 스케줄 실행 시간을 문자열로 반환합니다."""
    if not croniter or not pytz:
        return "스케줄러 라이브러리가 설치되지 않았습니다."

    common_settings = get_common_settings() or {}
    cron_key = f"SCHEDULE_CRON_{country_code.upper()}"

    default_cron = {
        "kor": "10 18 * * 1-5",
        "aus": "10 18 * * 1-5",
        "coin": "5 0 * * *",
    }.get(country_code, "0 * * * *")

    cron_value = common_settings.get(cron_key, default_cron)

    # scheduler.py의 로직과 일관성을 맞추기 위해 타임존을 설정합니다.
    # 참고: scheduler.py에서는 호주(aus) 스케줄에 'Asia/Seoul'을 사용하고 있습니다.
    tz_str_map = {
        "kor": "Asia/Seoul",
        "aus": "Asia/Seoul",
        "coin": "Asia/Seoul",
    }
    tz_str = tz_str_map.get(country_code, "Asia/Seoul")

    try:
        local_tz = pytz.timezone(tz_str)
        now_local = datetime.now(local_tz)

        if not croniter.is_valid(cron_value):
            return "설정된 스케줄(Crontab)이 올바르지 않습니다."

        cron = croniter(cron_value, now_local)
        next_run_time = cron.get_next(datetime)

        return _format_korean_datetime(next_run_time)

    except Exception as e:
        return f"다음 실행 시간 계산 중 오류 발생: {e}"


def get_cached_status_report(
    country: str,
    date_str: str,
    force_recalculate: bool = False,
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
):
    """
    MongoDB를 사용하여 현황 데이터를 캐시합니다.
    force_recalculate=True일 경우에만 다시 계산합니다.
    """
    try:
        report_date = pd.to_datetime(date_str).to_pydatetime()
    except (ValueError, TypeError):
        st.error(f"잘못된 날짜 형식입니다: {date_str}")
        return None

    if not force_recalculate:
        # 1. DB에서 먼저 찾아봅니다.
        report_from_db = get_status_report_from_db(country, report_date)
        if report_from_db:
            # DB에 저장된 형식은 딕셔너리, 반환 형식은 튜플이어야 합니다.
            return (
                report_from_db.get("header_line"),
                report_from_db.get("headers"),
                report_from_db.get("rows"),
            )
        else:
            # DB에 없으면 계산하지 않고 None을 반환합니다.
            return None

    # 2. 강제로 다시 계산해야 하는 경우
    try:
        with st.spinner(f"'{date_str}' 현황을 다시 계산하는 중..."):
            new_report_tuple = generate_status_report(
                country=country,
                date_str=date_str,
                prefetched_data=prefetched_data,
                notify_start=False,
            )
            if new_report_tuple:
                header_line, headers, rows, _ = new_report_tuple
                new_report = (header_line, headers, rows)
                # 3. 계산된 결과를 DB에 저장합니다.
                save_status_report_to_db(country, report_date, new_report)
            return new_report
    except ValueError as e:
        if str(e).startswith("PRICE_FETCH_FAILED:"):
            failed_tickers_str = str(e).split(":", 1)[1]
            st.error(f"{failed_tickers_str} 종목의 가격을 가져올 수 없습니다. 다시 시도하세요.")
            return None
        else:
            # 다른 ValueError는 기존처럼 처리
            print(f"오류: 현황 계산 오류: {country}/{date_str}: {e}")
            st.error(f"'{date_str}' 현황 계산 중 오류가 발생했습니다: {e}")
            return None
    except Exception as e:
        print(f"오류: 현황 계산 오류: {country}/{date_str}: {e}")
        st.error(
            f"'{date_str}' 현황 계산 중 오류가 발생했습니다. 자세한 내용은 콘솔 로그를 확인해주세요."
        )
        return None


def style_returns(val) -> str:
    """수익률 값(숫자)에 대해 양수는 빨간색, 음수는 파란색으로 스타일을 적용합니다."""
    color = ""
    if isinstance(val, (int, float)):
        if val > 0:
            color = "red"
        elif val < 0:
            color = "blue"
    return f"color: {color}"


def _display_status_report_df(df: pd.DataFrame, country_code: str):
    """
    현황 리포트 DataFrame에 종목 메타데이터(이름, 카테고리)를 실시간으로 병합하고 스타일을 적용하여 표시합니다.
    """
    # 1. 종목 메타데이터 로드
    etfs_data = get_etfs(country_code)
    if not etfs_data:
        meta_df = pd.DataFrame(columns=["ticker", "이름", "category"])
    else:
        meta_df = pd.DataFrame(etfs_data)
        required_cols = ["ticker", "name", "category"]
        for col in required_cols:
            if col not in meta_df.columns:
                meta_df[col] = None
        meta_df = meta_df[required_cols]
        meta_df.rename(columns={"name": "이름"}, inplace=True)

    # 호주 'IS' 종목의 이름을 수동으로 지정합니다.
    if country_code == "aus":
        # 'IS' 종목에 대한 메타데이터를 수동으로 추가합니다.
        is_meta = pd.DataFrame(
            [{"ticker": "IS", "이름": "International Shares", "category": "기타"}]
        )
        meta_df = pd.concat([meta_df, is_meta], ignore_index=True)

    # 2. 메타데이터 병합
    df_merged = pd.merge(df, meta_df, left_on="티커", right_on="ticker", how="left")

    # 메타데이터 병합 후 컬럼이 없을 경우를 대비
    if "이름" in df_merged.columns:
        df_merged["이름"] = df_merged["이름"].fillna(df_merged["티커"])
    else:
        df_merged["이름"] = df_merged["티커"]
    if "category" not in df_merged.columns:
        df_merged["category"] = ""  # 카테고리 정보가 없을 경우 빈 문자열로 채움

    # 3. 컬럼 순서 재정렬
    final_cols = [
        "#",
        "티커",
        "이름",
        "category",
        "상태",
        "매수일자",
        "보유일",
        "현재가",
        "일간수익률",
        "보유수량",
        "금액",
        "누적수익률",
        "비중",
        "고점대비",
        "점수",
        "지속",
        "문구",
    ]

    existing_cols = [col for col in final_cols if col in df_merged.columns]
    df_display = df_merged[existing_cols].copy()

    # 숫자형으로 변환해야 하는 컬럼 목록
    numeric_cols = ["현재가", "일간수익률", "보유수량", "금액", "누적수익률", "비중", "점수"]
    for col in numeric_cols:
        if col in df_display.columns:
            df_display[col] = pd.to_numeric(df_display[col], errors="coerce")

    # 4. 스타일 적용 및 표시
    if "#" in df_display.columns:
        df_display = df_display.set_index("#")

    style_cols = ["일간수익률", "누적수익률"]
    styler = df_display.style
    for col in style_cols:
        if col in df_display.columns:
            styler = styler.map(style_returns, subset=[col])

    formats = {
        "일간수익률": "{:+.2f}%",
        "누적수익률": "{:+.2f}%",
        "비중": "{:.1f}%",
        "점수": "{:.2f}",
    }

    # 국가별로 통화 형식 지정
    if country_code in ["kor", "coin"]:
        formats["현재가"] = "{:,.0f}"
        formats["금액"] = "{:,.0f}"
    if country_code in ["aus"]:
        formats["현재가"] = "{:,.2f}"
        formats["금액"] = "{:,.2f}"

    # 코인은 보유수량을 소수점 8자리로 표시
    if country_code == "coin" and "보유수량" in df_display.columns:
        formats["보유수량"] = "{:.8f}"
    styler = styler.format(formats, na_rep="-")

    num_rows_to_display = min(len(df_display), 15)
    height = (num_rows_to_display + 1) * 35 + 3

    shares_format_str = "%.8f" if country_code == "coin" else "%d"

    st.dataframe(
        styler,
        width="stretch",
        height=height,
        column_config={
            "이름": st.column_config.TextColumn("종목명", width=200),
            "category": st.column_config.TextColumn("카테고리", width=100),
            "상태": st.column_config.TextColumn(width="small"),
            "매수일자": st.column_config.TextColumn(width="small"),
            "보유": st.column_config.TextColumn(width=40),
            "보유수량": st.column_config.NumberColumn(format=shares_format_str),
            "일간수익률": st.column_config.TextColumn(width="small"),
            "금액": st.column_config.TextColumn(width="small"),
            "누적수익률": st.column_config.TextColumn(width="small"),
            "비중": st.column_config.TextColumn(width=50),
            "지속": st.column_config.TextColumn(width=50),
            "문구": st.column_config.TextColumn("문구", width="large"),
        },
    )


def render_master_etf_ui(country_code: str):
    """종목 마스터 조회 UI를 렌더링합니다."""
    # from utils.data_loader import fetch_crypto_name # 사용자 요청에 따라 제거됨

    country_name = COUNTRY_CODE_MAP.get(country_code, "기타")

    if country_code == "coin":
        st.info("이곳에서 가상화폐 종목을 조회할 수 있습니다.")
    else:
        st.info("이곳에서 투자 유니버스에 포함된 종목을 조회할 수 있습니다.")

    with st.spinner("종목 마스터 데이터를 불러오는 중..."):
        etfs_data = get_etfs(country_code)
        if not etfs_data:
            st.info("조회할 종목이 없습니다.")
            return

        df_etfs = pd.DataFrame(etfs_data)

        # 데이터 정합성을 위한 처리: 'name' 컬럼이 없거나 NaN 값이 있으면 오류가 발생할 수 있습니다.
        if "name" not in df_etfs.columns:
            df_etfs["name"] = ""
        df_etfs["name"] = df_etfs["name"].fillna("").astype(str)

        # 데이터 정합성을 위한 처리
        if country_code == "coin":
            if "type" not in df_etfs.columns:
                df_etfs["type"] = "crypto"
            df_etfs["type"] = df_etfs["type"].fillna("crypto")
        else:
            if "type" not in df_etfs.columns:
                df_etfs["type"] = ""

        # 'last_modified' 컬럼이 없는 구버전 데이터와의 호환성을 위해 추가
        if "last_modified" not in df_etfs.columns:
            df_etfs["last_modified"] = pd.NaT

        # 정렬 로직: 오래된 수정일자 우선
        df_etfs["modified_sort_key"] = pd.to_datetime(df_etfs["last_modified"], errors="coerce")

        df_etfs.sort_values(
            by=["modified_sort_key"],
            ascending=True,
            na_position="first",  # 수정일자가 없는 가장 오래된 데이터부터 표시
            inplace=True,
        )

        # 컬럼 순서 조정
        display_cols = ["ticker", "name", "category"]
        df_for_display = df_etfs.reindex(columns=display_cols)

        st.dataframe(
            df_for_display,
            width="stretch",
            hide_index=True,
            key=f"etf_viewer_{country_code}",
            column_config={
                "ticker": st.column_config.TextColumn("티커"),
                "name": st.column_config.TextColumn("종목명"),
            },
        )


def render_notification_settings_ui(country_code: str):
    """지정된 국가에 대한 알림 설정 UI를 렌더링합니다."""
    st.header(f"{COUNTRY_CODE_MAP.get(country_code, country_code.upper())} 국가 알림 설정")

    # .env 파일에서 웹훅 URL을 가져옵니다.
    from utils.notify import get_slack_webhook_url

    app_settings = get_app_settings(country_code) or {}

    with st.form(f"notification_settings_form_{country_code}"):
        st.subheader("슬랙 설정")
        slack_enabled = bool(app_settings.get("SLACK_ENABLED", False))
        slack_webhook_url = app_settings.get("SLACK_WEBHOOK_URL", "")

        new_slack_enabled = st.checkbox(
            "슬랙 알림 사용",
            value=slack_enabled,
            key=f"slack_enabled_{country_code}",
            help="이 국가의 현황 메시지를 슬랙으로 전송합니다.",
        )

        webhook_url_from_env = get_slack_webhook_url(country_code)
        if webhook_url_from_env:
            st.text_input(
                "웹훅 URL (.env)",
                value=webhook_url_from_env,
                disabled=True,
                help=f"{country_code.upper()}_SLACK_WEBHOOK 환경 변수에서 가져온 값입니다.",
            )
        else:
            st.warning(
                f"`.env` 파일에 `{country_code.upper()}_SLACK_WEBHOOK` 환경 변수를 설정해주세요."
            )

        st.caption("테스트는 스케줄과 무관하게 1회 계산 후 알림을 전송합니다.")

        cols = st.columns(2)
        with cols[0]:
            settings_save = st.form_submit_button("설정 저장")
        with cols[1]:
            test_send = st.form_submit_button("알림 테스트 전송")

    if settings_save:
        error = False
        # 웹훅 URL은 더 이상 DB에 저장하지 않고, '사용' 여부만 저장합니다.
        slack_settings_to_save = {"SLACK_ENABLED": new_slack_enabled}

        if not error:
            save_app_settings(country_code, slack_settings_to_save)
            st.success(f"{country_code.upper()} 국가의 슬랙 설정을 저장했습니다.")
            st.rerun()

    if test_send:
        # 테스트 전송은 현재 UI의 '사용' 여부만 저장하고 실행합니다.
        save_app_settings(country_code, {"SLACK_ENABLED": new_slack_enabled})
        if not webhook_url_from_env:
            st.error("테스트를 보내려면 .env 파일에 웹훅 URL을 먼저 설정해야 합니다.")
        else:
            result_tuple = generate_status_report(
                country=country_code, date_str=None, notify_start=True
            )
            if not result_tuple:
                st.error("현황 계산 실패로 테스트 전송을 건너뜁니다.")
            else:
                header_line, headers, rows_sorted, _ = result_tuple
                sent = _maybe_notify_detailed_status(
                    country_code, header_line, headers, rows_sorted, force=True
                )
                if sent:
                    st.success("알림 테스트 전송 완료. 슬랙 채널을 확인하세요.")
                else:
                    from utils.notify import get_last_error

                    err = get_last_error()
                    st.warning(f"전송 시도는 했지만 응답이 없었습니다. 상세: {err or '설정 확인'}")


def render_scheduler_tab():
    """스케줄러 설정을 위한 UI를 렌더링합니다."""
    st.header("스케줄러 설정 (모든 국가)")
    st.info("각 국가별 현황 계산 작업이 실행될 주기를 Crontab 형식으로 설정합니다.")

    common_settings = get_common_settings() or {}

    with st.form("scheduler_settings_form"):
        st.subheader("한국 (KOR)")
        kor_cron_key = "SCHEDULE_CRON_KOR"
        kor_default_cron = "10 18 * * 1-5"
        kor_cron_value = common_settings.get(kor_cron_key, kor_default_cron)
        render_cron_input("실행 주기", "cron_input_kor_scheduler", kor_cron_value, "kor")

        st.subheader("호주 (AUS)")
        aus_cron_key = "SCHEDULE_CRON_AUS"
        aus_default_cron = "10 18 * * 1-5"
        aus_cron_value = common_settings.get(aus_cron_key, aus_default_cron)
        render_cron_input("실행 주기", "cron_input_aus_scheduler", aus_cron_value, "aus")

        st.subheader("가상화폐 (COIN)")
        coin_cron_key = "SCHEDULE_CRON_COIN"
        coin_default_cron = "5 0 * * *"
        coin_cron_value = common_settings.get(coin_cron_key, coin_default_cron)
        render_cron_input("실행 주기", "cron_input_coin_scheduler", coin_cron_value, "coin")

        submitted = st.form_submit_button("스케줄러 설정 저장")

        if submitted:
            error = False
            cron_settings_to_save = {}

            # KOR
            new_kor_cron = st.session_state["cron_input_kor_scheduler"]
            if croniter and not croniter.is_valid(new_kor_cron):
                st.error("한국(KOR)의 Crontab 형식이 올바르지 않습니다.")
                error = True
            else:
                cron_settings_to_save[kor_cron_key] = new_kor_cron.strip()

            # AUS
            new_aus_cron = st.session_state["cron_input_aus_scheduler"]
            if croniter and not croniter.is_valid(new_aus_cron):
                st.error("호주(AUS)의 Crontab 형식이 올바르지 않습니다.")
                error = True
            else:
                cron_settings_to_save[aus_cron_key] = new_aus_cron.strip()

            # COIN
            new_coin_cron = st.session_state["cron_input_coin_scheduler"]
            if croniter and not croniter.is_valid(new_coin_cron):
                st.error("가상화폐(COIN)의 Crontab 형식이 올바르지 않습니다.")
                error = True
            else:
                cron_settings_to_save[coin_cron_key] = new_coin_cron.strip()

            if not error:
                if save_common_settings(cron_settings_to_save):
                    st.success("스케줄러 설정을 성공적으로 저장했습니다.")
                    st.rerun()
                else:
                    st.error("스케줄러 설정 저장에 실패했습니다.")


def _display_success_toast(country_code: str):
    """
    세션 상태에서 성공 메시지를 확인하고 토스트로 표시합니다.
    주로 다이얼로그가 닫힌 후 피드백을 주기 위해 사용됩니다.
    """
    keys_to_check = [
        f"buy_message_{country_code}",
        f"sell_message_{country_code}",
    ]
    for key in keys_to_check:
        if key in st.session_state:
            message = st.session_state[key]
            # 메시지가 (type, text) 튜플이고, type이 'success'인 경우에만 처리
            if isinstance(message, tuple) and len(message) == 2 and message[0] == "success":
                _, msg_text = st.session_state.pop(key)
                st.toast(msg_text)


def render_country_tab(country_code: str):
    """지정된 국가에 대한 탭의 전체 UI를 렌더링합니다."""

    @st.dialog("BUY")
    def show_buy_dialog(country_code_inner: str):
        """매수(BUY) 거래 입력을 위한 모달 다이얼로그를 표시합니다."""

        currency_str = f" ({'AUD' if country_code_inner == 'aus' else 'KRW'})"
        message_key = f"buy_message_{country_code_inner}"

        def on_buy_submit():
            # 한국 현지 시간으로 현재 시간을 가져옵니다.
            trade_time = datetime.now()
            if pytz:
                try:
                    korea_tz = pytz.timezone("Asia/Seoul")
                    # DB에 시간대 정보가 없는 순수한 한국 시간(naive)으로 저장해달라는 요청에 따라
                    # aware datetime에서 timezone 정보를 제거합니다.
                    trade_time = datetime.now(korea_tz).replace(tzinfo=None)
                except pytz.UnknownTimeZoneError:
                    # pytz가 설치되었지만 'Asia/Seoul'을 모르는 경우에 대한 폴백
                    pass

            # st.session_state에서 폼 데이터 가져오기
            ticker = st.session_state[f"buy_ticker_{country_code_inner}"].strip()
            shares = st.session_state[f"buy_shares_{country_code_inner}"]
            price = st.session_state[f"buy_price_{country_code_inner}"]

            if not ticker or not shares > 0 or not price > 0:
                st.session_state[message_key] = (
                    "error",
                    "종목코드, 수량, 가격을 모두 올바르게 입력해주세요.",
                )
                return

            etf_name = ""
            if country_code_inner == "kor" and _stock:
                # fetch_pykrx_name은 ETF와 ETN/주식 이름을 모두 조회 시도합니다.
                from utils.data_loader import fetch_pykrx_name

                etf_name = fetch_pykrx_name(ticker)
            elif country_code_inner == "aus":
                etf_name = fetch_yfinance_name(ticker)

            trade_data = {
                "country": country_code_inner,
                "date": trade_time,
                "ticker": ticker.upper(),
                "name": etf_name,
                "action": "BUY",
                "shares": float(shares),
                "price": float(price),
                "note": "Manual input from web app",
            }

            if save_trade(trade_data):
                st.session_state[message_key] = ("success", "거래가 성공적으로 저장되었습니다.")
            else:
                st.session_state[message_key] = (
                    "error",
                    "거래 저장에 실패했습니다. 콘솔 로그를 확인해주세요.",
                )

        # 다이얼로그 내에서 오류 메시지만 표시합니다. 성공 메시지는 메인 화면에서 토스트로 표시됩니다.
        if message_key in st.session_state:
            msg_type, msg_text = st.session_state[message_key]
            if msg_type == "success":
                st.error(msg_text)
                # 오류 메시지는 한 번만 표시되도록 세션에서 제거합니다.
                del st.session_state[message_key]

        with st.form(f"trade_form_{country_code_inner}"):
            st.text_input("종목코드 (티커)", key=f"buy_ticker_{country_code_inner}")
            shares_format_str = "%.8f" if country_code_inner == "coin" else "%d"
            st.number_input(
                "수량",
                min_value=0.00000001,
                step=0.00000001,
                format=shares_format_str,
                key=f"buy_shares_{country_code_inner}",
            )
            st.number_input(
                f"매수 단가{currency_str}",
                min_value=0.0,
                format=(
                    "%.4f"
                    if country_code_inner == "aus"
                    else ("%d" if country_code_inner in ["kor", "coin"] else "%d")
                ),
                key=f"buy_price_{country_code_inner}",
            )
            st.form_submit_button("거래 저장", on_click=on_buy_submit)

    @st.dialog("SELL", width="large")
    def show_sell_dialog(country_code_inner: str):
        """보유 종목 매도를 위한 모달 다이얼로그를 표시합니다."""
        currency_str = f" ({'AUD' if country_code_inner == 'aus' else 'KRW'})"
        message_key = f"sell_message_{country_code_inner}"

        from utils.data_loader import fetch_naver_realtime_price, fetch_ohlcv

        latest_date_str = (
            get_available_snapshot_dates(country_code_inner)[0]
            if get_available_snapshot_dates(country_code_inner)
            else None
        )
        if not latest_date_str:
            st.warning("보유 종목이 없어 매도할 수 없습니다.")
            return

        snapshot = get_portfolio_snapshot(country_code_inner, date_str=latest_date_str)
        if not snapshot or not snapshot.get("holdings"):
            st.warning("보유 종목이 없어 매도할 수 없습니다.")
            return

        holdings = snapshot.get("holdings", [])

        holdings_with_prices = []
        with st.spinner("보유 종목의 현재가를 조회하는 중..."):
            for h in holdings:
                price = None
                if country_code_inner == "kor":
                    price = fetch_naver_realtime_price(h["ticker"])
                    if not price:
                        df = fetch_ohlcv(h["ticker"], country="kor", months_back=1)
                        if df is not None and not df.empty:
                            if isinstance(df.columns, pd.MultiIndex):
                                df.columns = df.columns.get_level_values(0)
                                df = df.loc[:, ~df.columns.duplicated()]
                            price = df["Close"].iloc[-1]
                elif country_code_inner == "aus":
                    df = fetch_ohlcv(h["ticker"], country="aus", months_back=1)
                    if df is not None and not df.empty:
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                            df = df.loc[:, ~df.columns.duplicated()]
                        price = df["Close"].iloc[-1]

                # 불리언 평가 및 계산 전에 가격을 스칼라로 보장합니다.
                # 예: 중복 컬럼 등으로 인해 함수가 Series를 반환하는 경우를 처리합니다.
                price_val = price.item() if isinstance(price, pd.Series) else price

                if price_val and pd.notna(price_val):
                    value = h["shares"] * price_val
                    return_pct = (
                        (price_val / h["avg_cost"] - 1) * 100 if h.get("avg_cost", 0) > 0 else 0.0
                    )
                    holdings_with_prices.append(
                        {
                            "ticker": h["ticker"],
                            "name": h["name"],
                            "shares": h["shares"],
                            "price": price_val,
                            "value": value,
                            "return_pct": return_pct,
                        }
                    )

        if not holdings_with_prices:
            st.error("보유 종목의 현재가를 조회할 수 없습니다.")
            return

        df_holdings = pd.DataFrame(holdings_with_prices)

        def on_sell_submit():
            # 한국 현지 시간으로 현재 시간을 가져옵니다.
            trade_time = datetime.now()
            if pytz:
                try:
                    korea_tz = pytz.timezone("Asia/Seoul")
                    # DB에 시간대 정보가 없는 순수한 한국 시간(naive)으로 저장해달라는 요청에 따라
                    # aware datetime에서 timezone 정보를 제거합니다.
                    trade_time = datetime.now(korea_tz).replace(tzinfo=None)
                except pytz.UnknownTimeZoneError:
                    # pytz가 설치되었지만 'Asia/Seoul'을 모르는 경우에 대한 폴백
                    pass

            # st.session_state에서 폼 데이터 가져오기
            editor_state = st.session_state[f"sell_editor_{country_code_inner}"]

            # data_editor에서 선택된 행의 인덱스를 찾습니다.
            selected_indices = [
                idx for idx, edit in editor_state.get("edited_rows", {}).items() if edit.get("선택")
            ]

            if not selected_indices:
                st.session_state[message_key] = ("warning", "매도할 종목을 선택해주세요.")
                return

            selected_rows = df_holdings.loc[selected_indices]

            success_count = 0
            for _, row in selected_rows.iterrows():
                trade_data = {
                    "country": country_code_inner,
                    "date": trade_time,
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "action": "SELL",
                    "shares": row["shares"],
                    "price": row["price"],
                    "note": "Manual sell from web app",
                }
                if save_trade(trade_data):
                    success_count += 1

            if success_count == len(selected_rows):
                st.session_state[message_key] = (
                    "success",
                    f"{success_count}개 종목의 매도 거래가 성공적으로 저장되었습니다.",
                )
            else:
                st.session_state[message_key] = (
                    "error",
                    "일부 거래 저장에 실패했습니다. 콘솔 로그를 확인해주세요.",
                )

        # 다이얼로그 내에서 오류/경고 메시지만 표시합니다.
        if message_key in st.session_state:
            msg_type, msg_text = st.session_state[message_key]
            if msg_type == "success":
                if msg_type == "warning":
                    st.warning(msg_text)
                else:
                    st.error(msg_text)
                # 메시지는 한 번만 표시되도록 세션에서 제거합니다.
                del st.session_state[message_key]

        with st.form(f"sell_form_{country_code_inner}"):
            st.subheader("매도할 종목을 선택하세요 (전체 매도)")

            df_holdings["선택"] = False
            # 정렬이 필요한 컬럼은 숫자형으로 유지하고, column_config에서 포맷팅합니다.
            # 이렇게 하면 '평가금액' 등에서 문자열이 아닌 숫자 기준으로 올바르게 정렬됩니다.
            df_display = df_holdings[
                ["선택", "name", "ticker", "shares", "return_pct", "value", "price"]
            ].copy()
            value_col_name = f"평가금액{currency_str}"
            price_col_name = f"현재가{currency_str}"
            df_display.rename(
                columns={
                    "name": "종목명",
                    "ticker": "티커",
                    "shares": "보유수량",
                    "return_pct": "수익률",
                    "value": value_col_name,
                    "price": price_col_name,
                },
                inplace=True,
            )

            st.data_editor(
                df_display,
                hide_index=True,
                width="stretch",
                key=f"sell_editor_{country_code_inner}",
                disabled=["종목명", "티커", "보유수량", value_col_name, "수익률", price_col_name],
                column_config={
                    "선택": st.column_config.CheckboxColumn("삭제", required=True),
                    "보유수량": st.column_config.NumberColumn(format="%.8f"),
                    "수익률": st.column_config.NumberColumn(
                        format="%.2f%%",
                    ),
                    value_col_name: st.column_config.NumberColumn(
                        # 쉼표(,)를 포맷에 추가하여 3자리마다 구분자를 표시합니다.
                        format="%,.0f" if country_code_inner == "kor" else "%,.2f"
                    ),
                    price_col_name: st.column_config.NumberColumn(
                        format="%.4f" if country_code_inner == "aus" else "%d"
                    ),
                },
            )

            st.form_submit_button("선택 종목 매도", on_click=on_sell_submit)

    _display_success_toast(country_code)

    sub_tab_names = ["현황", "히스토리", "트레이드", "종목 관리", "설정", "알림"]
    (
        sub_tab_status,
        sub_tab_history,
        sub_tab_trades,
        sub_tab_etf_management,
        sub_tab_settings,
        sub_tab_notification,
    ) = st.tabs(sub_tab_names)

    # --- 공통 데이터 로딩 ---
    sorted_dates = get_available_snapshot_dates(country_code)

    # 오늘/다음 거래일을 목록에 반영
    today = pd.Timestamp.now().normalize()
    today_str = today.strftime("%Y-%m-%d")
    if country_code != "coin":
        # 한국/호주: 실제 거래일 캘린더로 오늘/다음 거래일을 판단 (실패 시 월~금 폴백)
        next_td_str_fallback = None
        try:
            lookahead_end = (today + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
            upcoming_days = get_trading_days(today_str, lookahead_end, country_code)
            is_trading_today = any(d.date() == today.date() for d in upcoming_days)
            if is_trading_today:
                if today_str not in sorted_dates:
                    sorted_dates.insert(0, today_str)
            else:
                # 다음 거래일을 찾아 추가 (예: 토요일이면 다음 월요일)
                next_td = next((d for d in upcoming_days if d.date() >= today.date()), None)
                if next_td is not None:
                    next_td_str = pd.Timestamp(next_td).strftime("%Y-%m-%d")
                    if next_td_str not in sorted_dates:
                        sorted_dates.insert(0, next_td_str)
        except Exception:
            # 무시하고 폴백 계산 수행
            pass
        # 캘린더 조회 실패 또는 주말일 때의 폴백: 다음 월~금
        if today.weekday() >= 5:  # 토/일
            delta = 7 - today.weekday()
            next_bday = today + pd.Timedelta(days=delta)
            next_td_str_fallback = next_bday.strftime("%Y-%m-%d")
            if next_td_str_fallback not in sorted_dates:
                sorted_dates.insert(0, next_td_str_fallback)
    # 코인: 현황 탭은 항상 오늘 날짜를 기준으로 하므로, 목록 맨 앞에 오늘 날짜를 추가합니다.
    elif country_code == "coin":
        if today_str not in sorted_dates:
            sorted_dates.insert(0, today_str)

    # --- 1. 현황 탭 (최신 날짜) ---
    with sub_tab_status:
        if not sorted_dates:
            st.warning(
                f"[{country_code.upper()}] 국가의 포트폴리오 데이터를 DB에서 찾을 수 없습니다."
            )
            st.info("먼저 '거래 입력' 버튼을 통해 거래 내역을 추가해주세요.")
        else:
            # '현황' 탭의 기준 날짜를 결정합니다.
            # - 코인: 항상 오늘 날짜
            # - 한국/호주: 오늘이 거래일이면 오늘, 아니면 다음 거래일
            today = pd.Timestamp.now().normalize()
            target_date_str = today.strftime("%Y-%m-%d")  # 기본값은 오늘

            if country_code != "coin":
                try:
                    lookahead_end = (today + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
                    # get_trading_days의 start_date는 target_date_str로 사용
                    upcoming_days = get_trading_days(target_date_str, lookahead_end, country_code)

                    # 오늘 또는 그 이후의 가장 가까운 거래일을 찾습니다.
                    next_td = next((d for d in upcoming_days if d.date() >= today.date()), None)

                    if next_td is not None:
                        target_date_str = pd.Timestamp(next_td).strftime("%Y-%m-%d")
                    else:
                        # 거래일 조회 실패 시 주말/평일로 폴백
                        if today.weekday() >= 5:  # 토/일
                            delta = 7 - today.weekday()
                            next_bday = today + pd.Timedelta(days=delta)
                            target_date_str = next_bday.strftime("%Y-%m-%d")
                except Exception:
                    # 예외 발생 시에도 주말/평일로 폴백
                    if today.weekday() >= 5:  # 토/일
                        delta = 7 - today.weekday()
                        next_bday = today + pd.Timedelta(days=delta)
                        target_date_str = next_bday.strftime("%Y-%m-%d")

            # 캐시된(또는 계산 완료된) 결과를 표시
            result = get_cached_status_report(
                country=country_code, date_str=target_date_str, force_recalculate=False
            )
            if result:
                header_line, headers, rows = result
                # 헤더(요약)과 경고를 분리합니다. '<br>' 이전은 요약, 이후는 경고 영역입니다.
                header_main = header_line
                warning_html = None
                if isinstance(header_line, str) and "<br>" in header_line:
                    parts = header_line.split("<br>", 1)
                    header_main = parts[0]
                    warning_html = parts[1]

                # 테이블 상단에 요약 헤더를 표시합니다.
                if header_main:
                    st.markdown(f":information_source: {header_main}", unsafe_allow_html=True)

                # 데이터와 헤더의 컬럼 수가 일치하는지 확인하여 앱 충돌 방지
                if rows and headers and len(rows[0]) != len(headers):
                    st.error(
                        f"데이터 형식 오류: 현황 리포트의 컬럼 수({len(headers)})와 데이터 수({len(rows[0])})가 일치하지 않습니다. '다시 계산'을 시도해주세요."
                    )
                    st.write("- 헤더:", headers)
                    st.write("- 첫 번째 행 데이터:", rows[0])
                else:
                    df = pd.DataFrame(rows, columns=headers)
                    _display_status_report_df(df, country_code)
                    # 테이블 아래에 경고(평가금액 대체 안내)가 있으면 표시합니다.
                    if warning_html:
                        st.markdown(warning_html, unsafe_allow_html=True)
                    # 테이블 아래에 벤치마크 대비 초과성과를 표시합니다.
                    benchmark_str = get_cached_benchmark_status(country_code)
                    if benchmark_str:
                        st.markdown(
                            f'<div style="text-align: left; padding-top: 0.5rem;">{benchmark_str}</div>',
                            unsafe_allow_html=True,
                        )
            else:
                next_run_time_str = get_next_schedule_time_str(country_code)
                st.info(
                    f"""
**{target_date_str}** 날짜의 현황 데이터가 아직 계산되지 않았습니다.

스케줄러에 의해 자동으로 계산될 예정입니다.

다음 예상 실행 시간: **{next_run_time_str}**
"""
                )

    with sub_tab_history:
        history_sub_tab_names = ["현황", "평가금액"]
        history_status_tab, history_equity_tab = st.tabs(history_sub_tab_names)

        with history_status_tab:
            # 히스토리 탭에서는 오늘 및 미래 날짜를 제외합니다.
            today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
            past_dates = [d for d in sorted_dates if d < today_str]

            # 코인: 시작일부터 어제까지 모든 날짜로 탭을 생성하고, 중복을 제거합니다.
            if country_code == "coin" and past_dates:
                try:
                    # Apply INITIAL_DATE floor
                    coin_settings = get_app_settings(country_code) or {}
                    initial_dt = None
                    if coin_settings.get("initial_date"):
                        try:
                            initial_dt = pd.to_datetime(
                                coin_settings.get("initial_date")
                            ).normalize()
                        except Exception:
                            initial_dt = None
                    oldest = pd.to_datetime(past_dates[-1]).normalize()
                    start_dt = max(oldest, initial_dt) if initial_dt is not None else oldest
                    yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
                    full_range = pd.date_range(start=start_dt, end=yesterday, freq="D")
                    # 최신이 먼저 오도록 내림차순 정렬 후 문자열로 변환
                    past_dates = [d.strftime("%Y-%m-%d") for d in full_range[::-1]]
                except Exception:
                    # 폴백: 중복 제거만 수행
                    seen = set()
                    uniq = []
                    # Also filter below INITIAL_DATE in fallback path
                    init_str = None
                    try:
                        if coin_settings.get("initial_date"):
                            init_str = pd.to_datetime(coin_settings.get("initial_date")).strftime(
                                "%Y-%m-%d"
                            )
                    except Exception:
                        init_str = None
                    for d in past_dates:
                        if init_str and d < init_str:
                            continue
                        if d not in seen:
                            seen.add(d)
                            uniq.append(d)
                    past_dates = uniq
            # 한국/호주: 히스토리의 첫 탭은 항상 '마지막 거래일'이 되도록 보정합니다.
            if country_code in ("kor", "aus") and past_dates:
                try:
                    today = pd.Timestamp.now().normalize()
                    lookback_start = (today - pd.Timedelta(days=21)).strftime("%Y-%m-%d")
                    trading_days = get_trading_days(lookback_start, today_str, country_code)
                    if trading_days:
                        # 오늘 '이전'의 마지막 거래일을 찾습니다.
                        past_trading_days = [d for d in trading_days if d.date() < today.date()]
                        if past_trading_days:
                            last_td = max(past_trading_days)
                            last_td_str = pd.Timestamp(last_td).strftime("%Y-%m-%d")
                            # 히스토리 탭 목록 맨 앞에 마지막 거래일이 오도록 정렬 보정
                            if last_td_str in past_dates:
                                past_dates = [last_td_str] + [
                                    d for d in past_dates if d != last_td_str
                                ]
                            elif not past_dates or last_td_str > past_dates[0]:
                                past_dates.insert(0, last_td_str)
                except Exception:
                    pass
            if not past_dates:
                st.info("과거 현황 데이터가 없습니다.")
            else:
                history_date_tabs = st.tabs(past_dates)
                for i, date_str in enumerate(past_dates):
                    with history_date_tabs[i]:
                        want_date = pd.to_datetime(date_str).to_pydatetime()
                        report_from_db = get_status_report_from_db(country_code, want_date)

                        if report_from_db:
                            # 데이터가 있으면 표시합니다.
                            header_line = report_from_db.get("header_line", "")
                            headers = report_from_db.get("headers")
                            rows = report_from_db.get("rows")

                            st.markdown(
                                f":information_source: {header_line}", unsafe_allow_html=True
                            )

                            # 데이터 형식 검증 (과거 데이터 호환용)
                            # "기준일:"로 시작하는 과거 형식의 헤더에 대해서만 날짜 일치 여부를 확인합니다.
                            if header_line.startswith("기준일:"):
                                expected_prefix = f"기준일: {date_str}("
                                if not header_line.startswith(expected_prefix):
                                    st.warning(
                                        "저장된 데이터의 날짜가 일치하지 않습니다. 재계산이 필요할 수 있습니다."
                                    )

                            if rows and headers and len(rows[0]) != len(headers):
                                st.error(
                                    f"데이터 형식 오류: 컬럼 수({len(headers)})와 데이터 수({len(rows[0])})가 일치하지 않습니다."
                                )
                                st.write("- 헤더:", headers)
                                st.write("- 첫 번째 행 데이터:", rows[0])
                            else:
                                df = pd.DataFrame(rows, columns=headers)
                                _display_status_report_df(df, country_code)
                        else:
                            # 데이터가 없으면 메시지를 표시합니다.
                            st.info(f"'{date_str}' 날짜의 현황 데이터가 없습니다.")

                        # 수동 재계산 버튼
                        if st.button(
                            "이 날짜 다시 계산하기",
                            key=f"recalc_hist_{country_code}_{date_str}_{i}",
                        ):
                            with st.spinner(f"'{date_str}' 기준 현황 데이터를 계산/저장 중..."):
                                calc_result = get_cached_status_report(
                                    country=country_code,
                                    date_str=date_str,
                                    force_recalculate=True,
                                )
                            if calc_result:
                                st.success("재계산 완료!")
                                st.rerun()

        with history_equity_tab:
            app_settings = get_app_settings(country_code)
            initial_date = (app_settings.get("initial_date") if app_settings else None) or (
                datetime.now() - pd.DateOffset(months=3)
            )

            currency_str = f" ({'AUD' if country_code == 'aus' else 'KRW'})"

            start_date_str = initial_date.strftime("%Y-%m-%d")
            end_date_str = datetime.now().strftime("%Y-%m-%d")

            with st.spinner("거래일 및 평가금액 데이터를 불러오는 중..."):
                all_trading_days = get_trading_days(start_date_str, end_date_str, country_code)
                if (
                    not all_trading_days
                    and country_code
                    == "kor"  # 호주는 yfinance가 주말을 건너뛰므로 거래일 조회가 필수는 아님
                ):
                    st.warning("거래일을 조회할 수 없습니다.")
                else:
                    start_dt_obj = pd.to_datetime(start_date_str).to_pydatetime()
                    end_dt_obj = pd.to_datetime(end_date_str).to_pydatetime()
                    existing_equities = get_all_daily_equities(
                        country_code, start_dt_obj, end_dt_obj
                    )
                    equity_data_map = {
                        pd.to_datetime(e["date"]).normalize(): e for e in existing_equities
                    }

                    # 거래일 조회가 실패한 경우(예: 호주), DB에 있는 날짜만 사용
                    if not all_trading_days:
                        all_trading_days = sorted(list(equity_data_map.keys()))

                    data_for_editor = []
                    for trade_date in all_trading_days:
                        existing_data = equity_data_map.get(trade_date, {})
                        row = {
                            "date": trade_date,
                            "total_equity": existing_data.get("total_equity", 0.0),
                            "updated_at": existing_data.get("updated_at"),
                            "updated_by": existing_data.get("updated_by"),
                        }
                        if country_code == "aus":
                            is_data = existing_data.get("international_shares", {})
                            row["is_value"] = is_data.get("value", 0.0)
                            row["is_change_pct"] = is_data.get("change_pct", 0.0)
                        data_for_editor.append(row)

                    df_to_edit = pd.DataFrame(data_for_editor)

                    column_config = {
                        "date": st.column_config.DateColumn(
                            "일자", format="YYYY-MM-DD", disabled=True
                        ),
                        "total_equity": st.column_config.NumberColumn(
                            f"총 평가금액{currency_str}",
                            format="%.2f" if country_code == "aus" else "%d",
                            required=True,
                        ),
                        "updated_at": st.column_config.DatetimeColumn(
                            "변경일시", format="YYYY-MM-DD HH:mm:ss", disabled=True
                        ),
                        "updated_by": st.column_config.TextColumn("변경자", disabled=True),
                    }
                    if country_code == "aus":
                        column_config["is_value"] = st.column_config.NumberColumn(
                            f"해외주식 평가액{currency_str}", format="%.2f"
                        )
                        column_config["is_change_pct"] = st.column_config.NumberColumn(
                            "해외주식 수익률(%)",
                            format="%.2f",
                            help="수익률(%)만 입력합니다. 예: 5.5",
                        )

                    st.info("총 평가금액을 수정한 후 아래 '저장하기' 버튼을 눌러주세요.")

                    edited_df = st.data_editor(
                        df_to_edit,
                        key=f"equity_editor_{country_code}",
                        width="stretch",
                        hide_index=True,
                        column_config=column_config,
                    )

                    if st.button("평가금액 저장하기", key=f"save_all_equities_{country_code}"):
                        with st.spinner("변경된 평가금액을 저장하는 중..."):
                            # st.data_editor의 변경 사항은 세션 상태에 저장됩니다.
                            # 전체 데이터프레임을 순회하는 대신, 변경된 행만 처리하여 불필요한 DB 업데이트를 방지합니다.
                            editor_state = st.session_state[f"equity_editor_{country_code}"]
                            edited_rows = editor_state.get("edited_rows", {})

                            saved_count = 0
                            for row_index, changes in edited_rows.items():
                                # 변경된 행의 원본 데이터를 가져옵니다.
                                original_row = df_to_edit.iloc[row_index]

                                # 저장할 데이터를 구성합니다.
                                date_to_save = original_row["date"].to_pydatetime()
                                equity_to_save = changes.get(
                                    "total_equity", original_row["total_equity"]
                                )
                                is_data_to_save = None
                                if country_code == "aus":
                                    is_data_to_save = {
                                        "value": changes.get("is_value", original_row["is_value"]),
                                        "change_pct": changes.get(
                                            "is_change_pct", original_row["is_change_pct"]
                                        ),
                                    }

                                if save_daily_equity(
                                    country_code,
                                    date_to_save,
                                    equity_to_save,
                                    is_data_to_save,
                                    updated_by="사용자",
                                ):
                                    saved_count += 1
                            if saved_count > 0:
                                st.success(f"{saved_count}개 날짜의 평가금액을 업데이트했습니다.")
                                st.rerun()
                            else:
                                st.info("변경된 내용이 없어 저장하지 않았습니다.")

    with sub_tab_trades:
        # 코인 탭: 거래 입력 대신 보유 현황/데이터 편집만 제공 (동기화 버튼 제거)
        if country_code == "coin":
            pass

        if country_code != "coin":
            col1, col2, _ = st.columns([1, 1, 8])
            with col1:
                if st.button("BUY", key=f"add_buy_btn_{country_code}"):
                    show_buy_dialog()
            with col2:
                if st.button("SELL", key=f"add_sell_btn_{country_code}"):
                    show_sell_dialog()

        all_trades = get_all_trades(country_code)
        if not all_trades:
            st.info("거래 내역이 없습니다.")
        else:
            df_trades = pd.DataFrame(all_trades)
            # 코인 전용: 티커 필터(ALL 포함)
            if country_code == "coin" and "ticker" in df_trades.columns:
                unique_tickers = sorted(
                    {str(t).upper() for t in df_trades["ticker"].dropna().tolist()}
                )
                options = ["ALL"] + unique_tickers
                selected = st.selectbox(
                    "티커 필터", options, index=0, key=f"coin_trades_filter_{country_code}"
                )
                if selected != "ALL":
                    df_trades = df_trades[df_trades["ticker"].str.upper() == selected]

            # 금액(수량*가격) 계산: 정수, 천단위 콤마
            try:
                amt = pd.to_numeric(df_trades.get("shares"), errors="coerce").fillna(
                    0.0
                ) * pd.to_numeric(df_trades.get("price"), errors="coerce").fillna(0.0)
                df_trades["amount"] = (
                    amt.round(0)
                    .astype("Int64")
                    .fillna(0)
                    .astype(object)
                    .apply(lambda x: f"{int(x):,}" if pd.notna(x) else "0")
                )
            except Exception:
                df_trades["amount"] = "0"

            # 삭제 선택을 위한 컬럼 추가
            df_trades["delete"] = False

            # 표시할 컬럼 순서 정의
            # 기록시간 대신 거래시간(빗썸 시간, 'date')을 우선 표시합니다.
            cols_to_show = [
                "delete",
                "date",
                "action",
                "ticker",
                "name",
                "shares",
                "price",
                "amount",
                "note",
                "id",
            ]
            # reindex를 사용하여 이전 데이터에 'created_at'이 없어도 오류가 발생하지 않도록 합니다.
            df_display = df_trades.reindex(columns=cols_to_show).copy()

            # 날짜 및 시간 포맷팅
            df_display["date"] = pd.to_datetime(df_display["date"], errors="coerce").dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            edited_df = st.data_editor(
                df_display,
                key=f"trades_editor_{country_code}",
                hide_index=True,
                width="stretch",
                column_config={
                    "delete": st.column_config.CheckboxColumn("삭제", required=True),
                    "id": None,  # ID 컬럼은 숨김
                    "date": st.column_config.TextColumn("거래시간"),
                    "action": st.column_config.TextColumn("종류"),
                    "ticker": st.column_config.TextColumn("티커"),
                    "name": st.column_config.TextColumn("종목명", width="medium"),
                    "shares": st.column_config.NumberColumn(
                        "수량", format="%.8f" if country_code in ["coin"] else "%.0f"
                    ),
                    "price": st.column_config.NumberColumn(
                        "가격", format="%.4f" if country_code == "aus" else "%d"
                    ),
                    "amount": st.column_config.NumberColumn("금액", format="%.0f"),
                    "note": st.column_config.TextColumn("비고", width="large"),
                },
                disabled=["date", "action", "ticker", "name", "shares", "price", "amount", "note"],
            )

            if st.button(
                "선택한 거래 삭제", key=f"delete_trade_btn_{country_code}", type="primary"
            ):
                trades_to_delete = edited_df[edited_df["delete"]]
                if not trades_to_delete.empty:
                    with st.spinner(f"{len(trades_to_delete)}개의 거래를 삭제하는 중..."):
                        deleted_count = 0
                        for trade_id in trades_to_delete["id"]:
                            if delete_trade_by_id(trade_id):
                                deleted_count += 1

                        st.success(f"{deleted_count}개의 거래를 성공적으로 삭제했습니다.")
                        st.rerun()
                else:
                    st.warning("삭제할 거래를 선택해주세요.")

    with sub_tab_etf_management:
        with st.spinner("종목 마스터 데이터를 불러오는 중..."):
            render_master_etf_ui(country_code)

    with sub_tab_notification:
        render_notification_settings_ui(country_code)

    with sub_tab_settings:
        # 1. DB에서 현재 설정값 로드
        db_settings = get_app_settings(country_code)
        current_capital = db_settings.get("initial_capital", 0) if db_settings else 0
        current_topn = db_settings.get("portfolio_topn") if db_settings else None
        current_ma = db_settings.get("ma_period") if db_settings else None
        current_replace_threshold = db_settings.get("replace_threshold") if db_settings else None
        current_replace_weaker = db_settings.get("replace_weaker_stock") if db_settings else None
        current_max_replacements = (
            db_settings.get("max_replacements_per_day") if db_settings else None
        )

        test_months_range = 12  # Default value
        default_date = pd.Timestamp.now() - pd.DateOffset(months=test_months_range)
        current_date = (
            db_settings.get("initial_date", default_date) if db_settings else default_date
        )

        with st.form(key=f"settings_form_{country_code}"):
            currency_str = f" ({'AUD' if country_code == 'aus' else 'KRW'})"

            new_capital = st.number_input(
                f"초기 자본금 (INITIAL_CAPITAL){currency_str}",
                value=float(current_capital) if country_code == "aus" else int(current_capital),
                format="%.2f" if country_code == "aus" else "%d",
                help="포트폴리오의 시작 자본금을 설정합니다. 누적 수익률 계산의 기준이 됩니다.",
            )

            new_date = st.date_input(
                "초기 자본 기준일 (INITIAL_DATE)",
                value=current_date,
                help="초기 자본금이 투입된 날짜를 설정합니다.",
            )

            if current_topn is None:
                st.warning("최대 보유 종목 수(PORTFOLIO_TOPN)를 설정해주세요.")

            new_topn_str = st.text_input(
                "최대 보유 종목 수 (PORTFOLIO_TOPN)",
                value=str(current_topn) if current_topn is not None else "",
                placeholder="예: 10",
                help="포트폴리오에서 최대로 보유할 종목의 개수를 설정합니다.",
            )

            st.markdown("---")
            st.subheader("전략 파라미터")

            if current_ma is None:
                st.warning("이동평균 기간(MA_PERIOD)을 설정해주세요.")
            new_ma_str = st.text_input(
                "이동평균 기간 (MA_PERIOD)",
                value=str(current_ma) if current_ma is not None else "75",
                placeholder="예: 15",
                help="종목의 추세 판단에 사용될 이동평균 기간입니다.",
            )

            # 교체 매매 사용 여부 (bool)
            replace_weaker_checkbox = st.checkbox(
                "교체 매매 사용 (REPLACE_WEAKER_STOCK)",
                value=bool(current_replace_weaker) if current_replace_weaker is not None else False,
                help="포트폴리오가 가득 찼을 때, 더 강한 후보가 있을 경우 약한 보유종목을 교체할지 여부",
            )

            # 하루 최대 교체 수
            max_replacements_str = st.text_input(
                "하루 최대 교체 수 (MAX_REPLACEMENTS_PER_DAY)",
                value=str(current_max_replacements) if current_max_replacements is not None else "",
                placeholder="예: 5",
                help="하루에 실행할 수 있는 교체 매매의 최대 종목 수",
            )

            # 교체 매매 임계값 설정 (DB에서 관리)
            new_replace_threshold_str = st.text_input(
                "교체 매매 점수 임계값 (REPLACE_SCORE_THRESHOLD)",
                value=(
                    "{:.2f}".format(float(current_replace_threshold))
                    if current_replace_threshold is not None
                    else ""
                ),
                placeholder="예: 1.5",
                help="교체 매매 실행 조건: 새 후보 점수가 기존 보유 점수보다 이 값만큼 높을 때 교체.",
            )

            # 코인 전용 임포트 기간 설정 제거됨 (트레이드 동기화 폐지)

            save_settings_submitted = st.form_submit_button("설정 저장하기")

            if save_settings_submitted:
                error = False
                if not new_topn_str or not new_topn_str.isdigit() or int(new_topn_str) < 1:
                    st.error("최대 보유 종목 수는 1 이상의 숫자여야 합니다.")
                    error = True
                if not new_ma_str or not new_ma_str.isdigit() or int(new_ma_str) < 1:
                    st.error("이동평균 기간은 1 이상의 숫자여야 합니다.")
                    error = True
                # max_replacements_per_day 검증 (정수 >= 0)
                if (
                    not max_replacements_str
                    or not max_replacements_str.isdigit()
                    or int(max_replacements_str) < 0
                ):
                    st.error("하루 최대 교체 수는 0 이상의 정수여야 합니다.")
                    error = True
                # replace_threshold 검증 (float 가능 여부)
                try:
                    _ = float(new_replace_threshold_str)
                except Exception:
                    st.error("교체 매매 점수 임계값은 숫자여야 합니다.")
                    error = True

                if not error:
                    new_topn = int(new_topn_str)
                    new_ma = int(new_ma_str)
                    new_max_replacements = int(max_replacements_str)
                    new_replace_threshold = float(new_replace_threshold_str)
                    settings_to_save = {
                        "country": country_code,
                        "initial_capital": new_capital,
                        "initial_date": pd.to_datetime(new_date).to_pydatetime(),
                        "portfolio_topn": new_topn,
                        "ma_period": new_ma,
                        "replace_weaker_stock": bool(replace_weaker_checkbox),
                        "max_replacements_per_day": new_max_replacements,
                        "replace_threshold": new_replace_threshold,
                    }
                    # 코인용 빗썸 임포트 기간 설정은 더 이상 사용하지 않습니다.
                    if save_app_settings(country_code, settings_to_save):
                        st.success("설정이 성공적으로 저장되었습니다.")
                        st.rerun()
                    else:
                        st.error("설정 저장에 실패했습니다.")


def main():
    """MomentumETF 오늘의 현황 웹 UI를 렌더링합니다."""
    # 페이지 설정은 Streamlit의 첫 명령으로 실행되어야 합니다.
    st.set_page_config(page_title="MomentumETF Status", layout="wide")

    # 페이지 상단 여백을 줄이기 위한 CSS 주입
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1rem; }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # --- DB 연결 확인 ---
    # 앱의 다른 부분이 실행되기 전에 DB 연결을 먼저 확인합니다.
    if get_db_connection() is None:
        st.error(
            """
            **데이터베이스 연결 실패**

            MongoDB 데이터베이스에 연결할 수 없습니다. 다음 사항을 확인해주세요:

            1.  **환경 변수**: `MONGO_DB_CONNECTION_STRING` 환경 변수가 올바르게 설정되었는지 확인하세요.
            2.  **IP 접근 목록**: 현재 서비스의 IP 주소가 MongoDB Atlas의 'IP Access List'에 추가되었는지 확인하세요.
            3.  **클러스터 상태**: MongoDB Atlas 클러스터가 정상적으로 실행 중인지 확인하세요.
            """
        )
        st.stop()  # DB 연결 실패 시 앱 실행 중단

    # 앱 가동시 거래일 캘린더 준비 상태 확인
    try:
        import pandas_market_calendars as _mcal  # type: ignore
    except Exception as e:
        st.error(
            "거래일 캘린더 라이브러리(pandas-market-calendars)를 불러올 수 없습니다.\n"
            "다음 명령으로 설치 후 다시 시도하세요: pip install pandas-market-calendars\n"
            f"상세: {e}"
        )
        st.stop()

    try:
        today = pd.Timestamp.now().normalize()
        start = (today - pd.DateOffset(days=30)).strftime("%Y-%m-%d")
        end = (today + pd.DateOffset(days=7)).strftime("%Y-%m-%d")
        problems = []
        for c in ("kor", "aus"):
            days = get_trading_days(start, end, c)
            if not days:
                problems.append(c)
        if problems:
            st.error(
                "거래일 캘린더를 조회하지 못했습니다: "
                + ", ".join({"kor": "한국", "aus": "호주"}[p] for p in problems)
                + "\nKOSPI/ASX 캘린더를 사용할 수 있는지 확인해주세요."
            )
            st.stop()
    except Exception as e:
        st.error(f"거래일 캘린더 초기화 중 오류가 발생했습니다: {e}")
        st.stop()

    # 제목과 시장 상태를 한 줄에 표시
    # "최근 중단" 기간이 길어지면서 줄바꿈되는 현상을 방지하기 위해
    # 오른쪽 컬럼의 너비를 늘립니다. (3:1 -> 2.5:1.5)
    col1, col2 = st.columns([2.5, 1.5])
    with col1:
        st.title("Momentum. ETF.")
    with col2:
        # 시장 상태는 한 번만 계산하여 10분간 캐시합니다.
        @st.cache_data(ttl=600)
        def _get_cached_market_status():
            return get_market_regime_status_string()

        market_status_str = _get_cached_market_status()
        if market_status_str:
            # st.markdown을 사용하여 오른쪽 정렬 및 상단 패딩을 적용합니다.
            st.markdown(
                f'<div style="text-align: right; padding-top: 1.5rem; font-size: 1.1rem;">{market_status_str}</div>',
                unsafe_allow_html=True,
            )

    tab_names = ["한국", "호주", "코인", "스케줄러", "설정"]
    tab_kor, tab_aus, tab_coin, tab_scheduler, tab_settings = st.tabs(tab_names)

    with tab_coin:
        render_country_tab("coin")

    with tab_kor:
        render_country_tab("kor")

    with tab_aus:
        render_country_tab("aus")

    with tab_scheduler:
        render_scheduler_tab()

    with tab_settings:
        st.header("공통 설정 (모든 국가 공유)")
        common = get_common_settings() or {}
        current_enabled = (
            bool(common.get("MARKET_REGIME_FILTER_ENABLED"))
            if "MARKET_REGIME_FILTER_ENABLED" in common
            else False
        )
        current_ticker = common.get("MARKET_REGIME_FILTER_TICKER")
        current_ma = common.get("MARKET_REGIME_FILTER_MA_PERIOD")
        current_stop = common.get("HOLDING_STOP_LOSS_PCT")
        current_cooldown = common.get("COOLDOWN_DAYS")
        current_atr = common.get("ATR_PERIOD_FOR_NORMALIZATION")

        with st.form("common_settings_form"):
            st.subheader("시장 레짐 필터")
            new_enabled = st.checkbox(
                "활성화 (MARKET_REGIME_FILTER_ENABLED)", value=current_enabled
            )
            new_ticker = st.text_input(
                "레짐 기준 지수 티커 (MARKET_REGIME_FILTER_TICKER)",
                value=str(current_ticker) if current_ticker is not None else "",
                placeholder="예: ^GSPC",
            )
            new_ma_str = st.text_input(
                "레짐 MA 기간 (MARKET_REGIME_FILTER_MA_PERIOD)",
                value=str(current_ma) if current_ma is not None else "",
                placeholder="예: 20",
            )

            st.subheader("위험 관리 및 지표")
            new_stop = st.number_input(
                "보유 손절 임계값 % (HOLDING_STOP_LOSS_PCT)",
                value=float(current_stop) if current_stop is not None else 0.0,
                step=0.1,
                format="%.2f",
                help="예: -10.0",
            )
            new_cooldown_str = st.text_input(
                "쿨다운 일수 (COOLDOWN_DAYS)",
                value=str(current_cooldown) if current_cooldown is not None else "",
                placeholder="예: 5",
            )
            new_atr_str = st.text_input(
                "ATR 기간 (ATR_PERIOD_FOR_NORMALIZATION)",
                value=str(current_atr) if current_atr is not None else "",
                placeholder="예: 14",
            )

            submitted = st.form_submit_button("공통 설정 저장")
            if submitted:
                error = False
                # Required validations
                if not new_ticker:
                    st.error("시장 레짐 필터 티커를 입력해주세요.")
                    error = True
                if not new_ma_str.isdigit() or int(new_ma_str) < 1:
                    st.error("레짐 MA 기간은 1 이상의 정수여야 합니다.")
                    error = True
                if not new_cooldown_str.isdigit() or int(new_cooldown_str) < 0:
                    st.error("쿨다운 일수는 0 이상의 정수여야 합니다.")
                    error = True
                if not new_atr_str.isdigit() or int(new_atr_str) < 1:
                    st.error("ATR 기간은 1 이상의 정수여야 합니다.")
                    error = True

                if not error:
                    # Normalize stop loss: interpret positive value as negative threshold
                    normalized_stop = -abs(float(new_stop))
                    to_save = {
                        "MARKET_REGIME_FILTER_ENABLED": bool(new_enabled),
                        "MARKET_REGIME_FILTER_TICKER": new_ticker,
                        "MARKET_REGIME_FILTER_MA_PERIOD": int(new_ma_str),
                        "HOLDING_STOP_LOSS_PCT": normalized_stop,
                        "COOLDOWN_DAYS": int(new_cooldown_str),
                        "ATR_PERIOD_FOR_NORMALIZATION": int(new_atr_str),
                    }
                    if save_common_settings(to_save):
                        st.success("공통 설정을 저장했습니다.")
                        st.rerun()
                    else:
                        st.error("공통 설정 저장에 실패했습니다.")


if __name__ == "__main__":
    main()
