import os
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# .env 파일이 있다면 로드합니다.
load_dotenv()


warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings(
    "ignore",
    message=r"\['break_start', 'break_end'\] are discontinued",
    category=UserWarning,
    module=r"^pandas_market_calendars\.",
)


try:
    from cron_descriptor import get_description as get_cron_description
except ImportError:
    get_cron_description = None
try:
    from croniter import croniter
except ImportError:
    croniter = None
try:
    import pytz
except ImportError:
    pytz = None
try:
    from pykrx import stock as _stock
except Exception:
    _stock = None

from signals import (
    generate_signal_report,
    get_market_regime_status_string,
    calculate_benchmark_comparison,
    get_next_trading_day,
)
from utils.account_registry import get_accounts_by_country, load_accounts
from utils.data_loader import (
    fetch_yfinance_name,
    get_trading_days,
    PykrxDataUnavailable,
)
from utils.db_manager import (
    delete_trade_by_id,
    get_account_settings,
    get_all_daily_equities,
    get_all_trades,
    get_available_snapshot_dates,
    get_common_settings,
    get_db_connection,
    get_portfolio_snapshot,
    get_signal_report_from_db,
    save_common_settings,
    save_daily_equity,
    save_portfolio_settings,
    save_signal_report_to_db,
    save_trade,
)
from utils.stock_list_io import get_etfs


COUNTRY_CODE_MAP = {"kor": "한국", "aus": "호주", "coin": "가상화폐"}

MARKET_DISPLAY_SETTINGS = {
    "kor": {"tz": "Asia/Seoul", "close": "15:30"},
    "aus": {"tz": "Australia/Sydney", "close": "16:00"},
}


def _get_local_now(country_code: str) -> Optional[datetime]:
    if not pytz:
        return None
    settings = MARKET_DISPLAY_SETTINGS.get(country_code)
    if not settings:
        return None
    try:
        tz = pytz.timezone(settings["tz"])
        return datetime.now(tz)
    except Exception:
        return None


def _get_status_target_date_str(country_code: str) -> str:
    if country_code == "coin":
        today = pd.Timestamp.now().normalize()
        return today.strftime("%Y-%m-%d")

    now_local = _get_local_now(country_code)
    if not now_local:
        today = pd.Timestamp.now().normalize()
        return today.strftime("%Y-%m-%d")

    local_today = now_local.date()
    today_str = pd.Timestamp(local_today).strftime("%Y-%m-%d")

    close_time = datetime.strptime(MARKET_DISPLAY_SETTINGS[country_code]["close"], "%H:%M").time()
    lookahead_end = pd.Timestamp(local_today) + pd.Timedelta(days=14)
    lookahead_end_str = lookahead_end.strftime("%Y-%m-%d")

    try:
        upcoming_days = get_trading_days(today_str, lookahead_end_str, country_code)
    except Exception:
        upcoming_days = []

    if not upcoming_days:
        return today_str

    is_trading_today = any(d.date() == local_today for d in upcoming_days)
    if is_trading_today and now_local.time() < close_time:
        return today_str

    next_day = next((d for d in upcoming_days if d.date() > local_today), None)
    if not next_day:
        fallback = get_next_trading_day(
            country_code, pd.Timestamp(local_today) + pd.Timedelta(days=1)
        )
        return pd.Timestamp(fallback).strftime("%Y-%m-%d")

    return pd.Timestamp(next_day).strftime("%Y-%m-%d")


def _ensure_header_has_date(header: str, date: datetime) -> str:
    """날짜 표시가 필요하면 붙이고, 이미 있다면 그대로 반환합니다."""
    if not header:
        return header

    normalized = header.strip()
    if "년" in normalized and "월" in normalized:
        return header

    date_display = f"{date.year}년 {date.month}월 {date.day}일"
    prefix = f"{date_display} | "

    if normalized.startswith(prefix):
        return header

    return prefix + header


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
    def get_cached_benchmark_comparison(
        country: str, date_str: str, account: str
    ) -> Optional[List[Dict[str, Any]]]:
        """벤치마크 비교 데이터를 캐시하여 반환합니다. (Streamlit용)"""
        try:
            return calculate_benchmark_comparison(country, account, date_str)
        except PykrxDataUnavailable as exc:
            start = exc.start_dt.strftime("%Y-%m-%d")
            end = exc.end_dt.strftime("%Y-%m-%d")
            return [{"name": "벤치마크", "error": f"데이터 없음 ({start}~{end})"}]

else:

    def get_cached_benchmark_comparison(
        country: str, date_str: str, account: str
    ) -> Optional[List[Dict[str, Any]]]:
        """벤치마크 비교 데이터를 반환합니다. (CLI용, 캐시 없음)"""
        try:
            return calculate_benchmark_comparison(country, account, date_str)
        except PykrxDataUnavailable as exc:
            start = exc.start_dt.strftime("%Y-%m-%d")
            end = exc.end_dt.strftime("%Y-%m-%d")
            return [{"name": "벤치마크", "error": f"데이터 없음 ({start}~{end})"}]


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


def get_cached_signal_report(
    country: str,
    account: str,
    date_str: str,
    force_recalculate: bool = False,
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Optional[tuple[Any, Any, Any]]:
    """
    MongoDB를 사용하여 매매 신호 리포트를 캐시합니다.
    force_recalculate=True일 경우에만 다시 계산합니다.
    """
    try:
        report_date = pd.to_datetime(date_str).to_pydatetime()
    except (ValueError, TypeError):
        st.error(f"잘못된 날짜 형식입니다: {date_str}")
        return None  # noqa: B901

    if not force_recalculate:
        # 1. DB에서 먼저 찾아봅니다.
        report_from_db = get_signal_report_from_db(country, account, report_date)
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
        with st.spinner(f"'{date_str}' 매매 신호를 다시 계산하는 중..."):
            new_report_tuple = generate_signal_report(
                country,
                account,
                date_str,
                prefetched_data=prefetched_data,
            )
            if new_report_tuple:
                header_line, headers, rows, _, _ = new_report_tuple
                new_report = (header_line, headers, rows)
                # 3. 계산된 결과를 DB에 저장합니다.
                save_signal_report_to_db(country, account, report_date, new_report)
            return new_report
    except ValueError as e:
        if str(e).startswith("PRICE_FETCH_FAILED:"):
            failed_tickers_str = str(e).split(":", 1)[1]
            st.error(f"{failed_tickers_str} 종목의 가격을 가져올 수 없습니다. 다시 시도하세요.")
            return None  # noqa: B901
        else:
            # 다른 ValueError는 기존처럼 처리
            print(f"오류: 신호 계산 오류: {country}/{date_str}: {e}")
            st.error(f"'{date_str}' 신호 계산 중 오류가 발생했습니다: {e}")
            return None
    except Exception as e:  # noqa: E722
        print(f"오류: 신호 계산 오류: {country}/{date_str}: {e}")
        st.error(
            f"'{date_str}' 신호 계산 중 오류가 발생했습니다. 자세한 내용은 콘솔 로그를 확인해주세요."
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


@st.cache_data
def get_cached_etfs(country_code: str) -> List[Dict[str, Any]]:
    """종목 마스터(etf.json) 데이터를 캐시하여 반환합니다."""
    return get_etfs(country_code) or []


def _display_status_report_df(df: pd.DataFrame, country_code: str):
    """
    시그널 리포트 DataFrame에 종목 메타데이터(이름, 카테고리)를 실시간으로 병합하고 스타일을 적용하여 표시합니다.
    """
    # 1. 종목 메타데이터 로드
    etfs_data = get_cached_etfs(country_code)
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
    numeric_cols = [
        "현재가",
        "일간수익률",
        "보유수량",
        "금액",
        "누적수익률",
        "비중",
        "점수",
    ]
    for col in numeric_cols:
        if col in df_display.columns:
            df_display[col] = pd.to_numeric(df_display[col], errors="coerce")

    # 4. 스타일 적용
    if "#" in df_display.columns:
        df_display = df_display.set_index("#")

    styler = df_display.style
    style_cols = ["일간수익률", "누적수익률"]
    for col in style_cols:
        if col in df_display.columns:
            styler = styler.map(style_returns, subset=[col])

    formats = {
        "일간수익률": "{:+.2f}%",
        "누적수익률": "{:+.2f}%",
        "비중": "{:.1f}%",
        "점수": lambda val: f"{val * 100:+.1f}%" if pd.notna(val) else "-",
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

    # 테이블 높이를 11개 행이 보이도록 고정합니다. (헤더 포함 11)
    height = (11) * 35 + 3

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

    if country_code == "coin":
        st.info("이곳에서 가상화폐 종목을 조회할 수 있습니다.")
    else:
        st.info("이곳에서 투자 유니버스에 포함된 종목을 조회할 수 있습니다.")

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


def render_scheduler_tab():
    """스케줄러 설정을 위한 UI를 렌더링합니다."""
    st.header("스케줄러 설정 (모든 국가)")
    st.info("각 국가별 시그널 계산 작업이 실행될 주기를 Crontab 형식으로 설정합니다.")

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


def _display_success_toast(key_prefix: str):
    """
    세션 상태에서 성공 메시지를 확인하고 토스트로 표시합니다.
    주로 다이얼로그가 닫힌 후 피드백을 주기 위해 사용됩니다.
    """
    keys_to_check = [
        f"buy_message_{key_prefix}",
        f"sell_message_{key_prefix}",
    ]
    for key in keys_to_check:
        if key in st.session_state:
            message = st.session_state[key]
            # 메시지가 (type, text) 튜플이고, type이 'success'인 경우에만 처리
            if isinstance(message, tuple) and len(message) == 2 and message[0] == "success":
                _, msg_text = st.session_state.pop(key)
                st.toast(msg_text)


def _prepare_account_entries(
    country_code: str, accounts: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for entry in accounts or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("is_active", True):
            entries.append(entry)
    if not entries:
        entries.append(
            {
                "account": None,
                "country": country_code,
                "display_name": COUNTRY_CODE_MAP.get(country_code, country_code.upper()),
            }
        )
    return entries


def _account_label(entry: Dict[str, Any]) -> str:
    label = entry.get("display_name") or entry.get("account")
    if not label:
        return "계좌"
    return str(label)


def _account_prefix(country_code: str, account_code: Optional[str]) -> str:
    return f"{country_code}_{account_code or 'default'}"


def _render_account_dashboard(country_code: str, account_entry: Dict[str, Any]):
    """지정된 계좌의 시그널/평가/거래/설정을 렌더링합니다."""

    account_code = account_entry.get("account")
    account_prefix = _account_prefix(country_code, account_code)

    if not account_code:
        st.info("활성 계좌가 없습니다. 계좌를 등록한 후 이용해주세요.")
        return

    @st.dialog("BUY")
    def show_buy_dialog(country_code_inner: str):
        """매수(BUY) 거래 입력을 위한 모달 다이얼로그를 표시합니다."""

        currency_str = f" ({'AUD' if country_code_inner == 'aus' else 'KRW'})"
        message_key = f"buy_message_{account_prefix}"

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
            ticker = st.session_state[f"buy_ticker_{account_prefix}"].strip()
            shares = st.session_state[f"buy_shares_{account_prefix}"]
            price = st.session_state[f"buy_price_{account_prefix}"]

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
                "account": account_code,
                "date": trade_time,
                "ticker": ticker.upper(),
                "name": etf_name,
                "action": "BUY",
                "shares": float(shares),
                "price": float(price),
                "note": "Manual input from web app",
            }

            if save_trade(trade_data):
                st.session_state[message_key] = (
                    "success",
                    "거래가 성공적으로 저장되었습니다.",
                )
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

        with st.form(f"trade_form_{account_prefix}"):
            st.text_input("종목코드 (티커)", key=f"buy_ticker_{account_prefix}")
            shares_format_str = "%.8f" if country_code_inner == "coin" else "%d"
            st.number_input(
                "수량",
                min_value=0.00000001,
                step=0.00000001,
                format=shares_format_str,
                key=f"buy_shares_{account_prefix}",
            )
            st.number_input(
                f"매수 단가{currency_str}",
                min_value=0.0,
                format=(
                    "%.4f"
                    if country_code_inner == "aus"
                    else ("%d" if country_code_inner in ["kor", "coin"] else "%d")
                ),
                key=f"buy_price_{account_prefix}",
            )
            st.form_submit_button("거래 저장", on_click=on_buy_submit)

    @st.dialog("SELL", width="large")
    def show_sell_dialog(country_code_inner: str):
        """보유 종목 매도를 위한 모달 다이얼로그를 표시합니다."""
        currency_str = f" ({'AUD' if country_code_inner == 'aus' else 'KRW'})"
        message_key = f"sell_message_{account_prefix}"

        from utils.data_loader import fetch_naver_realtime_price, fetch_ohlcv

        snapshot_dates = get_available_snapshot_dates(country_code_inner, account=account_code)
        latest_date_str = snapshot_dates[0] if snapshot_dates else None
        if not latest_date_str:
            st.warning("보유 종목이 없어 매도할 수 없습니다.")
            return

        snapshot = get_portfolio_snapshot(
            country_code_inner, date_str=latest_date_str, account=account_code
        )
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
            editor_state = st.session_state[f"sell_editor_{account_prefix}"]

            # data_editor에서 선택된 행의 인덱스를 찾습니다.
            selected_indices = [
                idx for idx, edit in editor_state.get("edited_rows", {}).items() if edit.get("선택")
            ]

            if not selected_indices:
                st.session_state[message_key] = (
                    "warning",
                    "매도할 종목을 선택해주세요.",
                )
                return

            selected_rows = df_holdings.loc[selected_indices]

            success_count = 0
            for _, row in selected_rows.iterrows():
                trade_data = {
                    "country": country_code_inner,
                    "account": account_code,
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

        with st.form(f"sell_form_{account_prefix}"):
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
                key=f"sell_editor_{account_prefix}",
                disabled=[
                    "종목명",
                    "티커",
                    "보유수량",
                    value_col_name,
                    "수익률",
                    price_col_name,
                ],
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

    _display_success_toast(account_prefix)

    sub_tab_names = ["시그널", "평가금액", "트레이드", "설정"]
    (
        sub_tab_signal,
        sub_tab_equity_history,
        sub_tab_trades,
        sub_tab_settings,
    ) = st.tabs(
        sub_tab_names
    )  # noqa: F841

    # 계좌 시작일 및 거래일 정보를 사용하여 날짜 선택 옵션을 필터링합니다.
    account_settings = get_account_settings(account_code)
    initial_date = None
    if account_settings and account_settings.get("initial_date"):
        try:
            initial_date = pd.to_datetime(account_settings["initial_date"]).normalize()
        except (ValueError, TypeError):
            pass

    raw_dates = get_available_snapshot_dates(country_code, account=account_code)
    sorted_dates = sorted(set(raw_dates), reverse=True)

    if country_code == "coin":
        today_ts = pd.Timestamp.now()
    else:
        local_now = _get_local_now(country_code)
        today_ts = pd.Timestamp(local_now) if local_now else pd.Timestamp.now()

    today_str = pd.Timestamp(today_ts.date()).strftime("%Y-%m-%d")
    target_date_str = _get_status_target_date_str(country_code)

    date_options: List[str] = []
    for candidate in [target_date_str] + sorted_dates:
        if candidate and candidate not in date_options:
            date_options.append(candidate)

    option_labels: Dict[str, str] = {}
    if date_options:
        if country_code == "coin":
            option_labels[target_date_str] = f"{target_date_str} (오늘)"
        else:
            if target_date_str == today_str:
                option_labels[target_date_str] = f"{target_date_str} (오늘)"
            else:
                option_labels[target_date_str] = f"{target_date_str} (다음 거래일)"
            if today_str in date_options and today_str != target_date_str:
                option_labels.setdefault(today_str, f"{today_str} (오늘)")

    with sub_tab_signal:
        if not date_options:
            # 데이터가 DB에 있지만 필터링되어 표시할 것이 없는 경우와,
            # DB에 데이터가 아예 없는 경우를 구분하여 메시지를 표시합니다.
            if not sorted_dates:
                st.warning(
                    f"[{country_code.upper()}] 국가의 포트폴리오 데이터를 DB에서 찾을 수 없습니다."
                )
                if country_code != "coin":
                    st.info("먼저 '거래 입력' 버튼을 통해 거래 내역을 추가해주세요.")
                else:
                    # 코인은 빗썸 동기화를 통해 거래 내역이 생성되므로, 그에 맞는 안내를 제공합니다.
                    st.info("빗썸 거래내역 동기화가 필요할 수 있습니다.")
            else:
                st.warning(f"[{country_code.upper()}] 표시에 유효한 시그널 데이터가 없습니다.")
                st.info("DB에 미래 날짜 또는 거래일이 아닌 날짜의 데이터만 존재할 수 있습니다.")
        else:
            selected_date_str = st.selectbox(
                "조회 날짜",
                date_options,
                format_func=lambda d: option_labels.get(d, d),
                key=f"signal_date_select_{account_prefix}",
            )

            result = get_cached_signal_report(
                country=country_code,
                account=account_code,
                date_str=selected_date_str,  # type: ignore
                force_recalculate=False,
            )

            if result:
                header_line, headers, rows = result
                header_main = header_line or ""
                warning_html = None
                if isinstance(header_main, str) and "<br>" in header_main:
                    parts = header_main.split("<br>", 1)
                    header_main = parts[0]
                    warning_html = parts[1]

                header_display = _ensure_header_has_date(
                    header_main, pd.to_datetime(selected_date_str).to_pydatetime()
                )
                if header_display:
                    safe_header = header_display.replace("$", "&#36;")
                    st.markdown(
                        f'<div class="status-summary">{safe_header}</div>',
                        unsafe_allow_html=True,
                    )

                if rows and headers and len(rows[0]) != len(headers):
                    st.error(
                        f"데이터 형식 오류: 시그널 리포트의 컬럼 수({len(headers)})와 데이터 수({len(rows[0])})가 일치하지 않습니다. '다시 계산'을 시도해주세요."
                    )
                    st.write("- 헤더:", headers)
                    st.write("- 첫 번째 행 데이터:", rows[0])
                else:
                    df = pd.DataFrame(rows, columns=headers)
                    _display_status_report_df(df, country_code)
                    if warning_html:
                        st.markdown(warning_html, unsafe_allow_html=True)
                # --- 벤치마크 비교 테이블 렌더링 ---
                benchmark_results = get_cached_benchmark_comparison(
                    country_code, selected_date_str, account=account_code  # type: ignore
                )
                if benchmark_results:
                    data_for_df = []
                    for res in benchmark_results:
                        if res.get("error"):
                            data_for_df.append(
                                {
                                    "티커": res.get("ticker", "-"),
                                    "벤치마크": res["name"],
                                    "누적수익률": res["error"],
                                    "초과수익률": "-",
                                }
                            )
                        else:
                            # header_line에서 실제 누적 수익률을 파싱하여 사용합니다.
                            # calculate_benchmark_comparison가 코인에 대해 잘못된 값을 반환하는 문제를 해결합니다.
                            portfolio_cum_ret_pct = None
                            if header_line:
                                try:
                                    # "누적: <span...>{+4.56}%...</span>" 형태에서 숫자 부분을 추출
                                    cum_ret_segment = [
                                        s for s in header_line.split("|") if "누적:" in s
                                    ][0]
                                    cum_ret_str = cum_ret_segment.split("%")[0].split("</span>")[-1]
                                    portfolio_cum_ret_pct = float(cum_ret_str)
                                except (IndexError, ValueError):
                                    pass  # 파싱 실패 시 기존 로직으로 폴백

                            excess_return_pct = res["excess_return_pct"]
                            if portfolio_cum_ret_pct is not None:
                                excess_return_pct = portfolio_cum_ret_pct - res["cum_ret_pct"]

                            data_for_df.append(
                                {
                                    "티커": res.get("ticker", "-"),
                                    "벤치마크": res["name"],
                                    "누적수익률": res["cum_ret_pct"],
                                    "초과수익률": excess_return_pct,
                                }
                            )
                    df_benchmark = pd.DataFrame(data_for_df)
                    st.dataframe(
                        df_benchmark,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "티커": st.column_config.TextColumn("티커"),
                            "누적수익률": st.column_config.NumberColumn(format="%.2f%%"),
                            "초과수익률": st.column_config.NumberColumn(format="%+.2f%%"),
                        },
                    )
            else:
                if selected_date_str == target_date_str:
                    next_run_time_str = get_next_schedule_time_str(country_code)
                    st.info(
                        f"""
    **{selected_date_str}** 날짜의 매매 신호가 아직 계산되지 않았습니다.

    스케줄러에 의해 자동으로 계산될 예정입니다.

    다음 예상 실행 시간: **{next_run_time_str}**
    """
                    )
                else:
                    st.info(f"'{selected_date_str}' 날짜의 매매 신호 데이터가 없습니다.")

            if st.button(
                "이 날짜 다시 계산하기",
                key=f"recalc_signal_{account_prefix}_{selected_date_str}",
            ):
                with st.spinner(f"'{selected_date_str}' 기준 매매 신호를 계산/저장 중..."):
                    calc_result = get_cached_signal_report(
                        country=country_code,
                        account=account_code,
                        date_str=selected_date_str,  # type: ignore
                        force_recalculate=True,
                    )
                if calc_result:
                    st.success("재계산 완료!")
                    st.rerun()

    with sub_tab_equity_history:

        if not account_code:
            st.info("활성 계좌가 없습니다. 계좌를 등록한 후 이용해주세요.")
        else:
            account_settings = get_account_settings(account_code)
            if not account_settings:
                st.warning(
                    f"'{account_code}' 계좌의 설정을 찾을 수 없습니다. 설정을 먼저 저장해주세요."
                )
                account_settings = {}

            initial_date = (account_settings.get("initial_date") if account_settings else None) or (
                datetime.now() - pd.DateOffset(months=3)
            )

            currency_str = f" ({'AUD' if country_code == 'aus' else 'KRW'})"

            start_date_str = initial_date.strftime("%Y-%m-%d")

            # 조회 종료일을 오늘과 DB에 저장된 최신 스냅샷 날짜 중 더 미래의 날짜로 설정합니다.
            # 이를 통해 미래 날짜로 백테스트/시그널조회한 데이터도 평가금액 탭에 표시될 수 있습니다.
            end_dt_candidates = [pd.Timestamp.now()]
            if sorted_dates:
                try:
                    latest_snapshot_dt = pd.to_datetime(sorted_dates[0])
                    end_dt_candidates.append(latest_snapshot_dt)
                except (ValueError, TypeError):
                    pass

            final_end_dt = max(end_dt_candidates)
            end_date_str = final_end_dt.strftime("%Y-%m-%d")

            with st.spinner("거래일 및 평가금액 데이터를 불러오는 중..."):
                all_trading_days = get_trading_days(start_date_str, end_date_str, country_code)
                trading_day_set = set()
                if not all_trading_days:
                    if country_code == "kor":
                        st.warning("거래일을 조회할 수 없습니다.")
                else:
                    trading_day_set = {pd.to_datetime(day).normalize() for day in all_trading_days}

                start_dt_obj = pd.to_datetime(start_date_str).to_pydatetime()
                end_dt_obj = pd.to_datetime(end_date_str).to_pydatetime()
                existing_equities = get_all_daily_equities(
                    country_code, account_code, start_dt_obj, end_dt_obj
                )
                equity_data_map = {
                    pd.to_datetime(e["date"]).normalize(): e for e in existing_equities
                }

                db_day_set = set(equity_data_map.keys())
                combined_days = sorted(trading_day_set.union(db_day_set))

                all_trading_days = combined_days

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
                    "date": st.column_config.DateColumn("일자", format="YYYY-MM-DD", disabled=True),
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
                    key=f"equity_editor_{account_prefix}",
                    width="stretch",
                    hide_index=True,
                    column_config=column_config,
                )

                if st.button("평가금액 저장하기", key=f"save_all_equities_{account_prefix}"):
                    with st.spinner("변경된 평가금액을 저장하는 중..."):
                        # st.data_editor의 변경 사항은 세션 상태에 저장됩니다.
                        # 전체 데이터프레임을 순회하는 대신, 변경된 행만 처리하여 불필요한 DB 업데이트를 방지합니다.
                        editor_state = st.session_state[f"equity_editor_{account_prefix}"]
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
                                    "value": changes.get("is_value", original_row.get("is_value")),
                                    "change_pct": changes.get(
                                        "is_change_pct",
                                        original_row.get("is_change_pct"),
                                    ),
                                }

                            if save_daily_equity(
                                country_code,
                                account_code,
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
        # 코인 탭: 거래 입력 대신 보유 시그널/데이터 편집만 제공 (동기화 버튼 제거)
        if country_code == "coin":
            pass

        if country_code != "coin":
            col1, col2, _ = st.columns([1, 1, 8])
            with col1:
                if st.button("BUY", key=f"add_buy_btn_{account_prefix}"):
                    show_buy_dialog(country_code)
            with col2:
                if st.button("SELL", key=f"add_sell_btn_{account_prefix}"):
                    show_sell_dialog(country_code)

        all_trades = get_all_trades(country_code, account_code)
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
                    "티커 필터",
                    options,
                    index=0,
                    key=f"coin_trades_filter_{account_prefix}",
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
                key=f"trades_editor_{account_prefix}",
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
                disabled=[
                    "date",
                    "action",
                    "ticker",
                    "name",
                    "shares",
                    "price",
                    "amount",
                    "note",
                ],
            )

            if st.button(
                "선택한 거래 삭제",
                key=f"delete_trade_btn_{account_prefix}",
                type="primary",
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

    with sub_tab_settings:
        if not account_code:
            st.info("활성 계좌가 없습니다. 계좌를 등록한 후 설정을 변경할 수 있습니다.")
            db_settings = {}
        else:
            db_settings = get_account_settings(account_code)
            if not db_settings:
                st.info(
                    "해당 계좌에 저장된 설정이 없습니다. 값을 입력 후 저장하면 계좌별 설정이 생성됩니다."
                )
                db_settings = {}

        current_capital = db_settings.get("initial_capital", 0)
        current_topn = db_settings.get("portfolio_topn")
        current_ma = db_settings.get("ma_period")
        current_replace_threshold = db_settings.get("replace_threshold")
        current_replace_weaker = db_settings.get("replace_weaker_stock")

        test_months_range = 12  # Default value
        default_date = pd.Timestamp.now() - pd.DateOffset(months=test_months_range)
        current_date = db_settings.get("initial_date", default_date)

        with st.form(key=f"settings_form_{account_prefix}"):
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
                # replace_threshold 검증 (float 가능 여부)
                try:
                    _ = float(new_replace_threshold_str)
                except Exception:
                    st.error("교체 매매 점수 임계값은 숫자여야 합니다.")
                    error = True

                if not error:
                    new_topn = int(new_topn_str)
                    new_ma = int(new_ma_str)
                    new_replace_threshold = float(new_replace_threshold_str)
                    settings_to_save = {
                        "country": country_code,
                        "initial_capital": new_capital,
                        "initial_date": pd.to_datetime(new_date).to_pydatetime(),
                        "portfolio_topn": new_topn,
                        "ma_period": new_ma,
                        "replace_weaker_stock": bool(replace_weaker_checkbox),
                        "replace_threshold": new_replace_threshold,
                    }
                    # 코인용 빗썸 임포트 기간 설정은 더 이상 사용하지 않습니다.
                    success = save_portfolio_settings(
                        country_code, settings_to_save, account=account_code
                    )

                    if success:
                        st.success("설정이 성공적으로 저장되었습니다.")
                        st.rerun()
                    else:
                        st.error("설정 저장에 실패했습니다.")


def _render_country_etf_management(country_code: str) -> None:
    render_master_etf_ui(country_code)


def render_country_tab(country_code: str, accounts: Optional[List[Dict[str, Any]]] = None):
    """국가 탭 내에서 계좌별 서브 탭과 종목 관리를 구성합니다."""

    account_entries = _prepare_account_entries(country_code, accounts)
    account_labels = [_account_label(entry) for entry in account_entries]

    tab_labels = account_labels + ["종목 관리"]
    tabs = st.tabs(tab_labels)

    for tab, entry in zip(tabs[: len(account_entries)], account_entries):
        with tab:
            with st.spinner(f"{_account_label(entry)} 계좌 데이터를 불러오는 중..."):
                _render_account_dashboard(country_code, entry)

    with tabs[-1]:
        with st.spinner("종목 관리 데이터를 불러오는 중..."):
            _render_country_etf_management(country_code)


def main():
    """MomentumETF 매매 신호 웹 UI를 렌더링합니다."""
    # 페이지 설정은 Streamlit의 첫 명령으로 실행되어야 합니다.  # noqa: E501
    st.set_page_config(page_title="MomentumETF Signal", layout="wide")

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
        pass  # type: ignore
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

    load_accounts(force_reload=False)
    account_map = {
        "kor": get_accounts_by_country("kor"),
        "aus": get_accounts_by_country("aus"),
        "coin": get_accounts_by_country("coin"),
    }

    tab_names = ["한국", "호주", "코인", "스케줄러", "설정"]
    tab_kor, tab_aus, tab_coin, tab_scheduler, tab_settings = st.tabs(tab_names)

    with tab_coin:
        render_country_tab("coin", accounts=account_map.get("coin"))

    with tab_kor:
        render_country_tab("kor", accounts=account_map.get("kor"))

    with tab_aus:
        render_country_tab("aus", accounts=account_map.get("aus"))

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

                if not error:
                    # Normalize stop loss: interpret positive value as negative threshold
                    normalized_stop = -abs(float(new_stop))
                    to_save = {
                        "MARKET_REGIME_FILTER_ENABLED": bool(new_enabled),
                        "MARKET_REGIME_FILTER_TICKER": new_ticker,
                        "MARKET_REGIME_FILTER_MA_PERIOD": int(new_ma_str),
                        "HOLDING_STOP_LOSS_PCT": normalized_stop,
                        "COOLDOWN_DAYS": int(new_cooldown_str),
                    }
                    if save_common_settings(to_save):
                        st.success("공통 설정을 저장했습니다.")
                        st.rerun()
                    else:
                        st.error("공통 설정 저장에 실패했습니다.")


if __name__ == "__main__":
    main()
