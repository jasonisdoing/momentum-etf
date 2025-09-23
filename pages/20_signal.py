import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

try:
    from croniter import croniter
except ImportError:
    croniter = None
try:
    import pytz
except ImportError:
    pytz = None

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals import (
    calculate_benchmark_comparison,
    generate_signal_report,
    get_next_trading_day,
)
from utils.account_registry import get_accounts_by_country, load_accounts
from utils.data_loader import PykrxDataUnavailable, get_trading_days
from utils.db_manager import (
    get_available_snapshot_dates,
    get_signal_report_from_db,
    save_signal_report_to_db,
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
        return pd.Timestamp.now().normalize().strftime("%Y-%m-%d")

    now_local = _get_local_now(country_code)
    if not now_local:
        return pd.Timestamp.now().normalize().strftime("%Y-%m-%d")

    local_today = now_local.date()
    today_str = pd.Timestamp(local_today).strftime("%Y-%m-%d")

    close_time_dt = datetime.strptime(MARKET_DISPLAY_SETTINGS[country_code]["close"], "%H:%M")
    close_time_with_buffer = (close_time_dt + pd.Timedelta(minutes=30)).time()
    lookahead_end = pd.Timestamp(local_today) + pd.Timedelta(days=14)

    try:
        upcoming_days = get_trading_days(
            today_str, lookahead_end.strftime("%Y-%m-%d"), country_code
        )
    except Exception:
        upcoming_days = []

    if not upcoming_days:
        return today_str

    is_trading_today = any(d.date() == local_today for d in upcoming_days)
    if is_trading_today and now_local.time() < close_time_with_buffer:
        return today_str

    next_day = next((d for d in upcoming_days if d.date() > local_today), None)
    if not next_day:
        fallback = get_next_trading_day(
            country_code, pd.Timestamp(local_today) + pd.Timedelta(days=1)
        )
        return pd.Timestamp(fallback).strftime("%Y-%m-%d")

    return pd.Timestamp(next_day).strftime("%Y-%m-%d")


def _ensure_header_has_date(header: str, date: datetime) -> str:
    if not header or ("년" in header and "월" in header):
        return header
    date_display = f"{date.year}년 {date.month}월 {date.day}일"
    prefix = f"{date_display} | "
    return prefix + header if not header.strip().startswith(prefix) else header


def _format_korean_datetime(dt: datetime) -> str:
    weekday_map = ["월", "화", "수", "목", "금", "토", "일"]
    ampm_str = "오후" if dt.hour >= 12 else "오전"
    hour12 = dt.hour - 12 if dt.hour > 12 else (12 if dt.hour == 0 else dt.hour)
    return f"{dt.strftime('%Y년 %m월 %d일')}({weekday_map[dt.weekday()]}) {ampm_str} {hour12}시 {dt.minute:02d}분"


@st.cache_data(ttl=600)
def get_cached_benchmark_comparison(
    country: str, date_str: str, account: str
) -> Optional[List[Dict[str, Any]]]:
    try:
        return calculate_benchmark_comparison(country, account, date_str)
    except PykrxDataUnavailable as exc:
        return [
            {
                "name": "벤치마크",
                "error": f"데이터 없음 ({exc.start_dt.strftime('%Y-%m-%d')}~{exc.end_dt.strftime('%Y-%m-%d')})",
            }
        ]


def get_next_schedule_time_str(country_code: str) -> str:
    if not croniter or not pytz:
        return "스케줄러 라이브러리가 설치되지 않았습니다."

    cron_key = f"SCHEDULE_CRON_{country_code.upper()}"
    default_cron = {"kor": "10 18 * * 1-5", "aus": "10 18 * * 1-5", "coin": "5 0 * * *"}.get(
        country_code, "0 * * * *"
    )
    cron_value = os.environ.get(cron_key, default_cron)
    tz_str = {"kor": "Asia/Seoul", "aus": "Asia/Seoul", "coin": "Asia/Seoul"}.get(
        country_code, "Asia/Seoul"
    )

    try:
        local_tz = pytz.timezone(tz_str)
        cron = croniter(cron_value, datetime.now(local_tz))
        return _format_korean_datetime(cron.get_next(datetime))
    except Exception as e:
        return f"다음 실행 시간 계산 중 오류 발생: {e}"


def get_cached_signal_report(
    country: str,
    account: str,
    date_str: str,
    force_recalculate: bool = False,
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Optional[tuple[Any, Any, Any]]:
    try:
        report_date = pd.to_datetime(date_str).to_pydatetime()
    except (ValueError, TypeError):
        st.error(f"잘못된 날짜 형식입니다: {date_str}")
        return None

    if not force_recalculate:
        report_from_db = get_signal_report_from_db(country, account, report_date)
        if report_from_db:
            return (
                report_from_db.get("header_line"),
                report_from_db.get("headers"),
                report_from_db.get("rows"),
            )
        return None

    try:
        with st.spinner(f"'{date_str}' 매매 신호를 다시 계산하는 중..."):
            new_report_tuple = generate_signal_report(
                country, account, date_str, prefetched_data=prefetched_data
            )
            if new_report_tuple:
                header_line, headers, rows, _, _ = new_report_tuple
                new_report = (header_line, headers, rows)
                save_signal_report_to_db(country, account, report_date, new_report)
                return new_report
            return None
    except ValueError as e:
        if str(e).startswith("PRICE_FETCH_FAILED:"):
            st.error(f"{str(e).split(':', 1)[1]} 종목의 가격을 가져올 수 없습니다. 다시 시도하세요.")
        else:
            st.error(f"'{date_str}' 신호 계산 중 오류가 발생했습니다: {e}")
        return None
    except Exception as e:
        st.error(f"'{date_str}' 신호 계산 중 오류가 발생했습니다. 자세한 내용은 콘솔 로그를 확인해주세요. {e}")
        return None


def style_returns(val) -> str:
    color = (
        "red"
        if isinstance(val, (int, float)) and val > 0
        else ("blue" if isinstance(val, (int, float)) and val < 0 else "")
    )
    return f"color: {color}"


@st.cache_data
def get_cached_etfs(country_code: str) -> List[Dict[str, Any]]:
    return get_etfs(country_code) or []


def _display_status_report_df(df: pd.DataFrame, country_code: str):
    etfs_data = get_cached_etfs(country_code)
    meta_df = (
        pd.DataFrame(etfs_data) if etfs_data else pd.DataFrame(columns=["ticker", "이름", "category"])
    )
    for col in ["ticker", "name", "category"]:
        if col not in meta_df.columns:
            meta_df[col] = None
    meta_df = meta_df[["ticker", "name", "category"]].rename(columns={"name": "이름"})

    if country_code == "aus":
        is_meta = pd.DataFrame([{"ticker": "IS", "이름": "International Shares", "category": "기타"}])
        meta_df = pd.concat([meta_df, is_meta], ignore_index=True)

    df_merged = pd.merge(df, meta_df, left_on="티커", right_on="ticker", how="left")
    df_merged["이름"] = df_merged["이름"].fillna(df_merged["티커"])
    df_merged["category"] = df_merged.get(
        "category", pd.Series(index=df_merged.index, dtype=str)
    ).fillna("")

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

    numeric_cols = ["현재가", "일간수익률", "보유수량", "금액", "누적수익률", "비중", "점수"]
    for col in numeric_cols:
        if col in df_display.columns:
            df_display[col] = pd.to_numeric(df_display[col], errors="coerce")

    if "#" in df_display.columns:
        df_display = df_display.set_index("#")

    styler = df_display.style
    for col in ["일간수익률", "누적수익률"]:
        if col in df_display.columns:
            styler = styler.map(style_returns, subset=[col])

    formats = {
        "일간수익률": "{:+.2f}%",
        "누적수익률": "{:+.2f}%",
        "비중": "{:.1f}%",
        "점수": lambda val: f"{val * 100:.1f}" if pd.notna(val) else "-",
    }
    if country_code in ["kor", "coin"]:
        formats.update({"현재가": "{:,.0f}", "금액": "{:,.0f}"})
    if country_code == "aus":
        formats.update({"현재가": "{:,.2f}", "금액": "{:,.2f}"})
    if country_code == "coin" and "보유수량" in df_display.columns:
        formats["보유수량"] = "{:.8f}"
    styler = styler.format(formats, na_rep="-")

    st.dataframe(
        styler,
        width="stretch",
        height=(16 * 35 + 3),
        column_config={
            "이름": st.column_config.TextColumn("종목명", width=200),
            "category": st.column_config.TextColumn("카테고리", width=100),
            "상태": st.column_config.TextColumn(width="small"),
            "매수일자": st.column_config.TextColumn(width="small"),
            "보유": st.column_config.TextColumn(width=40),
            "보유수량": st.column_config.NumberColumn(
                format="%.8f" if country_code == "coin" else "%d"
            ),
            "일간수익률": st.column_config.TextColumn(width="small"),
            "금액": st.column_config.TextColumn(width="small"),
            "누적수익률": st.column_config.TextColumn(width="small"),
            "비중": st.column_config.TextColumn(width=50),
            "지속": st.column_config.TextColumn(width=50),
            "문구": st.column_config.TextColumn("문구", width="large"),
        },
    )


def _prepare_account_entries(
    country_code: str, accounts: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    entries = [
        entry
        for entry in accounts or []
        if isinstance(entry, dict) and entry.get("is_active", True)
    ]
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
    return str(entry.get("display_name") or entry.get("account") or "계좌")


def _account_prefix(country_code: str, account_code: Optional[str]) -> str:
    return f"{country_code}_{account_code or 'default'}"


def render_signal_dashboard(
    country_code: str,
    account_entry: Dict[str, Any],
    prefetched_trading_days: Optional[List[pd.Timestamp]] = None,
):
    """지정된 계좌의 시그널 탭을 렌더링합니다."""
    account_code = account_entry.get("account")
    account_prefix = _account_prefix(country_code, account_code)

    if not account_code:
        st.info("활성 계좌가 없습니다. 계좌를 등록한 후 이용해주세요.")
        return

    raw_dates = get_available_snapshot_dates(country_code, account=account_code)
    sorted_dates = sorted(set(raw_dates), reverse=True)

    today_ts = (
        pd.Timestamp(_get_local_now(country_code))
        if country_code != "coin" and _get_local_now(country_code)
        else pd.Timestamp.now()
    )
    today_str = pd.Timestamp(today_ts.date()).strftime("%Y-%m-%d")
    target_date_str = _get_status_target_date_str(country_code)

    date_options = []
    if target_date_str:
        date_options.append(target_date_str)
    for d in sorted_dates:
        if d and d not in date_options:
            date_options.append(d)

    if country_code != "coin" and date_options:
        trading_day_set = {
            d.strftime("%Y-%m-%d")
            for d in (
                prefetched_trading_days
                or get_trading_days(min(date_options), max(date_options), country_code)
            )
        }
        date_options = [d for d in date_options if d in trading_day_set]

    option_labels = {}
    if date_options:
        if country_code == "coin":
            option_labels[target_date_str] = f"{target_date_str} (오늘)"
        else:
            option_labels[
                target_date_str
            ] = f"{target_date_str} ({'오늘' if target_date_str == today_str else '다음 거래일'})"
            if today_str in date_options and today_str != target_date_str:
                option_labels.setdefault(today_str, f"{today_str} (오늘)")

    if not date_options:
        if not sorted_dates:
            st.warning(f"[{country_code.upper()}] 국가의 포트폴리오 데이터를 DB에서 찾을 수 없습니다.")
            st.info(
                "먼저 '10_assets' 페이지에서 거래 내역을 추가해주세요."
                if country_code != "coin"
                else "빗썸 거래내역 동기화가 필요할 수 있습니다."
            )
        else:
            st.warning(f"[{country_code.upper()}] 표시에 유효한 시그널 데이터가 없습니다.")
        return

    selected_date_str = st.selectbox(
        "조회 날짜",
        date_options,
        format_func=lambda d: option_labels.get(d, d),
        key=f"signal_date_select_{account_prefix}",
    )

    result = get_cached_signal_report(
        country=country_code,
        account=account_code,
        date_str=selected_date_str,
        force_recalculate=False,
    )

    if result:
        header_line, headers, rows = result
        header_main, warning_html = (
            header_line.split("<br>", 1)
            if isinstance(header_line, str) and "<br>" in header_line
            else (header_line or "", None)
        )
        header_display = _ensure_header_has_date(
            header_main, pd.to_datetime(selected_date_str).to_pydatetime()
        )
        if header_display:
            st.markdown(
                f'<div class="status-summary">{header_display.replace("$", "&#36;")}</div>',
                unsafe_allow_html=True,
            )

        if rows and headers and len(rows[0]) != len(headers):
            st.error(f"데이터 형식 오류: 컬럼 수({len(headers)})와 데이터 수({len(rows[0])})가 일치하지 않습니다.")
        else:
            df = pd.DataFrame(rows, columns=headers)
            _display_status_report_df(df, country_code)
            if warning_html:
                st.markdown(warning_html, unsafe_allow_html=True)

        benchmark_results = get_cached_benchmark_comparison(
            country_code, selected_date_str, account_code
        )
        if benchmark_results:
            data_for_df = []
            for res in benchmark_results:
                row_data = {
                    "티커": res.get("ticker", "-"),
                    "벤치마크": res.get("name", "N/A"),
                    "누적수익률": res.get("cum_ret_pct") if not res.get("error") else res.get("error"),
                    "초과수익률": res.get("excess_return_pct") if not res.get("error") else "-",
                }
                data_for_df.append(row_data)
            st.dataframe(
                pd.DataFrame(data_for_df),
                hide_index=True,
                width="stretch",
                column_config={
                    "누적수익률": st.column_config.NumberColumn(format="%.2f%%"),
                    "초과수익률": st.column_config.NumberColumn(format="%+.2f%%"),
                },
            )
    else:
        if selected_date_str == target_date_str:
            st.info(
                f"**{selected_date_str}** 날짜의 매매 신호가 아직 계산되지 않았습니다.\n\n다음 예상 실행 시간: **{get_next_schedule_time_str(country_code)}**"
            )
        else:
            st.info(f"'{selected_date_str}' 날짜의 매매 신호 데이터가 없습니다.")

    if st.button("이 날짜 다시 계산하기", key=f"recalc_signal_{account_prefix}_{selected_date_str}"):
        if get_cached_signal_report(
            country=country_code,
            account=account_code,
            date_str=selected_date_str,
            force_recalculate=True,
        ):
            st.success("재계산 완료!")
            st.rerun()


def main():
    """매매 신호 페이지를 렌더링합니다."""
    st.title("📈 매매 신호 (Signal)")

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
            body {
                font-family: 'Noto Sans KR', sans-serif;
            }
            .block-container {
                max-width: 100%;
                padding-top: 1rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    print("[MAIN] 계좌 정보 로딩 시작...")
    start_time = time.time()
    with st.spinner("계좌 정보 로딩 중..."):
        load_accounts(force_reload=False)
        account_map = {
            "kor": get_accounts_by_country("kor"),
            "aus": get_accounts_by_country("aus"),
            "coin": get_accounts_by_country("coin"),
        }
    duration = time.time() - start_time
    print(f"[MAIN] 계좌 정보 로딩 완료 ({duration:.2f}초)")

    tab_kor, tab_aus, tab_coin = st.tabs(["한국", "호주", "코인"])

    for country_code, tab in [("kor", tab_kor), ("aus", tab_aus), ("coin", tab_coin)]:
        with tab:
            account_entries = _prepare_account_entries(country_code, account_map.get(country_code))
            if len(account_entries) == 1 and account_entries[0].get("account") is None:
                render_signal_dashboard(country_code, account_entries[0])
            else:
                account_labels = [_account_label(entry) for entry in account_entries]
                account_tabs = st.tabs(account_labels)
                for account_tab, entry in zip(account_tabs, account_entries):
                    with account_tab:
                        render_signal_dashboard(country_code, entry)


if __name__ == "__main__":
    main()
