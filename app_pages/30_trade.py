from __future__ import annotations

from datetime import datetime
from collections.abc import Mapping
from functools import lru_cache

import streamlit as st
import streamlit_authenticator as stauth

from utils.stock_list_io import get_etfs
from utils.trade_store import (
    fetch_recent_trades,
    insert_trade_event,
    list_open_positions,
    delete_trade,
)


if "trade_edit_id" not in st.session_state:
    st.session_state["trade_edit_id"] = None
if "trade_editing" not in st.session_state:
    st.session_state["trade_editing"] = None
if "trade_alerts" not in st.session_state:
    st.session_state["trade_alerts"] = []
if "trade_selected_country" not in st.session_state:
    st.session_state["trade_selected_country"] = "kor"
if "trade_delete_dialog" not in st.session_state:
    st.session_state["trade_delete_dialog"] = None


def _to_plain_dict(value):
    if isinstance(value, Mapping):
        return {k: _to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(v) for v in value]
    return value


def _load_authenticator() -> stauth.Authenticate:
    raw_config = st.secrets.get("auth")
    if not raw_config:
        st.error("인증 설정(st.secrets['auth'])이 구성되지 않았습니다.")
        st.stop()

    config = _to_plain_dict(raw_config)

    credentials = config.get("credentials")
    cookie = config.get("cookie") or {}
    preauthorized = config.get("preauthorized", {})

    required_keys = {"name", "key", "expiry_days"}
    if not credentials or not cookie or not required_keys.issubset(cookie):
        st.error("인증 설정 필드가 누락되었습니다. credentials/cookie 구성을 확인하세요.")
        st.stop()

    return stauth.Authenticate(
        credentials,
        cookie.get("name"),
        cookie.get("key"),
        cookie.get("expiry_days"),
        preauthorized,
    )


def _show_notification(message: str, *, kind: str = "info", icon: str | None = None) -> None:
    toast_fn = getattr(st, "toast", None)
    if callable(toast_fn):
        kwargs: dict[str, str] = {}
        if icon is not None:
            kwargs["icon"] = icon
        toast_fn(message, **kwargs)
        return

    kind = kind.lower()
    if kind == "success":
        st.success(message)
    elif kind == "warning":
        st.warning(message)
    elif kind == "error":
        st.error(message)
    else:
        st.info(message)


def _notify(
    message: str,
    *,
    kind: str = "info",
    icon: str | None = None,
    persist: bool = False,
) -> None:
    if persist:
        st.session_state.setdefault("trade_alerts", []).append(
            {"message": message, "kind": kind, "icon": icon}
        )
        return

    _show_notification(message, kind=kind, icon=icon)


def _clear_trade_edit_state() -> None:
    st.session_state["trade_editing"] = None


@lru_cache(maxsize=16)
def _ticker_name_map(country: str) -> dict[str, str]:
    country_norm = (country or "").strip().lower()
    if not country_norm:
        return {}

    try:
        items = get_etfs(country_norm)
    except Exception:
        return {}

    mapping: dict[str, str] = {}
    for item in items:
        ticker = str(item.get("ticker") or "").upper()
        name = str(item.get("name") or "").strip()
        if ticker and name and ticker not in mapping:
            mapping[ticker] = name
    return mapping


def _resolve_ticker_name(country: str, ticker: str, stored: str | None = None) -> str:
    if stored:
        return stored

    ticker_norm = (ticker or "").strip().upper()
    if not ticker_norm:
        return ""

    mapping = _ticker_name_map(country)
    return mapping.get(ticker_norm, "")


def _format_datetime(value) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            return value
    return "-"


def _flush_persisted_alerts() -> None:
    alerts = st.session_state.get("trade_alerts") or []
    if not alerts:
        return

    for alert in alerts:
        _show_notification(
            alert.get("message", ""),
            kind=alert.get("kind", "info"),
            icon=alert.get("icon"),
        )

    st.session_state["trade_alerts"] = []


def _country_options() -> list[str]:
    try:
        return [c.lower() for c in ["kor", "aus"]]
    except Exception:
        return ["kor", "aus"]


def _show_delete_dialog(
    checked_indices: list[int],
    table_rows: list,
    username: str,
    country_filter: str,
    editor_key: str,
) -> None:
    """삭제 다이얼로그를 표시합니다."""
    # 다이얼로그 컨테이너 생성
    dialog_container = st.empty()
    with dialog_container.container():
        st.warning("선택한 트레이드를 삭제하시겠습니까?", icon="⚠️")

        # 선택된 항목 표시
        for idx in checked_indices:
            if idx >= len(table_rows):
                continue

            row = table_rows[idx]
            ticker_value = row.get("티커", "")
            name_value = row.get("종목명", "")
            if ticker_value and name_value:
                display = f"{name_value}({ticker_value})"
            else:
                display = name_value or ticker_value or "선택한 트레이드"

            detail_parts = []
            if row.get("구분"):
                detail_parts.append(row.get("구분"))
            if row.get("실행일"):
                detail_parts.append(row.get("실행일"))

            if detail_parts:
                display += f" · {" / ".join(detail_parts)}"

            st.write(f"- {display}")

        col1, col2 = st.columns(2)

        # 버튼 클릭 처리
        if col1.button("취소", key=f"cancel_delete_{country_filter}"):
            st.session_state.pop(editor_key, None)
            st.rerun()

        if col2.button("삭제", type="primary", key=f"confirm_delete_{country_filter}"):
            success_count = 0
            error_count = 0

            # 트레이드 삭제 처리
            for idx in checked_indices:
                if idx >= len(table_rows):
                    continue

                trade = table_rows[idx]
                trade_id = trade.get("_id")
                if not trade_id:
                    continue

                try:
                    # 트레이드 삭제 (하드 딜리트)
                    delete_trade(trade_id)
                    success_count += 1
                except Exception as exc:
                    error_count += 1
                    _notify(f"삭제 실패 (ID: {trade_id}): {exc}", kind="error", icon="❌")

            # 결과 알림
            if success_count > 0:
                _notify(
                    f"총 {success_count}건의 트레이드를 삭제했습니다.", kind="success", icon="✅", persist=True
                )

            if error_count > 0:
                _notify(f"{error_count}건의 트레이드 삭제에 실패했습니다.", kind="warning", icon="⚠️")

            # 상태 정리 및 새로고침
            st.session_state.pop(editor_key, None)
            st.rerun()


def _render_trade_table() -> None:
    """트레이드 테이블을 렌더링합니다."""
    pass


def _render_trade_history(username: str, country_code: str) -> None:
    """트레이드 히스토리를 표시합니다.

    Args:
        username: 현재 로그인한 사용자명
        country_code: 필터링할 국가 코드 (예: 'kor', 'aus')
    """
    _flush_persisted_alerts()

    # 국가 코드가 유효한지 확인
    if not country_code or country_code not in ["kor", "aus"]:
        st.warning("유효한 국가 코드가 아닙니다.")
        return

    # 트레이드 목록 조회 (삭제된 항목은 제외, 국가별로 필터링)
    trades = fetch_recent_trades(country_code, limit=100, include_deleted=False)  # 국가 코드로 필터링

    if not trades:
        st.info("등록된 트레이드가 없습니다.")
        return

    # 트레이드 목록을 데이터프레임으로 변환
    trade_data = []
    for trade in trades:
        trade_id = trade.get("id", "")
        country = (trade.get("country") or "").upper()
        ticker = (trade.get("ticker") or "").upper()
        action = (trade.get("action") or "").upper()
        executed_at = trade.get("executed_at")
        memo = trade.get("memo", "")
        created_by = trade.get("created_by", "")

        executed_display = "-"
        if isinstance(executed_at, datetime):
            executed_display = executed_at.strftime("%Y-%m-%d %H:%M")
        elif executed_at:
            executed_display = str(executed_at)

        trade_name = _resolve_ticker_name(country, ticker, trade.get("name")) or ticker

        # 거래 데이터 추가
        trade_data.append(
            {
                "선택": False,  # 체크박스 용도
                "No.": len(trade_data) + 1,
                "티커": ticker,
                "종목명": trade_name,
                "구분": action,
                "거래일시": executed_display,
                "메모": memo,
                "작성자": created_by,
                "id": trade_id,  # 삭제를 위한 ID
            }
        )

    # 데이터프레임 생성
    import pandas as pd

    df = pd.DataFrame(trade_data)

    # 체크박스 컬럼을 가장 앞으로 이동
    cols = ["선택"] + [col for col in df.columns if col not in ["선택", "id", "수량", "가격"]]

    # 데이터 에디터로 표시 (체크박스 활성화)
    edited_df = st.data_editor(
        df[cols],  # id 컬럼을 제외하고 표시
        column_config={
            "선택": st.column_config.CheckboxColumn("선택", default=False),
            "구분": st.column_config.SelectboxColumn("구분", options=["BUY", "SELL"], required=True),
        },
        hide_index=True,
        width="stretch",
        key=f"trade_editor_{country_code}",
        disabled=["No.", "티커", "종목명", "구분", "거래일시", "메모", "작성자"],  # 선택 컬럼만 편집 가능
    )

    # 체크박스 선택 상태 확인
    if any(edited_df["선택"]):
        # 확장 패널에 삭제 확인/취소 버튼 표시
        with st.expander("선택한 항목 삭제", expanded=True):
            st.warning("정말로 선택한 항목을 삭제하시겠습니까?")

            # 확인/취소 버튼
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ 확인", key=f"confirm_delete_{country_code}"):
                    # 원본 데이터프레임에서 선택된 행의 인덱스 가져오기
                    selected_indices = edited_df[edited_df["선택"]].index
                    deleted_count = 0

                    for idx in selected_indices:
                        # 원본 데이터프레임에서 해당 인덱스의 id 가져오기
                        trade_id = df.iloc[idx]["id"]
                        if delete_trade(trade_id):
                            deleted_count += 1

                    if deleted_count > 0:
                        st.success(f"{deleted_count}건의 트레이드를 삭제했습니다.")
                        st.rerun()
                    else:
                        st.error("트레이드 삭제에 실패했습니다.")

            with col2:
                if st.button("❌ 취소", key=f"cancel_delete_{country_code}"):
                    # 체크박스 선택 해제
                    st.rerun()


def _render_buy_form(username: str, country: str) -> None:
    holdings = list_open_positions(country or "") if country else []
    holding_tickers = {
        (pos.get("ticker") or "").strip().upper() for pos in holdings if pos.get("ticker")
    }

    key_suffix = (country or "global").strip().lower() or "global"

    with st.form(f"buy-input-form-{key_suffix}", clear_on_submit=True):
        ticker = ""
        name: str | None = None
        ticker_map = _ticker_name_map(country)

        if ticker_map:
            placeholder_option = {
                "mode": "placeholder",
                "label": " ",
                "ticker": None,
                "name": None,
            }
            available_options = [
                {
                    "mode": "predefined",
                    "ticker": ticker_code,
                    "name": ticker_name,
                    "label": f"{ticker_code} - {ticker_name}",
                }
                for ticker_code, ticker_name in sorted(ticker_map.items())
                if (ticker_code or "").strip().upper() not in holding_tickers
            ]

            selection_options = [placeholder_option] + available_options

            selected_item = st.selectbox(
                "티커",
                options=selection_options,
                index=0,
                format_func=lambda item: item.get("label", " "),
                key=f"buy-ticker-select-{key_suffix}",
            )

            mode = selected_item.get("mode")
            if mode == "predefined" and selected_item.get("ticker"):
                ticker = str(selected_item["ticker"]).strip()
                name = selected_item.get("name", "")
                if name:
                    st.caption(f"선택한 종목명: {name}")

            st.caption("※ 보유중인 종목은 리스트에서 검색되지 않습니다.")
        else:
            ticker = st.text_input(
                "티커",
                placeholder="예: 005930",
                key=f"buy-ticker-manual-only-{key_suffix}",
            )
            name = None

        executed_date = st.date_input("실행 날짜", value=datetime.today(), key=f"buy-date-{key_suffix}")
        if isinstance(executed_date, datetime):
            executed_date = executed_date.date()
        memo = st.text_area(
            "메모",
            placeholder="메모가 있다면 입력하세요.",
            key=f"buy-memo-{key_suffix}",
        )

        submitted = st.form_submit_button("저장")
        if submitted:
            if not ticker:
                st.warning("티커를 입력하세요.")
                st.stop()

            resolved_name = name or _resolve_ticker_name(country, ticker)
            executed_date_value = executed_date
            if isinstance(executed_date_value, datetime):
                executed_date_value = executed_date_value.date()

            ticker_norm = ticker.strip().upper()
            if ticker_norm in holding_tickers:
                st.session_state["buy_duplicate_warning"] = {
                    "ticker": ticker_norm,
                    "country": (country or "").upper(),
                    "name": resolved_name or "",
                }
                st.rerun()

            executed_at = datetime.combine(executed_date_value, datetime.min.time())
            try:
                insert_trade_event(
                    country=country or "",
                    ticker=ticker.strip(),
                    name=(resolved_name or "").strip(),
                    action="BUY",
                    executed_at=executed_at,
                    memo=memo,
                    created_by=username,
                )
            except Exception as exc:
                _notify(f"매수 저장 실패: {exc}", kind="error", icon="❌")
            else:
                _notify("매수 이벤트가 저장되었습니다.", kind="success", icon="✅", persist=True)
                # 매수 폼을 닫기 위해 상태 업데이트
                st.session_state["show_buy_form_kor"] = False
                st.session_state["show_buy_form_aus"] = False
                st.rerun()

    warning_state = st.session_state.get("buy_duplicate_warning")
    if warning_state:
        ticker_label = warning_state.get("ticker") or "알 수 없음"
        country_label = warning_state.get("country") or (country or "").upper()
        name_label = warning_state.get("name") or ""

        warning_message = f"{country_label} 시장에서 {ticker_label}"
        if name_label:
            warning_message += f" ({name_label})"
        warning_message += " 종목은 이미 보유 중입니다."

        st.markdown(
            f"<div style='color:#d00000;font-weight:600;margin-top:0.75rem'>{warning_message}</div>",
            unsafe_allow_html=True,
        )
        if st.button("경고 닫기", key=f"buy-duplicate-warning-confirm-{key_suffix}"):
            st.session_state["buy_duplicate_warning"] = None
            st.rerun()


def _render_sell_section(username: str, country: str) -> None:
    positions = list_open_positions(country or "") if country else []

    if not positions:
        st.info("매도 가능한 종목이 없습니다.")
        return

    key_suffix = (country or "global").strip().lower() or "global"

    option_items: list[tuple[str, str, str]] = []
    for pos in positions:
        trade_id = pos.get("id") or ""
        if not trade_id:
            continue

        ticker = pos.get("ticker", "").upper()
        position_name = pos.get("name") or _resolve_ticker_name(country or "", ticker)
        executed_display = _format_datetime(pos.get("executed_at"))
        memo_text = pos.get("memo") or ""

        label_parts: list[str] = []
        if ticker:
            label_parts.append(ticker)
        if position_name:
            label_parts.append(position_name)

        label = " - ".join(label_parts) if label_parts else ticker or trade_id
        if executed_display and executed_display != "-":
            label += f" / 최근 매수일: {executed_display}"
        if memo_text:
            label += f" / 메모: {memo_text}"

        option_items.append((trade_id, label, ticker))

    if not option_items:
        st.info("삭제할 수 있는 보유 종목이 없습니다.")
        return

    with st.form(f"sell-input-form-{key_suffix}"):
        selected_option = st.selectbox(
            "보유 종목",
            options=option_items,
            format_func=lambda item: item[1],
            key=f"sell-position-select-{key_suffix}",
        )

        executed_date = st.date_input(
            "매도 날짜", value=datetime.today(), key=f"sell-date-{key_suffix}"
        )
        executed_date = (
            executed_date.date() if isinstance(executed_date, datetime) else executed_date
        )
        memo = st.text_area(
            "메모",
            placeholder="매도 사유 등을 기록하세요.",
            key=f"sell-memo-{key_suffix}",
        )

        submitted = st.form_submit_button("선택 종목 매도 처리")

        if submitted:
            trade_id, _, ticker = selected_option
            if not ticker:
                st.warning("매도할 종목을 선택하세요.")
                st.stop()

            executed_at = datetime.combine(executed_date, datetime.min.time())
            try:
                insert_trade_event(
                    country=country or "",
                    ticker=ticker,
                    name=None,
                    action="SELL",
                    executed_at=executed_at,
                    memo=memo,
                    created_by=username,
                )
            except Exception as exc:
                _notify(f"매도 저장 실패: {exc}", kind="error", icon="❌")
            else:
                _notify(
                    f"{ticker} 종목 SELL 데이터가 생성되었습니다.",
                    kind="success",
                    icon="✅",
                    persist=True,
                )
                # 매도 폼을 닫기 위해 상태 업데이트
                st.session_state["show_sell_form_kor"] = False
                st.session_state["show_sell_form_aus"] = False
                st.rerun()


authenticator = _load_authenticator()
_flush_persisted_alerts()
name, auth_status, username = authenticator.login(key="trade_login", location="sidebar")

if auth_status is False:
    st.error("로그인에 실패했습니다. 자격 증명을 확인하세요.")
elif auth_status is None:
    st.info("로그인이 필요합니다.")
else:
    authenticator.logout(button_name="로그아웃", location="sidebar")
    current_user = username or name or "unknown"

    # 국가별 탭 생성
    country_tabs = st.tabs(["🇰🇷 한국", "🇦🇺 호주"])

    # 한국 탭
    with country_tabs[0]:
        st.session_state["trade_selected_country"] = "kor"

        # 매수/매도 버튼
        col1, col2 = st.columns([1, 1], gap="small")
        with col1:
            if st.button("➕ 매수", key="toggle-buy-form-kor", width="stretch"):
                # 매수 버튼 클릭 시 매도 폼이 열려있으면 닫기
                if st.session_state.get("show_sell_form_kor", False):
                    st.session_state["show_sell_form_kor"] = False
                st.session_state["show_buy_form_kor"] = not st.session_state.get(
                    "show_buy_form_kor", False
                )
                st.rerun()
        with col2:
            if st.button("➖ 매도", key="toggle-sell-form-kor", width="stretch"):
                # 매도 버튼 클릭 시 매수 폼이 열려있으면 닫기
                if st.session_state.get("show_buy_form_kor", False):
                    st.session_state["show_buy_form_kor"] = False
                st.session_state["show_sell_form_kor"] = not st.session_state.get(
                    "show_sell_form_kor", False
                )
                st.rerun()

        # 한국 트레이드 히스토리 표시
        _render_trade_history(current_user, "kor")

        # 매수 입력 폼 (한국)
        if st.session_state.get("show_buy_form_kor", False):
            _render_buy_form(current_user, "kor")
            if st.button("닫기", key="close-buy-form-kor"):
                st.session_state["show_buy_form_kor"] = False
                st.rerun()
            st.write("---")

        # 매도 입력 폼 (한국)
        if st.session_state.get("show_sell_form_kor", False):
            _render_sell_section(current_user, "kor")
            if st.button("닫기", key="close-sell-form-kor"):
                st.session_state["show_sell_form_kor"] = False
                st.rerun()

    # 호주 탭
    with country_tabs[1]:
        st.session_state["trade_selected_country"] = "aus"

        # 매수/매도 버튼
        col1, col2 = st.columns([1, 1], gap="small")
        with col1:
            if st.button("➕ 매수", key="toggle-buy-form-aus", width="stretch"):
                # 매수 버튼 클릭 시 매도 폼이 열려있으면 닫기
                if st.session_state.get("show_sell_form_aus", False):
                    st.session_state["show_sell_form_aus"] = False
                st.session_state["show_buy_form_aus"] = not st.session_state.get(
                    "show_buy_form_aus", False
                )
                st.rerun()
        with col2:
            if st.button("➖ 매도", key="toggle-sell-form-aus", width="stretch"):
                # 매도 버튼 클릭 시 매수 폼이 열려있으면 닫기
                if st.session_state.get("show_buy_form_aus", False):
                    st.session_state["show_buy_form_aus"] = False
                st.session_state["show_sell_form_aus"] = not st.session_state.get(
                    "show_sell_form_aus", False
                )
                st.rerun()

        # 호주 트레이드 히스토리 표시
        _render_trade_history(current_user, "aus")

        # 매수 입력 폼 (호주)
        if st.session_state.get("show_buy_form_aus", False):
            _render_buy_form(current_user, "aus")
            if st.button("닫기", key="close-buy-form-aus"):
                st.session_state["show_buy_form_aus"] = False
                st.rerun()
            st.write("---")

        # 매도 입력 폼 (호주)
        if st.session_state.get("show_sell_form_aus", False):
            _render_sell_section(current_user, "aus")
            if st.button("닫기", key="close-sell-form-aus"):
                st.session_state["show_sell_form_aus"] = False
                st.rerun()
