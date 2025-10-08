from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping

import streamlit as st
import streamlit_authenticator as stauth

from utils.account_registry import list_available_accounts
from utils.trade_store import delete_account_trades
from utils.settings_loader import get_account_settings

SETTINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "settings" / "account"


def _to_plain_dict(value):
    if isinstance(value, Mapping):
        return {k: _to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
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


def _normalize_account_id(value: str) -> str:
    return (value or "").strip().lower()


def _delete_account_file(account_id: str) -> bool:
    target_path = SETTINGS_DIR / f"{account_id}.json"
    if not target_path.exists():
        return False
    target_path.unlink()
    return True


def render_account_delete_page() -> None:
    st.title("🗑️ 계정 삭제")
    st.caption("계정 설정 파일과 해당 계정의 모든 거래 기록(trades)을 완전히 삭제합니다.")
    st.warning("이 작업은 되돌릴 수 없습니다. 필요한 경우 백업을 먼저 수행하세요.")

    authenticator = _load_authenticator()
    _, auth_status, _ = authenticator.login(key="delete_login", location="sidebar")

    if not auth_status:
        st.warning("이 페이지에 접근하려면 로그인이 필요합니다.")
        return

    st.sidebar.write("")
    authenticator.logout(button_name="로그아웃", location="sidebar")

    try:
        available_accounts = list_available_accounts()
    except Exception as exc:
        st.error(f"계정 목록을 불러오지 못했습니다: {exc}")
        return

    if not available_accounts:
        st.info("삭제할 계정이 없습니다. `data/settings/account/*.json` 파일을 확인하세요.")
        return

    with st.form("account_delete_form"):
        account_to_delete = st.selectbox("삭제할 계정 ID", available_accounts)
        confirmation = st.text_input(
            "계정 ID를 다시 입력해 삭제를 확인하세요",
            placeholder="계정 ID를 입력하세요",
        )
        submitted = st.form_submit_button("계정 삭제", use_container_width=True)

    if not submitted:
        return

    account_id = _normalize_account_id(account_to_delete)
    confirmation_id = _normalize_account_id(confirmation)

    if not confirmation_id:
        st.warning("삭제 확인을 위해 계정 ID를 다시 입력하세요.")
        return

    if account_id != confirmation_id:
        st.warning("입력한 확인용 계정 ID가 선택한 계정과 일치하지 않습니다.")
        return

    try:
        trade_result = delete_account_trades(account_id)
    except Exception as exc:
        st.error(f"trades 컬렉션에서 데이터를 삭제하는 중 오류가 발생했습니다: {exc}")
        return

    file_removed = False
    try:
        file_removed = _delete_account_file(account_id)
    except Exception as exc:
        st.error(f"계정 설정 파일을 삭제하는 중 오류가 발생했습니다: {exc}")
        return

    # 설정 캐시 무효화
    try:
        get_account_settings.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    st.success(
        "계정 삭제가 완료되었습니다." f"\n- trades 삭제: {trade_result['deleted']}건" f"\n- 설정 파일 삭제: {'성공' if file_removed else '파일 없음'}"
    )
    st.info("사이드바 메뉴에서 다른 페이지로 이동하거나 새로고침하여 변경 사항을 반영하세요.")


render_account_delete_page()
