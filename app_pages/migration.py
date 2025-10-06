from __future__ import annotations

from pathlib import Path

import streamlit as st

import streamlit_authenticator as stauth

from collections.abc import Mapping

from utils.account_registry import list_available_accounts
from utils.trade_store import migrate_account_id
from utils.settings_loader import get_account_settings

SETTINGS_DIR = Path(__file__).resolve().parents[1] / "settings" / "account"


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


def _rename_account_file(old_account_id: str, new_account_id: str) -> None:
    old_path = SETTINGS_DIR / f"{old_account_id}.json"
    new_path = SETTINGS_DIR / f"{new_account_id}.json"

    if not old_path.exists():
        raise FileNotFoundError(f"계정 설정 파일을 찾을 수 없습니다: {old_path}")
    if new_path.exists():
        raise FileExistsError(f"이미 존재하는 계정 ID입니다: {new_account_id}")

    new_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.rename(new_path)


def render_migration_page() -> None:
    st.title("🛠️ 계정 ID 마이그레이션")
    st.caption("계정 설정 파일명을 변경하고 `trades` 컬렉션의 계정 ID도 함께 갱신합니다.")

    authenticator = _load_authenticator()
    name, auth_status, username = authenticator.login(key="migration_login", location="sidebar")

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
        st.info("마이그레이션할 계정이 없습니다. `data/settings/account/*.json` 파일을 추가하세요.")
        return

    with st.form("account_migration_form"):
        source_account = st.selectbox("변경할 계정 ID", available_accounts)
        target_account_input = st.text_input("새 계정 ID", placeholder="예: new_account")
        submitted = st.form_submit_button("확인", use_container_width=True)

    if not submitted:
        return

    source_account_id = _normalize_account_id(source_account)
    target_account_id = _normalize_account_id(target_account_input)

    if not target_account_id:
        st.warning("새 계정 ID를 입력하세요.")
        return

    if target_account_id == source_account_id:
        st.warning("새 계정 ID가 기존 ID와 동일합니다.")
        return

    try:
        _rename_account_file(source_account_id, target_account_id)
    except FileExistsError as exc:
        st.error(str(exc))
        return
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"계정 설정 파일 이름 변경에 실패했습니다: {exc}")
        return

    # 기존 설정 캐시 무효화
    try:
        get_account_settings.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        migration_result = migrate_account_id(source_account_id, target_account_id)
    except Exception as exc:
        st.error(f"trades 컬렉션 계정 갱신 중 오류가 발생했습니다: {exc}")
        return

    st.success(
        "계정 ID 마이그레이션이 완료되었습니다."
        f"\n- 설정 파일: {source_account_id}.json → {target_account_id}.json"
        f"\n- trades 업데이트: {migration_result['modified']}건"
        f"\n- legacy country 필드 업데이트: {migration_result['legacy_country_updated']}건"
    )
    st.info("변경 사항을 적용하려면 관리자/추천 페이지를 새로고침하세요.")


render_migration_page()
