from __future__ import annotations

import subprocess
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from app_pages.account_page import render_account_deleted_page, render_account_setup_page
from config import BUCKET_REVERSE_MAPPING
from utils.pool_rank_storage import fetch_latest_pool_rank
from utils.recommendations import recommendations_to_dataframe
from utils.ui import create_loading_status, format_relative_time, render_recommendation_table

try:
    from utils.data_loader import fetch_au_quoteapi_snapshot, fetch_naver_etf_inav_snapshot
except Exception:  # pragma: no cover
    fetch_naver_etf_inav_snapshot = None  # type: ignore
    fetch_au_quoteapi_snapshot = None  # type: ignore

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _format_updated_at_kst(value: Any) -> str | None:
    if value is None:
        return None

    ts: pd.Timestamp | None = None
    if isinstance(value, datetime):
        ts = pd.Timestamp(value)
    else:
        try:
            ts = pd.to_datetime(value)
        except Exception:
            return str(value)

    if ts is None:
        return None

    try:
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            ts = ts.tz_localize("UTC")
        ts = ts.tz_convert("Asia/Seoul")
    except Exception:
        # fallback: timezone conversion 실패 시 원본 표시
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


@st.cache_data(ttl=30, show_spinner=False)
def _load_pool_ranking(pool_id: str) -> tuple[list[dict[str, Any]] | None, str | None, str | None, str | None]:
    pool_norm = (pool_id or "").strip().lower()
    if not pool_norm:
        return None, "pool_id가 필요합니다.", None, None

    # 1) MongoDB 우선
    try:
        snapshot = fetch_latest_pool_rank(pool_norm)
    except Exception as exc:
        snapshot = None
        mongo_error = str(exc)
    else:
        mongo_error = None

    if snapshot:
        rows = snapshot.get("rows") or []
        country = str(snapshot.get("country_code") or snapshot.get("country") or "").strip().lower() or "kor"
        updated_by = str(snapshot.get("updated_by") or "").strip()
        updated = snapshot.get("updated_at") or snapshot.get("created_at")
        updated_at = _format_updated_at_kst(updated)
        return rows, updated_at, country, updated_by

    if mongo_error:
        return None, f"Mongo 랭킹 조회 실패: {mongo_error}", None, None
    return (
        None,
        "Mongo 랭킹 결과를 찾을 수 없습니다. `python rank.py <pool_id>`를 실행해 Mongo 저장을 생성하세요.",
        None,
        None,
    )


def _render_ranking_view(pool_id: str, selected_bucket: str = "전체") -> None:
    rows, updated_at, country, updated_by = _load_pool_ranking(pool_id)
    if rows is None:
        st.error(updated_at or "랭킹 데이터를 불러오지 못했습니다.")
        return

    if not rows:
        st.info("표시할 랭킹 데이터가 없습니다.")
        return

    country_norm = (country or "").strip().lower()
    rows_for_view = [dict(row) for row in rows]
    price_source = "Local Cache"

    # 추천 화면과 동일하게 한국 종목은 렌더링 시점에 Naver 실시간값으로 오버레이한다.
    if country_norm in ("kor", "kr") and fetch_naver_etf_inav_snapshot is not None:
        try:
            tickers = [r.get("ticker") for r in rows_for_view if r.get("ticker")]
            realtime_data = fetch_naver_etf_inav_snapshot(tickers)
            if realtime_data:
                for row in rows_for_view:
                    ticker = str(row.get("ticker") or "").strip().upper()
                    rt = realtime_data.get(ticker)
                    if not rt:
                        continue
                    if rt.get("nowVal") is not None:
                        row["price"] = float(rt["nowVal"])
                    if rt.get("changeRate") is not None:
                        row["daily_pct"] = float(rt["changeRate"])
                    if rt.get("nav") is not None:
                        row["nav_price"] = float(rt["nav"])
                    if rt.get("deviation") is not None:
                        row["price_deviation"] = float(rt["deviation"])
                    if rt.get("itemname"):
                        row["name"] = str(rt["itemname"])
                price_source = "Naver"
        except Exception:
            pass
    elif country_norm == "au" and fetch_au_quoteapi_snapshot is not None:
        try:
            tickers = [r.get("ticker") for r in rows_for_view if r.get("ticker")]
            realtime_data = fetch_au_quoteapi_snapshot(tickers)
            if realtime_data:
                for row in rows_for_view:
                    ticker = str(row.get("ticker") or "").strip().upper()
                    rt = realtime_data.get(ticker)
                    if not rt:
                        continue
                    if rt.get("nowVal") is not None:
                        row["price"] = float(rt["nowVal"])
                    if rt.get("changeRate") is not None:
                        row["daily_pct"] = float(rt["changeRate"])
                    elif rt.get("prevClose") is not None and rt.get("nowVal") is not None:
                        prev_close = float(rt["prevClose"])
                        now_val = float(rt["nowVal"])
                        if prev_close > 0:
                            row["daily_pct"] = ((now_val / prev_close) - 1.0) * 100.0
                price_source = "QuoteAPI"
        except Exception:
            pass

    df = recommendations_to_dataframe(country or "kor", rows_for_view)

    bucket_name = str(selected_bucket or "전체").strip()
    if bucket_name and bucket_name != "전체":
        bucket_id = BUCKET_REVERSE_MAPPING.get(bucket_name)
        if bucket_id is None:
            st.error(f"알 수 없는 버킷입니다: {bucket_name}")
            return
        df = df[df["bucket"] == bucket_id].copy()

    update_caption = None
    if updated_at:
        updated_rel = format_relative_time(updated_at)
        updated_display = f"{updated_at}{updated_rel}" if updated_rel else str(updated_at)
        if updated_by:
            updated_display = f"{updated_display}, {updated_by}"

        if country_norm in ("kor", "kr", "au"):
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            now_rel = format_relative_time(now_str)
            now_display = f"{now_str}{now_rel}" if now_rel else now_str
            update_caption = (
                f"랭킹 데이터 업데이트: {updated_display}  \n가격 데이터 업데이트: {now_display}, {price_source}"
            )
        else:
            update_caption = f"랭킹 데이터 업데이트: {updated_display}"

    # 추천 화면과 최대한 동일하게 렌더링하되, 불가 컬럼(#, 상태, 보유일, 문구)은 숨긴다.
    visible_cols = [
        "버킷",
        "티커",
        "종목명",
        "일간(%)",
        "현재가",
        "괴리율",
        "Nav",
        "1주(%)",
        "2주(%)",
        "1달(%)",
        "3달(%)",
        "6달(%)",
        "12달(%)",
        "고점대비",
        "점수",
        "RSI",
        "지속",
        "추세(3달)",
    ]
    render_recommendation_table(
        df=df,
        country_code=country or "kor",
        grouped_by_bucket=False,
        visible_columns=[c for c in visible_cols if c in df.columns],
        height=745,  # 최대 20행(헤더 제외) 표시
    )
    if update_caption:
        st.caption(update_caption)


def _render_pool_manual_actions(pool_id: str) -> None:
    """수동 액션 실행 (랭킹) 영역을 렌더링합니다."""
    st.subheader("🤖 수동 액션 실행")

    if st.button("🚀 랭크 시스템 즉시 실행", type="primary", width="stretch", key=f"btn_rank_{pool_id}"):
        try:
            subprocess.Popen(["python", "rank.py", pool_id])
            st.success(f"✅ `{pool_id}` 랭크 시스템 실행을 시작했습니다. (배경에서 처리가 완료됩니다)")
        except Exception as e:
            st.error(f"⚠️ 실행 시작 오류: {e}")


def render_pool_page(pool_id: str, view_mode: str | None = None, selected_bucket: str = "전체", loading=None) -> None:
    mode = view_mode or "1. 랭킹"
    owns_loading = loading is None
    loading = loading or create_loading_status()

    try:
        if mode == "2. 종목 관리":
            loading.update(f"{pool_id.upper()} 종목 관리 화면 준비")
            render_account_setup_page(pool_id)
            return

        if mode == "3. 삭제된 종목":
            loading.update(f"{pool_id.upper()} 삭제 종목 화면 준비")
            render_account_deleted_page(pool_id)
            return

        loading.update(f"{pool_id.upper()} 랭킹 데이터 조회")
        _render_ranking_view(pool_id, selected_bucket=selected_bucket)
        st.divider()
        _render_pool_manual_actions(pool_id)
    finally:
        if owns_loading:
            loading.clear()


__all__ = ["render_pool_page"]
