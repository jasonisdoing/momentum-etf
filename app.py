from __future__ import annotations

from typing import Any, Callable, Dict, Optional, List, Tuple
from datetime import datetime

import streamlit as st
import pandas as pd

from utils.notification import APP_VERSION

from app_pages.account_page import render_account_page
from logic.recommend.market import (
    get_market_regime_status_info,
    get_market_regime_aux_status_infos,
)
from utils.settings_loader import get_market_regime_settings, load_common_settings
from utils.data_loader import fetch_ohlcv
from utils.cache_utils import get_cache_path

try:  # pragma: no cover - yfinance는 환경에 따라 설치되지 않을 수 있음
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

from utils.account_registry import (
    get_icon_fallback,
    load_account_configs,
    pick_default_account,
)


def _build_account_page(page_cls: Callable[..., object], account: Dict[str, Any]):
    account_id = account["account_id"]
    icon = account.get("icon") or get_icon_fallback(account.get("country_code", ""))

    def _render(account_key: str = account_id) -> None:
        render_account_page(account_key)

    return page_cls(
        _render,
        title=account["name"],
        icon=icon,
        url_path=account_id,
    )


def _render_home_page() -> None:
    st.title("Momentum ETF")
    st.text(f"버전: Alpha-{APP_VERSION}")
    st.caption("서비스 진입점입니다. 좌측 메뉴에서 계정을 선택하세요.")

    _, default_ma_period, _, _, risk_off_ratio_common = get_market_regime_settings()

    risk_off_ratio = risk_off_ratio_common
    try:
        account_configs = load_account_configs()
        default_account = pick_default_account(account_configs)
    except Exception:
        default_account = None

    if default_account:
        account_settings = default_account.get("settings") or {}
        strategy_cfg = account_settings.get("strategy") or {}
        static_cfg = strategy_cfg.get("static") or {}
        tuning_cfg = strategy_cfg.get("tuning") or {}
        ratio_candidate = (
            static_cfg.get("MARKET_REGIME_RISK_OFF_EQUITY_RATIO")
            or tuning_cfg.get("MARKET_REGIME_RISK_OFF_EQUITY_RATIO")
            or strategy_cfg.get("MARKET_REGIME_RISK_OFF_EQUITY_RATIO")
        )
        try:
            ratio_candidate_int = int(ratio_candidate)
        except (TypeError, ValueError):
            ratio_candidate_int = None
        if ratio_candidate_int is not None:
            risk_off_ratio = ratio_candidate_int

    ratio_text = "?" if risk_off_ratio is None else str(risk_off_ratio)
    ma_period_input = st.number_input(
        f"레짐 이동평균 기간 (1-200) - 시스템은 {default_ma_period} 이용하고 있습니다. 경고 기간에는 현금의 {ratio_text}% 만 투자합니다",
        min_value=1,
        max_value=200,
        value=int(default_ma_period),
        step=1,
        help="전략 및 보조 지수 표시용 이동평균 기간을 설정합니다.",
    )
    ma_period = int(ma_period_input)
    try:
        common_settings = load_common_settings() or {}
    except Exception:
        common_settings = {}

    cache_start_cfg = common_settings.get("CACHE_START_DATE")
    cache_start_text = str(cache_start_cfg) if cache_start_cfg else "-"

    with st.spinner("시장 레짐 정보를 계산 중입니다..."):
        regime_info, regime_message = get_market_regime_status_info(ma_period_override=ma_period)
        aux_infos = get_market_regime_aux_status_infos(ma_period_override=ma_period)

    with st.container():
        st.markdown("**시장 레짐 요약**")
        if regime_info is None:
            st.markdown(regime_message, unsafe_allow_html=True)
        else:
            delay_days = int(common_settings.get("MARKET_REGIME_FILTER_DELAY_DAY", 0) or 0)
            st.caption(
                "디버그: "
                f"Ticker={regime_info.get('ticker')} | "
                f"MA Period={regime_info.get('ma_period')} | "
                f"Delay Days={delay_days} | "
                f"Last Date={regime_info.get('last_risk_off_start')} -> {regime_info.get('last_risk_off_end')} | "
                f"Divergence={regime_info.get('proximity_pct'):+.3f}%"
            )

            debug_lines = []
            ticker = regime_info.get("ticker")
            ma_period_debug = int(regime_info.get("ma_period") or ma_period)
            country = regime_info.get("country") or "us"

            try:
                df_debug = fetch_ohlcv(ticker, country=country, months_range=[12, 0], cache_country="common")
            except Exception as exc:  # pragma: no cover - 진단용 출력
                debug_lines.append(f"fetch_ohlcv 오류: {exc}")
                df_debug = None

            if df_debug is not None and not df_debug.empty:
                df_debug = df_debug.sort_index()
                try:
                    df_debug.index = pd.to_datetime(df_debug.index).normalize()
                except Exception:
                    pass
                df_debug = df_debug[~df_debug.index.duplicated(keep="last")]

                column_lookup: Dict[str, Any] = {}
                if isinstance(df_debug.columns, pd.MultiIndex):
                    for col in df_debug.columns:
                        if isinstance(col, tuple) and len(col) > 0 and col[0]:
                            column_lookup.setdefault(str(col[0]), col)
                else:
                    column_lookup = {str(col): col for col in df_debug.columns}

                price_col = None
                for candidate in ("Close", "Adj Close", "Price"):
                    if candidate in column_lookup:
                        price_col = column_lookup[candidate]
                        break

                if price_col is not None:
                    raw_close_series = df_debug.loc[:, price_col]
                    raw_ma_series = raw_close_series.rolling(window=ma_period_debug).mean()

                    latest_raw_date = raw_close_series.index[-1]
                    latest_raw_close = float(raw_close_series.iloc[-1])
                    ma_raw = float(raw_ma_series.iloc[-1])

                    debug_lines.append("Raw latest (yfinance): " f"{latest_raw_date} | Close={latest_raw_close:,.4f} | MA={ma_raw:,.4f}")

                    try:
                        row_raw_full = df_debug.loc[latest_raw_date]
                        debug_lines.append("Raw row data: " + row_raw_full.to_dict().__repr__())
                    except Exception:
                        pass

                    # Placeholder for future filtered data
                    debug_lines.append("Filtered latest: (미적용 - 향후 계산 값)")

                    raw_tail = raw_close_series.tail(5).to_string()
                    debug_lines.append("Raw tail (Close):\n" + raw_tail)

                    debug_lines.append("Filtered tail (미적용)")
                else:
                    debug_lines.append("가격 컬럼을 찾을 수 없습니다.")
            else:
                debug_lines.append("가격 데이터를 가져오지 못했습니다.")

            cache_path = get_cache_path("common", ticker)
            if cache_path.exists():
                mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                debug_lines.append(f"Cache file: {cache_path} (mtime={mtime})")
            else:
                debug_lines.append("Cache file not found.")

            if yf is not None:
                try:
                    yf_hist = yf.Ticker(ticker).history(period="5d", interval="1d")
                    if not yf_hist.empty:
                        last_idx = yf_hist.index[-1]
                        last_row = yf_hist.iloc[-1]
                        close_val = float(last_row.get("Close", float("nan")))
                        adj_val = float(last_row.get("Adj Close", close_val))
                        debug_lines.append(f"yfinance latest: {last_idx} | Close={close_val:,.4f} | Adj Close={adj_val:,.4f}")
                        debug_lines.append("yfinance tail (Close):\n" + yf_hist["Close"].tail(5).to_string())
                except Exception as exc:  # pragma: no cover - 진단용 출력
                    debug_lines.append(f"yfinance fetch 오류: {exc}")

            if debug_lines:
                st.code("\n\n".join(debug_lines), language="text")

            def _fmt_date(value: Any) -> Optional[str]:
                if value is None:
                    return None
                if hasattr(value, "strftime"):
                    try:
                        return value.strftime("%Y-%m-%d")
                    except Exception:  # pragma: no cover - 방어적 처리
                        pass
                return str(value)

            def _format_period(period: Optional[Tuple[pd.Timestamp, Optional[pd.Timestamp]]]) -> str:
                if not period or not period[0]:
                    if cache_start_text and cache_start_text != "-":
                        return f"{cache_start_text} 이후 없음"
                    return "-"
                start_dt, end_dt = period
                start_str = _fmt_date(start_dt) or "알 수 없음"
                if end_dt is None:
                    end_str = "현재"
                else:
                    end_str = _fmt_date(end_dt) or "알 수 없음"
                return f"{start_str} ~ {end_str}"

            def _build_row(label: str, payload: Dict[str, Any]) -> Dict[str, Any]:
                ticker_text = payload["ticker"]
                name_text = payload.get("name")
                if isinstance(name_text, str) and name_text.strip():
                    index_label = f"{name_text.strip()}({ticker_text})"
                else:
                    index_label = ticker_text

                proximity_pct = float(payload.get("proximity_pct", 0.0))
                direction = "아래" if proximity_pct < 0 else "위"
                basis_period = int(payload.get("ma_period", regime_info["ma_period"]))

                start_raw = payload.get("last_risk_off_start")
                risk_text = _format_period(
                    (
                        start_raw,
                        payload.get("last_risk_off_end"),
                    )
                    if start_raw is not None
                    else None
                )
                prev_text = _format_period(payload.get("last_risk_off_prev"))
                prev2_text = _format_period(payload.get("last_risk_off_prev2"))

                return {
                    "구분": label,
                    "지수": index_label,
                    "건강도": str(payload.get("status_label", "-")),
                    "위치": f"기준 {basis_period}일선 {abs(proximity_pct):.1f}% {direction}",
                    "마지막 시장위험 기간": risk_text,
                    "이전 시장위험 기간 1": prev_text,
                    "이전 시장위험 기간 2": prev2_text,
                    "_status_color": "red" if payload.get("status") == "warning" else "green",
                    "_position_color": "red" if proximity_pct < 0 else "green",
                }

            main_row = _build_row("메인", regime_info)
            main_df = pd.DataFrame([main_row])
            main_status = main_df["_status_color"].to_dict()
            main_position = main_df["_position_color"].to_dict()
            main_df = main_df.drop(columns=["_status_color", "_position_color"])

            def _style_status(col: pd.Series, mapping: Dict[Any, str]) -> List[str]:
                return [f"color: {mapping.get(idx, '')}" for idx in col.index]

            def _style_position(col: pd.Series, mapping: Dict[Any, str]) -> List[str]:
                return [f"color: {mapping.get(idx, '')}" for idx in col.index]

            if not main_df.empty:
                styled_main = main_df.style.apply(lambda col: _style_status(col, main_status), subset=["건강도"], axis=0)
                styled_main = styled_main.apply(lambda col: _style_position(col, main_position), subset=["위치"], axis=0)
                st.dataframe(styled_main, hide_index=True, width="stretch")

            if aux_infos:
                aux_rows: List[Dict[str, Any]] = []
                for offset, aux in enumerate(aux_infos, start=1):
                    aux_rows.append(_build_row(f"보조 {offset}", aux))

                aux_df = pd.DataFrame(aux_rows)
                aux_status = aux_df["_status_color"].to_dict()
                aux_position = aux_df["_position_color"].to_dict()
                aux_df = aux_df.drop(columns=["_status_color", "_position_color"])
                styled_aux = aux_df.style.apply(lambda col: _style_status(col, aux_status), subset=["건강도"], axis=0)
                styled_aux = styled_aux.apply(lambda col: _style_position(col, aux_position), subset=["위치"], axis=0)
                st.caption("아래 보조 지표는 단순 참고를 위한 정보입니다. 시스템은 메인 지수만을 이용하고 있습니다.")
                st.dataframe(styled_aux, hide_index=True, width="stretch")


def main() -> None:
    navigation = getattr(st, "navigation", None)
    page_cls = getattr(st, "Page", None)
    if navigation is None or page_cls is None:
        st.error("현재 설치된 Streamlit 버전이 `st.navigation`을 지원하지 않습니다.")
        st.stop()

    accounts = load_account_configs()
    if not accounts:
        st.error("사용할 수 있는 계정 설정이 없습니다. `data/settings/account` 폴더를 확인해주세요.")
        st.stop()

    default_icon = "📈"

    st.set_page_config(
        page_title="Momentum ETF",
        page_icon=default_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pages = [
        page_cls(
            _render_home_page,
            title="대시보드",
            icon="🏠",
            default=True,
        )
    ]
    for account in accounts:
        pages.append(_build_account_page(page_cls, account))

    pages.append(
        page_cls(
            "app_pages/trade.py",
            title="[Admin] trade",
            icon="📝",
            url_path="admin",
        )
    )

    # pages.append(
    #     page_cls(
    #         "app_pages/stocks.py",
    #         title="[Admin] 종목 정보",
    #         icon="📊",
    #         url_path="stocks",
    #     )
    # )

    # pages.append(
    #     page_cls(
    #         "app_pages/migration.py",
    #         title="[Admin] 마이그레이션",
    #         icon="🛠️",
    #         url_path="migration",
    #     )
    # )

    # pages.append(
    #     page_cls(
    #         "app_pages/delete.py",
    #         title="[Admin] 계정 삭제",
    #         icon="🗑️",
    #         url_path="delete",
    #     )
    # )

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 1.0rem !important;
            padding-right: 1.0rem !important;
        }

        .block-container h1,
        .block-container h2,
        .block-container h3 {
            margin-top: 0.5rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            margin-top: 0 !important;
        }

        section[data-testid="stSidebar"] {
            width: 12rem !important;
            min-width: 12rem !important;
        }

        section[data-testid="stSidebar"] > div {
            width: 12rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    navigation(pages).run()


if __name__ == "__main__":
    main()
