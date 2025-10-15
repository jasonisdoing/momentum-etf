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
    _overlay_recent_history,
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
            # delay_days = int(common_settings.get("MARKET_REGIME_FILTER_DELAY_DAY", 0) or 0)
            # st.caption(
            #     "디버그: "
            #     f"Ticker={regime_info.get('ticker')} | "
            #     f"MA Period={regime_info.get('ma_period')} | "
            #     f"Delay Days={delay_days} | "
            #     f"Last Date={regime_info.get('last_risk_off_start')} -> {regime_info.get('last_risk_off_end')} | "
            #     f"Divergence={regime_info.get('proximity_pct'):+.3f}%"
            # )

            debug_lines = []
            ticker = regime_info.get("ticker")
            # ma_period_debug = int(regime_info.get("ma_period") or ma_period)
            # country = regime_info.get("country") or "us"

            # 디버그 출력을 활성화하려면 아래 주석을 해제하세요.
            # try:
            #     df_debug = fetch_ohlcv(ticker, country=country, months_range=[12, 0], cache_country="regime")
            #     if df_debug is None or df_debug.empty:
            #         df_debug = fetch_ohlcv(ticker, country=country, months_range=[12, 0], cache_country="common")
            # except Exception as exc:  # pragma: no cover - 진단용 출력
            #     debug_lines.append(f"fetch_ohlcv 오류: {exc}")
            #     df_debug = None
            #
            # if df_debug is not None and not df_debug.empty:
            #     df_debug = df_debug.sort_index()
            #     try:
            #         df_debug.index = pd.to_datetime(df_debug.index).normalize()
            #     except Exception:
            #         pass
            #     df_debug = df_debug[~df_debug.index.duplicated(keep="last")]
            #     df_debug = _overlay_recent_history(df_debug, ticker)
            #
            #     column_lookup: Dict[str, Any] = {}
            #     if isinstance(df_debug.columns, pd.MultiIndex):
            #         for col in df_debug.columns:
            #             if isinstance(col, tuple) and len(col) > 0 and col[0]:
            #                 column_lookup.setdefault(str(col[0]), col)
            #     else:
            #         column_lookup = {str(col): col for col in df_debug.columns}
            #
            #     price_col = None
            #     for candidate in ("Close", "Adj Close", "Price"):
            #         if candidate in column_lookup:
            #             price_col = column_lookup[candidate]
            #             break
            #
            #     if price_col is not None:
            #         raw_close_series = df_debug.loc[:, price_col].astype(float)
            #         raw_ma_series = raw_close_series.rolling(window=ma_period_debug).mean()
            #
            #         latest_raw_date = raw_close_series.index[-1]
            #         latest_raw_close = float(raw_close_series.iloc[-1])
            #         ma_raw = float(raw_ma_series.iloc[-1])
            #
            #         debug_lines.append(
            #             "Raw latest (yfinance): "
            #             f"{latest_raw_date} | Close={latest_raw_close:,.4f} | MA={ma_raw:,.4f}"
            #         )
            #
            #         try:
            #             row_raw_full = df_debug.loc[latest_raw_date]
            #             debug_lines.append("Raw row data: " + row_raw_full.to_dict().__repr__())
            #         except Exception:
            #             pass
            #
            #         country_lower = (country or "").strip().lower()
            #         tz_map = {
            #             "us": "America/New_York",
            #             "usa": "America/New_York",
            #             "kor": "Asia/Seoul",
            #             "korea": "Asia/Seoul",
            #             "kr": "Asia/Seoul",
            #             "aus": "Australia/Sydney",
            #             "au": "Australia/Sydney",
            #         }
            #         tz_name = tz_map.get(country_lower, "UTC")
            #         cutoff = pd.Timestamp.now(tz=tz_name).normalize() - pd.Timedelta(days=int(delay_days))
            #         try:
            #             cutoff = cutoff.tz_localize(None)
            #         except AttributeError:
            #             pass
            #
            #         filtered_close_series = raw_close_series[raw_close_series.index <= cutoff]
            #
            #         if not filtered_close_series.empty:
            #             filtered_ma_series = filtered_close_series.rolling(window=ma_period_debug).mean()
            #             latest_filtered_date = filtered_close_series.index[-1]
            #             latest_filtered_close = float(filtered_close_series.iloc[-1])
            #             latest_filtered_ma = float(filtered_ma_series.iloc[-1]) if not pd.isna(filtered_ma_series.iloc[-1]) else float("nan")
            #             divergence_filtered = (
            #                 (latest_filtered_close / latest_filtered_ma) - 1
            #             ) * 100 if latest_filtered_ma else float("nan")
            #
            #             debug_lines.append(
            #                 "Filtered latest (used in regime): "
            #                 f"{latest_filtered_date} | Close={latest_filtered_close:,.4f} | "
            #                 f"MA={latest_filtered_ma:,.4f} | Divergence={divergence_filtered:+.3f}%"
            #             )
            #
            #             debug_lines.append("Filtered tail (Close):\n" + filtered_close_series.tail(5).to_string())
            #         else:
            #             debug_lines.append("Filtered latest (used in regime): 데이터 없음")
            #
            #         raw_tail = raw_close_series.tail(5).to_string()
            #         debug_lines.append("Raw tail (Close):\n" + raw_tail)
            #     else:
            #         debug_lines.append("가격 컬럼을 찾을 수 없습니다.")
            # else:
            #     debug_lines.append("가격 데이터를 가져오지 못했습니다.")

            cache_paths = [
                ("regime", get_cache_path("regime", ticker)),
                ("common", get_cache_path("common", ticker)),
            ]
            for label, cache_path in cache_paths:
                if cache_path.exists():
                    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                    debug_lines.append(f"Cache file[{label}]: {cache_path} (mtime={mtime})")
                else:
                    debug_lines.append(f"Cache file[{label}]: not found")

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

            # if debug_lines:
            #     st.code("\n\n".join(debug_lines), language="text")

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
                st.markdown("**ETF 추천 전략 설명**")
                with st.expander("요약 보기", expanded=False):
                    st.markdown(
                        """# ETF 추천 시스템 요약

## 시스템 개요
ETF의 가격 추세를 분석해 상승세가 강한 종목을 자동 추천합니다.
이동평균선 대비 가격 위치, 최근 수익률, 연속 상승 일수 등을 종합해 점수를 부여하고
상위 종목을 포트폴리오 후보로 제시합니다.

---

## 주요 기능
- **ETF 분석 및 점수화**: 상승 추세 강도 평가
- **자동 추천 및 교체**: 점수가 높은 ETF로 교체 제안
- **시장 위험 감시**: 리스크 신호 발생 시 매수 제한
- **자산 배분 관리**: 현금·보유 비중을 고려한 투자 제안

---

## 포지션 상태
| 상태 | 의미 |
|------|------|
| WAIT | 대기 상태 |
| BUY | 신규 매수 제안 |
| HOLD | 보유 유지 |
| SELL_TREND | 추세 약화로 매도 제안 |
| CUT_STOPLOSS | 손절 지시 |
| REPLACE | 교체 제안 |

---

## 리스크 관리
- 동일 카테고리 중복 방지
- 시장 리스크 시 신규 매수 차단
- 쿨다운(대기) 기간 중 거래 제한
- 데이터 누락 또는 현금 부족 시 대기 처리

---

## 핵심 요약
> ETF 추천 시스템은 **상승 추세 분석 + 리스크 제어 + 자동 교체 판단**을 통해
> 투자자의 포트폴리오를 **지속적으로 최적화**합니다.
                        """,
                        unsafe_allow_html=False,
                    )
                with st.expander("상세 보기", expanded=False):
                    st.markdown(
                        """# ETF 추천 시스템 매뉴얼

## 1. 시스템 개요
이 시스템은 여러 ETF의 가격 흐름을 분석해 **최근 상승 추세가 강한 종목을 자동으로 추천**합니다.
이동평균(MA) 대비 현재 가격 위치, 최근 1~3주의 수익률, 연속 상승 일수 등을 종합적으로 반영해
ETF별 ‘상승 강도 점수’를 계산합니다.

---

## 2. 작동 원리

### (1) 추세 분석
- 각 ETF의 현재 가격이 이동평균선 위에 얼마나 위치하는지 측정합니다.
- 최근 1·2·3주 수익률, 연속 상승 일수 등을 함께 반영해 종합 점수를 계산합니다.
- 점수가 높을수록 상승 추세가 뚜렷한 ETF로 간주합니다.

### (2) 추천 절차
1. 투자 가능한 ETF 후보군을 생성합니다.
2. 각 ETF의 점수를 계산해 상위권을 추천합니다.
3. 보유 중인 ETF라도 점수가 낮으면, 더 높은 점수를 가진 후보로 교체합니다.
4. 포트폴리오 여유가 있을 때는 신규 편입을 제안하고,
   여유가 없으면 기존 종목과 점수 차가 충분히 클 때만 교체합니다.
5. ETF 가격이 이동평균선 아래로 떨어지면 매도 신호로 판단합니다.

---

## 3. 시장 상황 반영
시장의 전반적인 리스크 신호를 실시간 감시합니다.

- 시장이 불안하거나 하락세로 전환될 경우:
  - 신규 매수를 중단하거나 경고 문구를 표시합니다.
- 시장이 안정적일 경우:
  - 상승세 ETF를 유지하거나 추가 편입합니다.

이를 통해 **개별 ETF의 추세**뿐만 아니라 **시장 전체 상황(레짐)** 도 함께 고려합니다.

---

## 4. 포지션 상태 로직
각 ETF는 시스템 내부에서 다음과 같은 상태를 가집니다.

| 상태 | 설명 |
|------|------|
| **WAIT** | 기본 대기 상태. 조건 미충족 또는 제약 발생. |
| **BUY** | 신규 매수 제안. 상승세 뚜렷, 리스크 낮음. |
| **HOLD** | 보유 유지. 여전히 상승세 유지 중. |
| **SELL_TREND** | 이동평균선 아래로 떨어져 추세 약화. |
| **CUT_STOPLOSS** | 손실률이 설정 한도를 초과해 손절 지시. |
| **SELL_REPLACE / BUY_REPLACE** | 교체 대상 선정 시 기존 종목 매도 및 신규 종목 편입. |

매수 예산은 `current_equity / portfolio_topn`과 보유 현금 중 더 작은 값으로 제한되어
**비중 과다 투자**를 방지합니다.

---

## 5. 리스크 및 제약 조건

- **카테고리 중복 방지**: 동일 카테고리 ETF가 이미 포트폴리오에 있으면 WAIT 상태로 유지.
- **시장 리스크 오프**: 전체 리스크가 높을 시 신규 매수 차단 및 주의 문구 표시.
- **쿨다운 제한**: 최근 매매 이후 일정 기간 동안 동일 방향 거래 금지.
- **데이터 부족 처리**: 가격 데이터가 없거나 현금이 부족할 경우 대기 상태로 복귀.
- **비추천 종목 제외**: 추천 우주에 없는 티커나 비활성화된 ETF는 제외.
- **포트폴리오 포화 처리**: 슬롯이 가득 차 있으면 신규 편입 중단, 후보만 표시.

---

## 6. 결론
ETF 추천 시스템은
**“지속적인 상승 추세를 유지하면서 시장 위험을 피하는 종목만 선별”**하는
**반(半)자동 포트폴리오 운용 시스템**입니다.

> 이 시스템은 사용자가 직접 시장을 모니터링하지 않아도,
> 상승 종목 유지 / 교체 / 매도 시점을 자동으로 제시해
> 투자 효율성과 안정성을 동시에 제공합니다.
                        """,
                        unsafe_allow_html=False,
                    )


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
