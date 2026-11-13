"""합성 전략 운용 현황 — 계좌 평가액 기준 목표·오늘의 액션·보유 차트.

계좌·슬리브 설정과 계좌 확정은 `utils.strategy_mix_service` 가 맡는다.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, NamedTuple

from core.strategy.mix.actions import build_action_groups
from core.strategy.mix.targets import dated_target_shares
from utils.logger import get_app_logger
from utils.mix_sleeve import SleeveSpec
from utils.stock_memo_store import attach_stock_memos
from utils.strategy_mix_service import mix_accounts, mix_weights_for_account, resolve_mix_account

logger = get_app_logger()


# 고정 자산(IS, International Shares) — 사용자가 화면에서 수량을 못 바꾸는 호주 계좌 항목이다.
# 원장 holdings 가 아니라 `intl_shares_value` 필드에 계좌 통화로 들어 있고, 합성은 이걸
# **굴리지 않되 총자산에는 넣는다**(자산 관리 화면과 같은 취급).
FIXED_ASSET_TICKER = "IS"
FIXED_ASSET_NAME = "International Shares"


def holding_charts_for_account(account_id: str | None, tickers: list[str]) -> list[dict[str, Any]]:
    """보유 종목 차트 — 슬리브별로 그 전략의 기준선을 그린다 (공용 `holding_chart_service`).

    모멘텀 슬리브 종목은 단기·장기, 신고가는 이탈 이평선, 포트폴리오는 풀 판정 이평선.
    같은 티커가 두 슬리브에 있으면 앞 슬롯(선정 우선) 기준 한 장만 그린다 — 행도 한 줄이다.
    """
    from utils.holding_chart_service import holding_charts as build_charts
    from utils.mix_sleeve import MOMENTUM, NEW_HIGH
    from utils.settings_loader import get_ticker_type_settings

    ctx = resolve_mix_account(account_id)
    wanted = [str(t).strip() for t in tickers if str(t or "").strip()]
    charts_by_ticker: dict[str, dict[str, Any]] = {}
    for spec in ctx["slots"]:
        remaining = [t for t in wanted if t not in charts_by_ticker]
        if not remaining:
            break
        if spec.strategy == MOMENTUM:
            ma_days = [int(spec.settings["short_ma_days"]), int(spec.settings["long_ma_days"])]
        elif spec.strategy == NEW_HIGH:
            ma_days = [int(spec.settings["exit_ma_days"])]
        else:
            # 포트폴리오(배당주) — 판정선이 없어 풀 설정 이평선(추세 배지 기준)을 참고로 그린다.
            pool_settings = get_ticker_type_settings(spec.pool) or {}
            ma_days = [int(pool_settings["SHORT_MA_DAYS"]), int(pool_settings["LONG_MA_DAYS"])]
        # 이 슬리브 풀에 없는 티커는 가격 프레임이 없어 자연히 건너뛴다(다음 슬리브가 잡는다).
        for chart in build_charts(spec.pool, remaining, ma_days):
            # 카드 제목의 [이름] — 사용자가 슬리브에 붙인 이름, 없으면 전략 이름(spec.label).
            chart["strategy_label"] = spec.label
            charts_by_ticker.setdefault(chart["ticker"], chart)
    return [charts_by_ticker[t] for t in wanted if t in charts_by_ticker]


def _load_account_state(account_id: str) -> dict[str, Any]:
    """적용 계좌의 실제 보유 수량·평단·현금 — portfolio_master 가 단일 소스다."""
    from utils.portfolio_io import load_portfolio_master

    master = load_portfolio_master(account_id) or {}
    holdings = {}
    for row in master.get("holdings") or []:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker or ticker in {FIXED_ASSET_TICKER, "__CASH__"}:
            continue
        try:
            quantity = float(row.get("quantity") or 0)
        except (TypeError, ValueError):
            quantity = 0.0
        if quantity <= 0:
            continue
        holdings[ticker] = {
            "quantity": quantity,
            "name": str(row.get("name") or ticker),
            "average_buy_price": float(row.get("average_buy_price") or 0) or None,
        }
    try:
        cash = float(master.get("cash_balance") or 0)
    except (TypeError, ValueError):
        cash = 0.0
    # 고정 자산(IS) — 원장 holdings 가 아니라 별도 필드에 계좌 통화로 들어 있다.
    # 합성이 굴리지는 않지만 **총자산에는 들어간다** — 빼면 목표 금액이 그만큼 작아져
    # 화면의 비중 합이 실제 계좌와 어긋난다.
    try:
        fixed_native = float(master.get("intl_shares_value") or 0)
    except (TypeError, ValueError):
        fixed_native = 0.0
    # 손익도 함께 저장돼 있다 — 원금 = 평가액 − 손익 (자산 관리 화면과 같은 규칙).
    try:
        fixed_change = float(master.get("intl_shares_change") or 0)
    except (TypeError, ValueError):
        fixed_change = 0.0
    return {
        "account_id": account_id,
        "holdings": holdings,
        "cash_balance": cash,
        "fixed_asset_native": fixed_native,
        "fixed_asset_change_native": fixed_change,
    }


def _slot_labels(slots: list[SleeveSpec], account_id: str) -> dict[str, str]:
    """슬롯 표시 이름 — 「A. 모멘텀」. 같은 전략이 두 슬롯에 올 수 있어 기호를 앞에 둔다.

    사고팔기가 있는 계좌는 사고팔기가 A 라 슬리브 기호가 B 부터다(`mix_slot_marks`).
    """
    from utils.account_settings_store import mix_slot_marks

    marks = mix_slot_marks(account_id)
    return {spec.key: f"{marks[spec.key]}. {spec.label}" for spec in slots}


def _attach_industry(holdings: list[dict[str, Any]], country: str) -> None:
    """행마다 업종을 붙인다 — 순위·모멘텀·신고가 화면과 **같은 공용 맵**이 단일 소스다.

    합성은 여러 종목풀이 섞이고 슬리브에 없는 계좌 보유분(전량 매도 대상)도 있어서, 풀
    하나가 아니라 그 계좌 **국가 전체**의 맵을 본다. 분류가 없는 종목은 빈 값으로 둔다.
    """
    from utils.industry_map import industry_map_for_country

    industry_by = industry_map_for_country(country)
    for row in holdings:
        row["industry"] = industry_by.get(str(row.get("ticker") or "").strip(), "")


def _attach_disparity(holdings: list[dict[str, Any]], pool_by_source: dict[str, str]) -> None:
    """행마다 단기·장기 이격(%)을 붙인다 — **그 종목이 속한 종목풀 설정**의 이평선이 기준.

    순위 화면(`/pools-rank`)·보유종목 알림과 같은 기준이다. 화면은 이 값으로 종목명 옆
    추세 이탈(행 전체 회색)을 표시하므로, 다른 기준으로 계산하면 같은 종목에 화면마다 다른
    배지가 붙는다. 계산 자체도 같은 함수(`momentum_metrics`)를 쓴다.

    행의 풀을 정하는 순서:
      1) 슬리브 목표 행 — `sources`(a/b)가 가리키는 풀. 둘 다면 A 슬리브의 풀(같은 풀일 때만 생긴다).
      2) 전량 매도 대상 행 — 슬리브 어디에도 없는 계좌 보유분이라 `sources` 가 비어 있다.
         한 티커는 한 풀에만 들어가므로 소속 풀을 직접 찾는다(보유종목 알림과 같은 함수).
    이평선이나 가격을 못 구하면 None 으로 둔다(임의 값으로 채우지 않는다).
    """
    import pandas as pd

    from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types
    from utils.momentum_service import momentum_metrics
    from utils.settings_loader import get_ticker_type_settings
    from utils.stock_list_io import pools_by_ticker

    tickers = [str(row["ticker"]).strip() for row in holdings if row.get("ticker")]
    if not tickers:
        return

    # 슬리브에 안 잡힌 행(전량 매도 대상)의 소속 풀 — 없으면 조회하지 않는다.
    orphan_tickers = [
        str(row["ticker"]).strip() for row in holdings if row.get("ticker") and not (row.get("sources") or [])
    ]
    pool_by_ticker = pools_by_ticker(orphan_tickers) if orphan_tickers else {}

    ma_by_pool: dict[str, tuple[int, int] | None] = {}

    def ma_days_of(pool: str) -> tuple[int, int] | None:
        if pool not in ma_by_pool:
            config = get_ticker_type_settings(pool) or {}
            short, long = config.get("SHORT_MA_DAYS"), config.get("LONG_MA_DAYS")
            ma_by_pool[pool] = (int(short), int(long)) if short and long else None
        return ma_by_pool[pool]

    frames = load_cached_frames_bulk_from_all_ticker_types(tickers)

    # 장중에는 실시간 가격을 마지막 봉으로 얹어 판정한다(strategy_logic.md 「장중 잠정 실행」) — 순위 화면과
    # 같은 공용 함수(`build_effective_close_series`)라 두 화면의 추세 이탈 표시가 갈리지 않는다.
    from services.price_service import get_realtime_snapshot
    from utils.effective_prices import bar_anchor
    from utils.moving_averages import pool_moving_average_type
    from utils.rankings import build_effective_close_series

    country_of_pool: dict[str, str] = {}

    def country_of(pool: str) -> str:
        if pool not in country_of_pool:
            config = get_ticker_type_settings(pool) or {}
            country_of_pool[pool] = str(config.get("country_code") or "").strip().lower()
        return country_of_pool[pool]

    def pool_of_row(row: dict[str, Any]) -> str:
        # 여러 슬리브에 함께 잡힌 종목은 **첫 슬리브**의 풀 이평선으로 본다 — 배지가
        # 하나뿐이라 기준도 하나여야 한다(슬롯 순서가 그 우선순위다).
        ticker = str(row.get("ticker") or "").strip()
        sources = row.get("sources") or []
        source = next((key for key in pool_by_source if key in sources), "")
        return pool_by_source.get(source, "") if source else pool_by_ticker.get(ticker.upper(), "")

    tickers_by_country: dict[str, list[str]] = {}
    for row in holdings:
        ticker = str(row.get("ticker") or "").strip()
        pool = pool_of_row(row)
        country = country_of(pool) if pool else ""
        if ticker and country:
            tickers_by_country.setdefault(country, []).append(ticker)
    realtime: dict[str, dict[str, Any]] = {}
    for country, country_tickers in tickers_by_country.items():
        try:
            realtime.update(get_realtime_snapshot(country, country_tickers))
        except Exception:
            logger.warning("[합성] 실시간 시세 조회 실패(%s) — 확정 종가로 판정", country, exc_info=True)

    for row in holdings:
        row["current_short_pct"] = None
        row["current_long_pct"] = None
        row["new_listing"] = None
        row["listing_months"] = None
        ticker = str(row.get("ticker") or "").strip()
        pool = pool_of_row(row)
        days = ma_days_of(pool) if pool else None
        if days is None:
            continue
        frame = frames.get(ticker)
        if frame is None or frame.empty or "Close" not in frame.columns:
            continue
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if close.empty:
            continue
        # 신규상장(🆕) — 전 화면 공용 판정(확정 시리즈 기준, 잠정 봉과 무관).
        from core.strategy.scoring import is_new_listing, listing_months

        row["new_listing"] = is_new_listing(close)
        row["listing_months"] = listing_months(close)
        entry = realtime.get(ticker)
        if entry:
            # 붙일 봉의 기준은 순위 화면과 같은 공용 앵커다 — 종목별 마지막 봉으로 정하면
            # 캐시가 종목마다 다른 날짜에서 끝날 때 합성의 이격이 순위와 갈린다.
            country = country_of(pool)
            effective = build_effective_close_series(close, entry, country, last_bar=bar_anchor(country))
            if effective is not None:
                close = effective
        metrics = momentum_metrics(
            close,
            short_ma_days=days[0],
            long_ma_days=days[1],
            ma_type=pool_moving_average_type(pool),
            as_of=None,
        )
        if not metrics:
            continue
        row["current_short_pct"] = round(metrics["short_disparity_pct"], 1)
        row["current_long_pct"] = round(metrics["disparity_pct"], 1)


def _krw_rate(currency: str) -> float:
    """종목 통화 → 원화 환율. 원화면 1.0.

    계좌 원장의 현금·평가액은 전부 원화 기준인데 종목 가격은 그 시장 통화다. 섞어서
    더하거나 나누면 미국·호주 풀에서 총자산과 목표 수량이 환율 배수만큼 어긋난다.
    환율을 못 받으면 0 을 돌려 호출부가 '계산 불가' 로 명시한다 — 1.0 으로 두면
    달러 가격을 원화로 착각한 값이 조용히 나간다.
    """
    code = str(currency or "KRW").strip().upper()
    if code == "KRW":
        return 1.0
    from services.price_service import get_exchange_rates

    try:
        return float(((get_exchange_rates() or {}).get(code) or {}).get("rate") or 0.0)
    except Exception:
        logger.warning("[STRATEGY-MIX] %s 환율 조회 실패 — 목표 수량을 계산하지 않는다", code, exc_info=True)
        return 0.0


# 슬리브별 값이 붙는 자리 — 내부 계산은 `a_weight` 처럼 평평하게 들고 다니고(키가 늘어도
# 코드가 그대로다), 화면에 내보낼 때만 `slots` 아래로 모은다. 화면은 슬롯 키를 돌며 읽는다.
# 슬리브별로 행에 실리는 값 — `plan`·`days`·`is_new`·`exit_reason` 은 각 전략 화면과
# 같은 상태 컬럼(`web/lib/grid-cells.slotStatusColumn`)이 요구하는 필드다.
_SLOT_ROW_FIELDS: tuple[str, ...] = (
    "weight",
    "status",
    "plan",
    "days",
    "is_new",
    "exit_reason",
    "return_pct",
    "held_label",
    "entry_date",
    "entry_price",
    # 행별 체결일 — 어제 확정 판정(오늘 체결)과 오늘 잠정 판정(내일 체결)을 가른다.
    "fill_date",
)


def _holding_payload(row: dict[str, Any], slot_keys: Sequence[str]) -> dict[str, Any]:
    """보유 행 하나를 화면 형태로 — 슬리브별 값(`a_weight` …)을 `slots[키]` 로 모은다."""
    flat_keys = {f"{key}_{field}" for key in slot_keys for field in _SLOT_ROW_FIELDS}
    payload = {name: value for name, value in row.items() if name not in flat_keys}
    payload["weight_pct"] = round(float(row["weight_pct"]), 2)
    payload["slots"] = {key: {field: row.get(f"{key}_{field}") for field in _SLOT_ROW_FIELDS} for key in slot_keys}
    return payload


def _attach_account_targets(
    holdings: list[dict[str, Any]],
    account: dict[str, Any],
    krw_rate: float = 1.0,
    slot_keys: Sequence[str] = (),
    target_shares: dict[str, int] | None = None,
    *,
    capital_krw: float,
) -> None:
    """계좌 보유와 목표를 대조해 수량 지시를 붙인다.

    종목별 목표 금액·주수 → 현재 보유와의 차이가 그대로 매매 지시가 된다.
    행이 종목 단위라 계좌 보유와 1:1 로 비교된다(겹치는 종목도 한 번만 센다).
    목표 포트폴리오에 없는 보유 종목은 전량 매도 대상 — 목표 비중 0% 행으로 표에
    함께 넣는다 (팔아야 할 종목이 표 밖에 있으면 계좌를 표 하나로 대조할 수 없다).
    """
    from utils.portfolio_io import return_pct_from_avg_price

    total_assets = float(account.get("total_assets") or 0)
    # 목표 금액은 원화, 가격은 그 시장 통화다 — 환율로 맞춘 뒤 나눠야 한다.
    price_krw_by_ticker: dict[str, float] = {}
    for row in holdings:
        held = account["holdings"].get(row["ticker"])
        row["held_quantity"] = held["quantity"] if held else 0.0
        row["held_value"] = (held or {}).get("value")
        # 실제 수익률 — 계좌 매입 평단 대비. 전략수익률(이론 편입가 대비)과 별개 컬럼이고,
        # 종목당 하나뿐이라 슬리브(A/B)로 나누지 않는다. 아직 안 산 종목은 평단이 없어 빈다.
        # 계산은 /assets 계좌 보유 표와 **같은 공용 함수**(utils.portfolio_io).
        # 평단도 함께 내려보낸다 — 화면이 현재가를 실시간으로 덮어쓰므로 그때 같은 평단으로
        # 다시 계산해야 한다. 안 그러면 현재가는 장중인데 수익률만 어제 종가 기준으로 남는다.
        row["average_buy_price"] = (held or {}).get("average_buy_price")
        actual = return_pct_from_avg_price(row.get("price"), row["average_buy_price"])
        row["return_pct"] = None if actual is None else round(actual, 2)
        row["current_weight_pct"] = (
            round(float(row["held_value"]) / total_assets * 100.0, 2)
            if row.get("held_value") and total_assets > 0
            else 0.0
        )
        # 주수를 정하기 전의 임시값(비중 기준). 아래에서 **목표 주수 × 1주 값**으로 덮어쓴다 —
        # 화면에 보이는 목표 금액은 실제로 주문할 금액이어야 한다.
        row["target_amount"] = round(capital_krw * row["weight_pct"] / 100.0, 2)
        price = row.get("price")
        if price and krw_rate > 0:
            price_krw_by_ticker[row["ticker"]] = float(price) * krw_rate

    # 목표 주수는 계좌 평가액을 기준으로 비싼 종목의 첫 1주를 확보한 뒤 남은 예산으로 정한다.
    # 실제 보유는 목표를 바꾸지 않고 회수·채우기 판단에만 쓴다.
    target_shares = target_shares or {}
    for row in holdings:
        price_krw = price_krw_by_ticker.get(row["ticker"])
        if row["ticker"] in target_shares:
            target_qty = target_shares[row["ticker"]]
        elif price_krw:
            # 포트폴리오 슬리브·목표 0 행 — 판정이 없어 비중으로만 잡는다.
            target_qty = int(float(row["target_amount"]) // price_krw)
        else:
            target_qty = None
        row["target_quantity"] = target_qty
        row["trade_quantity"] = None if target_qty is None else target_qty - int(row["held_quantity"])
        # 1주도 못 사는 종목 — 목록에서 지우지 않고 경고로 드러낸다. 계좌가 백테스트 자본보다
        # 작으면 도달할 수 없는 목표가 생기는데, 이걸 숨기면 백테스트 수익률을 낼 수 있다고
        # 착각하게 된다. 비중은 백테스트 것 그대로 둔다(종목·비중은 절대 바꾸지 않는다).
        row["unaffordable"] = bool(
            target_qty == 0 and float(row.get("weight_pct") or 0) > 0 and not row.get("is_sell_all") and price_krw
        )
        # **실제 가능한 비중** — 목표 주수 × 1주 값 ÷ 총자산. 계좌 금액과 단주 때문에 백테스트
        # 비중에 못 미치는 만큼이 여기서 드러난다(DELL 목표 5.97% ↔ 실제 3.75% = 1주).
        # 목표 비중(`weight_pct`)은 백테스트 값 그대로 둔다 — 덮어쓰면 전략이 원래 무엇을
        # 원했는지가 사라져, 못 맞추고 있다는 사실 자체가 안 보인다.
        row["actual_weight_pct"] = (
            round(target_qty * price_krw / capital_krw * 100.0, 4)
            if target_qty is not None and price_krw and total_assets > 0
            else None
        )
        # 목표 금액 = 목표 주수 × 1주 값 — 비중이 아니라 실제 주문 금액이다.
        if target_qty is not None and price_krw:
            row["target_amount"] = round(target_qty * price_krw, 2)
    target_tickers = {row["ticker"] for row in holdings}
    for ticker, item in sorted(account["holdings"].items()):
        if ticker in target_tickers:
            continue
        value = item.get("value")
        holdings.append(
            {
                "ticker": ticker,
                "name": item["name"],
                "sources": [],
                "weight_pct": 0.0,
                "price": item.get("price"),
                "change_pct": None,
                # 슬리브별 칸 — 이 행은 목표에 없는 보유라 어느 슬리브에도 안 걸린다.
                # 몫 0 을 명시해야 슬리브 현금 합계가 이 행에서 KeyError 없이 계산된다.
                **{f"{key}_weight": 0.0 for key in slot_keys},
                **{f"{key}_status": None for key in slot_keys},
                "is_sell_all": True,
                "actual_weight_pct": 0.0,
                "held_quantity": item["quantity"],
                "held_value": value,
                "average_buy_price": item.get("average_buy_price"),
                "return_pct": (
                    None
                    if (actual := return_pct_from_avg_price(item.get("price"), item.get("average_buy_price"))) is None
                    else round(actual, 2)
                ),
                "current_weight_pct": round(float(value) / total_assets * 100.0, 2)
                if value and total_assets > 0
                else 0.0,
                "target_amount": 0.0,
                "target_quantity": 0,
                "trade_quantity": -int(item["quantity"]),
            }
        )


def _block_trade_strategy_sells(holdings: list[dict[str, Any]], blocked: list[str]) -> None:
    """사고팔기 전략에 등록된 종목은 합성이 사고팔지 않는다 — 행에 표시하고 표의 매매 수량을 0 으로 둔다.

    사고팔기가 그 종목의 회차를 계좌 체결로 세므로, 합성이 팔면 사고팔기 원장이 깨진다.
    액션(화면·슬랙)에서의 제외는 `build_action_groups` 가 이 표시를 보고 한다.
    """
    for row in holdings:
        if row.get("ticker") not in blocked:
            continue
        row["is_trade_strategy"] = True
        for key in ("trade_quantity", "forecast_trade_quantity"):
            if row.get(key):
                row[key] = 0


def mix_positions(account_id: str | None = None) -> dict[str, Any]:
    """오늘 기준 합성 운영 상태 — 보유 목록(목표 비중)·현금 비중·오늘의 액션.

    개별 엔진의 종목·진입·청산을 받아 계좌 평가액 기준 목표와 회수·채우기 정책을 적용한다.
    포트폴리오는 저장 비중, 슬롯 전략은 슬리브 몫 ÷ 슬롯 수(빈 슬롯 = 현금).
    겹치는 종목은 한 행으로 합친다 — 계좌에는 그 종목이 하나뿐이라, 슬리브별로 나누면
    보유 수량이 두 번 세어지고 매매 지시가 반대로 나온다.
    """
    from utils.mix_sleeve import slot_state
    from utils.strategy_settings import require_start_date
    from utils.strategy_trade_service import trade_tickers

    ctx = resolve_mix_account(account_id)
    slots: list[SleeveSpec] = ctx["slots"]
    keys = [spec.key for spec in slots]
    for spec in slots:
        require_start_date(spec.settings)
    states = {spec.key: slot_state(spec) for spec in slots}
    _apply_target_weights(slots, states)

    base_weights = mix_weights_for_account(ctx["account_id"])
    shares = {key: base_weights[f"{key}_pct"] for key in keys}
    holdings = _merge_sleeve_targets(states, shares, keys)
    dates = _action_dates(ctx["country"], states)
    capital = _apply_account_capital(ctx, states, holdings, shares, dates)

    # 추세 이탈(행 전체 회색)용 — 행이 속한 슬리브의 **종목풀 설정** 이평선 기준.
    _attach_disparity(holdings, {spec.key: spec.pool for spec in slots})
    _attach_industry(holdings, ctx["country"])
    # 종목 메모 — 계좌가 아니라 **종목**에 붙는다(utils/stock_memo_store). 전량 매도 행까지 붙은 뒤에 한 번에 읽는다.
    attach_stock_memos(holdings)
    summary = _summary(holdings, capital.shares, states, keys)

    slot_actions = _slot_actions(states, _slot_labels(slots, ctx["account_id"]), keys)
    _attach_exit_forecast(holdings, slot_actions, keys)
    # 사고팔기 종목 — 보유가 없어도 등록돼 있으면 화면이 현금 몫 안의 사고팔기 칸을 보여준다.
    blocked = trade_tickers(ctx["account_id"])
    _block_trade_strategy_sells(holdings, blocked)
    # 오늘의 액션 — 화면·슬랙 알람이 같은 결과를 쓴다(조립 단일 소스). 슬리브별 값이 평평한 행을 읽는다.
    actions: dict[str, Any] = {"slots": slot_actions}
    actions["groups"] = build_action_groups(
        holdings,
        actions,
        dates.next_trading_day,
        harvest_pct=ctx["mix_harvest_pct"],
        refill_pct=ctx["mix_refill_pct"],
        cash_balance=capital.account["cash_balance"] / capital.krw_rate,
        currency=ctx["currency"],
        adjustment_day=dates.adjustment_day,
        adjustment_intraday=dates.adjustment_intraday,
        today=dates.today,
        target_schedule=capital.target_schedule,
    )
    return {
        "computed_at": datetime.now().astimezone().isoformat(),
        "currency": ctx["currency"],
        "krw_rate": capital.krw_rate,
        "capital_krw": capital.capital_krw,
        "account_id": ctx["account_id"],
        # 화면이 표시용 시세를 60초마다 갱신할 때 쓴다(시세 소스가 국가별로 다르다).
        "country": ctx["country"],
        "account": capital.account,
        # 데이터 기준일 — 그 값을 주는 전략이 있으면 그걸 쓰고, 없으면 오늘이다.
        "as_of": next((states[key].as_of for key in keys if states[key].as_of), None) or dates.today,
        "next_trading_day": dates.next_trading_day,
        # 장중 반영은 그 정보를 주는 전략에서만 온다. 슬리브 어디에도 없으면 거짓이다.
        "live": any(states[key].live for key in keys),
        "summary": summary,
        # 슬리브별 값을 `slots[키]` 로 모아 내보낸다 — 화면은 슬롯 키를 돌며 읽는다.
        "holdings": [_holding_payload(row, keys) for row in holdings],
        "actions": actions,
        "trade_tickers": blocked,
    }


def _apply_target_weights(slots: list[SleeveSpec], states: dict[str, Any]) -> None:
    """엔진 목표에 합성 배정 비중(`drift_pct`)을 붙인다 — 엔진은 종목·시점, 합성은 고정 배정 비중을 정한다."""
    from core.strategy.mix.capital_policy import internal_target_weights

    for spec in slots:
        weights = internal_target_weights(strategy=spec.strategy, settings=spec.settings)
        for target in states[spec.key].targets:
            weight = weights[str(target["ticker"])] if isinstance(weights, dict) else weights
            target["drift_pct"] = weight
            if spec.strategy == "portfolio":
                target["status"] = f"설정 비중 {weight:.2f}%"


def _merge_sleeve_targets(
    states: dict[str, Any], shares: dict[str, float], keys: Sequence[str]
) -> list[dict[str, Any]]:
    """슬리브 목표를 종목 한 행으로 합친다 — 비중은 더하고 슬리브별 값은 `{키}_{필드}` 로 평평하게 든다.

    매도 예정(자격 상실·이탈)은 목표 비중 0 이다 — 다음 시가에 전량 팔고 그 슬롯은
    다음 교체까지 현금이다. 비중을 남겨두면 팔아야 할 종목의 매매수량이 0 으로 보인다.
    """
    holdings: list[dict[str, Any]] = []
    by_ticker: dict[str, dict[str, Any]] = {}
    for key in keys:
        for target in states[key].targets:
            if target["is_exiting"]:
                weight = 0.0
            elif target.get("drift_pct") is not None:
                weight = shares[key] * float(target["drift_pct"]) / 100.0
            else:
                raise ValueError(f"{key}: 종목 기준 비중이 없습니다.")
            ticker = str(target["ticker"]).strip()
            row = by_ticker.get(ticker)
            if row is None:
                row = _new_holding_row(ticker, target, keys)
                by_ticker[ticker] = row
                holdings.append(row)
            _add_slot_target(row, key, target, weight)
    return holdings


def _new_holding_row(ticker: str, target: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    """합친 행의 빈 틀 — 슬리브마다 `_SLOT_ROW_FIELDS` 칸을 만든다(몫은 0, 나머지는 빈 값)."""
    row: dict[str, Any] = {
        "ticker": ticker,
        "name": target.get("name") or ticker,
        "sources": [],
        "weight_pct": 0.0,
        "price": target.get("price"),
        "change_pct": target.get("change_pct"),
    }
    for key in keys:
        for field in _SLOT_ROW_FIELDS:
            row[f"{key}_{field}"] = 0.0 if field == "weight" else None
    return row


def _add_slot_target(row: dict[str, Any], key: str, target: dict[str, Any], weight: float) -> None:
    """슬리브 하나의 목표를 행에 더한다 — 상태 컬럼·보유 기간·차트 진입 화살표 값은 그 슬리브 칸에 담는다."""
    if key not in row["sources"]:
        row["sources"].append(key)
    row["weight_pct"] += weight
    row[f"{key}_weight"] += weight
    if row.get("price") is None:
        row["price"] = target.get("price")
    if row.get("change_pct") is None:
        row["change_pct"] = target.get("change_pct")
    row[f"{key}_status"] = target.get("status")
    row[f"{key}_plan"] = target.get("plan")
    row[f"{key}_days"] = target.get("days")
    row[f"{key}_is_new"] = bool(target.get("is_new"))
    row[f"{key}_exit_reason"] = target.get("exit_reason")
    if target.get("return_pct") is not None:
        row[f"{key}_return_pct"] = round(float(target["return_pct"]), 2)
    # 보유 기간 표기 — 전략마다 단위가 달라("3주" vs "12일") 완성된 문자열로 받는다.
    if target.get("held_label"):
        row[f"{key}_held_label"] = target["held_label"]
    if target.get("entry_date"):
        row[f"{key}_entry_date"] = str(target["entry_date"])[:10]
    if target.get("entry_price") is not None:
        row[f"{key}_entry_price"] = float(target["entry_price"])
    if target.get("fill_date"):
        row[f"{key}_fill_date"] = str(target["fill_date"])[:10]


class _ActionDates(NamedTuple):
    """액션 묶음의 날짜 — 모두 그 시장 현지 날짜(YYYY-MM-DD)."""

    today: str
    # 엔진 판정이 체결되는 다음 시가 거래일.
    next_trading_day: str
    # 목표·보유 차이 조정을 낼 날 — 오늘 마감 전이면 오늘, 아니면 다음 거래일.
    adjustment_day: str
    # 조정일이 오늘이고 정규장이 이미 열렸는지 — 그룹 제목이 '시가'가 아니라 '장중'이 된다.
    adjustment_intraday: bool


def _action_dates(country: str, states: dict[str, Any]) -> _ActionDates:
    """오늘·다음 체결일·조정일·장중 여부를 정한다."""
    import pandas as pd

    from config import MARKET_SCHEDULES
    from utils.trading_calendar import get_trading_days, is_market_day_completed, is_market_day_started

    tz_name = str((MARKET_SCHEDULES.get(country) or {}).get("timezone") or "Asia/Seoul")
    today_local = pd.Timestamp.now(tz=tz_name).date()
    today = today_local.strftime("%Y-%m-%d")
    # 다음 체결일은 **슬리브 엔진이 낸 값 그대로**(`slot_positions.next_fill_session` / `_next_session`,
    # 전략 화면의 `next_session`). 합성이 따로 셈하면 같은 예정이 화면마다 다른 날로 갈린다(2026-10-26).
    sessions = {state.next_session for state in states.values()}
    if len(sessions) != 1:
        raise RuntimeError(f"슬리브마다 다음 체결일이 다릅니다: {sorted(str(s) for s in sessions)}")
    next_trading_day = sessions.pop()
    # 조정(목표·보유 차이)은 시가 체결 신호가 아니라 "지금 낼" 주문이다 — 오늘이 거래일이고 마감 전이면
    # (개장 전=오늘 시가, 장중=즉시) 오늘, 마감 후·휴장은 다음 거래일.
    ahead = get_trading_days(today, (today_local + timedelta(days=21)).strftime("%Y-%m-%d"), country)
    today_is_trading = bool(ahead) and ahead[0].date() == today_local
    adjustment_day = (
        today
        if today_is_trading and not is_market_day_completed(country, pd.Timestamp(today_local))
        else next_trading_day
    )
    # 장중 여부는 정규장이 이미 열렸는지로 본다(체결일 비교로 정하면 개장 전에도 '장중'이 됐다).
    adjustment_intraday = adjustment_day == today and is_market_day_started(country, pd.Timestamp(today_local))
    return _ActionDates(today, next_trading_day, adjustment_day, adjustment_intraday)


class _Capital(NamedTuple):
    """계좌 평가 결과와 그 기준의 목표 주수."""

    account: dict[str, Any]
    krw_rate: float
    # 운용 기준금액 = 확정일 계좌 평가액(현금 포함).
    capital_krw: float
    # IS 고정 자산 몫을 뺀 슬리브 몫(%).
    shares: dict[str, float]
    target_schedule: dict[str, Any]


def _apply_account_capital(
    ctx: dict[str, Any],
    states: dict[str, Any],
    holdings: list[dict[str, Any]],
    shares: dict[str, float],
    dates: _ActionDates,
) -> _Capital:
    """계좌를 평가해 목표 주수를 행에 붙인다 — IS 고정 자산은 현재 비중을 목표에도 보존하고 나머지만 배정한다."""
    account = _load_account_state(ctx["account_id"])
    krw_rate = _krw_rate(ctx["currency"])
    if krw_rate <= 0:
        raise RuntimeError(f"{ctx['currency']} 환율을 읽을 수 없습니다.")
    # 목표와 같은 확정일로 계좌를 평가한다. 표시용 실시간 가격은 수량 계산에 넣지 않는다.
    valuation_date = min(state.as_of for state in states.values() if state.as_of)
    _value_account(
        account,
        krw_rate,
        {row["ticker"]: row["price"] for row in holdings if row.get("price")},
        as_of=valuation_date,
    )
    # 운용 기준금액 = 이 확정일의 계좌 평가액(현금 포함). 저장한 금액이 아니라 평가액이 기준이다.
    capital_krw = account["total_assets"]
    if capital_krw <= 0:
        raise RuntimeError(f"'{ctx['account_id']}' 계좌 평가액이 없어 운용 기준금액을 정할 수 없습니다.")

    keys = list(shares)
    fixed_pct = float(account["fixed_asset_pct"])
    investable_ratio = 1.0 - fixed_pct / 100.0
    scaled = {key: value * investable_ratio for key, value in shares.items()}
    for row in holdings:
        row["weight_pct"] *= investable_ratio
        for key in keys:
            row[f"{key}_weight"] *= investable_ratio
    target_schedule = dated_target_shares(
        {key: state.targets for key, state in states.items()},
        {key: capital_krw * scaled[key] / 100.0 for key in keys},
        krw_rate,
        capital_krw,
        dates.next_trading_day,
        adjustment_day=dates.adjustment_day,
    )
    _attach_account_targets(
        holdings,
        account,
        krw_rate,
        slot_keys=keys,
        target_shares=target_schedule[max(target_schedule)]["quantities"],
        capital_krw=capital_krw,
    )
    if account["fixed_asset_value"]:
        holdings.append(_fixed_asset_row(account, fixed_pct, capital_krw, keys))
    return _Capital(account, krw_rate, capital_krw, scaled, target_schedule)


def _fixed_asset_row(
    account: dict[str, Any], fixed_pct: float, capital_krw: float, keys: Sequence[str]
) -> dict[str, Any]:
    """IS 고정 자산 행 — 합성이 사고팔지 않으므로 수량·매매 칸은 비운다."""
    return {
        "ticker": FIXED_ASSET_TICKER,
        "name": FIXED_ASSET_NAME,
        "sources": [],
        "is_fixed_asset": True,
        "price": None,
        "weight_pct": fixed_pct,
        "actual_weight_pct": fixed_pct,
        "held_value": account["fixed_asset_value"],
        "current_weight_pct": fixed_pct,
        "target_amount": capital_krw * fixed_pct / 100.0,
        "held_quantity": None,
        "target_quantity": None,
        "trade_quantity": None,
        **{f"{key}_weight": 0.0 for key in keys},
    }


def _summary(
    holdings: list[dict[str, Any]], shares: dict[str, float], states: dict[str, Any], keys: Sequence[str]
) -> dict[str, Any]:
    """비중 합계·슬리브별 현황 — 고정 자산 축소가 끝난 뒤의 값이라야 실제 계좌와 맞는다."""
    stock_pct = sum(row["weight_pct"] for row in holdings)
    # 실제 가능한 주식 비중 — 단주로 못 채운 나머지는 전부 현금이다.
    actual_stock_pct = sum(float(row.get("actual_weight_pct") or 0) for row in holdings)
    return {
        "stock_pct": round(stock_pct, 2),
        "cash_pct": round(100 - stock_pct, 2),
        # 계좌 금액·단주를 감안해 **실제로 도달할 수 있는** 비중. 목표(위)와의 차이가
        # 곧 단주로 못 채워 현금으로 남는 몫이다.
        "actual_stock_pct": round(actual_stock_pct, 2),
        "actual_cash_pct": round(100 - actual_stock_pct, 2),
        # slots_used = 목표가 찬 슬롯, held_count = 지금 실제로 들고 있는 종목 수. 둘이 다르면 아직 체결 전이다.
        # 슬리브 현금 = 그 슬리브 몫에서 담긴 종목 비중을 뺀 나머지(빈 슬롯 수로 세면 흘러간 비중과 맞지 않는다).
        "slots": {
            key: {
                "alloc_pct": round(shares[key], 2),
                "slots_used": states[key].active_count,
                "held_count": states[key].held_count,
                "top_n": states[key].top_n,
                "cash_pct": round(max(shares[key] - sum(row[f"{key}_weight"] for row in holdings), 0.0), 2),
            }
            for key in keys
        },
    }


def _slot_actions(states: dict[str, Any], labels: dict[str, str], keys: Sequence[str]) -> dict[str, Any]:
    """슬리브별 액션 — 전략에 없는 항목은 빈 목록/None 이다."""
    return {
        key: {
            "label": labels[key],
            # 장중 판정을 쓰는 전략인지 — 그 슬리브의 매도는 아직 '예상'이다.
            "live": states[key].live,
            # 다음 거래일 시가 매도(확정) — 자격 상실·이탈.
            "sells": states[key].sells,
            # 장중 판정 기준 이탈 **예상** — 오늘 종가로 확정된다. 화면 전용.
            "exit_forecast": states[key].exit_forecast,
            # 다음 거래일 시가에 새로 담는 것.
            "entries": states[key].entries,
            "engine_trades": states[key].engine_trades,
        }
        for key in keys
    }


def _attach_exit_forecast(holdings: list[dict[str, Any]], slot_actions: dict[str, Any], keys: Sequence[str]) -> None:
    """장중 이탈 예상 — 표의 매매수량·상태 칸에 예상을 겹쳐 보여주기 위한 행 플래그.

    목표수량·목표비중은 확정 기준 그대로 둔다(장중 값으로 표 전체를 뒤집지 않는다).
    """
    forecast_by_ticker = {row["ticker"]: key for key in keys for row in slot_actions[key]["exit_forecast"]}
    for row in holdings:
        slot = forecast_by_ticker.get(row["ticker"])
        if slot is None or float(row.get("held_quantity") or 0) <= 0:
            continue
        held_qty = float(row["held_quantity"])
        target_qty = float(row.get("target_quantity") or 0)
        weight_all = float(row.get("weight_pct") or 0)
        slot_w = float(row.get(f"{slot}_weight") or 0)
        # 예상 수량 = 보유 − 이탈 후 남을 목표(다른 슬리브 몫). 이미 팔아뒀으면 0 → 표시 없음.
        remain_qty = round(target_qty * (weight_all - slot_w) / weight_all) if weight_all > 0 else 0
        slot_qty = int(round(held_qty - remain_qty))
        row["is_exit_forecast"] = True
        # 이탈 후 남을 목표수량 — 화면이 목표수량 칸에 '(예상)' 으로 겹쳐 쓴다. 확정 목표는 그대로 둔다.
        row["forecast_target_quantity"] = int(remain_qty)
        # 매매수량(예상) — 팔 게 남았으면 그 수량, 이미 반영됐으면 0(매수는 유예되므로 부족분을 사라는 지시가 아니다).
        row["forecast_trade_quantity"] = -slot_qty if slot_qty > 0 else 0


def _value_account(
    account: dict[str, Any], krw_rate: float, price_hint: dict[str, float] | None = None, *, as_of: str | None = None
) -> None:
    """계좌 보유를 원화로 평가해 ``total_assets``·``stock_value``·고정 자산 몫을 채운다.

    ``price_hint`` 는 이미 알고 있는 현재가({티커: 그 시장 통화 가격})다. 없는 종목만
    가격 캐시에서 마지막 종가를 읽는다. 환율을 못 받으면(0) 평가액이 비고 총자산은 현금뿐이다
    — 임의 환율로 채우지 않는다.
    """
    import pandas as pd

    price_by_ticker = dict(price_hint or {})
    missing = [ticker for ticker in account["holdings"] if ticker not in price_by_ticker]
    if missing:
        from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types

        for ticker, frame in load_cached_frames_bulk_from_all_ticker_types(missing).items():
            if frame is None or frame.empty or "Close" not in frame.columns:
                continue
            if as_of is not None:
                frame = frame.loc[:as_of]
            close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            if not close.empty:
                price_by_ticker[ticker] = float(close.iloc[-1])

    stock_value = 0.0
    for ticker, item in account["holdings"].items():
        price = price_by_ticker.get(ticker)
        # 화면에 보이는 현재가는 그 시장 통화 그대로 둔다(달러 종목은 달러로 본다).
        item["price"] = round(float(price), 4) if price else None
        value_krw = item["quantity"] * float(price) * krw_rate if price and krw_rate > 0 else None
        item["value"] = round(value_krw, 2) if value_krw is not None else None
        if value_krw:
            stock_value += value_krw
    # 고정 자산(IS)은 계좌 통화로 들어 있어 여기서 원화로 맞춘다. 슬리브가 굴리지 않지만
    # 총자산에는 들어간다 — 빼면 목표 금액이 그만큼 작아져 실제 계좌와 합이 안 맞는다.
    fixed_value = float(account.get("fixed_asset_native") or 0) * krw_rate if krw_rate > 0 else 0.0
    account["fixed_asset_value"] = round(fixed_value, 2) if fixed_value else 0.0
    total_assets = stock_value + fixed_value + account["cash_balance"]
    account["stock_value"] = round(stock_value, 2)
    account["total_assets"] = round(total_assets, 2)
    # 고정 자산 몫(%) — 사용자가 정하는 값이 아니라 평가액에서 나온다.
    account["fixed_asset_pct"] = round(fixed_value / total_assets * 100.0, 4) if total_assets > 0 else 0.0


def _valued_account(account_id: str, currency: str) -> tuple[dict[str, Any], float]:
    """계좌 보유를 원화로 평가한 상태와 운용 자본(원) — 계좌 평가액(현금 포함)에서 IS 고정 자산을 뺀 금액."""
    krw_rate = _krw_rate(currency)
    if krw_rate <= 0:
        raise RuntimeError(f"{currency} 환율을 읽을 수 없습니다.")
    account = _load_account_state(account_id)
    _value_account(account, krw_rate)
    capital = account["total_assets"] - account["fixed_asset_value"]
    if capital <= 0:
        raise RuntimeError(f"'{account_id}' 계좌 평가액이 없어 운용 기준금액을 정할 수 없습니다.")
    return account, capital


def sleeve_actual_weights(strategy: str, pool: str, tickers: Sequence[str]) -> dict[str, Any]:
    """그 (전략, 종목풀) 슬리브를 쓰는 계좌의 **실제 보유 비중** — 보유 평가액 ÷ 슬리브 한도금액(%).

    슬리브 한도금액은 계좌 운용 자본 × 슬리브 비중이다. 계좌 보유는 슬리브별로 나눠 적히지 않으므로 계좌가 정확히
    하나이고 같은 종목풀에 다른 전략 슬리브가 없을 때만 구한다 — 아니면 `hidden_reason` 에 이유를 담고 비중은 비운다.
    `tickers` 만 센다(그 풀의 종목 — 계좌의 다른 보유는 제외).
    """
    users = [
        (account, [row for row in account["sleeves"] if row["strategy"] == strategy and row["pool"] == pool])
        for account in mix_accounts()
    ]
    users = [(account, mine) for account, mine in users if mine]
    hidden = {"account_name": None, "weight_by_ticker": None, "cash_weight_pct": None}
    if not users:
        return {
            **hidden,
            "hidden_reason": "이 풀의 포트폴리오 슬리브를 쓰는 계좌가 없어 운용 비중을 표시하지 않습니다.",
        }
    if len(users) > 1:
        return {**hidden, "hidden_reason": f"이 풀을 쓰는 계좌가 {len(users)}개라 운용 비중을 표시하지 않습니다."}
    account, mine = users[0]
    if any(row["pool"] == pool and row["strategy"] != strategy for row in account["sleeves"]):
        reason = (
            f"'{account['name']}' 은 같은 풀에 다른 슬리브가 있어 보유를 나눌 수 없어 운용 비중을 표시하지 않습니다."
        )
        return {**hidden, "hidden_reason": reason}
    state, capital = _valued_account(account["account_id"], account["currency"])
    budget = capital * sum(float(row["weight_pct"]) for row in mine) / 100.0
    wanted = {str(ticker).strip().upper() for ticker in tickers}
    # 보유하지 않은 풀 종목은 0% 다 — 모르는 값이 아니다.
    weight_by_ticker = {ticker: 0.0 for ticker in wanted}
    for ticker, item in state["holdings"].items():
        if ticker not in wanted:
            continue
        if item["value"] is None:
            raise RuntimeError(f"{ticker} 가격을 알 수 없어 '{account['name']}' 보유를 평가할 수 없습니다.")
        weight_by_ticker[ticker] = item["value"] / budget * 100.0
    return {
        "account_name": account["name"],
        "hidden_reason": None,
        "weight_by_ticker": weight_by_ticker,
        "cash_weight_pct": 100.0 - sum(weight_by_ticker.values()),
    }
