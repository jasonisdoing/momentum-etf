"""합성 전략 — 계좌 슬리브의 운용 현황과 고정 기준금액 백테스트.

종목과 진입·청산 시점은 각 전략 엔진이 정한다. 합성은 저장된 KRW 기준금액에
슬리브 배분과 회수·채우기 정책을 적용한다. 백테스트에서 회수 기준 매도 대금은
KRW 인출 누계로 격리하며 이후 매수에 쓰지 않는다.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from core.strategy.mix.actions import build_action_groups
from core.strategy.mix.targets import dated_target_shares
from utils.cash_model import currency_for_country
from utils.logger import get_app_logger
from utils.mix_sleeve import STRATEGY_LABELS, SleeveSpec
from utils.stock_memo_store import attach_stock_memos
from utils.trade_stats import summarize_trades

logger = get_app_logger()


def _sm_settings_map() -> dict[str, Any]:
    # SM 은 `{pool, settings_by_pool}` 래핑 구조라 풀 맵만 꺼낸다 (신고가는 평면 풀 맵).
    from utils.momentum_service import load_settings_map

    return dict(load_settings_map().get("settings_by_pool") or {})


def _nh_settings_map() -> dict[str, Any]:
    from utils.new_high_service import load_settings_map

    return load_settings_map()


def _all_active_pools() -> list[str]:
    """활성 종목풀 전체 — 계좌가 가리키는 슬리브 풀의 표기(이름·아이콘)를 붙이는 데 쓴다."""
    from utils.settings_loader import list_available_ticker_types

    return list(list_available_ticker_types())


def _pool_options(pools: list[str]) -> list[dict[str, Any]]:
    """화면 셀렉트용 풀 정보 — 공용 formatPoolLabel 이 쓰는 필드(ticker_type·name·icon·order)."""
    from utils.settings_loader import get_ticker_type_settings

    options = []
    for pool in pools:
        try:
            settings = get_ticker_type_settings(pool) or {}
        except Exception:
            settings = {}
        options.append(
            {
                "ticker_type": pool,
                "name": str(settings.get("name") or pool),
                "icon": str(settings.get("icon") or ""),
                "order": settings.get("order"),
                # 화면이 계좌 국가에 맞는 풀만 뿌리는 데 쓴다.
                "country_code": str(settings.get("country_code") or "").strip().lower(),
                "pool_kind": str(settings.get("pool_kind") or "").strip().lower(),
            }
        )
    options.sort(key=lambda o: (o["order"] is None, o["order"]))
    return options


# 백테스트 기간 선택지 — 신고가 화면과 동일한 목록 (상한 60개월 = 신고가 엔진의 최대).
def month_options() -> list[int]:
    """기간 선택지 — 종목풀 백테스트와 같은 목록(`get_month_options`)이 단일 소스.

    전략별로 목록을 따로 두면 화면마다 고를 수 있는 기간이 달라진다.
    """
    from utils.pool_signal_backtest_service import get_month_options

    return get_month_options()


# 고정 자산(IS, International Shares) — 사용자가 화면에서 수량을 못 바꾸는 호주 계좌 항목이다.
# 원장 holdings 가 아니라 `intl_shares_value` 필드에 계좌 통화로 들어 있고, 합성은 이걸
# **굴리지 않되 총자산에는 넣는다**(자산 관리 화면과 같은 취급).
FIXED_ASSET_TICKER = "IS"
FIXED_ASSET_NAME = "International Shares"


def mix_weights(account_settings: dict[str, Any]) -> dict[str, float]:
    """계좌 설정의 합성 배분(%) — {슬롯키_pct: 값} + cash_pct. 저장이 없으면 빈 배분.

    슬리브와 현금은 항상 함께 저장되므로(계좌 설정 검증), 슬리브가 없으면 미저장으로 보고
    현금 100% 를 돌려준다 — 일부만 읽어 섞으면 합이 100 이 아닌 배분이 만들어진다.
    """
    from utils.account_settings_store import mix_sleeves_of

    sleeves = mix_sleeves_of(account_settings)
    if not sleeves or account_settings.get("mix_cash_pct") is None:
        return {"cash_pct": 100.0}
    weights = {f"{row['key']}_pct": float(row["weight_pct"]) for row in sleeves}
    weights["cash_pct"] = float(account_settings["mix_cash_pct"])
    return weights


def mix_accounts() -> list[dict[str, Any]]:
    """합성 전략을 운용하는 계좌 목록 — 계좌 설정에서 **합성 사용**을 켠 계좌.

    슬리브는 각각 (전략, 종목풀) 쌍이고 `/strategy-mix` 화면에서 고른다. 하나만 쓰기,
    모멘텀 둘, 신고가 둘, 섞기 전부 가능하며 완전히 같은 조합 중복만 저장이 막는다.

    **조합이 아직 없어도 목록에 올린다** — 그래야 그 화면에서 고를 수 있다. 계산은
    `_resolve_mix_account` 가 조합이 갖춰졌는지 확인하고, 없으면 명시적으로 알린다.
    """
    from utils.account_settings_store import mix_sleeves_of
    from utils.settings_loader import get_account_settings, list_available_accounts

    pool_names = {option["ticker_type"]: option for option in _pool_options(_all_active_pools())}

    accounts: list[dict[str, Any]] = []
    for account_id in list_available_accounts():
        try:
            settings = get_account_settings(account_id) or {}
        except Exception:
            continue
        inner = settings.get("settings") or settings
        if not bool(inner.get("mix_enabled")):
            continue
        # 슬리브 목록 — 순서가 곧 슬롯(A·B·C). 키 부여는 계좌 설정 저장소가 한 곳에서 한다.
        sleeves = [
            {
                **row,
                # 화면에 실제로 쓸 이름 — 사용자가 붙인 이름이 있으면 그것, 없으면 전략 이름.
                "label": row["name"] or STRATEGY_LABELS.get(row["strategy"], row["strategy"]),
                "pool_label": pool_names.get(row["pool"]),
            }
            for row in mix_sleeves_of(inner)
        ]
        accounts.append(
            {
                "account_id": account_id,
                "name": str(inner.get("name") or account_id),
                "icon": str(inner.get("icon") or ""),
                "order": inner.get("order"),
                "currency": currency_for_country(inner["country_code"]),
                "country_code": str(inner.get("country_code") or "").strip().lower(),
                "sleeves": sleeves,
                # 조합이 다 갖춰졌는지 — 화면이 "고르세요" 상태를 판단하는 데 쓴다.
                "mix_ready": bool(sleeves) and all(row["strategy"] and row["pool"] for row in sleeves),
                # 오늘의 액션 슬랙 알람 토글 상태 — 화면 헤더가 그대로 보여준다.
                "mix_slack_enabled": bool(inner.get("mix_slack_enabled")),
                # 미설정 기존 계좌는 필터를 적용하지 않는다.
                **{key: inner.get(key) for key in ("mix_capital_krw", "mix_harvest_pct", "mix_refill_pct")},
                # 비워 두는 현금 몫(%) — 슬리브 배분은 sleeves 안에 있다.
                "mix_cash_pct": mix_weights(inner)["cash_pct"],
            }
        )
    accounts.sort(key=lambda item: (item["order"] is None, item["order"]))
    return accounts


def default_sleeves() -> list[dict[str, Any]]:
    """저장 이력이 없는 계좌에서 화면이 채울 슬리브 초안 — 최소 개수만큼 균등 배분.

    저장값이 아니라 **입력 초안**이다. 전략·종목풀은 비워 둔다(사용자가 고른다).
    """
    from utils.account_settings_store import MIN_MIX_SLEEVES, MIX_SLEEVE_KEYS

    share = round(100.0 / MIN_MIX_SLEEVES, 2)
    return [
        {"key": MIX_SLEEVE_KEYS[index], "strategy": "", "pool": "", "name": "", "weight_pct": share, "label": ""}
        for index in range(MIN_MIX_SLEEVES)
    ]


def _settings_summaries() -> dict[str, list[dict[str, str]]]:
    """`"전략:풀"` → 그 전략 화면에 저장된 설정 요약.

    계좌가 아니라 **조합**에 매단다 — 합성 화면에서 셀렉트를 바꾸는 즉시 그 조합의
    설정이 보여야 하는데, 계좌 슬리브에 붙이면 저장 전 초안에서는 값이 어긋난다.
    항목·라벨은 슬리브 어댑터(`utils.mix_sleeve.settings_summary`)가 단일 소스다.
    """
    from utils.mix_sleeve import STRATEGY_OPTIONS, settings_map, settings_summary, validate_settings

    summaries: dict[str, list[dict[str, str]]] = {}
    for strategy in STRATEGY_OPTIONS:
        try:
            stored_by_pool = settings_map(strategy)
        except Exception as error:
            # 조용히 빈칸으로 두지 않는다 — 설정이 깨졌으면 화면에서 보여야 고칠 수 있다.
            summaries[f"{strategy}:*"] = [{"label": "설정 오류", "value": str(error)}]
            continue
        for pool, stored in stored_by_pool.items():
            try:
                # 종목 수처럼 풀 설정에서 오는 값은 검증을 거쳐야 채워진다.
                checked = validate_settings(strategy, {"pool": pool, **stored})
                summaries[f"{strategy}:{pool}"] = settings_summary(strategy, checked)
            except Exception as error:
                summaries[f"{strategy}:{pool}"] = [{"label": "설정 오류", "value": str(error)}]
    return summaries


def mix_meta() -> dict[str, Any]:
    """화면 초기용 — 운용 계좌 목록과 기간 선택지 (백테스트 계산 없음)."""
    from utils.account_settings_store import MAX_MIX_SLEEVES, MIN_MIX_SLEEVES
    from utils.mix_sleeve import STRATEGY_OPTIONS

    accounts = mix_accounts()
    return {
        "accounts": accounts,
        "month_options": month_options(),
        # 슬리브 개수 제한·초안 — 백엔드 상수가 단일 소스(프론트에 복사본을 두지 않는다).
        "min_sleeves": MIN_MIX_SLEEVES,
        "max_sleeves": MAX_MIX_SLEEVES,
        "default_sleeves": default_sleeves(),
        # 조합 셀렉트 선택지 — 이 화면에서 전략·종목풀을 고른다(계좌 설정은 사용 여부만).
        "strategy_options": [{"value": key, "label": STRATEGY_LABELS[key]} for key in STRATEGY_OPTIONS],
        # 풀은 계좌 국가에 맞는 것만 고를 수 있어야 해서 국가 코드를 함께 준다.
        "pool_options": _pool_options(_all_active_pools()),
        # 조합별 저장 설정 요약 — 화면이 슬리브 오른쪽에 읽기 전용으로 보여준다.
        "settings_summaries": _settings_summaries(),
        # 기본 선택 — 목록의 첫 계좌. 화면이 마지막 선택을 로컬스토리지에 기억한다.
        "account_id": accounts[0]["account_id"] if accounts else "",
    }


def mix_weights_for_account(account_id: str) -> dict[str, float]:
    """그 계좌의 합성 배분(%) — {슬롯키_pct: 값} + cash_pct. 목록에 없으면 현금 100%.

    슬리브마다 풀이 다를 수 있으므로 풀로 계좌를 되찾을 수 없다 — 계좌가 기준이다.
    백테스트·운영 화면·슬랙이 모두 같은 배분을 쓰도록 여기서만 읽는다.
    """
    target = str(account_id or "").strip()
    for account in mix_accounts():
        if account["account_id"] == target:
            weights = {f"{row['key']}_pct": float(row["weight_pct"]) for row in account["sleeves"]}
            weights["cash_pct"] = float(account["mix_cash_pct"])
            return weights
    return {"cash_pct": 100.0}


def _resolve_mix_account(account_id: str | None) -> dict[str, Any]:
    """계좌를 확정하고 슬리브별 풀·저장 설정을 검증해 돌려준다 (백테스트·운영 공용).

    반환 컨텍스트를 두 진입점(`mix_positions`·`run_mix_backtest`)이 그대로 쓴다 —
    풀을 각자 다시 꺼내면 슬리브 배정이 갈릴 수 있다.
    국가·통화는 모든 풀이 같도록 계좌 설정이 강제하므로 첫 슬리브 풀에서 대표로 읽는다.
    """
    from utils.mix_sleeve import settings_map, validate_settings
    from utils.settings_loader import get_ticker_type_settings

    accounts = mix_accounts()
    if not accounts:
        raise RuntimeError("합성 전략을 운용하는 계좌가 없습니다 — 계좌 설정에서 합성 사용을 켜세요.")
    target = str(account_id or "").strip()
    if target:
        account = next((row for row in accounts if row["account_id"] == target), None)
        if account is None:
            raise ValueError(f"합성 전략을 운용하지 않는 계좌입니다: {account_id}")
    else:
        # 기본 계좌 — 계좌 order 순 첫 번째. 화면은 마지막 선택을 따로 기억한다.
        account = accounts[0]

    if not account.get("mix_ready"):
        raise RuntimeError(
            f"'{account['account_id']}' 의 합성 조합이 아직 정해지지 않았습니다 — "
            "화면 상단에서 각 슬리브의 전략과 종목풀을 고르고 저장하세요."
        )

    # 슬리브 — 키는 합성 안에서 이 슬리브를 가리키는 이름이다(순서대로 a·b·c).
    wanted = account["sleeves"]
    saved_by_slot = {row["key"]: settings_map(row["strategy"]).get(row["pool"]) for row in wanted}
    missing = [
        f"{STRATEGY_LABELS.get(row['strategy'], row['strategy'])}({row['pool']})"
        for row in wanted
        if not saved_by_slot[row["key"]]
    ]
    if missing:
        raise RuntimeError(f"{' · '.join(missing)} 설정이 저장돼 있지 않습니다 — 해당 전략 화면에서 먼저 저장하세요.")

    # 벤치마크는 **계좌** 것을 쓴다 — 합성은 한 계좌를 통째로 굴린 결과라 대조군도 계좌 단위여야
    # 한다. 슬리브 풀 것을 쓰면 모멘텀 풀을 바꾸는 것만으로 성과 비교 기준이 따라 바뀐다.
    from utils.settings_loader import get_account_settings

    account_settings = get_account_settings(account["account_id"]) or {}
    if any(account_settings.get(key) is None for key in ("mix_capital_krw", "mix_harvest_pct", "mix_refill_pct")):
        raise RuntimeError(
            "운용 기준금액(KRW)·회수 기준·채우기 기준을 먼저 저장하세요. 새 방식의 주문은 아직 생성하지 않습니다."
        )
    benchmark = account_settings.get("benchmark") or {}
    benchmark_ticker = str(benchmark.get("ticker") or "").strip().upper()
    if not benchmark_ticker:
        raise RuntimeError(
            f"'{account['account_id']}' 계좌에 벤치마크가 설정돼 있지 않습니다 — 계좌 설정에서 지정하세요."
        )

    # 슬리브 목록 — 합성은 이것만 보고 돈다. 전략 조합(모멘텀 둘·신고가 둘·섞기)에
    # 무관하게 시뮬레이션·백테스트가 같은 코드로 돈다.
    slots = [
        SleeveSpec(
            key=row["key"],
            strategy=row["strategy"],
            pool=row["pool"],
            settings=validate_settings(row["strategy"], {**saved_by_slot[row["key"]], "pool": row["pool"]}),
            name=row["name"],
        )
        for row in wanted
    ]

    # 국가·통화는 모든 슬리브 풀이 같도록 계좌 설정이 강제하므로 첫 슬리브에서 대표로 읽는다.
    country = str(account_settings["country_code"]).strip().lower()
    if country not in {"kor", "us", "au"}:
        raise RuntimeError(f"지원하지 않는 계좌 국가입니다: {country}")
    currency = currency_for_country(country)
    for spec in slots:
        pool_settings = get_ticker_type_settings(spec.pool) or {}
        if pool_settings.get("country_code") != country or pool_settings.get("currency") != currency:
            raise RuntimeError(f"{spec.pool}의 국가·통화가 계좌 국가 기준과 다릅니다.")
    return {
        "slots": slots,
        "account_id": account["account_id"],
        "account_name": account["name"],
        # 국가·통화 — 거래 달력(월초 리밸런싱 판정)과 원화 환산에 쓴다.
        "country": country,
        "currency": currency,
        **{key: float(account_settings[key]) for key in ("mix_capital_krw", "mix_harvest_pct", "mix_refill_pct")},
        "benchmark_ticker": benchmark_ticker,
        "benchmark_name": str(benchmark.get("name") or benchmark_ticker).strip(),
    }


def holding_charts_for_account(account_id: str | None, tickers: list[str]) -> list[dict[str, Any]]:
    """보유 종목 차트 — 슬리브별로 그 전략의 기준선을 그린다 (공용 `holding_chart_service`).

    모멘텀 슬리브 종목은 단기·장기, 신고가는 이탈 이평선, 포트폴리오는 풀 판정 이평선.
    같은 티커가 두 슬리브에 있으면 앞 슬롯(선정 우선) 기준 한 장만 그린다 — 행도 한 줄이다.
    """
    from utils.holding_chart_service import holding_charts as build_charts
    from utils.mix_sleeve import MOMENTUM, NEW_HIGH
    from utils.settings_loader import get_ticker_type_settings

    ctx = _resolve_mix_account(account_id)
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


def _slot_labels(slots: list[SleeveSpec]) -> dict[str, str]:
    """슬롯 표시 이름 — 「A. 모멘텀」. 같은 전략이 두 슬롯에 올 수 있어 기호를 앞에 둔다."""
    return {spec.key: f"{spec.key.upper()}. {spec.label}" for spec in slots}


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
) -> list[dict[str, Any]]:
    """계좌 보유와 목표를 대조해 수량 지시를 붙인다. 전량 매도 요약 목록을 돌려준다.

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

    # 목표 주수 = **고정 기준금액 ÷ 1주 값의 내림**(`capital_policy`). 계좌 평가액으로 예산을
    # 매일 다시 나누지 않는다 — 그러면 남는 돈이 그날그날 다른 종목에 얹혀, 엔진이 하지도
    # 않는 매매를 시킨다. 실제 보유는 목표를 바꾸지 않고 회수·채우기 판단에만 쓴다.
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
    sell_all: list[dict[str, Any]] = []
    for ticker, item in sorted(account["holdings"].items()):
        if ticker in target_tickers:
            continue
        value = item.get("value")
        sell_all.append({"ticker": ticker, "name": item["name"], "quantity": item["quantity"], "value": value})
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
    return sell_all


def mix_positions(account_id: str | None = None) -> dict[str, Any]:
    """오늘 기준 합성 운영 상태 — 보유 목록(목표 비중)·현금 비중·오늘의 액션.

    개별 엔진의 종목·진입·청산을 받아 고정 원화 기준금액과 회수·채우기 정책을 적용한다.
    포트폴리오는 저장 비중, 슬롯 전략은 슬리브 몫 ÷ 슬롯 수(빈 슬롯 = 현금).
    겹치는 종목은 한 행으로 합친다 — 계좌에는 그 종목이 하나뿐이라, 슬리브별로 나누면
    보유 수량이 두 번 세어지고 매매 지시가 반대로 나온다.
    """
    import pandas as pd

    from utils.mix_sleeve import slot_state

    ctx = _resolve_mix_account(account_id)
    slots: list[SleeveSpec] = ctx["slots"]
    keys = [spec.key for spec in slots]
    labels = _slot_labels(slots)
    from utils.strategy_settings import require_start_date

    for spec in slots:
        require_start_date(spec.settings)
    states = {spec.key: slot_state(spec) for spec in slots}

    from core.strategy.mix.capital_policy import internal_target_weights

    base_weights = mix_weights_for_account(ctx["account_id"])
    reserved_cash_share = base_weights["cash_pct"]
    shares = {key: base_weights[f"{key}_pct"] for key in keys}
    # 엔진은 종목·시점을 정하고 합성은 고정 배정 비중을 적용한다.
    for spec in slots:
        weights = internal_target_weights(strategy=spec.strategy, settings=spec.settings)
        for target in states[spec.key].targets:
            weight = weights[str(target["ticker"])] if isinstance(weights, dict) else weights
            target["drift_pct"] = weight
            if spec.strategy == "portfolio":
                target["status"] = f"설정 비중 {weight:.2f}%"

    holdings: list[dict[str, Any]] = []
    by_ticker: dict[str, dict[str, Any]] = {}

    def add_target(source: str, target: dict[str, Any], weight: float) -> None:
        ticker = str(target["ticker"]).strip()
        row = by_ticker.get(ticker)
        if row is None:
            row = {
                "ticker": ticker,
                "name": target.get("name") or ticker,
                "sources": [],
                "weight_pct": 0.0,
            }
            # 슬리브별 값 — 현금 비중·화면 요약·툴팁이 슬롯 키로 읽는다.
            for key in keys:
                row[f"{key}_weight"] = 0.0
                row[f"{key}_status"] = None
                # 각 전략 화면과 같은 상태 컬럼이 쓰는 값 — 문구는 화면이 공용 함수로 만든다.
                row[f"{key}_plan"] = None
                row[f"{key}_days"] = None
                row[f"{key}_is_new"] = None
                row[f"{key}_exit_reason"] = None
                # 전략 수익률(이론값) — 그 전략이 잡은 편입가 대비.
                row[f"{key}_return_pct"] = None
                # 보유 기간 표기 — 전략마다 단위가 다르다("3주" vs "12일"). 슬롯에 어느
                # 전략이 오든 맞게 읽히도록 숫자가 아니라 완성된 문자열로 내려준다.
                row[f"{key}_held_label"] = None
                # 차트의 진입 화살표 — 슬리브마다 편입 시점이 다르므로 슬롯별로 들고 간다.
                row[f"{key}_entry_date"] = None
                row[f"{key}_entry_price"] = None
                row[f"{key}_fill_date"] = None
            row["price"] = target.get("price")
            row["change_pct"] = target.get("change_pct")
            by_ticker[ticker] = row
            holdings.append(row)
        if source not in row["sources"]:
            row["sources"].append(source)
        row["weight_pct"] += weight
        row[f"{source}_weight"] += weight
        if row.get("price") is None:
            row["price"] = target.get("price")
        if row.get("change_pct") is None:
            row["change_pct"] = target.get("change_pct")
        row[f"{source}_status"] = target.get("status")
        row[f"{source}_plan"] = target.get("plan")
        row[f"{source}_days"] = target.get("days")
        row[f"{source}_is_new"] = bool(target.get("is_new"))
        row[f"{source}_exit_reason"] = target.get("exit_reason")
        if target.get("return_pct") is not None:
            row[f"{source}_return_pct"] = round(float(target["return_pct"]), 2)
        if target.get("held_label"):
            row[f"{source}_held_label"] = target["held_label"]
        if target.get("entry_date"):
            row[f"{source}_entry_date"] = str(target["entry_date"])[:10]
        if target.get("entry_price") is not None:
            row[f"{source}_entry_price"] = float(target["entry_price"])
        if target.get("fill_date"):
            row[f"{source}_fill_date"] = str(target["fill_date"])[:10]

    # 매도 예정(자격 상실·이탈)은 목표 비중 0 이다 — 다음 시가에 전량 팔고 그 슬롯은
    # 다음 교체까지 현금이다. 비중을 남겨두면 팔아야 할 종목의 매매수량이 0 으로 보인다.
    for key in keys:
        state = states[key]
        for target in state.targets:
            if target["is_exiting"]:
                weight = 0.0
            elif target.get("drift_pct") is not None:
                weight = shares[key] * float(target["drift_pct"]) / 100.0
            else:
                raise ValueError(f"{key}: 종목 기준 비중이 없습니다.")
            add_target(key, target, weight)

    # 매월 첫 거래일 = 슬리브 배분 리밸런싱 날 (그 시장 달력 기준).
    from config import MARKET_SCHEDULES
    from utils.trading_calendar import get_trading_days

    country = ctx["country"]
    # 종목 가격의 통화 — 계좌 원장(원화)과 맞추려면 환율이 필요하다.
    pool_currency = ctx["currency"]
    tz_name = str((MARKET_SCHEDULES.get(country) or {}).get("timezone") or "Asia/Seoul")
    today_local = pd.Timestamp.now(tz=tz_name).date()

    # 다음 거래일 — 모든 체결은 시가라 액션 묶음의 실제 날짜가 된다. 연휴가 끼면
    # 이 날짜가 교체일과 같아질 수 있고, 그러면 화면이 한 묶음으로 합친다.
    ahead = get_trading_days(
        today_local.strftime("%Y-%m-%d"), (today_local + timedelta(days=21)).strftime("%Y-%m-%d"), country
    )
    # 체결은 **시가**다 — 오늘 장이 아직 안 열렸으면 오늘 시가에 체결할 수 있다.
    # 오늘을 무조건 건너뛰면 개장 전인 시장에서 지시가 하루 뒤로 밀린다
    # (뉴욕 06:35 에 보는데 체결일이 다음 거래일로 나왔다).
    from utils.trading_calendar import is_market_day_started

    def _can_fill_at_open(day) -> bool:
        if day > today_local:
            return True
        if day < today_local:
            return False
        return not is_market_day_started(country, pd.Timestamp(day))

    next_trading_day = next((str(day.date()) for day in ahead if _can_fill_at_open(day.date())), None)

    # 이벤트 없는 조정(목표·보유 차이)의 기준일 — 시가 체결 신호가 아니라 "지금 낼" 주문이다.
    # 오늘이 거래일이고 아직 마감 전이면(개장 전=오늘 시가, 장중=즉시) 오늘, 마감 후·휴장은
    # 다음 거래일. 예전에는 장중이면 다음 거래일로 밀려, 이미 벌어진 수량 차이의 매수 지시가
    # 하루 뒤 시가로 나왔다.
    from utils.trading_calendar import is_market_day_completed

    today_is_trading = bool(ahead) and ahead[0].date() == today_local
    adjustment_day = (
        today_local.strftime("%Y-%m-%d")
        if today_is_trading and not is_market_day_completed(country, pd.Timestamp(today_local))
        else next_trading_day
    )
    # 장중 조정이면 그룹 제목이 '시가'가 아니라 '장중'이어야 한다.
    adjustment_intraday = adjustment_day is not None and adjustment_day != next_trading_day

    # ── 적용 계좌 — 이 계산의 기준 계좌 그대로다(슬리브별 풀이 여기서 나왔다).
    account = _load_account_state(ctx["account_id"])
    krw_rate = _krw_rate(pool_currency)
    if krw_rate <= 0:
        raise RuntimeError(f"{pool_currency} 환율을 읽을 수 없습니다.")
    if account is not None:
        # 목표와 같은 확정일로 계좌를 평가한다. 표시용 실시간 가격은 수량 계산에 넣지 않는다.
        valuation_date = min(state.as_of for state in states.values() if state.as_of)
        _value_account(
            account,
            krw_rate,
            {row["ticker"]: row["price"] for row in holdings if row.get("price")},
            as_of=valuation_date,
        )

        # IS는 매매하지 않고 현재 비중을 목표에도 보존한다. 남은 몫만 슬리브와 현금에 배정한다.
        fixed_pct = float(account["fixed_asset_pct"])
        investable_ratio = 1.0 - fixed_pct / 100.0
        shares = {key: value * investable_ratio for key, value in shares.items()}
        reserved_cash_share *= investable_ratio
        for row in holdings:
            row["weight_pct"] *= investable_ratio
            for key in keys:
                row[f"{key}_weight"] *= investable_ratio
        sleeve_amount_krw = {key: ctx["mix_capital_krw"] * shares[key] / 100.0 for key in keys}
        target_schedule = dated_target_shares(
            {key: state.targets for key, state in states.items()},
            sleeve_amount_krw,
            krw_rate,
            ctx["mix_capital_krw"],
            next_trading_day,
            adjustment_day=adjustment_day,
        )
        target_shares = target_schedule[max(target_schedule)]["quantities"]
        account["sell_all"] = _attach_account_targets(
            holdings, account, krw_rate, slot_keys=keys, target_shares=target_shares, capital_krw=ctx["mix_capital_krw"]
        )

        if account["fixed_asset_value"]:
            holdings.append(
                {
                    "ticker": FIXED_ASSET_TICKER,
                    "name": FIXED_ASSET_NAME,
                    "sources": [],
                    "is_fixed_asset": True,
                    "price": None,
                    "weight_pct": fixed_pct,
                    "actual_weight_pct": fixed_pct,
                    "held_value": account["fixed_asset_value"],
                    "current_weight_pct": fixed_pct,
                    "target_amount": ctx["mix_capital_krw"] * fixed_pct / 100.0,
                    "held_quantity": None,
                    "target_quantity": None,
                    "trade_quantity": None,
                    **{f"{key}_weight": 0.0 for key in keys},
                }
            )

    # 비중 합계·슬리브 현금 — 고정 자산 축소가 끝난 뒤의 값이라야 실제 계좌와 맞는다.
    stock_pct = sum(row["weight_pct"] for row in holdings)
    # 실제 가능한 주식 비중 — 단주로 못 채운 나머지는 전부 현금이다.
    actual_stock_pct = sum(float(row.get("actual_weight_pct") or 0) for row in holdings)
    # 슬리브 현금 = 그 슬리브 몫에서 담긴 종목 비중을 뺀 나머지. 빈 슬롯 수로 세면
    # 흘러간 비중과 맞지 않는다(종목이 오르면 남는 현금은 그만큼 줄어든다).
    sleeve_cash = {key: max(shares[key] - sum(row[f"{key}_weight"] for row in holdings), 0.0) for key in keys}

    # 추세 이탈(행 전체 회색)용 — 행이 속한 슬리브의 **종목풀 설정** 이평선 기준.
    _attach_disparity(holdings, {spec.key: spec.pool for spec in slots})
    _attach_industry(holdings, ctx["country"])

    # 종목 메모 — 계좌가 아니라 **종목**에 붙는다(utils/stock_memo_store). 순위·자산 관리·
    # 모멘텀 화면과 같은 값이다. 전량 매도 행까지 붙은 뒤에 한 번에 읽는다.
    attach_stock_memos(holdings)

    # 장중 반영은 그 정보를 주는 전략에서만 온다(신고가). 슬리브 어디에도 없으면 거짓이다.
    live = any(states[key].live for key in keys)
    # 데이터 기준일 — 그 값을 주는 전략(신고가)이 있으면 그걸 쓰고, 없으면 오늘이다.
    as_of_value = next((states[key].as_of for key in keys if states[key].as_of), None) or str(today_local)

    payload = {
        "computed_at": datetime.now().astimezone().isoformat(),
        "currency": ctx["currency"],
        "krw_rate": krw_rate,
        "capital_krw": ctx["mix_capital_krw"],
        "account_id": ctx["account_id"],
        # 화면이 표시용 시세를 60초마다 갱신할 때 쓴다(시세 소스가 국가별로 다르다).
        "country": country,
        "account": account,
        "as_of": as_of_value,
        "next_trading_day": next_trading_day,
        "live": live,
        "summary": {
            "stock_pct": round(stock_pct, 2),
            "cash_pct": round(100 - stock_pct, 2),
            # 계좌 금액·단주를 감안해 **실제로 도달할 수 있는** 비중. 목표(위)와의 차이가
            # 곧 단주로 못 채워 현금으로 남는 몫이다.
            "actual_stock_pct": round(actual_stock_pct, 2),
            "actual_cash_pct": round(100 - actual_stock_pct, 2),
            # 총 현금 중 **두 전략에 아예 주지 않고 비워 둔 몫**. 나머지는 빈 슬롯에서 생긴다.
            "reserved_cash_pct": round(reserved_cash_share, 2),
            # 월초에 되돌릴 배분 — 화면이 "지금 몫"과 "목표 배분"을 함께 보여준다.
            "base_weights": {name: round(value, 2) for name, value in base_weights.items()},
            # 슬리브별 현황 — 슬롯 키로 담는다(화면이 키 목록을 돌며 그린다).
            # slots_used = 목표가 찬 슬롯, held_count = 지금 실제로 들고 있는 종목 수.
            # 둘이 다르면 아직 체결 전이라는 뜻이라 화면이 구분해서 보여준다.
            "slots": {
                key: {
                    "alloc_pct": round(shares[key], 2),
                    "slots_used": states[key].active_count,
                    "held_count": states[key].held_count,
                    "top_n": states[key].top_n,
                    "cash_pct": round(sleeve_cash[key], 2),
                }
                for key in keys
            },
        },
        # 슬리브별 값은 여기서는 평평한 채로 둔다 — 아래 액션 조립이 그 형태를 읽는다.
        # 화면 형태(`slots`)로 모으는 것은 return 직전이다.
        "holdings": holdings,
        "actions": {
            # 슬리브별 액션 — 슬롯 키로 담는다. 전략에 없는 항목은 빈 목록/None 이다.
            "slots": {
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
            },
        },
    }
    # 주중 이탈 예상 — 표의 매매수량·상태 칸에 예상을 겹쳐 보여주기 위한 행 플래그.
    # 목표수량·목표비중은 확정 기준 그대로 둔다(장중 값으로 표 전체를 뒤집지 않는다).
    forecast_by_ticker: dict[str, str] = {}
    for key in keys:
        for row in payload["actions"]["slots"][key]["exit_forecast"]:
            forecast_by_ticker[row["ticker"]] = key
    for row in payload["holdings"]:
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
        # 이탈 후 남을 목표수량 — 화면이 목표수량 칸에 이 값을 '(예상)' 으로 겹쳐 쓴다.
        # 확정 목표(target_quantity)는 그대로 둔다: 예상이 풀리면 그 값으로 돌아간다.
        row["forecast_target_quantity"] = int(remain_qty)
        # 매매수량(예상) — 이대로 끝나면 오늘 할 일: 팔 게 남았으면 그 수량, 이미 반영됐으면
        # 0 (매수는 유예되므로 부족분을 사라는 지시가 아니다).
        row["forecast_trade_quantity"] = -slot_qty if slot_qty > 0 else 0

    # 오늘의 액션 — 화면·슬랙 알람이 같은 결과를 쓴다(조립 단일 소스).
    currency = ctx["currency"]
    payload["actions"]["groups"] = build_action_groups(
        payload["holdings"],
        payload["actions"],
        next_trading_day,
        harvest_pct=ctx["mix_harvest_pct"],
        refill_pct=ctx["mix_refill_pct"],
        cash_balance=account["cash_balance"] / krw_rate if account is not None else 0,
        currency=currency,
        adjustment_day=adjustment_day,
        adjustment_intraday=adjustment_intraday,
        today=today_local.strftime("%Y-%m-%d"),
        target_schedule=target_schedule if account is not None else {},
    )
    # 슬리브별 값을 `slots[키]` 로 모아 내보낸다 — 화면은 슬롯 키를 돌며 읽는다.
    payload["holdings"] = [_holding_payload(row, keys) for row in payload["holdings"]]
    return payload


def _simulate_mix(
    ctx: dict[str, Any], results: dict[str, dict[str, Any]], *, through_date: str | None
) -> dict[str, Any]:
    """전략별 엔진의 종목 편입 기간을 읽어 공통 고정 기준금액 재생에 전달한다."""
    import pandas as pd

    from core.strategy.mix.capital_policy import internal_target_weights
    from core.strategy.mix.capital_replay import replay_capital
    from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types
    from utils.data_loader import get_exchange_rate_series
    from utils.pool_settings_store import get_pool_slippage

    if through_date is not None:
        raise ValueError("고정 기준금액 합성은 미래 월초 재배분을 하지 않습니다.")
    base = mix_weights_for_account(ctx["account_id"])
    starts = [result["daily"][0]["date"] for result in results.values()]
    ends = [result["daily"][-1]["date"] for result in results.values()]
    start, end = max(starts), min(ends)
    intervals = []
    costs = {}
    for spec in ctx["slots"]:
        weights = internal_target_weights(strategy=spec.strategy, settings=spec.settings)
        cost = tuple(value / 100.0 for value in get_pool_slippage(spec.pool))
        for row in results[spec.key]["trades"]:
            if spec.strategy == "portfolio" and row["side"] != "buy":
                continue
            ticker = row["ticker"]
            weight = weights[ticker] if isinstance(weights, dict) else weights
            amount = ctx["mix_capital_krw"] * base[f"{spec.key}_pct"] / 100.0 * weight / 100.0
            entry = row["date"] if spec.strategy == "portfolio" else row["entry_date"]
            exit_date = None if spec.strategy == "portfolio" else row.get("exit_date")
            intervals.append((ticker, entry, exit_date, amount))
            if ticker in costs and costs[ticker] != cost:
                raise ValueError(f"중복 종목 {ticker}의 슬리브별 슬리피지가 다릅니다. 같은 값으로 설정하세요.")
            costs[ticker] = cost
    frames = load_cached_frames_bulk_from_all_ticker_types(sorted(costs))
    # 거래일은 엔진의 일별 결과 교집합이다. 특정 종목의 상장일로 전체 구간을 줄이지 않는다.
    common = set.intersection(*[{row["date"] for row in result["daily"]} for result in results.values()])
    index = pd.DatetimeIndex(sorted(day for day in common if start <= day <= end))
    if index.empty:
        raise ValueError("합성 재생의 공통 거래일이 없습니다.")
    close = pd.DataFrame(index=index)
    opened = pd.DataFrame(index=index)
    for ticker in costs:
        frame = frames.get(ticker)
        if frame is None or frame.empty or not {"Open", "Close"}.issubset(frame.columns):
            raise ValueError(f"합성 재생 가격이 없습니다: {ticker}")
        close[ticker] = frame["Close"].reindex(frame.index.union(index)).sort_index().ffill().reindex(index)
        opened[ticker] = frame["Open"].reindex(index)
    if ctx["currency"] == "KRW":
        fx = pd.Series(1.0, index=index)
    else:
        symbol = {"USD": "KRW=X", "AUD": "AUDKRW=X"}[ctx["currency"]]
        raw_fx = get_exchange_rate_series(index[0] - pd.Timedelta(days=10), index[-1], symbol=symbol)
        fx = raw_fx.reindex(raw_fx.index.union(index)).sort_index().ffill().reindex(index)
        if fx.isna().any() or (fx <= 0).any():
            raise ValueError("합성 백테스트의 일별 환율이 부족합니다.")
    targets = {}
    for day in index:
        date = str(day.date())
        amounts = {}
        for ticker, entry, exited, amount in intervals:
            if entry <= date and (exited is None or date < exited):
                amounts[ticker] = amounts.get(ticker, 0.0) + amount
        targets[date] = amounts
    replayed = replay_capital(
        close=close,
        opened=opened,
        fx=fx,
        targets=targets,
        costs=costs,
        capital_krw=ctx["mix_capital_krw"],
        harvest_pct=ctx["mix_harvest_pct"],
        refill_pct=ctx["mix_refill_pct"],
    )

    replayed["fx"] = {str(day.date()): float(rate) for day, rate in fx.items()}
    return replayed


def _value_account(
    account: dict[str, Any], krw_rate: float, price_hint: dict[str, float] | None = None, *, as_of: str | None = None
) -> None:
    """계좌 보유를 원화로 평가해 ``total_assets``·``stock_value``·고정 자산 몫을 채운다.

    운용 현황과 백테스트가 **같은 총자산**을 봐야 한다 — 백테스트도 이 돈으로 돌리기
    때문이다. 두 곳에서 따로 세면 목표 주수를 만든 자본과 성과를 낸 자본이 달라진다.

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


def run_mix_backtest(account_id: str | None = None, months: int | None = None) -> dict[str, Any]:
    """선택한 계좌의 슬리브별 저장 설정으로 A·B 백테스트를 각각 돌려 합성 결과를 만든다.

    ``months`` 를 주면 그 기간으로 슬리브 백테스트를 모두 돌린다(화면의 기간 셀렉트).
    없으면 저장 설정 중 **가장 짧은** 기간 — 한 슬리브라도 데이터가 없으면 합성이 안 된다.
    """
    import pandas as pd

    ctx = _resolve_mix_account(account_id)
    if months is None:
        months = min(int(spec.settings["backtest_months"]) for spec in ctx["slots"])
    months = int(months)
    allowed = month_options()
    if months not in allowed:
        raise ValueError(f"'months' 는 {allowed} 중 하나여야 합니다 (받은 값: {months})")

    logger.info(
        "[STRATEGY-MIX] %s 합성 백테스트 시작 (%s, %d개월)",
        ctx["account_id"],
        " · ".join(f"{_slot_labels(ctx['slots'])[spec.key]}/{spec.pool}" for spec in ctx["slots"]),
        months,
    )
    # 슬리브별 준비물·백테스트 — 어댑터가 두 엔진의 차이를 흡수한다.
    from utils.mix_sleeve import load_context as sleeve_context
    from utils.mix_sleeve import run_backtest as sleeve_backtest

    contexts = {spec.key: sleeve_context(spec) for spec in ctx["slots"]}
    results = {spec.key: sleeve_backtest(spec, months, contexts[spec.key]) for spec in ctx["slots"]}
    # 벤치마크 곡선 — **계좌 벤치마크**의 종가에서 직접 만든다. 합성은 한 계좌를 통째로 굴린
    # 결과라 대조군도 계좌 단위여야 한다(슬리브 풀 것을 쓰면 모멘텀 풀을 바꾸는 것만으로
    # 비교 기준이 따라 바뀐다). 값은 시작일 대비 누적 배수로 담고 아래에서 0% 로 재기준한다.
    from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types

    benchmark_ticker = ctx["benchmark_ticker"]
    frames = load_cached_frames_bulk_from_all_ticker_types([benchmark_ticker])
    bench_frame = frames.get(benchmark_ticker)
    if bench_frame is None or bench_frame.empty or "Close" not in bench_frame.columns:
        raise RuntimeError(f"벤치마크({ctx['benchmark_name']}) 가격 캐시를 불러올 수 없습니다.")
    bench_close = pd.to_numeric(bench_frame["Close"], errors="coerce").dropna()
    if bench_close.empty:
        raise RuntimeError(f"벤치마크({ctx['benchmark_name']}) 종가가 비어 있습니다.")
    bench_curve: dict[str, float] = {str(day.date()): float(value) for day, value in bench_close.items()}
    bench_index = bench_close.index

    # 슬리브별 곡선 — 화면이 "합성 vs 각 전략 단독" 을 한 표에서 비교한다.
    # **각 전략을 혼자 굴렸을 때의 곡선**이다(슬리브 간 월초 이관이 없는 상태). 그래야 각
    # 전략 화면의 백테스트와 값이 같고, 합성이 단독보다 나은지가 바로 읽힌다.
    # 형태 차이(모멘텀=전일 대비, 신고가=누적)는 어댑터가 누적 배수로 통일해 준다.
    from utils.mix_sleeve import daily_curve as sleeve_curve

    curves = {spec.key: sleeve_curve(spec, results[spec.key]) for spec in ctx["slots"]}

    # 합성 곡선은 개별 곡선의 가중 합이 아니라 종목별 회수·채우기 체결을 재생한다.
    replayed = _simulate_mix(ctx, results, through_date=None)
    mix_curve = replayed["curve"]
    withdrawn_curve = replayed["withdrawn_curve"]
    dates = [d for d in mix_curve.index if d in bench_curve]
    if len(dates) < 2:
        raise RuntimeError("슬리브 전략들의 공통 백테스트 구간이 부족합니다.")

    first_mix = 1.0  # 최초 매수 비용도 성과에 포함한다.
    # 벤치마크는 **시작일 시가**를 1 로 둔다 — 전략도 그날 시가에 사기 때문이다(공용 함수).
    from utils.benchmark_curve import growth_from_frame

    bench_growth = growth_from_frame(
        bench_frame,
        bench_index[bench_index.isin(pd.to_datetime(dates))],
        label=f"벤치마크({ctx['benchmark_name']})",
    )
    fx_by_day = replayed["fx"]
    initial_fx = fx_by_day[dates[0]]
    bench_curve = {
        str(day.date()): float(value) * fx_by_day[str(day.date())] / initial_fx for day, value in bench_growth.items()
    }
    # 비교 곡선도 같은 날짜별 환율로 원화 환산한다.
    curves = {
        key: {date: value * fx_by_day[date] / initial_fx for date, value in curve.items() if date in fx_by_day}
        for key, curve in curves.items()
    }
    first_bench = 1.0  # 시작 기준이 이미 시가라 곡선 자체가 1 에서 출발한다
    # 슬리브별 시작값 — 합성과 같은 시작일로 다시 맞추는 기준점.
    first_by_slot = {key: curve.get(dates[0]) for key, curve in curves.items()}

    def _rebased(curve: dict[str, float], base: float | None, date: str) -> float | None:
        """시작일을 0% 로 맞춘 누적(%). 그 날짜 값이 없으면 None — 임의로 채우지 않는다."""
        value = curve.get(date)
        if value is None or not base:
            return None
        return round((value / base - 1) * 100, 2)

    daily_rows = [
        {
            "date": date,
            "strategy_pct": round((float(mix_curve[date]) / first_mix - 1) * 100, 2),
            "withdrawn_to_initial_pct": round(float(withdrawn_curve[date]) * 100, 2),
            # 슬리브 단독 누적(%) — 합성과 같은 시작일 기준으로 다시 맞춘다.
            # 슬롯 키로 담는다(화면이 키 목록을 돌며 표를 만든다).
            "slots": {key: _rebased(curve, first_by_slot[key], date) for key, curve in curves.items()},
            "benchmark_pct": round((bench_curve[date] / first_bench - 1) * 100, 2),
        }
        for date in dates
    ]

    # 요약 지표 — 일별 누적 곡선 기준. 계산은 신고가 엔진과 같은 방식이다
    # (총수익·기간 CAGR·일별 곡선 MDD·일별 수익률 소르티노).
    def _summarize(curve: pd.Series) -> dict[str, Any]:
        total = float((curve.iloc[-1] - 1) * 100)
        returns = curve.pct_change().dropna()
        downside = returns[returns < 0]
        deviation = float((downside**2).mean() ** 0.5) if not downside.empty else 0.0
        sortino = (
            round(float(returns.mean()) / deviation * (252**0.5), 2) if deviation > 0 and len(returns) >= 2 else None
        )
        return {
            "total_pct": round(total, 2),
            "cagr_pct": round(((1 + total / 100) ** (12 / months) - 1) * 100, 2) if months > 0 else None,
            "mdd_pct": round(float(((curve / curve.cummax()) - 1).min() * 100), 2),
            "sortino": sortino,
        }

    strategy_curve = pd.Series([1 + row["strategy_pct"] / 100 for row in daily_rows])
    benchmark_curve = pd.Series([1 + row["benchmark_pct"] / 100 for row in daily_rows])
    strategy_stats, benchmark_stats = _summarize(strategy_curve), _summarize(benchmark_curve)
    merged_trades = [
        {
            "ticker": row["ticker"],
            "name": row["ticker"],
            "entry_date": row["date"],
            "exit_date": row["date"],
            "entry_price": row["price"],
            "exit_price": row["price"],
            "return_pct": None,
            "days": None,
            "reason": f"합성 {row['side']} {row['quantity']}주",
            "strategy": "mix",
        }
        for row in replayed["executions"]
    ]

    return {
        "computed_at": datetime.now().astimezone().isoformat(),
        "account_id": ctx["account_id"],
        "months": months,
        "start_date": dates[0],
        "end_date": dates[-1],
        "benchmark_name": ctx["benchmark_name"],
        "benchmark_ticker": ctx["benchmark_ticker"],
        "strategy_total_pct": strategy_stats["total_pct"],
        "strategy_cagr_pct": strategy_stats["cagr_pct"],
        "strategy_mdd_pct": strategy_stats["mdd_pct"],
        "strategy_sortino": strategy_stats["sortino"],
        "benchmark_total_pct": benchmark_stats["total_pct"],
        "benchmark_cagr_pct": benchmark_stats["cagr_pct"],
        "benchmark_mdd_pct": benchmark_stats["mdd_pct"],
        "benchmark_sortino": benchmark_stats["sortino"],
        # 일별 누적(%) — 화면이 연간·월간·주간·일간 표를 이 시계열에서 만든다.
        "daily": daily_rows,
        # 체결 목록 — 두 전략을 합쳐 보여준다(보유중 행이 위, 그 아래 청산일 최신순).
        "trades": merged_trades,
        # 거래 수·승률·평균 손익 — 각 전략 화면과 같은 공용 계산.
        **summarize_trades(merged_trades),
        "trade_count": len(merged_trades),
    }
