"""합성 전략 — 슬리브(전략 + 종목풀) 둘을 한 계좌에서 함께 굴린 백테스트·운용 현황.

`/strategy-mix` 열람 전용 화면의 백엔드. 설정은 이 화면이 갖지 않는다 —
**선택한 계좌의 슬리브들**(계좌 설정의 `mix_sleeves` 배열)에 저장된
각 전략 화면 설정을 그대로 가져와 두 백테스트를 돌리고,
매월 첫 거래일에 계좌 설정의 배분(A·B·비워 두는 현금)으로 되돌리되,
**현금 우선**으로 이관한다 —
넘기는 슬리브의 현금부터 쓰고, 모자랄 때만 주식을 비례 매도한다(오르는 종목 유지).
화면은 신고가 화면과 같은 방식으로 이 일별 누적에서 연간·월간·일간 표를 만든다.

캐시는 두지 않는다 — 각 전략 화면의 백테스트와 같은 패턴(요청 시 계산)이다.

**여기서 판정하지 않는다.** 보유·진입·이탈은 슬리브 엔진(`utils/slot_backtest.py`)이 정한
것을 읽어 온다. 합성이 하는 일은 슬리브 곡선에 월초 이관을 얹는 것과, 계좌 보유와 목표
주수의 차이를 지시로 내는 것뿐이다. 그 차이는 문턱 없이 전부 낸다(AGENTS.md 10).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from config import CACHE_TTL_COMPUTE
from utils.logger import get_app_logger
from utils.mix_sleeve import STRATEGY_LABELS, SleeveSpec
from utils.stock_memo_store import attach_stock_memos
from utils.trade_stats import summarize_trades
from utils.ttl_cache import TtlCache

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

    슬리브는 각각 (전략, 종목풀) 쌍이고 `/strategy-mix` 화면에서 고른다. 모멘텀 둘,
    신고가 둘, 섞기 전부 가능하며 둘이 완전히 같은 조합인 것만 저장이 막는다.

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
                "currency": str(inner.get("currency") or "").strip().upper(),
                "country_code": str(inner.get("country_code") or "").strip().lower(),
                "sleeves": sleeves,
                # 조합이 다 갖춰졌는지 — 화면이 "고르세요" 상태를 판단하는 데 쓴다.
                "mix_ready": bool(sleeves) and all(row["strategy"] and row["pool"] for row in sleeves),
                # 오늘의 액션 슬랙 알람 토글 상태 — 화면 헤더가 그대로 보여준다.
                "mix_slack_enabled": bool(inner.get("mix_slack_enabled")),
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
    first_pool_settings = get_ticker_type_settings(slots[0].pool) or {}
    return {
        "slots": slots,
        "account_id": account["account_id"],
        "account_name": account["name"],
        # 국가·통화 — 거래 달력(월초 리밸런싱 판정)과 원화 환산에 쓴다.
        "country": str(first_pool_settings.get("country_code") or "kor").strip().lower(),
        "currency": str(first_pool_settings.get("currency") or "KRW").strip().upper(),
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


# 슬리브 몫 캐시 — 이 값은 **종가 기준 백테스트 곡선**에서 나오므로 장중에는 바뀌지 않는데,
# 캐시가 없을 때는 요청마다 슬리브 수만큼 2개월 백테스트를 새로 돌려 17초 넘게 썼다.
# 계좌 잔고·보유 수량은 이 캐시 밖이라 그대로 매 요청 새로 읽는다.
_SHARES_CACHE = TtlCache(CACHE_TTL_COMPUTE, name="mix_sleeve_shares")


def _sleeve_shares(ctx: dict[str, Any]) -> dict[str, float]:
    """{슬롯키: 몫 %} + cash_pct — 직전 월초 배분 이후 흘러간 비율.

    각 슬리브의 일별 곡선에서 이번 달 첫 거래일 이후 성장률을 읽어, 월초 배분에 곱한 뒤
    전체를 100 으로 정규화한다. 비워 두는 현금은 자라지 않으므로 그 몫은 그대로 두고,
    슬리브가 오르면 상대적으로 현금 비중이 줄어든다.

    곡선 형태(전일 대비 vs 누적)는 어댑터가 **누적 배수**로 통일해 주므로 전략을 가리지 않는다.
    보유와 같은 운용 시작일로 곡선을 계산한다. 데이터 부족은 오류로 알린다.
    """
    # 캐시 키는 계좌 + **슬리브 구성**이다. 조합이나 배분을 바꾸면 몫이 달라지므로
    # 구성까지 키에 넣어야 저장 직후 옛 몫이 그대로 나오지 않는다.
    key = _SHARES_CACHE.make_key(
        ctx["account_id"],
        [(spec.key, spec.strategy, spec.pool, spec.settings) for spec in ctx["slots"]],
        mix_weights_for_account(ctx["account_id"]),
    )
    return _SHARES_CACHE.get_or_compute(key, lambda: _compute_sleeve_shares(ctx))


def _compute_sleeve_shares(ctx: dict[str, Any]) -> dict[str, float]:
    from utils.mix_sleeve import daily_curve, load_context, run_backtest
    from utils.strategy_settings import require_start_date

    base = mix_weights_for_account(ctx["account_id"])

    curves = {
        spec.key: daily_curve(
            spec, run_backtest(spec, 12, load_context(spec), start_date=require_start_date(spec.settings))
        )
        for spec in ctx["slots"]
    }
    if not curves or not all(curves.values()):
        raise RuntimeError("운용 시작일 이후 슬리브 가격 데이터가 부족합니다.")
    trimmed = curves
    # 기준 달 — 어느 슬리브든 마지막 날짜가 같은 달이다(같은 국가 달력).
    month = max(max(curve) for curve in trimmed.values())[:7]

    growths: dict[str, float] = {}
    for key, curve in trimmed.items():
        days = sorted(d for d in curve if d[:7] == month)
        # 월초 첫 거래일 값이 기준점이다 — 그날까지의 변동은 직전 달 몫이다.
        growths[key] = curve[days[-1]] / curve[days[0]] if len(days) >= 2 and curve[days[0]] > 0 else 1.0

    values = {key: base[f"{key}_pct"] * growths[key] for key in growths}
    cash_value = base["cash_pct"]  # 비워 둔 현금은 자라지 않는다.
    total = sum(values.values()) + cash_value
    if total <= 0:
        return base
    shares = {f"{key}_pct": round(value / total * 100.0, 4) for key, value in values.items()}
    shares["cash_pct"] = round(cash_value / total * 100.0, 4)
    return shares


_WEEKDAYS_KO = ("월", "화", "수", "목", "금", "토", "일")


def _format_date_weekday(date: str) -> str:
    from datetime import date as date_cls

    try:
        parsed = date_cls.fromisoformat(date)
    except ValueError:
        return date
    return f"{date} ({_WEEKDAYS_KO[parsed.weekday()]})"


def _slot_labels(slots: list[SleeveSpec]) -> dict[str, str]:
    """슬롯 표시 이름 — 「A. 모멘텀」. 같은 전략이 두 슬롯에 올 수 있어 기호를 앞에 둔다."""
    return {spec.key: f"{spec.key.upper()}. {spec.label}" for spec in slots}


def _build_action_groups(
    holdings: list[dict[str, Any]],
    actions: dict[str, Any],
    next_trading_day: str | None,
    *,
    currency: str = "KRW",
) -> list[dict[str, Any]]:
    """오늘의 액션 — 체결일 묶음(매도 먼저, 같은 방향은 티커 순).

    화면과 슬랙 알람이 **이 결과를 그대로** 쓴다 — 조립을 한 곳에 두어 둘이 어긋나지
    않게 한다. 규칙:
      · 계좌 보유와 목표 주수의 차이를 **거르지 않고 전부** 낸다. 예전에는 「목표비중의
        10% 이상 차이만」 이라는 문턱(밴드)을 종목마다 따로 걸었는데, 백테스트에 없는
        규칙이라 화면과 백테스트가 갈라졌다. 게다가 목표 주수 배분은 12종목을 한 번에
        계산해 매수 합이 매도 합 + 현금을 넘지 않는데, 문턱이 큰 매수만 통과시키고
        작은 매도를 걸러 **살 돈이 없는 매수 지시**를 만들었다.
      · 교체가 확정됐지만 미체결이면 그 슬리브 몫은 교체일 시가 그룹, 나머지는 다음 거래일 그룹.
    슬리브가 어떤 전략인지는 보지 않는다 — 있는 액션만 읽는다(교체가 없는 전략은 rebalance=None).
    """
    slots: dict[str, dict[str, Any]] = actions["slots"]

    # 교체가 아직 체결 전인 슬리브 — 그 슬리브 몫의 지시는 교체일 시가 그룹으로 간다.
    pending_fill: dict[str, str] = {}
    rebalance_tickers: dict[str, str] = {}  # ticker → 그 교체가 속한 슬롯
    rebalance_buys: set[str] = set()
    rebalance_sells: set[str] = set()
    for key, slot in slots.items():
        rebalance = slot.get("rebalance")
        if not rebalance:
            continue
        if not rebalance["is_filled"] and rebalance["fill_date"]:
            pending_fill[key] = rebalance["fill_date"]
        for row in rebalance["buys"]:
            rebalance_tickers[row["ticker"]] = key
            rebalance_buys.add(row["ticker"])
        for row in rebalance["sells"]:
            rebalance_tickers[row["ticker"]] = key
            rebalance_sells.add(row["ticker"])

    # 슬리브별 즉시 이벤트(진입·매도) — 그 종목은 교체일이 아니라 다음 거래일에 움직인다.
    event_tickers: dict[str, set[str]] = {
        key: {row["ticker"] for row in slot["entries"]} | {row["ticker"] for row in slot["sells"]}
        for key, slot in slots.items()
    }
    entry_tickers = {row["ticker"] for slot in slots.values() for row in slot["entries"]}
    live_entry_tickers = {row["ticker"] for slot in slots.values() if slot.get("live") for row in slot["entries"]}
    sell_pending = [row["ticker"] for slot in slots.values() for row in slot["sells"]]

    # 장중 판정은 오늘 종가로 확정되기 전이라 **예상**이다 — 문구로 구분한다.
    # 장중을 쓰는지는 슬리브마다 다르므로 그 슬리브의 플래그를 본다.
    sell_reason: dict[str, str] = {}
    forecast_sell_tickers: set[str] = set()
    for slot in slots.values():
        live_tag = " · 예상" if slot.get("live") else ""
        for row in slot["sells"]:
            # 수익률이 함께 오는 건 진입가를 아는 전략뿐이다(모멘텀 자격 상실은 사유만).
            suffix = f", {row['return_pct']:+.2f}%" if row.get("return_pct") is not None else ""
            sell_reason[row["ticker"]] = f"{row['reason']}{suffix}{live_tag}"
            if live_tag:
                forecast_sell_tickers.add(row["ticker"])

    def label(ticker: str, name: str, quantity: float | None) -> str:
        base = f"{name}({ticker})"
        return base if not quantity else f"{base} {abs(int(quantity)):,}주"

    row_by_ticker = {row["ticker"]: row for row in holdings}
    items: list[dict[str, Any]] = []
    for row in holdings:
        trade = row.get("trade_quantity")
        if not trade:
            continue
        ticker = row["ticker"]
        sources = row.get("sources") or []
        date = next_trading_day
        if ticker in rebalance_tickers:
            slot_key = rebalance_tickers[ticker]
            date = pending_fill.get(slot_key) or next_trading_day
        elif not row.get("is_sell_all"):
            # 교체 대기 중인 슬리브 몫이면 그 교체일에 함께 맞춘다 — 단, 다른 슬리브에서
            # 즉시 움직일 일이 잡혀 있으면 그쪽(다음 거래일)이 먼저다.
            for key, fill_date in pending_fill.items():
                if key not in sources:
                    continue
                if any(ticker in event_tickers[other] for other in event_tickers if other != key):
                    continue
                date = fill_date
                break
        reason = sell_reason.get(ticker)
        weight = float(row.get("weight_pct") or 0)
        held = float(row.get("held_quantity") or 0) > 0
        sell_reason_applies = bool(reason) and trade < 0 and weight <= 0
        if row.get("is_sell_all"):
            title = "교체 매도" if ticker in rebalance_sells else "전량 매도"
        elif trade < 0:
            if sell_reason_applies:
                title = "매도 예정(예상)" if ticker in forecast_sell_tickers else "매도 예정"
            else:
                title = "비중 조정 매도"
        elif held:
            title = "비중 조정 매수"
        elif ticker in rebalance_buys:
            title = "교체 매수"
        elif ticker in entry_tickers:
            title = "진입(예상)" if ticker in live_entry_tickers else "진입"
        else:
            title = "신규 매수"
        after = f" → 목표 {int(row['target_quantity']):,}주" if row.get("target_quantity") is not None else ""
        amount = _format_trade_amount(trade, row.get("price"), currency)
        amount_note = f" · {amount}" if amount else ""
        if sell_reason_applies:
            note = f"{after} ({reason}){amount_note}".strip()
        elif row.get("is_sell_all"):
            note = f"· 목표에 없는 보유 종목{amount_note}"
        else:
            note = f"{after} · {weight:.2f}%{amount_note}".strip()
        items.append(
            {
                "key": f"act-{ticker}",
                "ticker": ticker,
                "side": "buy" if trade > 0 else "sell",
                "title": title,
                "text": f"{label(ticker, row.get('name') or ticker, trade)} {note}".strip(),
                "date": date,
                # 알람 비교용 — 새 지시·수량 증가만 발송하고 감소(체결 반영)는 조용히 넘긴다.
                "quantity": abs(int(trade)),
            }
        )

    # 매도 예정인데 매매수량이 0인 경우(목표가 아직 그대로라 차이가 없음)도 알려야 한다.
    seen_keys = {item["key"] for item in items}
    for ticker in sell_pending:
        if f"act-{ticker}" in seen_keys:
            continue
        row = row_by_ticker.get(ticker)
        if not row or float(row.get("held_quantity") or 0) <= 0 or float(row.get("weight_pct") or 0) > 0:
            continue
        items.append(
            {
                "key": f"act-{ticker}",
                "ticker": ticker,
                "side": "sell",
                "title": "매도 예정",
                "text": (
                    f"{label(ticker, row.get('name') or ticker, row.get('held_quantity'))}"
                    f" ({sell_reason.get(ticker) or '이탈'})"
                    + (
                        f" · {amt}"
                        if (amt := _format_trade_amount(row.get("held_quantity"), row.get("price"), currency))
                        else ""
                    )
                ),
                "date": next_trading_day,
                "quantity": abs(int(float(row.get("held_quantity") or 0))),
            }
        )
        seen_keys.add(f"act-{ticker}")

    by_date: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_date.setdefault(item["date"] or "", []).append(item)
    # 교체일 묶음에는 어느 슬리브의 교체인지 적는다 — 한 계좌에 교체가 둘일 수 있다.
    rebalance_note: dict[str, list[str]] = {}
    for key, fill_date in pending_fill.items():
        rebalance_note.setdefault(fill_date, []).append(slots[key]["label"])
    groups = []
    for date in sorted(by_date):
        group_items = sorted(by_date[date], key=lambda x: (0 if x["side"] == "sell" else 1, x["ticker"]))
        if date:
            note = rebalance_note.get(date)
            title = f"{_format_date_weekday(date)} 시가" + (f" · {' · '.join(note)} 교체 포함" if note else "")
        else:
            title = "체결일 미정"
        groups.append({"key": date or "unscheduled", "title": title, "items": group_items})

    # 주중 이탈 **예상** 그룹 — 판정(오늘 종가) 확정 전의 미리보기. 같은 체결일(다음 거래일)
    # 시가지만 확정 지시와 섞이지 않게 별도 그룹으로 뒤에 둔다. 슬랙 알람은 이 그룹을 보내지
    # 않는다(장중 출렁일 때마다 알람이 나가면 노이즈 — notify 가 forecast 그룹을 거른다).
    # 이탈 예상 종목의 **매수** 지시는 유예한다 — "오늘 사서 내일 팔라"가 된다. 조용히 지우지
    # 않고 '(예상)' 그룹에 유예 사유를 남긴다(수량이 부족해 보여도 오늘은 사지 말라는 안내).
    forecast_rows = [(key, row) for key, slot in slots.items() for row in slot["exit_forecast"]]
    forecast_tickers = {row["ticker"] for _, row in forecast_rows}
    suspended_buys: list[dict[str, Any]] = []
    if forecast_tickers:
        for group in groups:
            kept = []
            for item in group["items"]:
                if item["side"] == "buy" and item["ticker"] in forecast_tickers:
                    suspended_buys.append(item)
                else:
                    kept.append(item)
            group["items"] = kept
        groups = [group for group in groups if group["items"]]

    confirmed_sells = {item["ticker"] for group in groups for item in group["items"] if item["side"] == "sell"}
    forecast_items = []
    forecast_slot_by_ticker: dict[str, str] = {}
    for key, row in forecast_rows:
        ticker = row["ticker"]
        held = row_by_ticker.get(ticker)
        if ticker in confirmed_sells or not held:
            continue
        # 파는 건 **그 슬리브 몫**뿐이다 — 예상 수량 = 보유 − 이탈 후 남을 목표(다른 슬리브 몫).
        # 계좌가 이미 그 몫을 팔아뒀으면 0 이 되어 자동으로 표시되지 않는다.
        held_qty = float(held.get("held_quantity") or 0)
        target_qty = float(held.get("target_quantity") or 0)
        weight_all = float(held.get("weight_pct") or 0)
        slot_weight = float(held.get(f"{key}_weight") or 0)
        remain_qty = round(target_qty * (weight_all - slot_weight) / weight_all) if weight_all > 0 else 0
        slot_qty = int(round(held_qty - remain_qty))
        if slot_qty <= 0:
            continue
        both = slot_weight > 0 and (weight_all - slot_weight) > 0
        reason = f"{row.get('reason')} · {slots[key]['label']} 몫" if both else str(row.get("reason"))
        forecast_slot_by_ticker[ticker] = key
        forecast_items.append(
            {
                "key": f"forecast-{ticker}",
                "ticker": ticker,
                "side": "sell",
                "title": "매도 예정(예상)",
                "text": (
                    f"{label(ticker, row.get('name') or ticker, slot_qty)} ({reason})"
                    + (f" · {amt}" if (amt := _format_trade_amount(slot_qty, held.get("price"), currency)) else "")
                ),
                "date": next_trading_day,
                # 알람 상태 비교용 — 예상도 '처음 등장할 때 1건' 발송되도록 실제 수량을 싣는다.
                "quantity": abs(slot_qty),
            }
        )
    # 매수 유예 안내는 매도 예상이 **없는** 종목에만 — 전량 매도 예상이 이미 '사지 말라'를
    # 내포하므로 같은 종목에 두 줄이 나오면 소음이다.
    forecast_sold = {item["ticker"] for item in forecast_items}
    forecast_slot_all = {row["ticker"]: key for key, row in forecast_rows}
    for item in suspended_buys:
        if item["ticker"] in forecast_sold:
            continue
        slot_label = slots[forecast_slot_all[item["ticker"]]]["label"]
        forecast_items.append(
            {
                **item,
                "key": f"suspend-{item['ticker']}",
                "title": "매수 유예(예상)",
                "text": f"{item['text']} — 내일 {slot_label} 몫 매도 예상이라 오늘은 사지 않음",
                "date": next_trading_day,
            }
        )
    if forecast_items and next_trading_day:
        groups.append(
            {
                "key": f"{next_trading_day}-forecast",
                "title": f"{_format_date_weekday(next_trading_day)} 시가 (예상 — 오늘 종가 확정 시)",
                "forecast": True,
                "items": sorted(forecast_items, key=lambda x: x["ticker"]),
            }
        )
    return groups


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
    추세 이탈 배지(❗)를 붙이므로, 다른 기준으로 계산하면 같은 종목에 화면마다 다른
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
    for row in holdings:
        row["current_short_pct"] = None
        row["current_long_pct"] = None
        ticker = str(row.get("ticker") or "").strip()
        # 여러 슬리브에 함께 잡힌 종목은 **첫 슬리브**의 풀 이평선으로 본다 — 배지가
        # 하나뿐이라 기준도 하나여야 한다(슬롯 순서가 그 우선순위다).
        sources = row.get("sources") or []
        source = next((key for key in pool_by_source if key in sources), "")
        pool = pool_by_source.get(source, "") if source else pool_by_ticker.get(ticker.upper(), "")
        days = ma_days_of(pool) if pool else None
        if days is None:
            continue
        frame = frames.get(ticker)
        if frame is None or frame.empty or "Close" not in frame.columns:
            continue
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if close.empty:
            continue
        metrics = momentum_metrics(close, short_ma_days=days[0], long_ma_days=days[1], as_of=None)
        if not metrics:
            continue
        row["current_short_pct"] = round(metrics["short_disparity_pct"], 1)
        row["current_long_pct"] = round(metrics["disparity_pct"], 1)


def _format_trade_amount(quantity: float | None, price: float | None, currency: str) -> str:
    """지시 금액 표기 — 원화는 'N억 1,234만원', 미국·호주는 현지 통화($ / A$)."""
    if not quantity or not price or float(price) <= 0:
        return ""
    amount = abs(float(quantity)) * float(price)
    code = str(currency or "KRW").strip().upper()
    if code == "KRW":
        if amount >= 1_0000_0000:
            uk = int(amount // 1_0000_0000)
            man = int(round((amount - uk * 1_0000_0000) / 1_0000))
            return f"{uk}억 {man:,}만원" if man else f"{uk}억원"
        if amount >= 1_0000:
            return f"{int(round(amount / 1_0000)):,}만원"
        return f"{amount:,.0f}원"
    symbol = {"USD": "$", "AUD": "A$"}.get(code, f"{code} ")
    return f"{symbol}{amount:,.0f}"


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
_SLOT_ROW_FIELDS: tuple[str, ...] = ("weight", "status", "return_pct", "held_label", "entry_date", "entry_price")


def _holding_payload(row: dict[str, Any], slot_keys: Sequence[str]) -> dict[str, Any]:
    """보유 행 하나를 화면 형태로 — 슬리브별 값(`a_weight` …)을 `slots[키]` 로 모은다."""
    flat_keys = {f"{key}_{field}" for key in slot_keys for field in _SLOT_ROW_FIELDS}
    payload = {name: value for name, value in row.items() if name not in flat_keys}
    payload["weight_pct"] = round(float(row["weight_pct"]), 2)
    payload["slots"] = {key: {field: row.get(f"{key}_{field}") for field in _SLOT_ROW_FIELDS} for key in slot_keys}
    return payload


def _sleeve_target_shares(
    states: dict[str, Any],
    sleeve_amount_krw: dict[str, float],
    krw_rate: float,
) -> dict[str, int]:
    """백테스트 목표 비중을 계좌 금액으로 환산하고 정수 주수로 배분한다.

    실제 보유 수량은 입력하지 않는다. 계좌 규모를 바꿔 백테스트를 다시 실행하면
    비싼 종목의 진입 가능 여부가 바뀌므로, 엔진 결과를 비례 환산만 한다.
    """
    from math import fsum

    from utils.share_allocation import ShareTarget, allocate_integer_shares

    # 소수 목표를 **티커별로 합산**한다 — 두 슬리브가 같은 종목을 담으면 몫이 더해진다.
    # 슬리브마다 따로 정하면 뒤에 온 슬리브가 앞의 값을 덮어써, 목표비중(합산)과 목표 주수가
    # 어긋난다(kor_test 에서 125.5주 + 111.1주가 113주가 되어 115주를 팔라는 지시가 났다).
    amount_by_ticker: dict[str, float] = {}
    unit_by_ticker: dict[str, float] = {}
    for key, state in states.items():
        budget = sleeve_amount_krw.get(key, 0.0)
        if budget <= 0 or krw_rate <= 0:
            continue
        # 슬리브 안 비중 — 이미 산 종목은 흘러간 실제 비중(drift_pct), 진입 예정은 슬롯 1칸.
        slot_pct = 100.0 / state.top_n if state.top_n else 0.0
        for row in state.targets:
            price = row.get("price")
            if not price or row.get("is_exiting"):
                continue
            weight = row.get("drift_pct")
            weight = slot_pct if weight is None else float(weight)
            if weight <= 0:
                continue
            ticker = str(row["ticker"]).strip()
            unit_by_ticker[ticker] = float(price) * krw_rate
            amount_by_ticker[ticker] = amount_by_ticker.get(ticker, 0.0) + budget * weight / 100.0

    return allocate_integer_shares(
        [
            ShareTarget(key=ticker, target_amount=amount, price=unit_by_ticker[ticker])
            for ticker, amount in amount_by_ticker.items()
        ],
        # 합성 유보 현금은 슬리브 배정액에서 이미 제외된다. 전략 내부 현금도 보호하려면
        # 주식 목표 금액까지만 쓸 수 있다. 추가 주수에는 내림으로 생긴 단주 잔여만 쓴다.
        # 동시에 전체 슬리브 배정액을 넘지 않도록 계좌 예산 한도도 유지한다.
        budget=min(fsum(amount_by_ticker.values()), fsum(sleeve_amount_krw.values())),
    )


def _attach_account_targets(
    holdings: list[dict[str, Any]],
    account: dict[str, Any],
    krw_rate: float = 1.0,
    slot_keys: Sequence[str] = (),
    target_shares: dict[str, int] | None = None,
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
        row["target_amount"] = round(total_assets * row["weight_pct"] / 100.0, 2)
        price = row.get("price")
        if price and krw_rate > 0:
            price_krw_by_ticker[row["ticker"]] = float(price) * krw_rate

    # 목표 주수 = **계좌 금액으로 돌린 백테스트가 지금 들고 있는 주수**(`_sleeve_target_shares`).
    #
    # 예전에는 여기서 `allocate_integer_shares` 로 예산을 끝까지 소진하는 배분을 매일 돌렸다.
    # 그건 백테스트가 **진입할 때** 쓰는 규칙이다. 백테스트는 진입 후 이탈까지 주수를
    # 건드리지 않는데, 매일 다시 배분하면 남는 돈이 그날그날 다른 종목에 얹혀 백테스트가
    # 하지도 않는 매매를 시킨다.
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
            round(target_qty * price_krw / total_assets * 100.0, 4)
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


def _build_next_week_preview(
    states: dict[str, Any],
    holdings: list[dict[str, Any]],
    actions: dict[str, Any],
    slot_weight: dict[str, float],
    account: dict[str, Any] | None,
    ahead: list,
    today_local,
    currency: str,
    krw_rate: float,
) -> dict[str, Any] | None:
    """다음 교체를 **지금 순위 그대로 확정된다고 가정**했을 때의 전체 액션.

    오늘의 액션과 **같은 조립기**(`_build_action_groups`)를 가정 목표에 돌린다 —
    전량 매도(목표에 없는 보유 종목)까지 포함한 완전한 그림이라, 종목 교체가 없으면
    오늘의 액션과 똑같이 나온다. 주기적 교체가 있는 슬리브만 다음 교체 예상
    (`next_expected` — 실시간 순위 기준)으로 목표를 바꾸고, 나머지 슬리브는 그대로 둔다.

    이번 교체가 아직 체결 전이면(오늘의 액션에 교체일 그룹이 이미 있으면) 만들지
    않는다 — 같은 교체가 두 번 보이면 헷갈린다. 수량은 현재가·현재 총자산 기준
    추정치라 실제 체결 수량과 다를 수 있다.
    """
    if account is None:
        return None
    # 교체가 있고 이미 체결된 슬리브만 대상 — 미체결이면 오늘의 액션이 이미 보여준다.
    rotating = [
        key for key, state in states.items() if state.rebalance and state.rebalance["is_filled"] and state.next_expected
    ]
    if not rotating:
        return None

    expected_by_slot = {key: states[key].next_expected for key in rotating}

    # ── 가정 목표 — 현재 목표에서 교체 슬리브 몫만 다음 예상으로 바꾼다 ──
    keys = list(states)
    hypo: list[dict[str, Any]] = []
    for row in holdings:
        if row.get("is_sell_all"):
            continue  # 전량 매도 행은 계좌 대조가 다시 만든다
        copy = {
            "ticker": row["ticker"],
            "name": row.get("name"),
            "sources": list(row.get("sources") or []),
            "weight_pct": float(row.get("weight_pct") or 0),
            "price": row.get("price"),
        }
        for key in keys:
            copy[f"{key}_weight"] = float(row.get(f"{key}_weight") or 0)
        for key in rotating:
            if key in copy["sources"] and copy["ticker"] not in expected_by_slot[key]:
                # 편출 예상 — 그 슬리브 몫을 뺀다. 다른 슬리브 몫이 없으면 행 자체가 빠진다.
                copy["weight_pct"] -= copy[f"{key}_weight"]
                copy[f"{key}_weight"] = 0.0
                copy["sources"] = [s for s in copy["sources"] if s != key]
        if not copy["sources"] and copy["weight_pct"] <= 0:
            continue
        hypo.append(copy)

    preview_buys: list[dict[str, Any]] = []
    preview_sells: list[dict[str, Any]] = []
    for key in rotating:
        expected = expected_by_slot[key]
        for ticker in sorted(expected):
            existing = next((r for r in hypo if r["ticker"] == ticker), None)
            if existing is not None and key in existing["sources"]:
                continue  # 유지 — 이미 그 슬리브 몫이 있다
            row = expected[ticker]
            if existing is None:
                existing = {
                    "ticker": ticker,
                    "name": row.get("name") or ticker,
                    "sources": [],
                    "weight_pct": 0.0,
                    "price": row.get("price"),
                    **{f"{k}_weight": 0.0 for k in keys},
                }
                hypo.append(existing)
            existing["sources"].append(key)
            existing["weight_pct"] += slot_weight[key]
            existing[f"{key}_weight"] += slot_weight[key]
            preview_buys.append({"ticker": ticker, "name": row.get("name") or ticker, "price": row.get("price")})
        preview_sells.extend(
            {"ticker": row["ticker"], "name": row.get("name") or row["ticker"]}
            for row in holdings
            if not row.get("is_sell_all") and key in (row.get("sources") or []) and row["ticker"] not in expected
        )

    # 환율을 반드시 넘긴다 — 계좌 원장은 원화이고 종목 가격은 그 시장 통화라, 빠뜨리면
    # 원화 총자산을 달러 가격으로 나눠 수량이 환율 배수만큼 부풀어 오른다.
    _attach_account_targets(hypo, account, krw_rate, slot_keys=keys)

    # 다음 교체 체결일 — 주간 리듬 기준 다음주 첫 거래일.
    this_week = today_local.isocalendar()[:2]
    fill_date = next((str(day.date()) for day in ahead if day.date().isocalendar()[:2] > tuple(this_week)), None)

    # 오늘의 액션과 같은 조립 — 교체 예상분만 rebalance 로 넘겨 '교체 매수/매도' 라벨을 받는다.
    hypo_slots: dict[str, dict[str, Any]] = {}
    for key in keys:
        slot = actions["slots"][key]
        hypo_slots[key] = {
            "label": slot["label"],
            "live": slot.get("live", False),
            "sells": slot["sells"],
            "entries": slot["entries"],
            "exit_forecast": [],
            "rebalance": (
                {"is_filled": True, "fill_date": None, "buys": preview_buys, "sells": preview_sells}
                if key == rotating[0]
                else None
            ),
        }
    groups = _build_action_groups(hypo, {"slots": hypo_slots}, fill_date, currency=currency)
    # 주중 이탈 예상 종목은 뺀다 — 다음주가 아니라 내일 시가에 팔릴 예상이라 오늘의 액션의
    # '(예상)' 그룹이 담당한다. 여기 두면 '교체 매도 · 체결일 미정' 으로 잘못 안내된다.
    forecast_tickers = {row["ticker"] for slot in actions["slots"].values() for row in slot["exit_forecast"]}
    if forecast_tickers:
        groups = [
            {**group, "items": [item for item in group["items"] if item["ticker"] not in forecast_tickers]}
            for group in groups
            if not group.get("forecast")
        ]
        groups = [group for group in groups if group["items"]]
    return {"fill_date": fill_date, "groups": groups}


def mix_positions(account_id: str | None = None) -> dict[str, Any]:
    """오늘 기준 합성 운영 상태 — 보유 목록(목표 비중)·현금 비중·오늘의 액션.

    각 슬리브의 전략 화면이 계산하는 것을 어댑터(`utils.mix_sleeve.slot_state`)로 같은
    형태로 받아 합칠 뿐, 새 판정 로직은 없다. 비중은 슬리브 몫 ÷ 슬롯 수(빈 슬롯 = 현금).
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

    # ── 슬리브 몫 — 월초 배분에서 각 슬리브가 흘러간 비율을 역산한다 ──
    # 재조정은 매월 첫 거래일에만 하므로, 그 사이에는 잘 나간 슬리브의 몫이 커진 채로
    # 가는 것이 백테스트다. 항상 월초 배분으로 보면 승자 슬리브를 매주 깎는 지시가 나온다.
    base_weights = mix_weights_for_account(ctx["account_id"])
    drifted = _sleeve_shares(ctx)
    reserved_cash_share = drifted["cash_pct"]
    shares = {key: drifted[f"{key}_pct"] for key in keys}

    # 새로 담는 슬롯의 몫 — 진입·교체 시점에는 그 슬리브 몫의 1/N 을 배정한다.
    # 이미 들고 있는 종목은 **흘러간 실제 비중**(drift_pct)을 목표로 쓴다. 진입할 때
    # 1/N 이었다가 시세대로 벌어진 값이고, 백테스트도 그 상태를 그대로 들고 간다.
    # 고정 1/N 을 목표로 두면 목표와 보유가 매일 어긋나 실제로는 하지 않을 매매가 나온다.
    slot_weight = {key: shares[key] / states[key].top_n for key in keys}

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
                # 전략 수익률(이론값) — 그 전략이 잡은 편입가 대비.
                row[f"{key}_return_pct"] = None
                # 보유 기간 표기 — 전략마다 단위가 다르다("3주" vs "12일"). 슬롯에 어느
                # 전략이 오든 맞게 읽히도록 숫자가 아니라 완성된 문자열로 내려준다.
                row[f"{key}_held_label"] = None
                # 차트의 진입 화살표 — 슬리브마다 편입 시점이 다르므로 슬롯별로 들고 간다.
                row[f"{key}_entry_date"] = None
                row[f"{key}_entry_price"] = None
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
        if target.get("return_pct") is not None:
            row[f"{source}_return_pct"] = round(float(target["return_pct"]), 2)
        if target.get("held_label"):
            row[f"{source}_held_label"] = target["held_label"]
        if target.get("entry_date"):
            row[f"{source}_entry_date"] = str(target["entry_date"])[:10]
        if target.get("entry_price") is not None:
            row[f"{source}_entry_price"] = float(target["entry_price"])

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
                weight = slot_weight[key]
            add_target(key, target, weight)

    # 매월 첫 거래일 = 슬리브 배분 리밸런싱 날 (그 시장 달력 기준).
    from config import MARKET_SCHEDULES
    from utils.trading_calendar import get_trading_days

    country = ctx["country"]
    # 종목 가격의 통화 — 계좌 원장(원화)과 맞추려면 환율이 필요하다.
    pool_currency = ctx["currency"]
    tz_name = str((MARKET_SCHEDULES.get(country) or {}).get("timezone") or "Asia/Seoul")
    today_local = pd.Timestamp.now(tz=tz_name).date()
    month_start = today_local.replace(day=1)
    month_days = get_trading_days(month_start.strftime("%Y-%m-%d"), today_local.strftime("%Y-%m-%d"), country)
    sleeve_rebalance_today = bool(month_days) and month_days[0].date() == today_local

    # ── 월초 배분 되돌리기는 **현금 우선**으로 이관한다 (백테스트와 같은 규칙) ──
    # 종목 비중은 흘러간 그대로 두고(오르는 종목 유지), 장부상 현금만 슬리브 사이에서
    # 옮긴다. 한 슬리브의 주식만으로 제 몫을 넘을 때만 — 현금으로 이관액을 다 못 채울
    # 때만 — 초과분을 비례 매도한다. 그래서 여기서는 주식 비중을 그 몫에 맞춰 깎고,
    # 슬리브 몫 표시는 월초 배분으로 되돌린다. 비워 두는 현금 몫도 여기서 함께 복구된다.
    if sleeve_rebalance_today:
        for key in keys:
            target_pct = base_weights[f"{key}_pct"]
            sleeve_stock = sum(row[f"{key}_weight"] for row in holdings)
            if sleeve_stock > target_pct:
                scale = (target_pct / sleeve_stock) if sleeve_stock > 0 else 0.0
                for row in holdings:
                    trimmed = row[f"{key}_weight"] * scale
                    row["weight_pct"] += trimmed - row[f"{key}_weight"]
                    row[f"{key}_weight"] = trimmed
        shares = {key: base_weights[f"{key}_pct"] for key in keys}
        reserved_cash_share = base_weights["cash_pct"]

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

    # ── 적용 계좌 — 이 계산의 기준 계좌 그대로다(슬리브별 풀이 여기서 나왔다).
    account = _load_account_state(ctx["account_id"])
    if account is not None:
        krw_rate = _krw_rate(pool_currency)
        _value_account(account, krw_rate, {row["ticker"]: row["price"] for row in holdings if row.get("price")})

        # ── 고정 자산 몫만큼 슬리브·현금 비중을 줄인다 ──
        # 슬리브 배분(50:50 등)은 **고정 자산을 뺀 나머지**에 대한 비율이다. 고정 자산은
        # 사용자가 못 바꾸는 값이라 배분 대상이 아니고, 총자산 대비로 두면 슬리브 합 + 고정
        # 자산이 100% 를 넘는다. 여기서 줄여야 목표 금액이 실제 계좌와 맞는다.
        fixed_pct = float(account["fixed_asset_pct"])
        if fixed_pct > 0:
            scale = max(1.0 - fixed_pct / 100.0, 0.0)
            for row in holdings:
                row["weight_pct"] *= scale
                for key in keys:
                    row[f"{key}_weight"] *= scale
            shares = {key: value * scale for key, value in shares.items()}
            reserved_cash_share *= scale
            base_weights = {name: value * scale for name, value in base_weights.items()}
            # 고정 자산 행 — 표에서 비중 합이 100% 가 되게 한다. 목표 = 현재라 매매 지시가
            # 나오지 않는다(수량·목표수량을 아래에서 같은 값으로 채운다).
            holdings.append(
                {
                    "ticker": FIXED_ASSET_TICKER,
                    "name": FIXED_ASSET_NAME,
                    "sources": [],
                    "weight_pct": fixed_pct,
                    "price": None,
                    "change_pct": None,
                    **{f"{key}_weight": 0.0 for key in keys},
                    **{f"{key}_status": None for key in keys},
                    "is_fixed_asset": True,
                }
            )

        # 목표 주수 — 슬리브마다 자기 몫 예산 안에서 배분. 종목·비중은 백테스트 것 그대로다.
        total_assets = float(account.get("total_assets") or 0)
        sleeve_amount_krw = {key: total_assets * shares[key] / 100.0 for key in keys}
        target_shares = _sleeve_target_shares(
            states,
            sleeve_amount_krw,
            krw_rate,
        )
        account["sell_all"] = _attach_account_targets(
            holdings, account, krw_rate, slot_keys=keys, target_shares=target_shares
        )

        # 고정 자산 행 마무리 — 목표 대조(위)는 계좌 원장 holdings 만 보므로 이 행은 비어 있다.
        # 평가액은 그대로 채우고 수량 지시는 만들지 않는다(살 수도 팔 수도 없는 자산이다).
        for row in holdings:
            if not row.get("is_fixed_asset"):
                continue
            row["held_value"] = account["fixed_asset_value"]
            row["current_weight_pct"] = fixed_pct
            row["target_amount"] = account["fixed_asset_value"]
            row["held_quantity"] = None
            row["target_quantity"] = None
            row["trade_quantity"] = None
            # 살 수도 팔 수도 없는 자산이라 목표 = 현재다. 실제목표에도 같은 값을 넣어야
            # 두 합계가 같은 기준이 된다(안 넣으면 고정 자산 몫만큼 실제목표가 낮게 보인다).
            row["actual_weight_pct"] = fixed_pct
            # 수익률 — 평단이 없는 항목이라 평가액과 손익으로 낸다(원금 = 평가액 − 손익).
            principal = float(account.get("fixed_asset_native") or 0) - float(
                account.get("fixed_asset_change_native") or 0
            )
            row["return_pct"] = (
                round(float(account["fixed_asset_change_native"]) / principal * 100.0, 2) if principal > 0 else None
            )

    # 비중 합계·슬리브 현금 — 고정 자산 축소가 끝난 뒤의 값이라야 실제 계좌와 맞는다.
    stock_pct = sum(row["weight_pct"] for row in holdings)
    # 실제 가능한 주식 비중 — 단주로 못 채운 나머지는 전부 현금이다.
    actual_stock_pct = sum(float(row.get("actual_weight_pct") or 0) for row in holdings)
    # 슬리브 현금 = 그 슬리브 몫에서 담긴 종목 비중을 뺀 나머지. 빈 슬롯 수로 세면
    # 흘러간 비중과 맞지 않는다(종목이 오르면 남는 현금은 그만큼 줄어든다).
    sleeve_cash = {key: max(shares[key] - sum(row[f"{key}_weight"] for row in holdings), 0.0) for key in keys}

    # 종목명 옆 추세 이탈 배지(❗)용 — 행이 속한 슬리브의 **종목풀 설정** 이평선 기준.
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
                    # 주기적 교체가 있는 전략만 — 판정은 끝났고 체결만 남았다.
                    "rebalance": states[key].rebalance,
                }
                for key in keys
            },
            "sleeve_rebalance_today": sleeve_rebalance_today,
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
    currency = next((states[key].currency for key in keys if states[key].currency), "KRW")
    payload["actions"]["groups"] = _build_action_groups(
        payload["holdings"],
        payload["actions"],
        next_trading_day,
        currency=currency,
    )
    # 다음주 교체 가정 미리보기 — 실시간 순위 기준 잠정치.
    payload["actions"]["next_week_preview"] = _build_next_week_preview(
        states,
        payload["holdings"],
        payload["actions"],
        slot_weight,
        account,
        ahead,
        today_local,
        currency,
        _krw_rate(pool_currency),
    )
    # 슬리브별 값을 `slots[키]` 로 모아 내보낸다 — 화면은 슬롯 키를 돌며 읽는다.
    payload["holdings"] = [_holding_payload(row, keys) for row in payload["holdings"]]
    return payload


def _merge_trades(slots: list[SleeveSpec], results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """슬리브들의 체결을 한 목록으로 — 보유중 행이 먼저, 그 아래는 청산일 최신순.

    엔진별 형태 차이는 어댑터(`utils.mix_sleeve.trade_rows`)가 흡수한다 — 여기서는 슬롯
    키만 붙여 합친다. ``strategy`` 에 전략 이름이 아니라 **슬롯 키**를 담는 것은 같은
    전략이 두 슬롯에 올 수 있어서다(어느 슬리브인지는 화면이 계좌 설정으로 풀어 쓴다).
    """
    from utils.mix_sleeve import trade_rows

    merged: list[dict[str, Any]] = []
    for spec in slots:
        for row in trade_rows(spec, results[spec.key]):
            merged.append({**row, "strategy": spec.key})
    holding = sorted(
        (row for row in merged if row.get("exit_date") is None), key=lambda row: row["entry_date"], reverse=True
    )
    closed = sorted((row for row in merged if row.get("exit_date")), key=lambda row: row["exit_date"], reverse=True)
    return holding + closed


def _simulate_mix_daily(ctx: dict[str, Any], curves: dict[str, dict[str, float]]) -> Any:
    """한 계좌에서 슬리브들을 함께 굴린 일별 자산 곡선(시작 1.0). 반환은 pandas.Series.

    **판정을 다시 하지 않는다.** 슬리브별 곡선(`utils.mix_sleeve.daily_curve`)은 각 전략
    엔진이 낸 결과 그대로이고, 여기서는 그 위에 **월초 이관**만 얹는다. 예전에는 이 함수가
    슬롯 엔진의 청산·진입 루프를 통째로 다시 구현했는데, 그러면 엔진을 고칠 때마다 합성이
    조용히 갈라진다(실제로 포트폴리오 슬리브는 이 루프에서 아무것도 사지 않아 현금으로만
    남아 있었다). 판정은 엔진 한 곳, 합성은 읽기만 — 이게 이 파일이 지켜야 할 선이다.

    월초 첫 거래일에 계좌 설정 배분으로 되돌린다. 넘치는 슬리브에서 빼 모은 뒤 모자란
    슬리브에 넘기고, 남는 것이 비워 두는 현금 몫이 된다. 이관에는 넘기는 슬리브 종목풀의
    **매도 슬리피지**를 물린다(받는 쪽이 그 돈을 굴리는 비용은 이미 그 슬리브 곡선 안에 있다).
    """
    import pandas as pd

    from utils.pool_settings_store import get_pool_slippage

    slots: list[SleeveSpec] = ctx["slots"]
    weights = mix_weights_for_account(ctx["account_id"])
    keys = [spec.key for spec in slots]
    sell_slip = {spec.key: get_pool_slippage(spec.pool)[1] / 100.0 for spec in slots}

    # 슬리브 곡선을 한 표로 — 거래일이 다른 풀이 섞이면 쉬는 날은 직전 값을 이어간다
    # (그 시장이 닫힌 날 그 슬리브의 평가액은 변하지 않는다). 모든 슬리브에 값이 생기기
    # 전의 앞부분은 버린다 — 한 슬리브만으로 합성을 만들 수는 없다.
    frame = pd.DataFrame({key: pd.Series(curves[key]) for key in keys}).sort_index().ffill().dropna()
    if len(frame) < 2:
        raise RuntimeError("슬리브 전략들의 공통 백테스트 구간이 부족합니다.")
    growth = (frame / frame.shift(1)).iloc[1:]

    values = {key: weights[f"{key}_pct"] / 100.0 for key in keys}
    # 두 전략에 주지 않고 비워 두는 몫 — 자라지 않고, 월초에만 다시 맞춘다.
    cash_reserved = weights["cash_pct"] / 100.0
    days = list(frame.index)
    curve: dict[str, float] = {days[0]: sum(values.values()) + cash_reserved}

    for i, day in enumerate(days[1:], start=1):
        for key in keys:
            values[key] *= float(growth.at[day, key])

        # 월초 배분 되돌리기 — 현금 우선 이관.
        if day[:7] != days[i - 1][:7]:
            total = sum(values.values()) + cash_reserved
            targets = {key: total * weights[f"{key}_pct"] / 100.0 for key in keys}
            pool_cash = cash_reserved
            cash_reserved = 0.0
            for key in keys:
                excess = values[key] - targets[key]
                if excess <= 1e-12:
                    continue
                values[key] -= excess
                pool_cash += excess * (1 - sell_slip[key])
            for key in keys:
                need = targets[key] - values[key]
                if need <= 1e-12:
                    continue
                give = min(need, pool_cash)
                values[key] += give
                pool_cash -= give
            cash_reserved = pool_cash

        # 비워 둔 현금도 계좌 자산이다 — 곡선에서 빼면 배분을 늘릴수록 총자산이 줄어 보인다.
        curve[day] = sum(values.values()) + cash_reserved

    return pd.Series(curve).sort_index()


def _value_account(account: dict[str, Any], krw_rate: float, price_hint: dict[str, float] | None = None) -> None:
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

    # 합성 곡선 — 위 슬리브 곡선 위에 월초 배분 이관만 얹는다(판정은 엔진이 이미 했다).
    mix_curve = _simulate_mix_daily(ctx, curves)
    dates = [d for d in mix_curve.index if d in bench_curve]
    if len(dates) < 2:
        raise RuntimeError("슬리브 전략들의 공통 백테스트 구간이 부족합니다.")

    first_mix = float(mix_curve[dates[0]])
    # 벤치마크는 **시작일 시가**를 1 로 둔다 — 전략도 그날 시가에 사기 때문이다(공용 함수).
    from utils.benchmark_curve import growth_from_frame

    bench_growth = growth_from_frame(
        bench_frame,
        bench_index[bench_index.isin(pd.to_datetime(dates))],
        label=f"벤치마크({ctx['benchmark_name']})",
    )
    bench_curve = {str(day.date()): float(value) for day, value in bench_growth.items()}
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
    merged_trades = _merge_trades(ctx["slots"], results)

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
    }
