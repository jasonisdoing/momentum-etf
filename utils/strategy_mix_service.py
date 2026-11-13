"""합성 전략 — 계좌·슬리브 설정과 화면 메타.

종목과 진입·청산 시점은 각 전략 엔진이 정한다. 합성은 계좌 평가액(현금 포함)에
슬리브 배분과 회수·채우기 정책을 적용한다. 합성 자체의 백테스트는 두지 않는다.
운용 현황(목표·액션)은 `utils.strategy_mix_positions` 에 있다.
"""

from __future__ import annotations

from typing import Any

from core.strategy.mix.capital_policy import HARVEST_OFF
from utils.cash_model import currency_for_country
from utils.mix_sleeve import STRATEGY_LABELS, SleeveSpec
from utils.pool_strategy_use import pool_options


def _all_active_pools() -> list[str]:
    """활성 종목풀 전체 — 계좌가 가리키는 슬리브 풀의 표기(이름·아이콘)를 붙이는 데 쓴다."""
    from utils.settings_loader import list_available_ticker_types

    return list(list_available_ticker_types())


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
    `resolve_mix_account` 가 조합이 갖춰졌는지 확인하고, 없으면 명시적으로 알린다.
    """
    from utils.account_settings_store import MIX_SLEEVE_KEYS, TRADE_SLOT_MARK, mix_sleeves_of, mix_slot_marks
    from utils.settings_loader import get_account_settings, list_available_accounts

    pool_names = {option["ticker_type"]: option for option in pool_options(_all_active_pools())}

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
        marks = mix_slot_marks(account_id)
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
                **{key: inner.get(key) for key in ("mix_harvest_pct", "mix_refill_pct")},
                # 비워 두는 현금 몫(%) — 슬리브 배분은 sleeves 안에 있다.
                "mix_cash_pct": mix_weights(inner)["cash_pct"],
                # 슬롯 키 → 표시 기호. 사고팔기가 있으면 사고팔기가 A 라 슬리브는 B 부터다.
                "slot_marks": marks,
                # 사고팔기 기호 — 사고팔기 종목이 없는 계좌는 None(사고팔기 칸이 없다).
                "trade_mark": TRADE_SLOT_MARK if marks[MIX_SLEEVE_KEYS[0]] != TRADE_SLOT_MARK else None,
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
    """화면 초기용 — 운용 계좌 목록과 선택지."""
    from utils.account_settings_store import MAX_MIX_SLEEVES, MIN_MIX_SLEEVES
    from utils.mix_sleeve import STRATEGY_OPTIONS

    accounts = mix_accounts()
    return {
        "accounts": accounts,
        # 슬리브 개수 제한·초안 — 백엔드 상수가 단일 소스(프론트에 복사본을 두지 않는다).
        "min_sleeves": MIN_MIX_SLEEVES,
        "max_sleeves": MAX_MIX_SLEEVES,
        "default_sleeves": default_sleeves(),
        # 조합 셀렉트 선택지 — 이 화면에서 전략·종목풀을 고른다(계좌 설정은 사용 여부만).
        "strategy_options": [{"value": key, "label": STRATEGY_LABELS[key]} for key in STRATEGY_OPTIONS],
        # 풀은 계좌 국가에 맞는 것만 고를 수 있어야 해서 국가 코드를 함께 준다.
        "pool_options": pool_options(_all_active_pools()),
        # 조합별 저장 설정 요약 — 화면이 슬리브 오른쪽에 읽기 전용으로 보여준다.
        "settings_summaries": _settings_summaries(),
        # 기본 선택 — 목록의 첫 계좌. 화면이 마지막 선택을 로컬스토리지에 기억한다.
        "account_id": accounts[0]["account_id"] if accounts else "",
    }


def mix_weights_for_account(account_id: str) -> dict[str, float]:
    """그 계좌의 합성 배분(%) — {슬롯키_pct: 값} + cash_pct. 목록에 없으면 현금 100%.

    슬리브마다 풀이 다를 수 있으므로 풀로 계좌를 되찾을 수 없다 — 계좌가 기준이다.
    운영 화면·슬랙이 모두 같은 배분을 쓰도록 여기서만 읽는다.
    """
    target = str(account_id or "").strip()
    for account in mix_accounts():
        if account["account_id"] == target:
            weights = {f"{row['key']}_pct": float(row["weight_pct"]) for row in account["sleeves"]}
            weights["cash_pct"] = float(account["mix_cash_pct"])
            return weights
    return {"cash_pct": 100.0}


def resolve_mix_account(account_id: str | None) -> dict[str, Any]:
    """계좌를 확정하고 슬리브별 풀·저장 설정을 검증해 돌려준다(`mix_positions` 가 쓴다).

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

    from utils.settings_loader import get_account_settings

    account_settings = get_account_settings(account["account_id"]) or {}
    harvest = account_settings.get("mix_harvest_pct")
    refill = account_settings.get("mix_refill_pct")
    if harvest is None or refill is None:
        raise RuntimeError("초과 비중 매도와 보유 종목 추가 매수 방식을 먼저 저장하세요.")

    # 슬리브 목록 — 합성은 이것만 보고 돈다. 전략 조합(모멘텀 둘·신고가 둘·섞기)과 무관하다.
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
        # 국가·통화 — 거래 달력과 원화 환산에 쓴다.
        "country": country,
        "currency": currency,
        "mix_harvest_pct": None if harvest == HARVEST_OFF else float(harvest),
        "mix_refill_pct": float(refill),
    }
