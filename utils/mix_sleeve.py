"""합성 슬리브 어댑터 — 모멘텀·신고가 엔진을 **같은 얼굴**로 감싼다.

합성(`utils/strategy_mix_service`)은 예전에 슬리브 둘을 `sm`/`nh` 로 하드코딩했다. 그래서
"모멘텀 2개" 나 "신고가 2개" 같은 조합을 만들 수 없었다. 여기서 두 엔진의 차이를 흡수해,
합성은 **슬롯 목록**만 알고 어느 전략인지는 신경 쓰지 않게 한다.

두 엔진이 원래 다른 점(이 모듈이 흡수하는 것):
  - 현재 상태 함수 이름과 반환 형태 (`compute_picks` vs `current_positions`)
  - 일별 곡선 형태 — 모멘텀은 전일 대비 변동률(%), 신고가는 구간 시작 대비 누적(%)
  - 사전 준비 — 신고가만 가격 패널(`load_context`)이 필요하다
  - 백테스트 인자 순서

일별 곡선은 **누적 배수**(시작 1.0)로 통일해서 돌려준다 — 합성이 형태를 다시 따질 일이 없다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()

MOMENTUM = "momentum"
NEW_HIGH = "new_high"
PORTFOLIO = "portfolio"

# 화면 셀렉트·라벨의 단일 소스. 값은 계좌 설정에 그대로 저장된다.
STRATEGY_OPTIONS: tuple[str, ...] = (MOMENTUM, NEW_HIGH, PORTFOLIO)
STRATEGY_LABELS: dict[str, str] = {MOMENTUM: "모멘텀", NEW_HIGH: "신고가", PORTFOLIO: "포트폴리오"}


@dataclass(frozen=True)
class SleeveSpec:
    """슬리브 하나 — 어떤 전략을 어떤 종목풀로 굴리는지.

    `key` 는 합성 안에서 이 슬리브를 가리키는 이름이다(행의 `sources`, 배분 필드 등).
    """

    key: str
    strategy: str
    pool: str
    settings: dict[str, Any]
    # 사용자가 붙인 표시 이름 — 비면 전략 이름을 쓴다. 같은 전략을 두 슬롯에 올릴 수
    # 있어서, 화면에서 둘을 구분하려면 이름을 직접 붙일 수 있어야 한다.
    name: str = ""

    @property
    def label(self) -> str:
        return self.name or STRATEGY_LABELS.get(self.strategy, self.strategy)


def normalize_strategy(value: Any) -> str:
    """전략 이름을 정규화한다. 모르는 값은 명시적 에러 — 임의로 고르지 않는다."""
    strategy = str(value or "").strip().lower()
    if strategy not in STRATEGY_OPTIONS:
        raise ValueError(f"알 수 없는 전략입니다: {value} (가능: {', '.join(STRATEGY_OPTIONS)})")
    return strategy


def settings_map(strategy: str) -> dict[str, Any]:
    """그 전략의 풀별 저장 설정 {풀: 설정}."""
    strategy = normalize_strategy(strategy)
    if strategy == MOMENTUM:
        from utils.momentum_service import load_settings_map

        # 모멘텀은 `{pool, settings_by_pool}` 로 한 겹 감싸 저장한다(나머지는 평면).
        return dict(load_settings_map().get("settings_by_pool") or {})
    if strategy == PORTFOLIO:
        from utils.portfolio_service import load_settings_map as portfolio_map

        return dict(portfolio_map())
    from utils.new_high_service import load_settings_map

    return dict(load_settings_map())


def settings_summary(strategy: str, settings: dict[str, Any]) -> list[dict[str, str]]:
    """그 슬리브의 저장 설정을 화면에 읽기 전용으로 보여줄 `[{label, value}]`.

    전략마다 항목이 다르다 — 어떤 항목을 어떤 말로 부르는지는 여기 한 곳에만 둔다.
    각 전략 화면의 셀렉트 라벨과 같은 말을 쓴다(같은 값을 다른 이름으로 부르지 않는다).
    값이 없는 설정은 임의로 채우지 않고 '없음'·'안 씀'으로 그대로 드러낸다.
    """
    strategy = normalize_strategy(strategy)

    def optional(value: Any, suffix: str = "", *, empty: str = "없음") -> str:
        return empty if value is None else f"{value:g}{suffix}" if isinstance(value, (int, float)) else str(value)

    if strategy == MOMENTUM:
        return [
            {"label": "종목 수", "value": optional(settings.get("top_n"), "개")},
            {"label": "이평선", "value": f"{settings.get('short_ma_days')}/{settings.get('long_ma_days')}일"},
            {"label": "ADR 하한", "value": optional(settings.get("adr_floor"))},
        ]
    if strategy == NEW_HIGH:
        return [
            {"label": "종목 수", "value": optional(settings.get("top_n"), "개")},
            {"label": "이탈 이평", "value": optional(settings.get("exit_ma_days"), "일")},
            {"label": "거래대금 하한", "value": optional(settings.get("min_value_mult"), "배")},
            {"label": "ADR 하한", "value": optional(settings.get("adr_floor"))},
        ]
    from config import REBALANCE_LABELS

    weights = list(settings.get("weights") or [])
    rebalance = str(settings.get("rebalance") or "none")
    return [
        {"label": "종목 수", "value": f"{len(weights)}개"},
        {"label": "리밸런싱", "value": REBALANCE_LABELS.get(rebalance, rebalance)},
        {"label": "허용 밴드", "value": optional(settings.get("band_pct"), "%p")},
        {"label": "현금", "value": optional(settings.get("cash_weight_pct"), "%")},
    ]


def validate_settings(strategy: str, settings: dict[str, Any]) -> dict[str, Any]:
    strategy = normalize_strategy(strategy)
    if strategy == MOMENTUM:
        from utils.momentum_service import validate_settings as validate

        return validate(settings)
    if strategy == PORTFOLIO:
        from utils.portfolio_service import validate_settings as validate_portfolio

        return validate_portfolio(settings)
    from utils.new_high_service import validate_settings as validate

    return validate(settings)


def load_context(spec: SleeveSpec) -> dict[str, Any] | None:
    """백테스트·현재 상태가 함께 쓰는 무거운 준비물(가격 패널). 필요 없는 전략은 None.

    패널 생성이 이 계산에서 가장 비싼 부분이라, 한 번 만들어 재사용해야 한다. 모멘텀·신고가는
    같은 슬롯 엔진을 쓰므로 둘 다 필요하고, 포트폴리오는 판정이 없어 필요 없다.
    """
    if spec.strategy == MOMENTUM:
        from utils.momentum_backtest import load_context as sm_load_context

        return sm_load_context(spec.settings)
    if spec.strategy == NEW_HIGH:
        from utils.new_high_backtest import load_context as nh_load_context

        return nh_load_context(spec.settings)
    return None


def current_state(spec: SleeveSpec) -> dict[str, Any]:
    """오늘 기준 이 슬리브의 선정·보유 상태 — 엔진 원본 형태 그대로."""
    if spec.strategy == MOMENTUM:
        from utils.momentum_backtest import current_positions as sm_positions

        return sm_positions(spec.settings)
    if spec.strategy == PORTFOLIO:
        # 포트폴리오는 판정이 없다 — 저장된 목표 비중이 곧 현재 상태다.
        return dict(spec.settings)
    from utils.new_high_backtest import current_positions

    return current_positions(spec.settings)


def _held_label(value: Any, *, unit: str, zero: str) -> str:
    """보유 기간 표기 — 전략마다 단위가 다르다(모멘텀 "주", 신고가 "일").

    화면이 슬롯 A·B 만 알고 어느 전략인지 모르게 하려면, 숫자가 아니라 단위까지 붙은
    문자열로 내려줘야 한다. 값이 없으면 빈 문자열(표시 안 함).

    ``zero`` 는 0 이하일 때 쓸 문구다 — 아직 안 산 종목("0주")에 쓴다.
    """
    if value is None:
        return ""
    try:
        count = int(value)
    except (TypeError, ValueError):
        return ""
    return zero if count <= 0 else f"{count}{unit}"


@dataclass
class SlotState:
    """슬리브 하나의 **오늘 상태**를 전략과 무관한 한 형태로 담는다.

    합성 운용 현황(`mix_positions`)이 이것만 보고 돈다. 전략마다 다른 것:
      - 슬롯을 채우는 방식 — 모멘텀은 순위 상위 N, 신고가는 보유 + 진입 예정
      - 매도 리듬 — 모멘텀은 주간 교체 + 주중 자격 상실, 신고가는 이탈·손절
      - 보유 기간 단위 — "주" vs "일"
    이 차이는 전부 `slot_state()` 안에서 흡수하고, 밖으로는 아래 필드만 낸다.
    """

    spec: SleeveSpec
    top_n: int
    currency: str
    # 목표 슬롯 — 이 슬리브가 (다음 체결 후) 들고 있어야 할 종목들.
    # ticker·name·price·change_pct·status·return_pct·held_label·is_exiting·drift_pct
    targets: list[dict[str, Any]] = field(default_factory=list)
    held_tickers: set[str] = field(default_factory=set)
    held_count: int = 0
    # 슬롯을 실제로 채운 수 — 나머지는 그 슬리브의 현금이다.
    active_count: int = 0
    # 다음 거래일 시가에 파는 것(확정) — {ticker, name, reason, return_pct}
    sells: list[dict[str, Any]] = field(default_factory=list)
    # 장중 판정 기준 이탈 **예상** — 오늘 종가로 확정된다. 화면 전용(알람 제외).
    exit_forecast: list[dict[str, Any]] = field(default_factory=list)
    # 다음 거래일 시가에 새로 담는 것 — {ticker, name, price, change_pct, value_mult}
    entries: list[dict[str, Any]] = field(default_factory=list)
    # 주기적 교체가 있는 전략만 — {is_filled, fill_date, signal_date, portfolio_week, buys, sells}
    rebalance: dict[str, Any] | None = None
    # 다음 교체 예상 종목 {ticker: row} — 교체가 있는 전략만. 다음주 가정 미리보기가 쓴다.
    next_expected: dict[str, dict[str, Any]] = field(default_factory=dict)
    # 데이터 기준일·장중 반영 — 이 정보를 주는 전략만 채운다.
    as_of: str | None = None
    live: bool = False
    # 백테스트가 지금 굴리고 있는 슬리브 평가액과 슬롯 하나의 몫(그 풀 통화).
    # 합성이 계좌 금액을 역산할 때 쓴다 — 판정이 없는 포트폴리오는 0.
    sleeve_value: float = 0.0
    slot_amount: float = 0.0


def slot_state(spec: SleeveSpec) -> SlotState:
    """엔진 원본 상태를 `SlotState` 로 정규화한다 — 합성이 전략을 가리지 않게."""
    raw = current_state(spec)
    # 포트폴리오는 슬롯 개수 개념이 없다 — 담은 종목 수가 곧 슬롯 수다.
    top_n = len(spec.settings.get("weights") or []) if spec.strategy == PORTFOLIO else int(spec.settings["top_n"])
    if spec.strategy == PORTFOLIO:
        return _portfolio_slot_state(spec, raw)
    # 모멘텀·신고가는 같은 슬롯 엔진이라 상태 형태도 같다.
    return _slot_state_from_positions(spec, raw, top_n)


def _portfolio_slot_state(spec: SleeveSpec, raw: dict[str, Any]) -> SlotState:
    """포트폴리오 — 저장된 목표 비중이 그대로 슬롯이다.

    이 전략에는 판정이 없어 `SlotState` 의 대부분이 빈다: 교체(`rebalance`)·이탈(`sells`)·
    진입 예정(`entries`)·이탈 예상이 모두 없다. 대신 **종목마다 비중이 다르므로**
    `drift_pct` 에 저장 비중을 그대로 싣는다 — 합성이 `슬리브 몫 × drift_pct / 100` 으로
    목표를 잡으므로 균등 분배 전략과 같은 코드로 굴러간다.

    남는 몫(현금)은 슬롯을 채우지 않는 것으로 표현된다 — 합성이 이미 그렇게 센다.
    """
    # 현재가·일간은 /strategy-portfolio 화면과 **같은 공용 함수**로 구한다. 판정이 없다고
    # 시세까지 비워 두면 합성이 목표수량·매매수량을 못 내고(가격 없는 종목은 수량 계산에서
    # 빠진다) 그 슬롯의 오늘의 액션이 통째로 사라진다.
    from utils.portfolio_service import universe_metrics

    metrics_by = {row["ticker"]: row for row in universe_metrics(spec.pool)}
    targets: list[dict[str, Any]] = []
    for row in list(raw.get("weights") or []):
        ticker = str(row["ticker"]).strip()
        metrics = metrics_by.get(ticker) or {}
        targets.append(
            {
                "ticker": ticker,
                "name": metrics.get("name") or ticker,
                "price": metrics.get("current_price"),
                "change_pct": metrics.get("daily_change_pct"),
                "status": f"목표 {float(row['weight_pct']):.2f}%",
                "return_pct": None,
                "held_label": "",
                # 포트폴리오는 진입 판정이 없다 — 매수 시점을 만들어 내지 않는다.
                "entry_date": None,
                "entry_price": None,
                "is_exiting": False,
                # 저장 비중을 그대로 싣는다 — 합성이 `슬리브 몫 × drift_pct / 100` 으로 목표를
                # 잡으므로, 종목 30% + 현금 10% 는 슬리브 몫의 30%·10% 가 된다(비율 유지).
                "drift_pct": float(row["weight_pct"]),
            }
        )
    # 종목 비중 합이 100 미만이면 나머지가 현금이다 — 슬리브 몫에서 그만큼이 안 채워진다
    # (합성이 `슬리브 몫 − 담긴 종목 비중 합` 을 그 슬리브의 현금으로 센다).
    # 통화는 종목풀 설정 것 — 비워 두면 두 슬롯이 모두 포트폴리오일 때 합성이 KRW 로
    # 떨어져 미국 풀의 액션 금액이 원화로 찍힌다.
    from utils.settings_loader import get_ticker_type_settings

    return SlotState(
        spec=spec,
        top_n=len(targets),
        currency=str((get_ticker_type_settings(spec.pool) or {}).get("currency") or "KRW").strip().upper(),
        targets=targets,
        held_tickers={row["ticker"] for row in targets},
        held_count=len(targets),
        active_count=len(targets),
        sells=[],
        exit_forecast=[],
        entries=[],
        rebalance=None,
    )


def _slot_state_from_positions(spec: SleeveSpec, raw: dict[str, Any], top_n: int) -> SlotState:
    held = list(raw.get("holdings") or [])
    # 빈 슬롯을 채울 진입 예정 — 다음 시가에 사므로 목표에 포함한다. 매도 예정(이탈·손절)
    # 종목은 같은 시가에 슬롯이 비므로 빈 슬롯으로 센다 — 엔진의 pick_entries 와 같은
    # 계산이라 신고가 화면의 '진입 예정' 과 어긋나지 않는다.
    exiting_count = sum(1 for row in held if str(row.get("status")) == "sell")
    free = max(top_n - (len(held) - exiting_count), 0)
    planned = list(raw.get("planned_entries") or [])[:free]

    targets: list[dict[str, Any]] = []
    for row in held:
        status = "오늘 진입" if row.get("is_new") else f"{row.get('days')}일째"
        exiting = str(row.get("status")) == "sell"
        if exiting:
            status += f" · 매도 예정({row.get('exit_reason') or '이탈'})"
        weight = row.get("sleeve_weight_pct")
        targets.append(
            {
                "ticker": str(row["ticker"]).strip(),
                "name": row.get("name"),
                "price": row.get("price"),
                "change_pct": row.get("change_pct"),
                "status": status,
                "return_pct": row.get("return_pct"),
                "held_label": _held_label(row.get("days"), unit="일", zero="진입"),
                # 차트의 진입 화살표용.
                "entry_date": row.get("entry_date"),
                "entry_price": row.get("entry_price"),
                "is_exiting": exiting,
                "drift_pct": float(weight) if weight is not None else None,
            }
        )
    for row in planned:
        targets.append(
            {
                "ticker": str(row["ticker"]).strip(),
                "name": row.get("name"),
                "price": row.get("price"),
                "change_pct": row.get("change_pct"),
                "status": "진입 예정 (다음 시가 매수)",
                "return_pct": None,
                # 아직 안 샀다 — 빈칸으로 두면 보유일 컬럼·차트 배지가 통째로 사라진다.
                "held_label": "0일",
                # 아직 안 샀다.
                "entry_date": None,
                "entry_price": None,
                "is_exiting": False,
                "drift_pct": None,
            }
        )

    return SlotState(
        spec=spec,
        top_n=top_n,
        currency=str(raw.get("currency") or "KRW"),
        targets=targets,
        held_tickers={str(row["ticker"]).strip() for row in held},
        held_count=len(held),
        active_count=sum(1 for row in targets if not row["is_exiting"]),
        sells=[
            {
                "ticker": str(row["ticker"]).strip(),
                "name": row.get("name") or row["ticker"],
                "reason": row.get("exit_reason") or "이탈",
                "return_pct": row.get("return_pct"),
            }
            for row in held
            if str(row.get("status")) == "sell"
        ],
        exit_forecast=[],
        entries=[
            {
                "ticker": str(row["ticker"]).strip(),
                "name": row.get("name") or row["ticker"],
                "price": row.get("price"),
                "change_pct": row.get("change_pct"),
                "value_mult": row.get("value_mult"),
            }
            for row in planned
        ],
        rebalance=None,
        as_of=raw.get("as_of"),
        live=bool(raw.get("live")),
        sleeve_value=float(raw.get("sleeve_value") or 0.0),
        slot_amount=float(raw.get("slot_amount") or 0.0),
    )


def run_backtest(
    spec: SleeveSpec,
    months: int,
    context: dict[str, Any] | None = None,
    initial_capital: float | None = None,
) -> dict[str, Any]:
    """이 슬리브를 **혼자** 굴린 백테스트 — 엔진 원본 형태 그대로.

    ``initial_capital`` 을 주면 그 돈으로 돌린다. 합성 운용 현황이 계좌 금액을 역산해 넘겨
    마지막 날 보유 주수를 그대로 목표 주수로 쓴다. 포트폴리오는 소수 주수로 비중만 맞추는
    전략이라 시작 자본이 결과를 바꾸지 않는다 — 받지 않는다.
    """
    if spec.strategy == PORTFOLIO:
        from utils.portfolio_backtest import run_backtest as portfolio_backtest

        return portfolio_backtest(months, spec.settings)
    if spec.strategy == MOMENTUM:
        from utils.momentum_backtest import run_backtest as sm_backtest

        return sm_backtest(months, spec.settings, context, initial_capital)
    from utils.new_high_backtest import run_backtest as nh_backtest

    return nh_backtest(months, spec.settings, context, initial_capital)


def trade_rows(spec: SleeveSpec, result: dict[str, Any]) -> list[dict[str, Any]]:
    """백테스트 결과의 체결을 **한 형태로** 맞춘다 — 합성 화면의 체결 목록이 이걸 쓴다.

    모멘텀·신고가는 같은 슬롯 엔진이라 형태가 같다(보유중은 `open_positions`, 청산은 `trades`).
    포트폴리오만 '보유 기간' 개념이 없다 — 체결이 종목 교체가 아니라 **비중 되돌리기**라
    진입·청산일이 같은 한 줄로 만든다(매매 기록으로 읽힌다).

    공통 키: ticker · name · entry_date · entry_price · exit_date · exit_price ·
             return_pct · days · reason
    """
    if spec.strategy == PORTFOLIO:
        rows: list[dict[str, Any]] = []
        for row in result.get("trades") or []:
            side = "매수" if row.get("side") == "buy" else "매도"
            rows.append(
                {
                    "ticker": row["ticker"],
                    "name": row.get("name") or row["ticker"],
                    "entry_date": row["date"],
                    "entry_price": row.get("price"),
                    "exit_date": row["date"],
                    "exit_price": row.get("price"),
                    "return_pct": None,
                    "days": None,
                    "reason": f"{row.get('reason') or '리밸런싱'} {side} "
                    f"({row.get('weight_before_pct')}% → {row.get('weight_after_pct')}%)",
                }
            )
        return rows

    rows = []
    for row in result.get("open_positions") or []:
        rows.append(
            {
                "ticker": row["ticker"],
                "name": row["name"],
                "entry_date": row["entry_date"],
                "entry_price": row["entry_price"],
                "exit_date": None,
                "exit_price": row.get("price"),
                "return_pct": row.get("return_pct"),
                "days": row.get("days"),
                "reason": "보유중",
            }
        )
    rows.extend(dict(row) for row in result.get("trades") or [])
    return rows


def daily_curve(spec: SleeveSpec, result: dict[str, Any]) -> dict[str, float]:
    """백테스트 결과의 일별을 **누적 배수(시작 1.0)** 로 통일한다.

    세 엔진 모두 구간 시작 대비 누적(%)을 주므로 그대로 배수로 바꾼다.
    """
    del spec
    rows = sorted(result.get("daily") or [], key=lambda row: row["date"])
    curve: dict[str, float] = {}
    for row in rows:
        if row.get("strategy_pct") is not None:
            curve[row["date"]] = 1 + row["strategy_pct"] / 100
    return curve
