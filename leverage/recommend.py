import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from leverage.constants import CONFIG_DIR, MARKET_SCHEDULES, STATE_DIR, ZRESULTS_DIR
from leverage.engine.backtest.runner import run_backtest
from leverage.engine.backtest.settings import load_settings
from leverage.notify import send_slack_recommendation


def get_market_status(market: str) -> str:
    """현재 시간 기준 장 상태를 반환합니다.

    인자 market 은 MARKET_SCHEDULES 의 키(kor/us)입니다.

    반환값:
        "OPEN"            - 장중
        "CLOSED_JUST_NOW" - 장 마감 후 75분 이내
        "PRE_OPEN"        - 당일 장 시작 전
        "CLOSED"          - 장 마감 후 75분 초과 (전날 마감 이후 ~ 당일 개장 전 아닌 경우 포함)
    """
    from datetime import timedelta

    schedule = MARKET_SCHEDULES.get(market)
    if not schedule:
        return "OPEN"

    tz = ZoneInfo(schedule["timezone"])
    now = datetime.now(tz)

    # 주말 체크 (월=0, ..., 일=6)
    if now.weekday() >= 5:
        return "CLOSED"

    current_time = now.time()
    open_time = schedule["open"]
    close_time = schedule["close"]

    if open_time <= current_time <= close_time:
        return "OPEN"

    # 장 마감 후 75분 이내
    close_dt = datetime.combine(now.date(), close_time, tzinfo=tz)
    time_since_close = now - close_dt
    if timedelta(0) <= time_since_close <= timedelta(minutes=75):
        return "CLOSED_JUST_NOW"

    # 당일 개장 전
    open_dt = datetime.combine(now.date(), open_time, tzinfo=tz)
    if now < open_dt:
        return "PRE_OPEN"

    return "CLOSED"


MARKET_PHASE_LABEL = {
    "OPEN": "장중",
    "CLOSED_JUST_NOW": "장 마감 직후",
    "PRE_OPEN": "장전",
    "CLOSED": "장 마감 후",
}


def _market_label(market: str) -> str:
    return "🇺🇸 미국" if market == "us" else "🇰🇷 한국"


def load_previous_state(profile: str) -> dict:
    """저장된 이전 추천 상태를 로드합니다."""
    state_path = STATE_DIR / f"last_recommendation_{profile}.json"
    if not state_path.exists():
        return {}
    try:
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_current_state(profile: str, state: dict) -> None:
    """현재 추천 상태를 저장합니다."""
    state_dir = STATE_DIR
    state_dir.mkdir(exist_ok=True)
    state_path = state_dir / f"last_recommendation_{profile}.json"
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)


def _format_metric_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:+.2f}%"


def _format_asset_price(value: float | None, prefix: str, suffix: str, fmt: str) -> str:
    if value is None:
        return "-"
    return f"{prefix}{format(value, fmt)}{suffix}"


def _format_display_name(ticker: str, name: str | None) -> str:
    if name and name != ticker:
        return f"{name}({ticker})"
    return ticker


def _build_ticker_names(settings: dict, prev_state: dict, display_target: str | None) -> dict[str, str]:
    ticker_names = {
        settings["offense_ticker"]: settings.get("offense_name", settings["offense_ticker"]),
        settings["defense_ticker"]: settings.get("defense_name", settings["defense_ticker"]),
        settings["signal_ticker"]: settings.get("signal_name", settings["signal_ticker"]),
    }

    for entry in settings.get("benchmarks", []):
        if isinstance(entry, dict):
            ticker = entry.get("ticker")
            name = entry.get("name")
            if ticker and name:
                ticker_names[ticker] = name

    if display_target and display_target not in ticker_names:
        ticker_names[display_target] = prev_state.get("target_name", display_target)

    return ticker_names


def main() -> None:
    parser = argparse.ArgumentParser(description="추천 실행 엔트리 포인트")
    parser.add_argument("profile", nargs="?", default="switch", help="전략 프로파일 (switch)")
    parser.add_argument("--slack", action="store_true", help="결과를 Slack으로 전송")
    args = parser.parse_args()

    profile = args.profile
    config_path = CONFIG_DIR / f"{profile}.json"
    if not config_path.exists():
        print(f"설정 파일을 찾을 수 없습니다: {config_path}")
        return

    settings = load_settings(config_path)
    market = settings.get("market", "kor")

    schedule = MARKET_SCHEDULES.get(market, {})
    tz_name = schedule.get("timezone", "UTC")
    now_local = datetime.now(ZoneInfo(tz_name)).strftime("%Y-%m-%d %H:%M %Z")
    status = get_market_status(market)
    market_phase = MARKET_PHASE_LABEL.get(status, "장 마감 후")
    print(f"[{profile}] 실행 시작 (현지시각: {now_local}, status: {status} [{market_phase}], slack={args.slack})")

    try:
        _recommend_switch(profile, settings, market, status, market_phase, args)
    except Exception as exc:
        if "YFRateLimitError" in repr(exc) or "rate limit" in repr(exc).lower():
            print("YFRateLimitError: 요청이 너무 많습니다. 잠시 후 다시 실행하세요.")
            return
        raise


def _recommend_switch(profile: str, settings: dict, market: str, status: str, market_phase: str, args) -> None:
    is_warning = status == "OPEN"

    # 장중에는 오늘의 미완성 봉을 제외하고 "마지막으로 닫힌 거래일 종가"로 신호를 확정한다.
    # (설계 원칙: 당일 종가로 확정 → 익일 시초가 실행. 장중에는 포지션을 바꾸지 않는다.)
    result = run_backtest(settings, drop_today=is_warning)

    # 마지막(확정) 거래일 추천 정보 추출
    last_target = result["last_target"]
    rec_data = result["recommendation_data"]

    # 장중(is_warning)일 때 오늘 아침에 이미 실행된 매매와 보유를 반영하기 위해 날짜와 보유일 보정
    display_holding_days = result.get("holding_days", 0)
    if is_warning:
        tz_name = "Asia/Seoul" if market == "kor" else "America/New_York"
        end_date = datetime.now(ZoneInfo(tz_name)).date().isoformat()
        display_holding_days += 1
    else:
        end_date = rec_data["last_date"]

    # 확정된 보유 종목 = 마지막으로 닫힌 거래일의 신호 (장중에도 뒤집지 않음)
    display_target = last_target

    # 이전 상태 로드 및 변경 여부 확인 (확정은 장 마감 후에만 의미가 있음)
    prev_state = load_previous_state(profile)
    prev_target = prev_state.get("target")
    is_changed = (not is_warning) and (prev_target is not None) and (prev_target != last_target)

    # 상태 저장: 장중이 아닐 때(종가 확정)만 저장
    if status != "OPEN":
        current_state = {
            "date": end_date,
            "target": last_target,
            "target_name": settings.get(
                "offense_name" if last_target == settings["offense_ticker"] else "defense_name",
                last_target,
            ),
            "updated_at": datetime.now().isoformat(),
        }
        save_current_state(profile, current_state)

    offense_ticker = settings["offense_ticker"]
    offense_name = settings.get("offense_name", offense_ticker)
    defense_ticker = settings["defense_ticker"]
    defense_name = settings.get("defense_name", defense_ticker)

    last_prices = rec_data["last_prices"]
    daily_returns = rec_data.get("daily_returns", {})
    cum_returns = rec_data.get("cum_returns", {})
    holding_start_prices = rec_data.get("holding_start_prices", {})
    buy_cutoff = rec_data["buy_cutoff"]
    sell_cutoff = rec_data["sell_cutoff"]

    # 장중(C안): 보유 종목·보유일·매매 실행은 어제 종가로 확정 유지하지만,
    # 그 외 표시값(현재가·일간·누적·드로다운·회복/하락 필요·설명)은 오늘 실시간으로 매시간 갱신한다.
    live_prices = result.get("live_prices", {}) if is_warning else {}
    if is_warning and "live_drawdown" in rec_data:
        current_dd = rec_data["live_drawdown"]
        buy_cut_frac = -buy_cutoff / 100
        needed_recovery = (buy_cut_frac - current_dd) * 100 if current_dd < buy_cut_frac else 0.0
    else:
        current_dd = rec_data["current_drawdown"]
        needed_recovery = rec_data["needed_recovery"]

    if market == "kor":
        currency_prefix = ""
        currency_suffix = "원"
        price_fmt = ",.0f"
    else:
        currency_prefix = "$"
        currency_suffix = ""
        price_fmt = ",.2f"

    ticker_names = _build_ticker_names(settings, prev_state, display_target)

    table_lines = []
    assets = []
    if display_target and display_target not in (offense_ticker, defense_ticker):
        assets.append(display_target)
    for sym in [offense_ticker, defense_ticker]:
        if sym not in assets:
            assets.append(sym)

    for sym in assets:
        name = ticker_names.get(sym, sym)
        display_name = _format_display_name(sym, name)

        has_market_data = sym in last_prices
        price = last_prices.get(sym)
        day_ret = daily_returns.get(sym) if has_market_data else None
        c_ret = cum_returns.get(sym) if has_market_data else None

        # 장중: 현재가/일간/누적을 오늘 실시간값으로 표시 (보유 종목·보유일은 확정 유지)
        is_live_price = sym in live_prices and last_prices.get(sym)
        if is_live_price:
            confirmed_close = last_prices[sym]
            price = live_prices[sym]
            day_ret = live_prices[sym] / confirmed_close - 1
            # 보유 종목의 누적 수익률도 실시간(오늘가 / 보유 시작가)으로 갱신
            if sym == display_target and holding_start_prices.get(sym):
                c_ret = live_prices[sym] / holding_start_prices[sym] - 1

        sell_cutoff_val = -sell_cutoff / 100
        needed_drop = (current_dd - sell_cutoff_val) * 100 if current_dd > sell_cutoff_val else 0

        if sym == display_target:
            status_text = "BUY"
            status_emoji = "✅️"
        else:
            status_text = "WAIT"
            status_emoji = "⏳️"

        signal_name = settings.get("signal", {}).get("name", "신호")
        note = ""
        if sym == offense_ticker:
            if display_target == offense_ticker:
                # 공격 자산 보유 중 → 매도 기준까지 남은 하락폭 (실시간)
                if needed_drop > 0:
                    note = f"{signal_name}가 {needed_drop:.2f}% 더 하락 시 매도"
                else:
                    note = f"{signal_name} 매도 기준 도달(실시간) → 장 마감 종가 확정 시 방어 전환 예정"
            else:
                # 방어 자산 보유 중 → 매수 기준까지 남은 회복폭 (실시간)
                if needed_recovery > 0:
                    note = f"{signal_name}가 {needed_recovery:+.2f}% 더 회복 시 매수"
                else:
                    note = f"{signal_name} 매수 기준 도달(실시간) → 장 마감 종가 확정 시 공격 전환 예정"
        else:
            note = ""

        table_lines.append(f"{status_emoji} {display_name}")
        table_lines.append(f"  상태: {status_text}")
        table_lines.append(f"  일간: {_format_metric_pct(day_ret)}")

        cum_text = f"  누적: {_format_metric_pct(c_ret)}"
        if sym == display_target:
            h_days = result.get("holding_days", 0)
            if is_warning:
                h_days += 1
            if h_days > 0:
                cum_text += f"({h_days}거래일째 보유중)"
        else:
            cum_text += "(미보유)"
        table_lines.append(cum_text)

        price_str = _format_asset_price(price, currency_prefix, currency_suffix, price_fmt)
        if is_live_price:
            price_str += " (장중 실시간)"
        table_lines.append(f"  현재가: {price_str}")
        if note:
            table_lines.append(f"  비고: {note}")
        table_lines.append("")

    target_name = ticker_names.get(display_target, display_target)
    target_display = _format_display_name(display_target, target_name)

    warning_target_display = None

    out_dir = ZRESULTS_DIR / profile
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"recommend_{datetime.now().date()}.log"

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"추천 로그 생성: {datetime.now().isoformat()}\n")
        f.write(f"프로파일: {profile} | 시장: {market}\n\n")
        f.write("=== 추천 목록 ===\n")
        for line in table_lines:
            f.write(line + "\n")
        f.write("\n")
        f.write(f"[INFO] 기준일: {end_date}\n")
        f.write(f"[INFO] 최종 타깃: {target_display}\n")
        f.write(f"[INFO] 적용 파라미터: {defense_ticker} / Buy {buy_cutoff}% / Sell {sell_cutoff}%\n")

    print(f"\n추천 결과 저장: {out_path}")

    if is_changed:
        print(f"⚠️ 포지션 변경 감지: {prev_target} -> {target_display}")
    else:
        print(f"ℹ️ 포지션 유지: {target_display}")

    market_name = _market_label(market)
    header_text = f"{market_name} 스위칭 {'포지션 변경 알림' if is_changed else '정기 보고'}"
    print("\n=== Slack 전송 요약 ===")
    print(f"{header_text} (기준일: {end_date})")
    print(f"🏆 최적 파라미터 (CAGR: {result.get('cagr', 0) * 100:.2f}%)")
    for line in table_lines:
        if line.strip():
            print(line.strip())
    print(f"🎯 최종 타깃: {target_display}")
    print("========================\n")

    if args.slack:
        tuning_meta = {
            "offense_ticker": offense_ticker,
            "offense_name": offense_name,
            "defense_ticker": defense_ticker,
            "defense_name": defense_name,
            "buy_cutoff": buy_cutoff,
            "sell_cutoff": sell_cutoff,
            "cagr": result.get("cagr", 0.0),
            "period_start": result.get("start"),
            "period_end": result.get("end"),
            "signal_name": settings.get("signal", {}).get("name", "신호 자산"),
            "current_drawdown": current_dd,
            "needed_recovery": needed_recovery,
        }
        send_slack_recommendation(
            country=market,
            as_of=end_date,
            target_display=target_display,
            table_lines=table_lines,
            tuning_meta=tuning_meta,
            is_changed=is_changed,
            holding_days=display_holding_days,
            is_warning=is_warning,
            warning_target_display=warning_target_display,
            market_phase=market_phase,
        )

if __name__ == "__main__":
    main()
