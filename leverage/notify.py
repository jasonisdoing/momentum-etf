"""Slack 알림 전송 유틸리티.

전송은 momentum-etf 의 공용 Slack 전송기(utils.notification.send_slack_message_v2)를 사용한다.
(봇 토큰 + settings_loader.get_slack_channel 기준 채널)
"""

from datetime import datetime
from typing import Any

from utils.notification import send_slack_message_v2


def _format_display_name(ticker: str, name: str | None) -> str:
    if name and name != ticker:
        return f"{name}({ticker})"
    return ticker


def _post(text: str, blocks: list[dict], label: str) -> bool:
    """momentum-etf 공용 전송기로 blocks 메시지를 보낸다."""
    ts = send_slack_message_v2(text=text, blocks=blocks)
    if ts:
        print(f" [SLACK] {label} 전송 완료")
        return True
    print(f" [SLACK] {label} 전송 실패")
    return False


def send_slack_recommendation(
    country: str,
    as_of: str,
    target_display: str,
    table_lines: list[str],
    tuning_meta: dict[str, Any] | None = None,
    is_changed: bool = False,
    holding_days: int = 0,
    is_warning: bool = False,
    warning_target_display: str | None = None,
    market_phase: str = "장 마감 후",
) -> bool:
    """나스닥 스위칭 추천 결과를 Slack으로 전송합니다."""
    market_name = "🇺🇸 미국" if country.lower() == "us" else "🇰🇷 한국"
    phase_tag = f"[{market_phase}]"

    # 이모지 및 타이틀 분기
    if is_warning:
        # 장중
        if is_changed:
            header_emoji = "⚠️"
            header_text = f"{market_name} {phase_tag} 포지션 변경 예상 (경고)"
        else:
            header_emoji = "📊"
            header_text = f"{market_name} {phase_tag} 정기 보고"
    else:
        # 장전 / 장 마감 직후 / 장 마감 후
        if is_changed:
            header_emoji = "🚨"
            header_text = f"{market_name} {phase_tag} 포지션 변경 확정! (다음 거래일 시초가 매매)"
        else:
            header_emoji = "✅"
            header_text = f"{market_name} {phase_tag} 정기 보고"

    # 메시지 블록 구성
    blocks = []

    # 1. 헤더
    blocks.append(
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{header_emoji} {header_text}",
                "emoji": True,
            },
        }
    )

    # 2. 최적 파라미터 정보 (최근 튜닝 결과)
    if tuning_meta:
        offense_ticker = tuning_meta.get("offense_ticker", "N/A")
        offense_name = tuning_meta.get("offense_name", "")
        offense_display = _format_display_name(offense_ticker, offense_name)

        defense_ticker = tuning_meta.get("defense_ticker", "N/A")
        defense_name = tuning_meta.get("defense_name", "")
        defense_display = _format_display_name(defense_ticker, defense_name)

        tuning_text = (
            f"*🏆 최적 파라미터 (CAGR 기준)*\n"
            f"• 공격 자산: {offense_display}\n"
            f"• 방어 자산: {defense_display}\n"
            f"• 매수 컷: {tuning_meta.get('buy_cutoff', 0):.1f}%\n"
            f"• 매도 컷: {tuning_meta.get('sell_cutoff', 0):.1f}%\n"
            f"• CAGR: {tuning_meta.get('cagr', 0) * 100:.2f}%"
        )

        period_start = tuning_meta.get("period_start")
        period_end = tuning_meta.get("period_end")
        if period_start and period_end:
            try:
                from datetime import date as _date

                _s = _date.fromisoformat(period_start)
                _e = _date.fromisoformat(period_end)
                _months = max(1, int(round((_e - _s).days / 30)))
                tuning_text += f"\n• 백테스트 기간: {period_start} ~ {period_end} ({_months} 개월)"
            except Exception:
                tuning_text += f"\n• 백테스트 기간: {period_start} ~ {period_end}"

        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": tuning_text}})
        blocks.append({"type": "divider"})

    # 3. 추천 목록 (상세 테이블)
    if table_lines:
        # 가독성을 위해 간결하게 변환
        clean_lines = []
        for line in table_lines:
            if line.strip().startswith("📌"):
                clean_lines.append(f"*{line.strip()}*")
            elif line.strip():
                clean_lines.append(f"  {line.strip()}")

        table_text = "\n".join(clean_lines)
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*=== 추천 목록 ===*\n{table_text}",
                },
            }
        )
        blocks.append({"type": "divider"})

    # 4. 요약 정보
    summary_text = f"ℹ️ *기준일*: {as_of}"
    holding_days_text = f"\n⏳ *보유일*: *{holding_days}거래일째*"

    if is_warning and warning_target_display:
        # 경고 모드: 현재 보유 + 전환 가능성 안내
        summary_text += f"\n💼 *현재 보유*: *{target_display}*"
        summary_text += holding_days_text
        summary_text += (
            f"\n\n*⚠️ 장중 경고*: 이대로 장 마감 시 "
            f"*{warning_target_display}*(으)로 전환될 수 있습니다. "
            "장 마감 후 최종 확정 알림을 기다려주세요."
        )
    elif is_warning:
        # 경고 모드이지만 변경 없음
        summary_text += f"\n🎯 *현재 보유*: *{target_display}*"
        summary_text += holding_days_text
        summary_text += "\n\n*ℹ️ 안내*: 장 마감 시까지 변동될 수 있습니다. 장 마감 후 최종 확정 알림을 기다려주세요."
    elif is_changed:
        # 확정 모드에서 변경됨
        summary_text += f"\n🎯 *최종 타깃*: *{target_display}*"
        summary_text += holding_days_text
        summary_text += (
            "\n\n*🔔 실행 안내*: 오늘 종가 기준으로 시그널이 확정되었습니다. "
            "내일(다음 거래일) 아침 시초가에 해당 종목을 매매하세요."
        )
    else:
        # 확정 모드에서 변경 없음
        summary_text += f"\n🎯 *최종 타깃*: *{target_display}*"
        summary_text += holding_days_text

    # 방어 자산 보유 시, 공격 자산으로 전환하기 위해 필요한 시그널 회복률에 대한 설명 추가
    if tuning_meta and "defense_ticker" in tuning_meta:
        defense_ticker = tuning_meta.get("defense_ticker")
        defense_name = tuning_meta.get("defense_name", "")
        defense_display = _format_display_name(defense_ticker, defense_name)

        n_rec = tuning_meta.get("needed_recovery", 0.0)
        # 방어 자산 보유 중일 때 설명 표시.
        # 장중에는 실시간 드로다운으로 매시간 갱신되며, 매수 기준에 도달(n_rec<=0)해도 안내한다.
        if target_display == defense_display and (n_rec > 0 or is_warning):
            sig_name = tuning_meta.get("signal_name", "신호 자산")
            offense_ticker = tuning_meta.get("offense_ticker", "")
            offense_name = tuning_meta.get("offense_name", "")
            off_display = _format_display_name(offense_ticker, offense_name)
            curr_dd = tuning_meta.get("current_drawdown", 0.0) * 100
            b_cut = tuning_meta.get("buy_cutoff", 0.0)
            live_tag = " (장중 실시간)" if is_warning else ""

            if n_rec > 0:
                explanation = (
                    f"\n\n💡 *설명{live_tag}*: 현재 {sig_name}가 최고점 대비 {curr_dd:+.2f}% 하락해 있는 상태이므로, "
                    f"공격 자산({off_display})으로 전환하기 위한 매수 기준인 *{-b_cut:+.1f}%*에 도달하기 위해선 "
                    f"지수가 *{n_rec:+.2f}%만큼 추가 회복(상승)*해야 합니다."
                )
            else:
                explanation = (
                    f"\n\n💡 *설명{live_tag}*: 현재 {sig_name}가 최고점 대비 {curr_dd:+.2f}% 로 "
                    f"매수 기준 *{-b_cut:+.1f}%* 에 도달했습니다. "
                    f"장 마감 종가로 확정되면 공격 자산({off_display})으로 전환됩니다."
                )
            summary_text += explanation

    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": summary_text}})

    # 5. 채널 맨션 (확정 알림에서 변경이 있을 때만, 경고 모드에서는 멘션 안 함)
    if is_changed and not is_warning:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "<!channel> 포지션이 변경되었습니다! 확인해주세요.",
                },
            }
        )

    return _post(f"[{market_name}] {header_text} ({as_of})", blocks, "스위칭 추천")


def send_slack_tuning_result(
    country: str,
    started_at: datetime,
    ended_at: datetime,
    elapsed: str,
    best_result: dict[str, Any],
    table_lines: list[str],
    meta: dict[str, Any] | None = None,
    log_path: str | None = None,
) -> bool:
    """튜닝 완료 결과를 Slack으로 전송합니다."""

    market_name = "🇺🇸 미국" if country.lower() == "us" else "🇰🇷 한국"
    params = best_result.get("params", {})
    offense_display = _format_display_name(
        str(params.get("offense_ticker", "N/A")),
        params.get("offense_name"),
    )
    defense_display = _format_display_name(
        str(params.get("defense_ticker", "N/A")),
        params.get("defense_name"),
    )

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🏆 {market_name} 튜닝 완료",
                "emoji": True,
            },
        }
    ]

    period_start = meta.get("period_start") if meta else None
    period_end = meta.get("period_end") if meta else None
    period_text = f"\n• 백테스트 기간: {period_start} ~ {period_end}" if period_start and period_end else ""
    log_text = f"\n• 로그: `{log_path}`" if log_path else ""

    summary_text = (
        f"*최적 파라미터 (CAGR 기준)*\n"
        f"• 공격 자산: {offense_display}\n"
        f"• 방어 자산: {defense_display}\n"
        f"• 매수 컷: {float(params.get('drawdown_buy_cutoff', 0)):.1f}%\n"
        f"• 매도 컷: {float(params.get('drawdown_sell_cutoff', 0)):.1f}%\n"
        f"• 기간 수익률: {best_result.get('period_return', 0.0) * 100:.2f}%\n"
        f"• CAGR: {best_result.get('cagr', 0.0) * 100:.2f}%\n"
        f"• MDD: {best_result.get('mdd', 0.0) * 100:.2f}%\n"
        f"• Sharpe: {best_result.get('sharpe', 0.0):.2f}\n"
        f"• 시작: {started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"• 종료: {ended_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"• 소요: {elapsed}"
        f"{period_text}"
        f"{log_text}"
    )
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": summary_text}})

    if table_lines:
        top_text = "\n".join(table_lines[:12])
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*상위 튜닝 결과*\n```{top_text}```"},
            }
        )

    return _post(f"[{market_name}] 튜닝 완료 ({ended_at.date().isoformat()})", blocks, "튜닝 결과")


def send_slack_buy_recommendation(
    market: str,
    as_of: str,
    target_display: str,
    recommendation: dict,
    table_lines: list[str],
    strategy_meta: dict[str, Any] | None = None,
    is_changed: bool = False,
    is_warning: bool = False,
    market_phase: str = "장 마감 후",
) -> bool:
    """무한매수법(buy) 추천 결과를 Slack으로 전송합니다.

    기존 스위칭 추천 알림(send_slack_recommendation)과 동일한 스타일을 따른다.
    """

    market_name = "🇺🇸 미국" if market.lower() == "us" else "🇰🇷 한국"
    phase_tag = f"[{market_phase}]"

    # 헤더 이모지/문구 분기 (스위칭 알림과 동일한 규칙)
    if is_warning:
        # 장중
        if is_changed:
            header_emoji = "⚠️"
            header_text = f"{market_name} 무한매수법 {phase_tag} 행동 변경 예상 (경고)"
        else:
            header_emoji = "📊"
            header_text = f"{market_name} 무한매수법 {phase_tag} 정기 보고"
    else:
        # 장전 / 장 마감 직후 / 장 마감 후
        if is_changed:
            header_emoji = "🚨"
            header_text = f"{market_name} 무한매수법 {phase_tag} 행동 변경 확정! (다음 거래일 시초가 매매)"
        else:
            header_emoji = "✅"
            header_text = f"{market_name} 무한매수법 {phase_tag} 정기 보고"

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{header_emoji} {header_text}", "emoji": True},
        }
    ]

    # 전략 정보 블록 (스위칭의 '최적 파라미터' 블록에 대응)
    if strategy_meta:
        meta_text = (
            f"*🏆 전략 파라미터*\n"
            f"• 대상: {target_display}\n"
            f"• 분할 수: {strategy_meta.get('divisions', 'N/A')}\n"
            f"• 익절률: {float(strategy_meta.get('take_profit_pct', 0)):.1f}%\n"
            f"• CAGR: {strategy_meta.get('cagr', 0) * 100:.2f}% | "
            f"MDD: {strategy_meta.get('mdd', 0) * 100:.2f}% | "
            f"익절 {strategy_meta.get('cycles', 0)}회"
        )
        period_start = strategy_meta.get("period_start")
        period_end = strategy_meta.get("period_end")
        if period_start and period_end:
            meta_text += f"\n• 백테스트 기간: {period_start} ~ {period_end}"
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": meta_text}})
        blocks.append({"type": "divider"})

    # 추천 목록 (상세)
    if table_lines:
        clean_lines = []
        for line in table_lines:
            if line.strip().startswith("🎯"):
                clean_lines.append(f"*{line.strip()}*")
            elif line.strip():
                clean_lines.append(f"  {line.strip()}")
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*=== 추천 목록 ===*\n" + "\n".join(clean_lines)},
            }
        )
        blocks.append({"type": "divider"})

    # 요약 정보 (오늘의 행동)
    rec = recommendation or {}
    action = rec.get("action", "HOLD")
    message = rec.get("message", "")
    summary_text = f"ℹ️ *기준일*: {as_of}\n🛒 *오늘 행동*: *[{action}] {message}*"
    if is_warning:
        summary_text += (
            "\n\n*ℹ️ 안내*: 장중에는 종가 확정 전이라 행동이 바뀔 수 있습니다. 장 마감 후 최종 확정 알림을 기다려주세요."
        )
    elif is_changed:
        summary_text += (
            "\n\n*🔔 실행 안내*: 오늘 종가 기준으로 행동이 확정되었습니다. 다음 거래일 아침 시초가에 실행하세요."
        )
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": summary_text}})

    # 채널 맨션 (확정 모드에서 변경이 있을 때만)
    if is_changed and not is_warning:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "<!channel> 오늘 행동이 변경되었습니다! 확인해주세요."},
            }
        )

    return _post(f"[{market_name}] {header_text} ({as_of})", blocks, "무한매수법 추천")


def send_slack_buy_tuning_result(
    market: str,
    started_at: datetime,
    ended_at: datetime,
    elapsed: str,
    best_result: dict,
    table_lines: list[str],
    target_name: str,
    meta: dict | None = None,
    log_path: str | None = None,
) -> bool:
    """무한매수법(buy) 튜닝 결과를 Slack으로 전송합니다."""

    market_name = "🇺🇸 미국" if market.lower() == "us" else "🇰🇷 한국"
    params = best_result.get("params", {})
    period_text = ""
    if meta and meta.get("period_start") and meta.get("period_end"):
        period_text = (
            f"\n• 백테스트 기간: {meta['period_start']} ~ {meta['period_end']} ({meta.get('period_days', '?')} 거래일)"
        )
    log_text = f"\n• 로그: `{log_path}`" if log_path else ""

    summary_text = (
        f"*최적 파라미터 (CAGR 기준)*\n"
        f"• 대상: {target_name}\n"
        f"• 분할 수: {params.get('divisions', 'N/A')}\n"
        f"• 익절률: {float(params.get('take_profit_pct', 0)):.1f}%\n"
        f"• 기간 수익률: {best_result.get('period_return', 0.0) * 100:.2f}%\n"
        f"• CAGR: {best_result.get('cagr', 0.0) * 100:.2f}%\n"
        f"• MDD: {best_result.get('mdd', 0.0) * 100:.2f}%\n"
        f"• Sharpe: {best_result.get('sharpe', 0.0):.2f}\n"
        f"• 익절 횟수: {best_result.get('cycles', 0)}회\n"
        f"• 시작: {started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"• 종료: {ended_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"• 소요: {elapsed}"
        f"{period_text}"
        f"{log_text}"
    )

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"🏆 {market_name} 무한매수법 튜닝 완료", "emoji": True},
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": summary_text}},
    ]
    if table_lines:
        top_text = "\n".join(table_lines[:12])
        blocks.append({"type": "divider"})
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*상위 튜닝 결과*\n```{top_text}```"}})

    return _post(
        f"[{market_name}] 무한매수법 튜닝 완료 ({ended_at.date().isoformat()})", blocks, "무한매수법 튜닝 결과"
    )
