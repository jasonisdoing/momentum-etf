"""계정별 추천 실행 스크립트."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from logic.recommend import RecommendationReport, generate_recommendation_report
from logic.recommend.reporting import (
    dump_json,
    dump_recommendation_log,
    print_result_summary,
    print_run_header,
)
from utils.account_registry import (
    get_account_settings,
    get_strategy_rules,
    list_available_accounts,
)
from utils.data_loader import MissingPriceDataError
from utils.logger import get_app_logger
from utils.notification import (
    compose_recommendation_slack_message,
    send_recommendation_slack_notification,
)
from utils.recommendation_storage import save_recommendation_report

RESULTS_DIR = Path(__file__).resolve().parent / "zaccounts"


def _available_account_choices() -> list[str]:
    choices = list_available_accounts()
    if not choices:
        raise SystemExit("계정 설정(JSON)이 존재하지 않습니다. zsettings/account/*.json 파일을 확인하세요.")
    return choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MomentumEtf 계정 추천 실행기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("account", choices=_available_account_choices(), help="실행할 계정 ID")
    parser.add_argument(
        "--date",
        help="YYYY-MM-DD 형식의 기준일 (미지정 시 최신 거래일)",
    )
    parser.add_argument(
        "--output",
        help="결과 JSON 저장 경로",
    )
    return parser


def main() -> None:
    logger = get_app_logger()

    parser = build_parser()
    args = parser.parse_args()

    account_id = args.account.lower()

    try:
        account_settings = get_account_settings(account_id)
        get_strategy_rules(account_id)
    except Exception as exc:  # pragma: no cover - 잘못된 입력 방어 전용 처리
        parser.error(f"계정 설정을 로드하는 중 오류가 발생했습니다: {exc}")

    account_country = str((account_settings or {}).get("country_code", "") or "")

    print_run_header(account_id, date_str=args.date)
    start_time = time.time()

    try:
        report = generate_recommendation_report(account_id=account_id, date_str=args.date)
    except MissingPriceDataError as exc:
        logger.error(str(exc))
        raise SystemExit(1)

    if not isinstance(report, RecommendationReport):
        logger.error(
            "예상한 RecommendationReport 타입이 아닙니다 (account=%s, got=%s)",
            account_id,
            type(report).__name__,
        )
        return

    if not report.recommendations:
        logger.warning("%s에 대한 추천 결과가 비어 있습니다.", account_id.upper())
        return

    duration = time.time() - start_time
    items = list(report.recommendations)

    print_result_summary(items, account_id, args.date)

    try:
        meta = save_recommendation_report(report)
        logger.info(
            "✅ %s 추천 결과를 MongoDB에 저장했습니다. document_id=%s",
            account_id.upper(),
            meta.get("document_id") if isinstance(meta, dict) else meta,
        )
    except Exception:
        logger.error(
            "기본 추천 결과 저장에 실패했습니다 (account=%s)",
            account_id,
            exc_info=True,
        )
        meta = None

    if args.output:
        custom_path = Path(args.output)
        dump_json(items, custom_path)
        logger.info("📄 커스텀 JSON을 '%s'에 저장했습니다.", custom_path)

    # 로그 파일 저장
    try:
        log_path = dump_recommendation_log(report, results_dir=RESULTS_DIR)
        logger.info("✅ 추천 로그를 '%s'에 저장했습니다.", log_path)
    except Exception:
        logger.error("추천 로그 저장에 실패했습니다 (account=%s)", account_id, exc_info=True)

    slack_payload = compose_recommendation_slack_message(
        account_id,
        report,
        duration=duration,
    )

    target_country = (getattr(report, "country_code", "") or account_country or "").strip().lower()
    notified = send_recommendation_slack_notification(
        account_id,
        slack_payload,
    )

    base_date = getattr(report, "base_date", None)
    base_date_str = base_date.strftime("%Y-%m-%d") if hasattr(base_date, "strftime") else str(base_date)
    if notified:
        logger.info(
            "[%s/%s] Slack 알림 전송이 완료되었습니다 (소요 %.1fs)",
            (target_country or account_country or "").upper() or "UNKNOWN",
            base_date_str,
            duration,
        )
    else:
        logger.info(
            "[%s/%s] Slack 알림이 전송되지 않았습니다.",
            (target_country or account_country or "").upper() or "UNKNOWN",
            base_date_str,
        )

    elapsed_total = int(time.time() - start_time)
    hours, remainder = divmod(elapsed_total, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}시간")
    if minutes:
        parts.append(f"{minutes}분")
    if seconds or not parts:
        parts.append(f"{seconds}초")
    logger.info(
        "[%s] 총 소요 시간: %s",
        account_id.upper(),
        " ".join(parts),
    )


if __name__ == "__main__":
    main()
