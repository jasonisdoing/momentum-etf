"""Pool RANK ranking entrypoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from core.rank import RankConfig, run_pool_ranking, save_rank_result
from utils.data_loader import MissingPriceDataError
from utils.env import load_env_if_present
from utils.identifier_guard import ensure_account_pool_id_separation
from utils.logger import get_app_logger
from utils.pool_rank_storage import save_pool_rank_payload
from utils.pool_registry import get_pool_dir, list_available_pools

RESULTS_DIR = Path(__file__).resolve().parent / "zpools"
logger = get_app_logger()


def _load_pool_rank_config(pool_id: str) -> RankConfig:
    pool_dir = get_pool_dir(pool_id)
    config_path = pool_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"종목풀 설정 파일이 없습니다: {config_path}")

    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"종목풀 설정 파일을 읽을 수 없습니다: {config_path} ({exc})") from exc

    rank_data = config_data.get("rank")
    if not isinstance(rank_data, dict):
        raise SystemExit(f"'{config_path}'에 'rank' 설정 객체가 필요합니다.")

    try:
        country = str(rank_data["country"]).strip().lower()
        months = int(rank_data["months"])
        ma_type = str(rank_data["ma_type"]).strip().upper()
    except KeyError as exc:
        raise SystemExit(f"'{config_path}' rank 설정 필수 키 누락: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"'{config_path}' rank 설정 타입 오류: {exc}") from exc

    return RankConfig(
        country=country,
        months=months,
        ma_type=ma_type,
    )


def _available_pool_choices() -> list[str]:
    choices = list_available_pools()
    if not choices:
        raise SystemExit("사용 가능한 종목풀 디렉토리가 없습니다. `zpools/<order>_<pool_id>`를 확인하세요.")
    return choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MomentumEtf 풀 랭킹 실행기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("pool", choices=_available_pool_choices(), help="실행할 종목군 ID")
    return parser


def print_run_header(pool_id: str, *, config: RankConfig) -> None:
    banner = f"=== {pool_id.upper()} 랭킹 생성 ==="
    logger.info("%s", banner)
    logger.info(
        "설정: country=%s, ma=%s(%sm)",
        config.country,
        config.ma_type.upper(),
        config.months,
    )


def print_result_summary(result) -> None:
    from collections import Counter

    items = list(result.rows or [])
    if not items:
        logger.warning("%s에 대한 랭킹 결과가 비어 있습니다.", result.pool_id.upper())
        return

    base_date = result.base_date.strftime("%Y-%m-%d")
    bucket_counts = Counter(int(item.get("bucket", 1) or 1) for item in items)
    bucket_summary = ", ".join(f"bucket{bucket}: {count}" for bucket, count in sorted(bucket_counts.items()))

    logger.info("=== %s 랭킹 요약 (기준일: %s) ===", result.pool_id.upper(), base_date)
    if bucket_summary:
        logger.info("버킷 분포: %s", bucket_summary)
    logger.info("결과가 성공적으로 생성되었습니다. (총 %d개 항목)", len(items))


def main() -> int:
    load_env_if_present()
    ensure_account_pool_id_separation()
    args = build_parser().parse_args()
    started = time.time()

    pool_id = args.pool.strip().lower()
    config = _load_pool_rank_config(pool_id)

    print_run_header(pool_id, config=config)

    try:
        result = run_pool_ranking(pool_id, config)
    except MissingPriceDataError as exc:
        logger.error(
            "[%s] 가격 캐시 누락: %d개 종목 (%s ~ %s)",
            pool_id.upper(),
            len(exc.tickers),
            exc.start_date,
            exc.end_date,
        )
        preview = ", ".join(exc.tickers[:10])
        if preview:
            suffix = " ..." if len(exc.tickers) > 10 else ""
            logger.error("누락 티커: %s%s", preview, suffix)
        logger.error("먼저 가격 캐시를 업데이트한 뒤 다시 실행하세요.")
        logger.error("예시: python scripts/update_price_cache.py %s", pool_id)
        return 2
    print_result_summary(result)

    outputs = save_rank_result(result)
    mongo_meta = None
    try:
        mongo_meta = save_pool_rank_payload(
            pool_id=result.pool_id,
            country_code=result.country,
            rows=result.rows,
            base_date=result.base_date,
            config={
                "country": result.country,
                "months": result.months,
                "ma_type": result.ma_type,
            },
        )
        logger.info(
            "✅ %s 랭킹 결과를 MongoDB에 저장했습니다. document_id=%s",
            result.pool_id.upper(),
            mongo_meta.get("document_id") if isinstance(mongo_meta, dict) else None,
        )
    except Exception as exc:
        logger.warning("MongoDB 저장 실패 (pool=%s): %s", result.pool_id, exc)

    if result.missing_tickers:
        logger.warning("가격 누락 %d개", len(result.missing_tickers))

    logger.info("✅ 랭킹 로그를 '%s'에 저장했습니다.", outputs.log_path)
    logger.info("[%s] 랭킹 생성 완료 (소요 %.1fs)", result.pool_id.upper(), time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
