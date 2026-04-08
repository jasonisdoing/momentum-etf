from __future__ import annotations

import argparse
from collections.abc import Iterable

import pandas as pd

from services.etf_holdings_service import login_krx_session, sync_korean_etf_holdings_cache
from utils.data_loader import get_trading_days
from utils.stock_list_io import get_etfs
from utils.ticker_registry import load_ticker_type_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="한국 ETF 구성종목 캐시를 MongoDB에 저장합니다.")
    parser.add_argument(
        "--date",
        default=None,
        help="조회 기준일 YYYYMMDD. 미지정 시 최근 한국 거래일을 사용합니다.",
    )
    return parser.parse_args()


def resolve_target_date(value: str | None) -> str:
    normalized = str(value or "").strip()
    if normalized:
        return normalized

    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    trading_days = get_trading_days(
        (today - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d"),
        "kor",
    )
    if not trading_days:
        raise RuntimeError("최근 한국 거래일을 계산할 수 없습니다.")
    return pd.Timestamp(trading_days[-1]).strftime("%Y%m%d")


def iter_korean_etfs() -> Iterable[dict[str, str]]:
    seen: set[str] = set()
    for config in load_ticker_type_configs():
        if str(config.get("country_code") or "").strip().lower() != "kor":
            continue
        ticker_type = str(config["ticker_type"]).strip().lower()
        for item in get_etfs(ticker_type):
            ticker = str(item.get("ticker") or "").strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            yield {
                "ticker": ticker,
                "name": str(item.get("name") or "").strip(),
            }


def main() -> int:
    args = parse_args()
    target_date = resolve_target_date(args.date)
    print(f"target_date={target_date}")

    # KRX 로그인 세션을 먼저 주입해 이후 pykrx 호출이 동일 세션을 사용하게 한다.
    login_krx_session()
    print("KRX 로그인 세션 설정 완료")

    etfs = list(iter_korean_etfs())
    if not etfs:
        raise RuntimeError("한국 ETF 목록이 비어 있습니다.")

    success_count = 0
    failures: list[tuple[str, str]] = []
    for index, etf in enumerate(etfs, start=1):
        ticker = etf["ticker"]
        name = etf["name"]
        print(f"[{index}/{len(etfs)}] {name}({ticker}) 처리 중")
        try:
            document = sync_korean_etf_holdings_cache(
                ticker=ticker,
                etf_name=name,
                as_of_date=target_date,
            )
        except Exception as exc:
            failures.append((ticker, str(exc)))
            print(f"  실패: {exc}")
            continue
        success_count += 1
        print(f"  저장 완료: count={document['holdings_count']}")

    print("-" * 80)
    print(f"success={success_count}")
    print(f"failed={len(failures)}")
    if failures:
        for ticker, message in failures:
            print(f"{ticker}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
