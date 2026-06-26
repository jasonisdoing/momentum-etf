"""튜닝 실행 엔트리 포인트 (스위칭 전략의 파라미터를 튜닝).

시장(market)은 config 파일의 'market' 필드로 결정된다.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from leverage.config_store import load_leverage_config_raw, save_leverage_config_raw
from leverage.constants import CONFIG_DIR, ZRESULTS_DIR
from leverage.engine.backtest.settings import normalize_settings
from leverage.engine.tune.runner import render_top_table, run_tuning
from leverage.notify import send_slack_tuning_result
from leverage.recommend import get_market_status

# 전략 프로파일별 튜닝 탐색 공간
TUNING_CONFIG: dict[str, dict] = {
    # 스위칭 전략: 매수/매도 컷오프 + 공격 자산 후보 + 방어 자산 후보
    "switch": {
        "drawdown_buy_cutoff": np.arange(1, 6, 1),
        "drawdown_sell_cutoff": np.arange(1, 6, 1),
        "offense": [
            {"ticker": "122630", "name": "KODEX 레버리지"},
            {"ticker": "243880", "name": "TIGER 200IT레버리지"},
            # {"ticker": "0193W0", "name": "KODEX 삼성전자단일종목레버리지"},
            # {"ticker": "0193T0", "name": "KODEX SK하이닉스단일종목레버리지"},
        ],
        "defense": [
            {"ticker": "CASH", "name": "현금"},
            {"ticker": "237350", "name": "KODEX 코스피100"},
            {"ticker": "161510", "name": "PLUS 고배당주"},
            {"ticker": "091170", "name": "KODEX 은행"},
            {"ticker": "279530", "name": "KODEX 고배당주"},
            {"ticker": "484880", "name": "SOL 금융지주플러스고배당"},
            {"ticker": "140700", "name": "KODEX 보험"},
            {"ticker": "498860", "name": "RISE 코리아금융고배당"},
            {"ticker": "466940", "name": "TIGER 은행고배당플러스TOP10"},
        ],
    },
}


def format_seconds(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h}시간 {m}분 {s}초"


def main() -> None:
    parser = argparse.ArgumentParser(description="튜닝 실행 엔트리 포인트")
    parser.add_argument("profile", nargs="?", default="switch", help="전략 프로파일 (switch)")
    parser.add_argument("--auto", action="store_true", help="자동 실행 모드 (장 운영 시간 체크 수행)")
    parser.add_argument("--slack", action="store_true", help="튜닝 결과를 Slack으로 전송")
    args = parser.parse_args()

    profile = args.profile

    if profile not in TUNING_CONFIG:
        print(f"지원하지 않는 전략 프로파일입니다: {profile}")
        print(f"지원 프로파일: {list(TUNING_CONFIG.keys())}")
        return

    try:
        raw_config = load_leverage_config_raw(profile)
    except Exception as exc:
        print(f"설정을 불러올 수 없습니다: {exc}")
        return
    # 엔진(run_tuning)이 설정 파일을 읽으므로 DB 설정을 파일로 미러링한다(파일은 임시 작업본).
    config_path = CONFIG_DIR / f"{profile}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(raw_config, f, ensure_ascii=False, indent=4)

    settings = normalize_settings(dict(raw_config))
    market = settings.get("market", "kor")

    # 자동 실행 모드일 때만 장 운영 시간 체크
    if args.auto and get_market_status(market) == "CLOSED":
        print(f"[{profile}] 장 운영 시간이 아닙니다. 튜닝을 건너뜁니다.")
        return

    _tune_switch(profile, config_path, settings, market, args)


def _tune_switch(profile: str, config_path: Path, settings: dict, market: str, args) -> None:
    """스위칭 전략 튜닝 (전수 조사 + config 갱신)."""
    tuning_config = TUNING_CONFIG[profile]
    start_ts = datetime.now()

    out_dir = ZRESULTS_DIR / profile
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tune_{start_ts.date()}.log"

    if "months_range" in settings:
        months_range = settings["months_range"]
    else:
        start_dt = datetime.strptime(settings["start_date"], "%Y-%m-%d")
        months_range = int((datetime.now() - start_dt).days / 30)

    def write_partial(results: list[dict], completed: int, total: int) -> None:
        results.sort(key=lambda x: x["cagr"], reverse=True)
        table_lines = render_top_table(results, top_n=10, months_range=months_range)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"실행 시각: {start_ts.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"프로파일: {profile} | 시장: {market}\n")
            f.write(f"진행률: {completed}/{total} ({completed / total * 100:.1f}%)\n\n")
            f.write("=== 중간 결과 - 상위 10개 ===\n")
            for line in table_lines:
                f.write(line + "\n")

    def progress_cb(completed: int, total: int) -> None:
        pct = int(completed / total * 100)
        print(f"[튜닝 진행률] {pct}% ({completed}/{total})")

    print(f"[튜닝 시작] {start_ts.strftime('%Y-%m-%d %H:%M:%S')} ({profile})")
    total_cases = 1
    for arr in tuning_config.values():
        total_cases *= len(arr)
    print(f"[튜닝 설정] 총 조합: {total_cases}개, 워커: auto (CPU)")

    try:
        results, meta = run_tuning(
            tuning_config,
            config_path=config_path,
            months_range=months_range,
            max_workers=None,
            progress_cb=progress_cb,
            partial_cb=write_partial,
        )
    except SystemExit as exc:
        msg = str(exc)
        if "YFRateLimitError" in msg or "rate" in msg.lower():
            print("YFRateLimitError: 요청이 너무 많습니다. 잠시 후 다시 실행하세요.")
            raise SystemExit(1) from exc
        raise

    results.sort(key=lambda x: x["cagr"], reverse=True)
    top_n = results[:100]

    if not results:
        print(f"튜닝 결과가 없습니다. {config_path}을 변경하지 않습니다.")
        raise SystemExit(1)

    best_params = results[0]["params"]
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    config["drawdown_buy_cutoff"] = round(float(best_params["drawdown_buy_cutoff"]), 2)
    config["drawdown_sell_cutoff"] = round(float(best_params["drawdown_sell_cutoff"]), 2)

    offense_obj = best_params.get("_offense_obj")
    if offense_obj and isinstance(offense_obj, dict):
        config["offense"] = {
            "ticker": offense_obj.get("ticker", ""),
            "name": offense_obj.get("name", ""),
        }

    defense_obj = best_params.get("_defense_obj")
    if defense_obj and isinstance(defense_obj, dict):
        config["defense"] = {
            "ticker": defense_obj.get("ticker", ""),
            "name": defense_obj.get("name", ""),
        }

    ordered_config = {"backtested_date": datetime.now().date().isoformat()}
    for key, value in config.items():
        if key != "backtested_date":
            ordered_config[key] = value

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(ordered_config, f, ensure_ascii=False, indent=4)
    # 단일 소스(DB)에도 최적 파라미터를 반영한다(추천 배치가 DB 를 읽으므로).
    save_leverage_config_raw(profile, ordered_config)
    print(f"최적 파라미터로 업데이트했습니다 (DB+파일, backtested_date={ordered_config['backtested_date']})")

    table_lines = render_top_table(results, top_n=100, months_range=months_range)
    end_ts = datetime.now()
    elapsed = format_seconds((end_ts - start_ts).total_seconds())

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"실행 시각: {start_ts.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"종료 시각: {end_ts.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"프로파일: {profile} | 시장: {market}\n")
        f.write(f"걸린 시간: {elapsed}\n\n")
        f.write("=== 튜닝 설정 ===\n")
        if meta and meta.get("period_start") and meta.get("period_end"):
            f.write(f"기간: {meta['period_start']} ~ {meta['period_end']} ({meta['period_months']} 개월)\n")
        else:
            f.write(f"기간: {start_ts.date()} ~ {end_ts.date()}\n")
        f.write("탐색 공간: ")
        parts = [f"{k} {len(v)}개" for k, v in tuning_config.items()]
        f.write(" × ".join(parts) + f" = {total_cases}개 조합\n\n")
        f.write(f"=== 결과 - 기간: {months_range} 개월 | 정렬 기준: CAGR ===\n")
        for line in table_lines[:200]:
            f.write(line + "\n")
        if len(results) > len(top_n):
            f.write(f"... (총 {total_cases}개 중 상위 {len(top_n)}개 표시)\n")

    print(f"튜닝 결과 저장: {out_path}")

    if args.slack:
        send_slack_tuning_result(
            country=market,
            started_at=start_ts,
            ended_at=end_ts,
            elapsed=elapsed,
            best_result=results[0],
            table_lines=render_top_table(results, top_n=10, months_range=months_range),
            meta=meta,
            log_path=str(out_path),
        )


if __name__ == "__main__":
    main()
