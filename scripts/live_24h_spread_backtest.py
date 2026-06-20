#!/usr/bin/env python
"""삼성+하이닉스 vs 코스피200 스프레드 평균회귀 백테스트 (Hyperliquid 시봉, 탐색용).

아이디어: 삼성+하이닉스 ≈ 코스피200의 절반. 반도체가 지수보다 많이 오르면(스프레드↑)
"나머지 절반"이 뒤쳐진 것 → 지수 매수/반도체 매도로 수렴 베팅(평균회귀).

⚠️ 탐색용. 데이터가 짧고(코스피200 상장이 늦어 시봉 겹치는 구간 ~2.4개월), 펀딩/슬리피지는
근사다. 결과가 좋아도 "검증된 엣지"가 아니라 "과최적화 의심"으로 봐야 한다.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

INFO_URL = "https://api.hyperliquid.xyz/info"
COINS = {"SMSN": "xyz:SMSN", "SKHX": "xyz:SKHX", "KR200": "xyz:KR200"}
BARS_PER_YEAR = 24 * 365  # 24시간 시장


def _candles(coin: str, interval: str, days_back: int) -> list[dict]:
    end = int(time.time() * 1000)
    start = end - days_back * 86400 * 1000
    body = json.dumps(
        {"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": start, "endTime": end}}
    ).encode()
    req = urllib.request.Request(INFO_URL, data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=20))


def load_prices() -> pd.DataFrame:
    """3종목 시봉 종가를 공통 타임스탬프로 정렬한 DataFrame."""
    series = {}
    for name, coin in COINS.items():
        rows = _candles(coin, "1h", 400)
        s = pd.Series({int(r["t"]): float(r["c"]) for r in rows}).sort_index()
        series[name] = s
        t0 = datetime.fromtimestamp(s.index[0] / 1000, timezone.utc).strftime("%Y-%m-%d")
        print(f"  {name:6} {len(s):5}봉  시작 {t0}")
    df = pd.DataFrame(series).dropna()  # 공통 구간(코스피200 상장 이후)
    return df


def backtest(
    df: pd.DataFrame,
    *,
    samsung_w: float = 0.8,
    window: int = 48,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
    cost_per_turn: float = 0.0010,  # |Δpos| 1당 비용(2개 레그 한 방향 ≈ 10bps)
    funding_bps_hr: float = 0.0,  # 시간당 펀딩 드래그(보유 포지션에)
    momentum: bool = False,  # True 면 추세추종(부호 반대)
) -> dict:
    log = np.log(df)
    semi = samsung_w * log["SMSN"] + (1 - samsung_w) * log["SKHX"]
    idx = log["KR200"]
    spread = semi - idx  # 반도체 - 지수 (로그가격 차)

    roll = spread.rolling(window)
    z = (spread - roll.mean()) / roll.std(ddof=0)

    # 스프레드 단순수익률 변화 (다음 봉)
    semi_ret = samsung_w * df["SMSN"].pct_change() + (1 - samsung_w) * df["SKHX"].pct_change()
    idx_ret = df["KR200"].pct_change()
    d_spread = (semi_ret - idx_ret).shift(-1)  # 이 봉 신호 → 다음 봉 수익

    pos = pd.Series(0.0, index=df.index)
    cur = 0.0
    for t in df.index:
        zv = z.loc[t]
        if np.isnan(zv):
            pos.loc[t] = 0.0
            continue
        # 평균회귀: z 높으면 스프레드 숏(=지수 롱/반도체 숏), z 낮으면 스프레드 롱
        if cur == 0.0:
            if zv > entry_z:
                cur = -1.0
            elif zv < -entry_z:
                cur = 1.0
        else:
            if abs(zv) < exit_z:
                cur = 0.0
        pos.loc[t] = cur
    if momentum:
        pos = -pos

    gross = pos * d_spread
    turn = pos.diff().abs().fillna(pos.abs())
    cost = turn * cost_per_turn
    funding = pos.abs() * (funding_bps_hr / 10000.0)
    net = (gross - cost - funding).dropna()

    eq = (1 + net).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    sharpe = (net.mean() / net.std() * np.sqrt(BARS_PER_YEAR)) if net.std() > 0 else 0.0
    trades = int((pos.diff().abs() > 0).sum())
    win = float((net[net != 0] > 0).mean()) if (net != 0).any() else 0.0
    ann = (eq.iloc[-1]) ** (BARS_PER_YEAR / len(net)) - 1 if len(net) > 0 else 0.0
    return {
        "총수익": eq.iloc[-1] - 1 if len(eq) else 0.0,
        "연환산": ann,
        "Sharpe": sharpe,
        "MDD": mdd,
        "거래수": trades,
        "승률": win,
        "회전율(연)": turn.sum() / len(net) * BARS_PER_YEAR if len(net) else 0.0,
    }


def main():
    print("=== 시봉 수집 ===")
    df = load_prices()
    t0 = datetime.fromtimestamp(df.index[0] / 1000, timezone.utc).strftime("%Y-%m-%d")
    t1 = datetime.fromtimestamp(df.index[-1] / 1000, timezone.utc).strftime("%Y-%m-%d")
    print(f"  공통 구간: {t0} ~ {t1}  ({len(df)}봉 ≈ {len(df)/24:.0f}일)\n")

    # 참고: 단순 보유 수익
    print("=== 참고: 보유(buy&hold) ===")
    for n in COINS:
        r = df[n].iloc[-1] / df[n].iloc[0] - 1
        print(f"  {n:6} {r*100:+.1f}%")
    corr = np.log(df).diff().corr()
    print(f"  상관(시봉 로그수익): 삼성-하이닉스 {corr.loc['SMSN','SKHX']:.2f}  반도체-지수 추정\n")

    base = dict(samsung_w=0.8, cost_per_turn=0.0010, funding_bps_hr=0.0)
    print("=== 기본 파라미터 (window=48, entry=1.5, exit=0.5, 비용 10bps/turn, 펀딩 0) ===")
    r = backtest(df, window=48, entry_z=1.5, exit_z=0.5, **base)
    for k, v in r.items():
        print(f"  {k:10}: {v:+.3f}" if isinstance(v, float) and k not in ("거래수",) else f"  {k:10}: {v}")

    print("\n=== 파라미터 스윕 (네트 Sharpe) — 골고루 높아야 신뢰, 몇 개만 높으면 과최적 ===")
    print("  window  entry  exit |  평균회귀Sharpe  추세Sharpe   총수익%")
    sharpes = []
    for window in (24, 48, 72, 168):
        for entry in (1.0, 1.5, 2.0):
            for ex in (0.0, 0.5):
                rmr = backtest(df, window=window, entry_z=entry, exit_z=ex, momentum=False, **base)
                rmo = backtest(df, window=window, entry_z=entry, exit_z=ex, momentum=True, **base)
                sharpes.append(rmr["Sharpe"])
                print(f"  {window:6} {entry:5.1f} {ex:5.1f} |  {rmr['Sharpe']:+10.2f}  {rmo['Sharpe']:+10.2f}   {rmr['총수익']*100:+7.1f}")
    arr = np.array(sharpes)
    print(f"\n  평균회귀 Sharpe — 중앙값 {np.median(arr):+.2f}, >1 비율 {np.mean(arr>1)*100:.0f}%, 최고 {arr.max():+.2f}")

    print("\n=== 펀딩 민감도 (window=48,entry=1.5,exit=0.5) ===")
    for f in (0.0, 0.5, 1.0, 2.0):
        r = backtest(df, window=48, entry_z=1.5, exit_z=0.5, samsung_w=0.8, cost_per_turn=0.0010, funding_bps_hr=f)
        print(f"  펀딩 {f:.1f}bps/h → Sharpe {r['Sharpe']:+.2f}, 총수익 {r['총수익']*100:+.1f}%")


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
