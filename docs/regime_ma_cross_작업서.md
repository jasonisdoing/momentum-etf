# 작업서: 시장지수 레짐 → "MA20/60 교차 + 지수별 N일 확인"

> 상태: 착수 전 / 담당: 미정
> 관련 화면: `/market-trend`(시장지수 추세 + 백테스트 탭), 탑픽 현금제어

## 0. 목표
- 시장지수 레짐 판정을 현재의 **슈퍼트렌드(ST) + ALMA 이동선 ± 버퍼** 방식에서
  **MA20/60 교차 + N거래일 확인** 방식으로 **완전히 교체**한다.
- 운영에서 사람이 조정하는 튜닝값은 **지수별 N(확인 거래일 수)** 하나뿐.
- 흐름:
  `/market-trend` 백테스트 탭에서 N 검증 → **config의 지수별 N을 사람이 직접 수정** →
  라이브(시장지수 추세 화면 + 탑픽 현금제어)가 그 N으로 동작.

## 1. 레짐 규칙 (라이브·백테스트 공통 정본)
MA20 = SMA(종가, 20), MA60 = SMA(종가, 60).

```
원본 레짐:
  MA20 < MA60 (데드크로스)        → 하락
  MA20 ≥ MA60 이고 종가 > MA20     → 상승
  MA20 ≥ MA60 이고 종가 < MA60     → 하락
  그 외 (두 선 사이)               → 중립

확인 필터(N):
  새 레짐이 (N+1)거래일 연속 나올 때만 전환. N=0 은 즉시 전환.
  전체 과거를 순방향으로 처리하며 '확정 상태'를 유지 → 오늘값이 라이브 레짐. (룩어헤드 없음)
```

> 백테스트에 이미 구현된 `utils/market_trend_service.py`의
> `_regime_series_ma_cross()` + `_apply_confirmation()`가 이 규칙의 **정본**이다.
> 이 둘을 **공개 함수로 승격**해 라이브도 재사용할 것. (예: `compute_ma_cross_regime(df, confirm_days)`)

## 2. config.py
**추가**
```python
MARKET_TREND_REGIME_MA_SHORT = 20   # 단기선
MARKET_TREND_REGIME_MA_LONG  = 60   # 장기선

# 지수별 확인 거래일 수 (백테스트로 검증 후 지수마다 조정)
MARKET_TREND_REGIME_CONFIRM_DAYS = {
    "^KS11": 2, "^KS200": 2, "^DJI": 3, "^GSPC": 3, "^NDX": 2, "^SOX": 2,
}
```
- `_resolve_confirm_days(ticker)` 헬퍼 추가 — 미등록 지수는 **명시적 에러**(암묵적 기본값 금지).
- **제거** — 레짐 전용으로 더 이상 쓰지 않는 예전 설정:
  - 지수별 버퍼 설정
  - 예전 레짐 MA 타입/단기 MA 기간 설정
- **유지** — SuperTrend는 레짐 판단에서는 제외하지만 차트 보조선/화살표 표시에 사용:
  - `MARKET_TREND_SUPERTREND_PERIOD`
  - `MARKET_TREND_SUPERTREND_MULTIPLIER`(dict)

## 3. 백엔드 — utils/market_trend_service.py
1. **정본 함수 승격**: `_regime_series_ma_cross`, `_apply_confirmation`를 라이브에서 쓸 수 있게 공개.
2. **`_build_item` / `_regime_step` 교체**: 지수별 일별 레짐을 새 규칙 + `_resolve_confirm_days(ticker)`로 계산.
3. **`compute_market_trend`(표) / `compute_index_history`(행 펼침 차트)**: `current_regime`·일별 regime을 새 규칙으로.
4. **forecast(전환 예측) 재정의** (`_forecast_thresholds`, 약 647행):
   방향별로 `(가격 조건, 남은 K거래일)`을 반환.
   - 상승 전환: "앞으로 K거래일 동안 종가가 **MA20(≈X원) 위**를 유지"
   - 하락 전환: "앞으로 K거래일 동안 종가가 **MA60(≈Y원) 아래**"
   - K = (N+1) − (현재 후보 레짐이 연속으로 나온 일수)
   - 임계가 X/Y는 MA선이 매일 움직이므로 **오늘 기준 근사**임을 표기.
5. **trend_score(추세 점수) 재정의**: 기준을 ALMA-48 → **MA20 대비 괴리율**로 바꿔 −100~+100 정규화.
6. **제거**: 예전 버퍼 조회, 단일 스텝 레짐 판정, 예전 벡터 레짐 함수 (grep 로 잔참조 확인).
   `_resolve_supertrend_params`, `_calculate_supertrend`는 차트 보조선/화살표 표시용으로 유지.
7. **`/defaults` 라우트 응답** 정리 → 새 값(MA 20/60, confirm) 노출.

## 4. 백엔드 — utils/top_pick_service.py (탑픽 현금제어)
- `_calculate_benchmark_regimes(bench_df, ticker)`를 새 규칙 + `_resolve_confirm_days(ticker)`로 교체. 지수별 튜닝 제거.
- 현금제어 로직(accel_up→현금0%, accel_down→현금확보, neutral→기본)과 `_load_regime_benchmark_ohlc`는 **그대로**. 레짐 **소스만** 교체.

## 5. 프론트 — web/app/market-trend/MarketTrendClient.tsx
- 하단 레짐 설명(`REGIME_DESCRIPTIONS`)·"추세 점수" 컬럼 설명의 "슈퍼트렌드/버퍼/ALMA" 문구를
  **"MA20/60 교차 + N일 확인 / MA20 기준"** 으로 재작성.
- **백테스트 탭 "현재 config" 행 제거**(ST 기반이라 폐기).
  `buy&hold` 행은 `ma_variants[0].bh_*` 에서 가져오도록 배선. 백엔드 응답의 `current`/관련 계산 제거.

## 6. 프론트 — web/app/market-trend/MarketTrendChart.tsx
- 차트의 레짐 음영은 MA20/60 기준으로 교체하되, **슈퍼트렌드 선/화살표는 보조 시각화로 유지**.
- forecast 표시 블록을 §3-4 형식("K거래일 동안 X원 유지 시 전환")으로 갱신.

## 7. 정리·검증
- 죽은 상수/함수 전수 제거 — grep: `BUFFER_PCT`, `_regime_step`.
- **수용 기준**
  1. `/market-trend` 표·차트 레짐이 백테스트 탭의 같은 지수·기간 **"+N일" 행과 정확히 일치**(N = 해당 지수 config).
  2. 탑픽 현금제어가 같은 규칙·같은 지수별 N 사용.
  3. config의 지수별 N만 바꾸면 라이브 화면과 탑픽이 함께 바뀐다.
  4. 버퍼 참조가 코드에 남지 않음. ST/곱수 참조는 차트 보조 시각화에만 남음.
  5. 확인 필터가 과거만 사용(룩어헤드 없음).

## 8. 참고
- 백테스트 탭(읽기 전용, `compute_regime_confirm_backtest`)은 검증 도구로 **유지**한다.
  운영 반영은 오직 사람이 config의 지수별 N을 수정하는 것으로만 이뤄진다(자동 저장 없음).
