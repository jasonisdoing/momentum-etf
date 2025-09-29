# 시그널 규칙 명세 (Signal Rules)

본 문서는 시그널 계산 로직 중 "규칙(Decision Rules)"만을 일관되게 관리하기 위한 명세입니다. 구현 상세(데이터 파이프라인, IO, 인프라)는 다루지 않으며, 규칙 변경 시 이 문서를 우선 업데이트합니다.

## 1. 용어와 범위
- 상태(State): HOLD, WAIT, SOLD
- 부분 매수(Partial Buy): 기존 보유 상태에서 추가 매수 발생
- 신규 매수(New Buy): 당일 최초 매수로 보유 상태 진입
- 부분 매도(Partial Sell): 보유 수량의 일부만 매도
- 전량 매도(Full Sell): 보유 수량 전부 매도하여 보유 종료
- 코인 최소치 임계값: `COIN_ZERO_THRESHOLD`

## 2. 컬럼/표시 규칙(SIGNAL_TABLE_HEADERS 매핑)
- 인덱스 2: 종목명
- 인덱스 4: 상태(HOLD/WAIT/SOLD)
- 마지막 컬럼: 메시지(예: 신규/부분 매수·매도 문구)
- 규칙 변경 시 컬럼 인덱스와의 정합성을 반드시 유지한다.

### 표시 문구 관리
- 모든 표시 문구는 상수 `logic/strategies/momentum/constants.py`의 `DECISION_MESSAGES`에서 중앙 관리한다.
- 사용 키:
  - `NEW_BUY`: "✅ 신규 매수"
  - `PARTIAL_BUY`: "🌗 부분 매수({amount})" — amount는 `format_kr_money()` 결과 문자열
  - `PARTIAL_SELL`: "⚠️ 부분 매도 ({amount})" — 금액 기준 표시(수량 아님)
  - `FULL_SELL`: "🔚 매도 완료"

## 3. 상태 결정 규칙
- 추천(recommendation) 결과라도 실제 보유 수량에 따라 표시 상태 보정:
  - 보유 수량 > 0 (코인: `> COIN_ZERO_THRESHOLD`) → 상태를 HOLD로 보정
  - 보유 수량 == 0 (코인: `<= COIN_ZERO_THRESHOLD`) → 상태를 WAIT로 보정
- 당일 매도 체결 존재 시:
  - 남은 보유 수량 > 0 → 상태는 HOLD, 메시지에 부분 매도 문구 첨부
  - 남은 보유 수량 == 0 → 상태는 SOLD, 메시지는 "매도 완료"

## 4. 매수 규칙
  - 기존 보유가 있었거나 동일 일자에 2회 이상 분할 매수 → "부분 매수"로 표시(금액 포함 가능)
  - 그 외 → "신규 매수"로 표시
- 전략별 캡/슬롯/쿨다운이 있을 경우, 각 전략 규칙에 따라 후보 선정 및 실행 제한 적용

## 5. 매도 규칙
- 부분 매도: 남은 수량이 0이 아니면 상태 HOLD, 메시지에 "부분 매도(금액)" 표시
  - 전량 매도: 남은 수량이 0이면 상태 SOLD, 메시지 "매도 완료"

## 6. 예외/엣지 케이스
- 휴장일: `generate_signal_report()` 호출 레벨에서 생성 금지(코인 제외)
- 실시간 시세 실패: 가능하면 과거 종가로 계산, 불가 시 해당 종목 실패 처리
- 데이터 부족: 최소 필요 길이 미만이면 오류/스킵 처리
## 7. 관련 구현 포인터
- 상태/문구 최종 적용: `logic/signals/pipeline.py`
- 헤더 정의: `logic/strategies/momentum/shared.py` 의 `SIGNAL_TABLE_HEADERS`
- 실시간 시세: `utils/data_loader.py`
- 포트폴리오/체결: `utils/db_manager.py`, `utils/transaction_manager.py`

## 8. 테스트/검증 가이드
- 당일 전량 매도 케이스: 상태가 4번 컬럼에 SOLD, 종목명(2번)은 유지되어야 함
- 당일 부분 매도: 상태 HOLD, 마지막 컬럼에 부분 매도 문구(수량 표기)
- 신규/부분 매수 케이스: 마지막 컬럼 문구 및 금액/수량 표시 일관성 확인

## 9. 변경 이력(요약)
- 2025-09-29: 상태 컬럼 인덱스 오류 수정(2 → 4). SOLD가 종목명 컬럼에 덮어쓰이는 문제 해결.
- 2025-09-29: 정렬 정책 업데이트
  - `DECISION_CONFIG`에서 `SOLD.order`를 `60 → 50`으로 조정하여 `WAIT(order=50)`와 동일 그룹으로 병합 정렬
  - `WAIT` 항목 개수 제한 제거(기존 상한 100 → 무제한)

## 10. 정렬 규칙
- 모든 항목은 `DECISION_CONFIG[state].order` 오름차순으로 정렬한다.
- 동일 `order` 내에서는 `score` 내림차순 정렬한다.
- `order`가 같은 서로 다른 상태(예: WAIT, SOLD)는 하나의 그룹으로 병합되어 점수 기준으로 함께 정렬된다.
- WAIT 항목의 개수 제한은 두지 않는다(무제한 포함).

참고: 현재 설정에서 `SOLD`와 `WAIT`는 모두 `order=50`으로 동일 그룹에 속한다.
