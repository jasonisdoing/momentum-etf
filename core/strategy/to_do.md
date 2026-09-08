# 전략 통합·리팩토링 — 남은 작업

최종 갱신: 2026-09-08. 큰 구조 이동은 완료됐으나 아래 후속 수정·검증은 남아 있다.
작업 단위마다 현재 상태·변경 파일·검증 결과·다음 행동을 갱신한다.

## 재개 지점

- 상태: 1번(확정 주문 날짜·예상 표시)·2번(시가 미상 진입 예산) 수정 완료, 검증까지 끝(다른
  세션의 미실행 검증을 이어받아 실행).
- 검증(2026-09-08): 핵심 테스트 8개 성공(139260 조합·날짜별 목표 고정 입력 포함).
  kor_test 실확인 — 보유 표와 오늘의 액션이 정합: 오늘(9/8) 체결 예정이던 확정 주문
  (0008T0 매도, 0005G0·307520 진입)은 계좌가 이미 반영해 지시가 없고, 남은 지시는 전부
  9/9 시가(조정 + 잠정 진입 예상 — 474590 진입(예상)·사유 표기 정상). 139260은
  목표 34주 = 보유 34주라 지시 없음(정상). 신호 없는 전량 매도·재매수 증상 재현 안 됨.
- 다음 행동: 3번(반올림 전 계산값)부터.

## 실행 순서

1. **확정 주문의 날짜·예상 표시**
   - [x] 엔진의 `fill_date`를 합성 이벤트·액션까지 전달하고 이벤트별로 확정/예상을 구분.
   - [x] 고정 입력 엔진→슬리브→합성 액션에서 확정 매수·매도의 날짜와 잠정 진입 날짜 검증.
   - [x] 동일·반대 방향의 복수 체결일을 날짜별 엔진 목표로 분리. 임시 충돌 오류 경로 제거.
     날짜별 정수 목표의 차이를 사용하고 최종 표의 목표와 일치하도록 연결.
   - [x] 같은 날짜 상계, 앞선 주문 처리 후 후속 주문 수량·키 유지, 중복 종목 정수 배분 검증.
2. [x] **시가 미상 진입의 예산** — 확정 미체결 매수 우선 예약·비용 제외 목표 비중 전달.
   잠정 진입은 예약 후 잔여만 사용한다. kor_test의 날짜별 예산 초과 해소 및
   기존 엔진→합성 일치 테스트에 전체 목표 비중 검증 추가.
   - [x] **첫 운용일 공통 구간** — 한 슬리브의 전략 시작일이 오늘이면 공통 일별 결과가
     한 행이 된다. `mix/simulate.replay_mix`는 이를 데이터 부족으로 처리하지 않고 저장된
     합성 배분을 첫날 상태로 반환한다. 가격·현금 비중을 추정하거나 과거를 채우지 않는다.
3. [ ] **반올림 전 계산값** — 포트폴리오 일별 `strategy_pct`는 2자리, 슬롯은 6자리이고
   합성은 이를 다시 배수로 변환한다. 계산용 원값과 표시값을 분리해 세 전략·합성에 연결한다.
4. [ ] **실제 장중 검증** — 고정 입력 테스트와 별개로 개장 후 모멘텀·신고가·합성 확인.
   국내 ETF 시가 미상 확정 주문의 날짜·표시, 체결 시각 미제공 시 세션 시계 live 처리 포함.
   실제 검증 시각·시장·계좌·확인 결과를 기록한다. 과거 작업일 전체가 휴장일이었다고 단정하지 않는다.
5. [ ] **실시간 마지막 봉 헬퍼 통일** — 4번 검증 후 `utils/rankings.build_effective_close_series`와
   `core/strategy/intraday.effective_close_frame` 통합. 순위의 서울 날짜 고정과 전략의 시장 세션 날짜 차이를 해소한다.

## 완료·유지 사항

- [x] 기존 큰 구조 이동·장중 엔진 연결·옛 경로 정리 커밋: `93b4c873` (2026-09-07).
- 공통 엔진·신호·패널·포트폴리오 시뮬레이션·합성 계산과 전략 문서는 `core/strategy/`에 있다.
  수집·설정 로딩·페이로드 조립은 외부 계층에 남아 있다.
- 포트폴리오 현금 초과 매수는 수정됐다. 실제 보유로 목표를 보정하지 않는 원칙은 유지한다.
- 현재 운용 API의 `daily`는 합성 배분과 같은 실행 결과를 공유하기 위한 계약이다.
- 이번 후속 작업은 요청 없이 커밋·푸시하지 않는다.

## 검증 기록

- 이전 리팩토링: 핵심 테스트 8개·pre-commit·Next 빌드·tsc 및 계좌 읽기 확인 기록은 git 이력 참고.
- 2026-09-08: `.venv/bin/python -m unittest tests.test_screen_matches_backtest.SlotEngineProvisionalBarTest`
  2개 성공. 새 파일 없이 기존 엔진 잠정 상태 테스트에 합성 액션 연결 검증을 포함.
- `.venv/bin/python -m unittest tests.test_screen_matches_backtest`: 로컬 DB 접근 실행에서 8개 성공(38.159초).
- `SKIP=update-app-datetime pre-commit run --files utils/mix_sleeve.py core/strategy/mix/actions.py tests/test_screen_matches_backtest.py core/strategy/strategy_logic.md core/strategy/to_do.md`:
  Next 빌드 성공. 첫 실행은 테스트 파일 공백·포맷 자동 수정으로 종료.
- 같은 파일 목록에 `SKIP=update-app-datetime,next-build`로 재실행: flake8·ruff·포맷 성공.
  `git diff --check` 성공. 주문 실행·슬랙 발송·실제 장중 화면 검증은 하지 않음.
- 작업 도중 다른 수정이 `config.py`, 모멘텀 신호·서비스·튜닝·API·풀 설정에 함께 나타남.
  이번 작업의 변경은 위 검증 대상 5파일이며 다른 변경은 수정하거나 되돌리지 않았다.
  다음 검증 시 현재 작업 트리 상태를 다시 확인한다.

### 2026-09-08 복수 체결일 단위

- `.venv/bin/python -m unittest tests.test_screen_matches_backtest`: 8개 성공(36.382초, 로컬 DB 접근).
- 이후 테스트 사례만 추가하여 `.venv/bin/python -m unittest tests.test_screen_matches_backtest.SlotEngineProvisionalBarTest`:
  2개 성공. 매수/매도 네 조합, 같은 날 상계, 선행 목표 달성 후 후속 주문 유지, 중복 종목 정수 배분 포함.
- `SKIP=update-app-datetime pre-commit run --files core/strategy/mix/actions.py core/strategy/mix/targets.py utils/strategy_mix_service.py tests/test_screen_matches_backtest.py core/strategy/to_do.md core/strategy/strategy_logic.md`:
  Next 빌드 성공. 최초 실행과 테스트 추가 후 실행에서 테스트 파일 공백·포맷 자동 수정이 발생.
- 위 파일 목록으로 `SKIP=update-app-datetime,next-build` 재실행: flake8·ruff·포맷 성공.
  `git diff --check` 성공. 실거래·슬랙 발송은 하지 않음.
- 액션 키가 종목+날짜로 바뀌므로 기존 알림 비교 캐시와 처음 비교할 때 새 항목으로 인식될 수 있다.
  이후에는 앞 날짜 액션이 사라져도 후속 날짜 액션 키가 유지된다. 알림 캐시를 임의 삭제하지 않았다.

### 2026-09-08 kor_test 예산 초과 수정

- `PYTHONPATH=. .venv/bin/python /tmp/diagnose_mix_budget.py`: 로컬 DB를 읽고
  `mix_positions('kor_test')`의 날짜별 배분을 검사. 수정 전 두 번째 날짜 예산 초과,
  수정 후 `STAGE 1 OK`, `STAGE 2 OK`, 정상 종료(한국 시간 09:15~09:16).
  임시 스크립트는 유지보수 의존성이 아니며 같은 검증은 `mix_positions('kor_test')`로 재실행 가능.
- `.venv/bin/python -m unittest tests.test_screen_matches_backtest`: 8개 성공(45.756초).
  기존 잠정 상태 테스트에 유지 목표+확정 미체결 목표+잠정 진입 목표 합계 검증을 포함.
- `SKIP=update-app-datetime pre-commit run --files core/strategy/slot_backtest.py tests/test_screen_matches_backtest.py core/strategy/strategy_logic.md core/strategy/to_do.md`:
  Next 빌드·flake8·ruff 성공. 포맷이 테스트 파일 1개 수정 후 종료.
- 위 파일 목록으로 `SKIP=update-app-datetime,next-build` 재실행: 전체 코드 검사 성공.
  `git diff --check` 성공. 브라우저 화면·슬랙 발송·주문 실행은 하지 않음.
  따라서 4번 실제 장중 화면 검증 전체를 완료 처리하지 않는다.

### 2026-09-08 us_test 첫 운용일 공통 구간 수정

- 원인: A 슬리브(`us_stock`) 전략 시작일이 2026-09-08이라 일별 결과가 당일 한 행뿐이었다.
  B·C 슬리브는 9월 1일부터라 공통 프레임도 한 행이 됐고, 합성 재생의 두 행 이상 조건이
  `합성에 필요한 공통 가격·현금 비중 데이터가 부족합니다.` 오류를 냈다.
- 조치: `core/strategy/mix/simulate.py`에서 빈 프레임·현금 비중 누락만 오류로 두고,
  한 행은 저장된 슬리브 배분을 그대로 첫 운용일 상태로 반환하도록 변경.
- 검증: `MixRebalanceMatchesBacktest` 2개 성공. 실제 `mix_positions('us_test')`는
  보유 행 17개·액션 그룹 1개로 정상 반환됐고, 슬리브 초기 배분 A 40%·B 30%·C 30%를 확인.
