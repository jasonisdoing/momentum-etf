# 전략 통합·리팩토링 — 남은 작업

최종 갱신: 2026-09-08. 큰 구조 이동은 완료됐으나 아래 후속 수정·검증은 남아 있다.
작업 단위마다 현재 상태·변경 파일·검증 결과·다음 행동을 갱신한다.

## 재개 지점

- 상태: 1번 복수 체결일 목표·액션 분리까지 구현·검증 완료. 현재 변경은 미커밋.
- 변경 파일: `core/strategy/mix/targets.py`, `core/strategy/mix/actions.py`,
  `utils/strategy_mix_service.py`, `tests/test_screen_matches_backtest.py`, 전략 문서와 이 문서.
- 결과: 날짜별 엔진 목표를 기존 정수 배분으로 계산. 최초 액션은 실제 보유, 이후는 직전 목표와 비교한다.
  최종 날짜 목표를 표에서도 사용하고, 동일 종목의 액션 키는 날짜를 포함한다.
- 검증: 핵심 테스트 8개·Next 빌드·변경 파일 코드 검사 성공. 추가 고정 입력 검증도 성공(아래 기록).
- 다음 행동: 2번 `core/strategy/slot_backtest.py`의 `_entry_quantities`, `pending_entries`,
  `planned_entry_weights`를 함께 점검한다. 시가 미상 진입이 자금 예약 없이 1/N로 전달되는 경로를
  엔진의 동일한 현금 제한 안에서 처리하고, 기존 잠정 상태 일치 테스트를 확장한다.
- 동작 전제: 이후 날짜 주문은 앞선 날짜 목표 달성을 가정한 계획이다. 실제 체결 이력 저장을 추가하지 않았다.
  소액 조정 필터는 날짜별 액션에 기존 방식으로 적용된다.

## 실행 순서

1. **확정 주문의 날짜·예상 표시**
   - [x] 엔진의 `fill_date`를 합성 이벤트·액션까지 전달하고 이벤트별로 확정/예상을 구분.
   - [x] 고정 입력 엔진→슬리브→합성 액션에서 확정 매수·매도의 날짜와 잠정 진입 날짜 검증.
   - [x] 동일·반대 방향의 복수 체결일을 날짜별 엔진 목표로 분리. 임시 충돌 오류 경로 제거.
     날짜별 정수 목표의 차이를 사용하고 최종 표의 목표와 일치하도록 연결.
   - [x] 같은 날짜 상계, 앞선 주문 처리 후 후속 주문 수량·키 유지, 중복 종목 정수 배분 검증.
2. [ ] **시가 미상 진입의 예산** — `pending_entries`의 고정 `100 / slots` 비중은
   합성 목표에도 사용된다. 현금·다른 진입의 자금 예약을 고려하는 엔진 배분으로 점검·수정한다.
   일반 `planned_entries`는 이미 잔여 현금을 고려하므로 둘을 혼동하지 않는다.
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
