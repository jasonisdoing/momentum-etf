# momentum-pilot
ETF 추세추종 전략을 기반으로 한 자동매매 엔진


간단한 운용, 시뮬레이션, 일일 액션 산출을 위한 스크립트 모음입니다. 실제 운용 전 반드시 충분한 검증과 점검을 해 주세요.

구성 개요 / 폴더 구조
---------------------

- `data/`: 사용자 데이터 폴더
  - `tickers.txt`: 운용/시뮬레이션 대상 티커 목록. (포맷: `티커` `이름`)
  - `holdings.csv`: 현재 보유 현황. 웹앱에서 작성/저장. (포맷: `ticker,name,shares,amount`)
  - `data.json`: 웹앱에서 사용하는 데이터 저장소 (예: 평가금액).
- `core/`: 핵심 엔진
  - `backtester.py`: 개별 종목 백테스팅 실행기.
  - `portfolio.py`: 포트폴리오 백테스팅 실행기.
- `logics/`: 매매 전략(로직) 정의
  - `jason.py`: 현재 기본 전략을 정의하기 위한 파일 (현재는 core에 로직 통합).
- `utils/`: 공용 유틸리티 함수
  - `data_loader.py`: 데이터 로딩 및 API 호출.
  - `indicators.py`: 보조지표 계산 (SuperTrend 등).
  - `report.py`: 리포트, 로그 포맷팅 및 테이블 렌더링.
- `main.py`: 프로젝트 메인 실행 진입점.
- `test.py`: 과거 구간 백테스트 실행 및 `logs/test.log` 생성.
- `today.py`: 당일/익일 매매 액션 계산 및 `logs/today.log` 생성.
- `web_app.py`: Streamlit 기반 웹 UI.
- `settings.py`: 전략 파라미터 및 전역 설정.

전략(Strategy) 구조
-------------------

현재 핵심 매매 로직은 `core/backtester.py`와 `core/portfolio.py`에 통합되어 있습니다. 이 로직은 `settings.py` 파일의 파라미터를 통해 제어됩니다.

향후 여러 전략을 지원하도록 확장할 경우, `logics/` 디렉터리 아래에 새로운 전략 파일을 추가하고 `main.py`에서 이를 선택적으로 로드하도록 수정할 수 있습니다.

설치 및 준비
------------

1) Python 가상환경 구성(권장)

    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install --upgrade pip

2) 라이브러리 설치

    pip install pandas streamlit pykrx

3) 파일 준비

- `data/tickers.txt`에 운용 티커/명칭을 한 줄씩 등록합니다.
- (선택) `data/holdings.csv`에 현재 보유 현황을 저장합니다 (웹앱에서 작성 가능).
- (권장) `data/data.json`의 `equity_by_date`에 최신 거래일의 평가금액을 기록합니다.

실행 방법
--------

1) 오늘/다음 거래일 액션 산출

경량 모드로, 과거 시뮬레이션 없이 “현재 보유 + 오늘 신호”를 바탕으로 다음 거래일에 해야 할 액션을 제안합니다.

    python main.py --today

- 출력: `logs/today.log`
- 헤더 표기: pykrx 거래일 캘린더로 오늘이 휴장/영업일인지 판별해
  - 영업일이면: “오늘 YYYY-MM-DD”
  - 휴장 등이면: “다음 거래일 YYYY-MM-DD”
- 사전 조건: `data/data.json`의 `equity_by_date`에 해당 거래일의 평가금액이 있어야 하며, 없으면 경고 후 종료합니다.
- 표: 액션의 예상 체결금액(notional) 내림차순 정렬. 상태(SELL/CUT/TRIM/BUY/HOLD/WAIT), 비중, 1주/2주/합계, ST, 문구(사유) 표시

2) 백테스트 로그 생성
 
과거 구간에 대해 백테스트를 실행합니다.

    # 특정 전략(예: jason)의 상세 백테스트 실행
    python main.py --test jason

    # 'jason'과 'dummy' 전략의 성과 요약 비교
    python main.py --test

- 상세 백테스트 실행 시: `logs/test.log`에 일별 상세 로그가 기록됩니다.
- 요약 비교 실행 시: 두 전략의 최종 성과(CAGR, MDD 등)만 간략히 표로 보여줍니다.

3) 웹앱 실행

평가금액/보유 입력을 브라우저에서 편하게 관리합니다.

    streamlit run web_app.py

- 상단: 평가금액(원) 입력 → 저장 시 data.json에 initial_capital로 저장(토스트 알림)
- 보유 입력(수동) 테이블
  - 티커 입력 시 `data/tickers.txt`를 참조하여 명칭 자동 채움(포커스 아웃 시 반영)
  - 수량(정수), 매수단가(원) 입력 → 오른쪽 상단 저장(`data/holdings.csv`) 클릭 시 저장
  - 테이블 아래 현재 저장된 `data/holdings.csv`의 유효 행(티커 있고 수량>0) 수 표시

전략/로직 요약
-------------

- 신호 산출(경량 today.py)
  - 1주 수익률 p1 = (t / t-5)
  - 2주 수익률 p2 = (t-5 / t-10)
  - 합계 s2 = p1 + p2
  - ST 방향(슈퍼트렌드) = +1/-1/0
- 의사결정(경량 today.py)
  - 보유 중
    - CUT: 보유수익률 ≤ HOLDING_STOP_LOSS_PCT
    - SELL: s2 + 보유수익률 < SELL_SUM_THRESHOLD
    - TRIM: 보유가치 > MAX_POSITION_PCT × 총자산 → 초과분 부분매도
  - 비보유
    - BUY: s2 > BUY_SUM_THRESHOLD AND ST=+1 AND MIN_POSITION_PCT 비중을 정수 주식으로 충족 가능
    - 불가 시 사유: ‘현금 부족’ 또는 ‘상한 제한’
- 공통 규칙(시뮬레이션)
  - 급락 매도 금지: 일간 등락률 ≤ BIG_DROP_PCT이면 당일 포함 BIG_DROP_SELL_BLOCK_DAYS 매도 차단
  - 쿨다운: 매수/매도 후 COOLDOWN_DAYS 동안 반대 방향 거래 금지

설정 파일
--------

- settings.py: 기본 파라미터(전역)
  - 예) SELL_SUM_THRESHOLD, BUY_SUM_THRESHOLD, HOLDING_STOP_LOSS_PCT, COOLDOWN_DAYS, BIG_DROP_PCT, BIG_DROP_SELL_BLOCK_DAYS, MONTHS_RANGE 등

데이터 파일 포맷
--------------

- `data/tickers.txt`: 예) `449450 PLUS K방산` (공백/탭/콤마 구분 허용)
- `data/holdings.csv`: 저장 형식 `ticker,name,shares,amount` (표준 CSV, 이름에 공백/쉼표 허용)
- `data/data.json`: `equity_by_date`에 거래일별 평가금액 기록, 웹앱의 `initial_capital`도 함께 저장

주의/제약
--------

- today.py는 pykrx를 이용하여 거래일 판정을 수행합니다(네트워크 필요). pykrx/API 오류 시 휴장 판단이 달라질 수 있습니다.
- today.py 경량 모드는 “내일 할 일” 계산에 특화되어 있습니다(보유일/매수일 등 장기 이력은 출력하지 않음).
- 모든 결과는 참고용이며, 실거래 적용 전 리스크/수수료/세금/체결 슬리피지 등을 반드시 반영해 재검증하세요.
