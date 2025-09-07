# momentum-pilot
ETF 추세추종 전략을 기반으로 한 자동매매 엔진

간단한 운용, 시뮬레이션, 일일 액션 산출을 위한 스크립트 모음입니다. 실제 운용 전 반드시 충분한 검증과 점검을 해 주세요.

구성 개요 / 폴더 구조
---------------------

- `data/`: 사용자 데이터 폴더
  - `tickers.txt`: 운용/시뮬레이션 대상 티커 목록. (포맷: `티커 이름`)
  - `portfolio_YYYY-MM-DD.json`: 특정 날짜의 포트폴리오 스냅샷(평가금액, 보유종목). 웹 UI를 통해 생성/저장됩니다.
- `core/`: 핵심 엔진
  - `backtester.py`: (사용되지 않음) 개별 종목 백테스팅 실행기.
  - `portfolio.py`: (사용되지 않음) 포트폴리오 백테스팅 실행기.
- `logics/`: 매매 전략(로직) 정의. 각 전략은 자체 폴더를 가집니다.
  - `<strategy_name>/strategy.py`: 전략의 핵심 백테스팅 로직.
  - `<strategy_name>/settings.py`: 해당 전략에만 사용되는 파라미터.
- `utils/`: 공용 유틸리티 함수
  - `data_loader.py`: 데이터 로딩 및 API 호출.
  - `indicators.py`: 보조지표 계산 (SuperTrend 등).
  - `report.py`: 리포트, 로그 포맷팅 및 테이블 렌더링.
- `main.py`: 프로젝트 메인 실행 진입점.
- `test.py`: 과거 구간 백테스트 실행 및 `logs/test.log` 생성.
- `today.py`: 당일/익일 매매 액션 계산 및 `logs/today.log` 생성.
- `tune_seykota.py`: `seykota` 전략의 파라미터를 최적화하는 스크립트.
- `web_app.py`: Streamlit 기반 웹 UI.
- `settings.py`: 모든 전략에 공통으로 적용되는 전역 설정.

전략(Strategy) 구조
-------------------

각 투자 전략은 `logics/` 디렉토리 아래에 자체 폴더로 구성됩니다. 예를 들어, `jason` 전략은 `logics/jason/` 폴더에 위치합니다.
- `logics/jason/strategy.py`: `jason` 전략의 매수/매도 로직을 구현합니다.
- `logics/jason/settings.py`: `jason` 전략에만 사용되는 파라미터(예: `BUY_SUM_THRESHOLD`)를 정의합니다.
- `settings.py`: 모든 전략에 공통으로 적용되는 파라미터(예: `INITIAL_CAPITAL`, `PORTFOLIO_TOPN`)를 정의합니다.

설치 및 준비
------------

1) Python 가상환경 구성(권장)

    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install --upgrade pip

2) 라이브러리 설치

    pip install pandas streamlit pykrx tqdm

3) 파일 준비

- `data/tickers.txt`에 운용 티커/명칭을 한 줄씩 등록합니다.
- (권장) `web_app.py`를 실행하여 초기 포트폴리오(`portfolio_YYYY-MM-DD.json`)를 생성합니다. 이 파일에는 평가금액과 현재 보유 종목 정보가 포함됩니다.

실행 방법
--------

1) 오늘/다음 거래일 액션 산출

경량 모드로, 과거 시뮬레이션 없이 “현재 보유 + 오늘 신호”를 바탕으로 다음 거래일에 해야 할 액션을 제안합니다.

    python main.py --today <전략이름>  # 예: jason, seykota

- 출력: `logs/today_전략이름.log` (예: `logs/today_jason.log`).
- 헤더 표기: pykrx 거래일 캘린더로 오늘이 휴장/영업일인지 판별해
  - 영업일이면: “오늘 YYYY-MM-DD”
  - 휴장 등이면: “다음 거래일 YYYY-MM-DD”
- 사전 조건: `data/portfolio_YYYY-MM-DD.json` 파일에 해당 거래일의 평가금액과 보유종목이 있어야 합니다.
- 표: `jason`, `seykota` 전략은 신호 점수가 높은 순으로 정렬됩니다. 상태(SELL/CUT/TRIM/BUY/HOLD/WAIT), 비중, 전략별 신호, 문구(사유) 등을 표시합니다.

2) 백테스트 로그 생성

과거 구간에 대해 백테스트를 실행합니다.

    # 특정 전략(예: jason)의 상세 백테스트 실행
    python main.py --test jason

    # 'jason'과 'seykota' 전략의 성과 요약 비교
    python main.py --test

- 상세 백테스트 실행 시: `logs/test_전략이름.log` (예: `logs/test_jason.log`)에 일별 상세 로그가 기록됩니다.
- 요약 비교 실행 시: 두 전략의 최종 성과(CAGR, MDD 등)만 간략히 표로 보여줍니다.

3) 전략 파라미터 튜닝 (예: seykota)

`tune_seykota.py` 스크립트를 실행하여 `seykota` 전략의 최적 이동평균 기간을 찾습니다.

    python tune_seykota.py

- 주의: 매우 많은 조합을 테스트하므로 실행에 오랜 시간이 걸릴 수 있습니다.
- 스크립트 내에서 테스트할 파라미터 범위를 조절할 수 있습니다.
- 최종적으로 최고 CAGR, 최저 MDD, 최고 Calmar Ratio(위험 조정 수익률)를 기록한 파라미터와 성과를 각각 출력합니다.

4) 웹앱 실행

평가금액/보유 입력을 브라우저에서 편하게 관리합니다.

    streamlit run web_app.py

- 상단: 평가금액(원) 입력.
- 보유 입력(수동) 테이블
  - 티커 입력 시 `data/tickers.txt`를 참조하여 명칭 자동 채움(포커스 아웃 시 반영)
  - 수량(정수), 매수단가(원) 입력.
  - '포트폴리오 스냅샷 저장' 버튼 클릭 시, 입력된 평가금액과 보유 종목 정보가 `data/portfolio_YYYY-MM-DD.json` 파일로 저장됩니다.

전략/로직 요약
-------------

### `jason` 전략 (모멘텀 + 추세추종)
- **매수 신호**:
  - 모멘텀 점수(`score`)가 `BUY_SUM_THRESHOLD`를 초과하고,
  - 슈퍼트렌드(`filter`)가 상승 추세(+1)일 때.
  - *참고: `score`는 최근 1주 수익률과 2주 수익률의 가중합으로 계산됩니다.*
- **매도 신호**:
  - **모멘텀소진**: 보유 수익률과 모멘텀 점수의 합이 `SELL_SUM_THRESHOLD` 미만일 때.

### `seykota` 전략 (이동평균선 추세추종)
- **매수 신호**:
  - 단기 이동평균선이 장기 이동평균선 위에 있을 때 (골든 크로스).
- **매도 신호**:
  - **추세이탈**: 단기 이동평균선이 장기 이동평균선 아래로 내려갈 때 (데드 크로스).

- 공통 규칙(시뮬레이션)
  - **가격기반손절(CUT)**: 보유수익률 ≤ `HOLDING_STOP_LOSS_PCT`
  - **비중조절(TRIM)**: 보유가치 > `MAX_POSITION_PCT` × 총자산 → 초과분 부분매도
  - 쿨다운: 매수/매도 후 COOLDOWN_DAYS 동안 반대 방향 거래 금지

설정 파일
--------

- **`settings.py`**: 모든 전략에 공통으로 적용되는 전역 설정.
  - `INITIAL_CAPITAL`: 초기 자본금
  - `TEST_DATE_RANGE`: 백테스트 기간
  - `PORTFOLIO_TOPN`: 최대 보유 종목 수
  - `MIN_POSITION_PCT`, `MAX_POSITION_PCT`: 최소/최대 포지션 비중
  - `HOLDING_STOP_LOSS_PCT`: 공통 손절매 비율
- **`logics/<strategy_name>/settings.py`**: 특정 전략에만 적용되는 설정.
  - 예) `jason` 전략의 `BUY_SUM_THRESHOLD`, `seykota` 전략의 `SEYKOTA_FAST_MA` 등

데이터 파일 포맷
--------------

- `data/tickers.txt`: 예) `449450 PLUS K방산` (공백/탭/콤마 구분 허용)
- `data/portfolio_YYYY-MM-DD.json`: 웹 UI에서 저장되는 포트폴리오 스냅샷.
  ```json
  {
    "date": "2024-05-24",
    "total_equity": 100000000,
    "holdings": [
      {
        "ticker": "005930",
        "name": "삼성전자",
        "shares": 10,
        "avg_cost": 75000
      }
    ]
  }
  ```

주의/제약
--------

- today.py는 pykrx를 이용하여 거래일 판정을 수행합니다(네트워크 필요). pykrx/API 오류 시 휴장 판단이 달라질 수 있습니다.
- today.py 경량 모드는 “내일 할 일” 계산에 특화되어 있습니다.
- 모든 결과는 참고용이며, 실거래 적용 전 리스크/수수료/세금/체결 슬리피지 등을 반드시 반영해 재검증하세요.
