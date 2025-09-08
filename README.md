# momentum-pilot
ETF 추세추종 전략 기반의 트레이딩 시뮬레이션 및 분석 도구

간단한 운용, 시뮬레이션, 일일 액션 산출을 위한 스크립트 모음입니다. 모든 결과는 투자 참고용이며, 실제 투자 결정에 대한 책임은 투자자 본인에게 있습니다.

구성 개요 / 폴더 구조
---------------------

- `data/`: 사용자 데이터 폴더
  - `tickers.txt`: 운용/시뮬레이션 대상 티커 목록. (포맷: `티커 이름`).
  - `portfolio_raw.txt`: 사용자가 직접 관리하는 포트폴리오 원본 파일.
  - `portfolio_YYYY-MM-DD.json`: `convert_portfolio.py`를 통해 생성되는 포트폴리오 스냅샷.
- `logics/`: 매매 전략(로직) 정의. 각 전략은 자체 폴더를 가집니다.
  - `<strategy_name>/strategy.py`: 전략의 핵심 백테스팅 로직.
  - `<strategy_name>/settings.py`: 해당 전략에만 사용되는 파라미터.
- `utils/`: 공용 유틸리티 함수
  - `data_loader.py`: 데이터 로딩 및 API 호출.
  - `indicators.py`: 보조지표 계산 (SuperTrend 등).
  - `report.py`: 리포트, 로그 포맷팅 및 테이블 렌더링.
- `main.py`: 프로젝트의 메인 실행 진입점.
- `aus.py`: 호주 시장용 메인 실행 진입점.
- `test.py`: 과거 구간 백테스트 실행 및 `logs/test.log` 생성.
- `status.py`: 당일/익일 매매 액션 계산 및 `logs/status.log` 생성.
- `tune_seykota.py`: `seykota` 전략의 파라미터를 최적화하는 스크립트.
- `web_app.py`: Streamlit 기반 웹 UI. 오늘의 현황을 시각적으로 보여줍니다.
- `convert_portfolio.py`: `portfolio_raw.txt`를 `portfolio_YYYY-MM-DD.json` 형식으로 변환합니다.
- `settings.py`: 모든 전략에 공통으로 적용되는 전역 설정.

전략(Strategy) 구조
-------------------

각 투자 전략은 `logics/` 디렉토리 아래에 자체 폴더로 구성됩니다. 예를 들어, `jason` 전략은 `logics/jason/` 폴더에 위치합니다.
- `logics/jason/strategy.py`: `jason` 전략의 매수/매도 로직을 구현합니다.
- `logics/jason/settings.py`: `jason` 전략에만 사용되는 파라미터(예: `BUY_SUM_THRESHOLD`)를 정의합니다. 전역 설정에 정의된 값보다 우선 적용될 수 있습니다.
- `settings.py`: 모든 전략에 공통으로 적용되는 파라미터(예: `INITIAL_CAPITAL`, `PORTFOLIO_TOPN`)를 정의합니다.

설치 및 준비
------------

1) Python 가상환경 구성(권장)

    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install --upgrade pip

2) 라이브러리 설치

    pip install pandas pykrx tqdm requests beautifulsoup4
3) 파일 준비

- **티커 목록**: `data/<국가코드>/tickers.txt`에 운용할 종목의 티커와 이름을 한 줄씩 등록합니다. (예: `data/kor/tickers.txt`)
- **포트폴리목
--------

1) 포트폴리오 원본 파일 변환 (필요시)

`data/<국가코드>/portfolio_raw.txt` 파일의 내용을 시스템이 사용하는 `portfolio_YYYY-MM-DD.json` 형식으로 변환합니다.

    python kor.py --convert
a/kr/portfolio_raw.txt` 파일이 없으면 예시 파일이 자동 생성됩니다.
- 실행 후 생성된 `portfolio_YYYY-MM-DD.json` 파일의 `total_equity` 값을 실제 총평가액으로 직접 수정해야 합니다.

2) 오늘/다음 거래일 현황 조회

경량 모드로, 과거 시뮬레이션 없이 “현재 보유 + 오늘 신호”를 바탕으로 다음 거래일에 해야 할 액션을 제안합니다.

    python <국가코드>.py --status <전략이름>  # 예: python kor.py --status jason

- 출력: `logs/status_전략이름.log` (예: `logs/status_jason.log`).
- 헤더 표기: pykrx 거래일 캘린더로 오늘이 휴장/영업일인지 판별해
  - 영업일이면:Y“다음 거래일 YYYY-MM-DD”
- 사전 조건: `data/portfolio_YYYY-MM-DD.json` 파일에 해당 거래일의 평가금액과 보유 종목이 있어야 합니다.
- 표: `jason`, `seykota` 전략은 신호 점수가 높은 순으로 정렬됩니다. 상태(SELL/CUT/TRIM/BUY/HOLD/WAIT), 비중, 전략별 신호, 문구(사유) 등을 표시합니다.

3) 백테스트 로그 생성

과거 구간에 대해 백테스트를 실행합니다.

    # 특정 전략(예: jason)의 상세 백테스트 실행
    python <국가코드>.py --test jason

    # 'jason'과 'seykota' 전략의 성과 요약 비교
    python <국가코드>.py --test
름.log` (예: `logs/test_jason.log`)에 일별 상세 로그가 기록됩니다.
- 요약 비교 실행 시: 두 전략의 최종 성과(CAGR, MDD 등)만 간략히 표로 보여줍니다.

4) 전략 파라미터 튜닝 (예: seykota)
`tune_seykota.py` 스크립트를 실행하여 `seykota` 전략의 최적 이동평균 기간을 찾습니다.

    python tune_seykota.py

- 주의: 매우 많은 조합을 테스트하므로 실행에 오랜 시간이 걸릴 수 있습니다.
- 스크립트 내에서 테스트할 파라미터 범위를 조절할 수 있습니다.
- 최종적으로 최고 CAGR, 최저 MDD, 최고 Calmar Ratio(위험 조정 수익률)를 기록한 파라미터와 성과를 각각 출력합니다.

5) 웹앱으로 현황 확인

웹 브라우저를 통해 오늘의 현황을 시각적으로 확인합니다.

    streamlit run web_app.py

- 각 전략과 국가 조합([KOR] Jason, [AUS] Jason 등)이 탭으로 구분되어 표시됩니다.
- 테이블의 수익률 항목은 양수일 경우 <span style="color:red">빨간색</span>, 음수일 경우 <span style="color:blue">파란색</span>으로 표시됩니다.

전략/로직 요약
-------------

### `jason` 전략 (모멘텀 + 추세추종)
- **매수 신호**:
  - **모멘텀 점수**: 최근 1주 수익률과 그 이전 1주(총 2주) 수익률의 합이 `BUY_SUM_THRESHOLD`를 초과.
  - **추세 필터**: 슈퍼트렌드 지표가 상승 추세(+1)일 때.
- **매도 신호**:
  - **모멘텀소진**: 보유 수익률과 모멘텀 점수의 합이 `SELL_SUM_THRESHOLD` 미만일 때.

### `seykota` 전략 (이동평균선 추세추종)
- **매수 신호**:
- **매수 신호**:
  - 단기 이동평균선이 장기 이동평균선 위에 있을 때 (골든 크로스).
- **매도 신호**:
  - **추세이탈**: 단기 이동평균선이 장기 이동평균선 아래로 내려갈 때 (데드 크로스).
- **매수 신호**:
  - 가격이 지정된 기간의 이동평균선 위에 있을 때.
- **매도 신호**:
  - **추세이탈**: 가격이 이동평균선 아래로 내려갈 때.

### 공통 리스크 관리 규칙 (백테스트)
  - **가격기반손절(CUT)**: 보유수익률 ≤ `HOLDING_STOP_LOSS_PCT`
  - **비중조절(TRIM)**: 보유가치 > `MAX_POSITION_PCT` × 총자산 → 초과분 부분매도
  - **쿨다운**: 매수/매도 후 `COOLDOWN_DAYS` 동안 반대 방향 거래 금지

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

- `data/<국가코드>/tickers.txt`: 운용할 종목의 티커 목록. 한 줄에 티커 하나씩 입력합니다. (예: `005930` 또는 `ASX:BHP`). 호주(`aus`)의 경우 `.AX` 접미사는 자동으로 추가되며, 종목명이 없으면 자동으로 조회하여 **파일에 다시 기록합니다**.
- `data/<국가코드>/portfolio_raw.txt`: 사용자가 직접 관리하는 포트폴리오 원본 파일.
  ```
  # 총평가액 (주석이 아닌 첫 번째 줄에 숫자만 입력)
  100000000

  # 보유종목 (티커 수량 매수단가)
  # 각 종목은 한 줄에 하나씩 입력합니다.
  005930 10 75000
  000660 5 120000
  ```
- `data/portfolio_YYYY-MM-DD.json`: `convert_portfolio.py`로 생성되는 포트폴리오 스냅샷.
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

- status.py는 pykrx를 이용하여 거래일 판정을 수행합니다(네트워크 필요). pykrx/API 오류 시 휴장 판단이 달라질 수 있습니다.
- status.py 경량 모드는 “내일 할 일” 계산에 특화되어 있습니다.
- status.py는 장중 실행 시, 네이버 금융 웹 스크레이핑을 통해 실시간 가격을 가져오려고 시도합니다. 이 기능은 네이버 웹사이트 구조 변경 시 동작하지 않을 수 있으며, 비공식적인 방법이므로 안정성을 보장하지 않습니다.
- 모든 결과는 참고용이며, 실거래 적용 전 리스크/수수료/세금/체결 슬리피지 등을 반드시 반영해 재검증하세요.
