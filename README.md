# momentum-pilot
ETF 추세추종 전략 기반의 트레이딩 시뮬레이션 및 분석 도구

간단한 운용, 시뮬레이션, 일일 액션 산출을 위한 스크립트 모음입니다. 모든 결과는 투자 참고용이며, 실제 투자 결정에 대한 책임은 투자자 본인에게 있습니다.

구성 개요 / 폴더 구조
---------------------

- `logic/`: 매매 전략(로직) 정의.
- `utils/`: 공통 유틸리티 모듈
  - `data_loader.py`: 데이터 로딩 및 API 호출.
  - `indicators.py`: 보조지표 계산 (SuperTrend 등).
  - `report.py`: 리포트, 로그 포맷팅 및 테이블 렌더링.
- `scripts/`: 각종 유틸리티 및 분석 스크립트 모음.
- `run.py`: 메인 실행 진입점.
- `test.py`: 과거 구간 백테스트 실행 및 `logs/test.log` 생성.
- `web_app.py`: Streamlit 기반 웹 UI. 오늘의 현황을 시각적으로 보여줍니다.
- `settings.py`: 모든 전략에 공통으로 적용되는 전역 설정.
설치 및 준비
------------

1) Python 가상환경 구성(권장)
2) 데이터베이스 준비
- **MongoDB**: 연결 정보(`MONGO_DB_CONNECTION_STRING`)를 `settings.py` 또는 환경 변수에 설정해야 합니다. 모든 거래 내역, 종목 마스터, 평가금액 등은 DB에 저장됩니다.
- **종목 마스터**: 웹 앱의 `마스터정보` 탭을 통해 투자 유니버스에 포함할 종목을 관리합니다.

주요 사용법
-----------

1) 오늘/다음 거래일 현황 조회

과거 시뮬레이션 없이 “현재 보유 + 오늘 신호”를 바탕으로 다음 거래일에 대한 현황을 제안합니다.

    python run.py <국가코드> --status

예: `python run.py kor --status`

2) 백테스트 로그 생성

과거 구간에 대해 백테스트를 실행합니다.

    python run.py <국가코드> --test

예: `python run.py coin --test`

3) 파라미터 튜닝

`tune.py` 스크립트를 실행하여 전략의 최적 파라미터를 찾습니다.

    python scripts/tune.py

- 주의: 매우 많은 조합을 테스트하므로 실행에 오랜 시간이 걸릴 수 있습니다.
- 스크립트 내에서 테스트할 파라미터 범위를 조절할 수 있습니다.
- 최종적으로 최고 CAGR, 최저 MDD, 최고 Calmar Ratio(위험 조정 수익률)를 기록한 파라미터와 성과를 각각 출력합니다.

4) 웹앱으로 현황 확인

웹 브라우저를 통해 오늘의 현황을 시각적으로 확인하고, 거래 내역, 종목, 업종 등 모든 데이터를 관리합니다.
    
    streamlit run web_app.py

5) (선택) 종목 업종 일괄 초기화

모든 종목의 업종을 공백으로 초기화하여 재분류할 수 있도록 준비합니다.

    python scripts/reset_stock_sectors.py

6) (선택) 업종 목록 재설정

모든 기존 업종을 삭제하고, 스크립트 내에 정의된 새로운 표준 업종 목록으로 교체합니다.

    python scripts/reset_and_seed_sectors.py

7) (선택) 급등주 찾기

pykrx 라이브러리를 사용하여 한국 시장의 급등 ETF를 섹터별로 찾아봅니다.

    python scripts/find.py --type etf --min-change 5.0


전략/로직 요약
-------------


  - 가격이 지정된 기간의 이동평균선 위에 있을 때.
- **매도 신호**:
  - **추세이탈**: 가격이 이동평균선 아래로 내려갈 때.

### 공통 리스크 관리 규칙 (백테스트)
  - **가격기반손절(CUT)**: 보유수익률 ≤ `HOLDING_STOP_LOSS_PCT`
  - **쿨다운**: 매수/매도 후 `COOLDOWN_DAYS` 동안 반대 방향 거래 금지

설정 체계 (DB 기반)
-------------------

- 인프라 설정(`settings.py`): DB 연결, 웹앱 비밀번호 등 인프라 관련 설정.
- 공통 설정(웹앱 ‘설정’ 탭 → 공통 설정)
  - 모든 국가가 동일하게 사용하는 전역 파라미터를 DB에 저장하여 관리합니다.
  - MARKET_REGIME_FILTER_ENABLED / MARKET_REGIME_FILTER_TICKER / MARKET_REGIME_FILTER_MA_PERIOD
  - HOLDING_STOP_LOSS_PCT / COOLDOWN_DAYS / ATR_PERIOD_FOR_NORMALIZATION
  - 주의: HOLDING_STOP_LOSS_PCT는 양수로 입력해도 자동으로 음수로 저장/해석됩니다. (예: 10 → -10)
- 국가별 전략 파라미터(각 국가 탭 → 설정)
  - portfolio_topn, ma_period_etf, ma_period_stock
  - replace_weaker_stock, max_replacements_per_day, replace_threshold
  - 각 국가별로 DB에 저장되어 해당 국가 현황/백테스트에 반영됩니다.

주의/제약
--------

- status.py는 pykrx를 이용하여 거래일 판정을 수행합니다(네트워크 필요). pykrx/API 오류 시 휴장 판단이 달라질 수 있습니다.
- status.py 경량 모드는 “내일 할 일” 계산에 특화되어 있습니다.
- status.py는 장중 실행 시, 네이버 금융 웹 스크레이핑을 통해 실시간 가격을 가져오려고 시도합니다. 이 기능은 네이버 웹사이트 구조 변경 시 동작하지 않을 수 있으며, 비공식적인 방법이므로 안정성을 보장하지 않습니다.
- 모든 결과는 참고용이며, 실거래 적용 전 리스크/수수료/세금/체결 슬리피지 등을 반드시 반영해 재검증하세요.
