# 사용자 가이드 (User Guide)

이 문서는 Momentum ETF 시스템의 설치, 설정, 실행 방법을 설명합니다.

현재 이 시스템은 한국 상장 ETF와 호주 ETF를 운용 대상으로 사용합니다.
계좌 설정의 `country_code`는 `kor` 또는 `au`만 허용합니다.

## 1. 설치 및 실행

### 필수 요구사항
*   Python 3.9 이상
*   Docker (선택 사항, 배포 시 필요)

### 실행 명령어

**1. 종목풀 랭킹 생성**
종목풀에서 모멘텀 랭킹을 생성합니다.
```bash
python rank.py kor
python rank.py tax
python rank.py au
```

**2. 튜닝 (계좌 리밸런싱 파라미터 탐색)**
최적의 리밸런싱 파라미터를 찾기 위해 튜닝을 수행합니다. 완료 후 계좌 설정이 **자동으로 업데이트**됩니다.
```bash
python tune.py kor_account
python tune.py core_account
```

**3. 백테스트 실행**
과거 데이터를 바탕으로 전략의 성과를 시뮬레이션합니다.
```bash
python backtest.py kor_account
python backtest.py core_account
```

**4. 추천 실행 (매일 아침)**
백테스트 마지막 거래일 스냅샷을 기반으로 현재 시점의 비중 조절 결과를 생성하고 Slack으로 알림을 보냅니다.
```bash
python recommend.py kor_account
python recommend.py core_account
```

현재 운영 식별자:
* 종목풀: `kor`, `tax`, `au`
* 계좌: `kor_account`, `isa_account`, `pension_account`, `core_account`, `aus_account`

## 2. 설정 가이드

### 기본 설정 (`config.py`)
시스템 전반에 걸친 기본 설정을 변경할 수 있습니다.
*   `SLACK_BOT_TOKEN`: 알림을 봇을 통해 발송하기 위한 슬랙 봇 토큰
*   `SLACK_CHANNEL_ID`: 알림을 받을 슬랙 채널 ID

### 계좌 설정 (`zaccounts/<order>_<account_id>/config.json`)
계좌는 등록된 전체 종목을 상시 보유하고, 활성 버킷 수 기준으로 버킷 `weight`를 균등 계산하는 방식으로 동작합니다. 계산된 버킷 비중은 버킷 내부 종목 수 `N`으로 다시 균등 분배하며, 종목별 수동 `weight` 입력은 사용하지 않습니다.
계좌 `country_code`는 현재 `kor` 또는 `au`만 사용합니다.

```json
{
  "country_code": "kor",
  "strategy": {
    "TUNE_MONTHS": 12,
    "OPTIMIZATION_METRIC": "CAGR",
    "REBALANCE_MODE": "QUARTERLY"
  }
}
```

### 종목풀 설정 (`zpools/<order>_<pool_id>/config.json`)
종목풀은 랭킹 계산 파라미터를 관리합니다.

```json
{
  "name": "국내상장 국내 ETF",
  "desc": "국내상장 국내 ETF 종목풀",
  "rank": {
    "country": "kor",
    "months": 12,
    "ma_type": "HMA"
  }
}
```

> 참고: 계좌 추천/백테스트는 종목풀 랭킹을 자동 병합하지 않습니다. 랭킹 결과를 보고 사용자가 계좌 종목을 직접 교체하면, 시스템은 그 종목 집합을 기준으로 균등 비중만 다시 계산합니다.
> 참고: `tax` 종목풀은 절세 계좌용 ETF 분류이며, 계좌 `country_code`와는 별개인 운영 식별자입니다.

## 3. 대시보드 및 계좌 관리

### 대시보드 구성
대시보드는 **[요약(Summary)]** 탭과 **[상세(Details)]** 탭으로 나뉩니다.
*   **요약 탭**: 포트폴리오 전체 수익률, 원금 대비 순이익, 현금 비중 등 핵심 지표를 확인합니다.
*   **상세 탭**: 개별 종목의 현재가, 평가손익, 현재 비중, 타겟비중, 예정 비중조절 문구를 상세히 확인합니다.
*   계좌 운용 상태는 `BUY`, `HOLD`만 사용합니다. 비중 축소/확대는 상태가 아니라 문구에만 표시됩니다.

### 계좌 관리 (원금 및 현금)
사이드바의 **[계좌 관리]** 메뉴를 통해 각 계좌의 기초 데이터를 수정할 수 있습니다.
*   **💵 원금 및 현금 관리**: 각 계좌의 '총 투자 원금'과 '남은 현금 잔고'를 입력합니다.
*   입력된 데이터는 실시간으로 대시보드의 **총자산 및 수익률 계산**에 반영됩니다.

### 캐시 알림 (Alerts)
대시보드 상단에는 가격 데이터의 최신 정합성을 체크하는 경고창이 뜹니다.
*   **계좌별 그룹화**: 어떤 계좌의 종목이 갱신이 필요한지 계좌별로 친절하게 안내합니다.
*   해당 경고가 뜨면 `python scripts/update_price_cache.py`를 실행하여 데이터를 갱신하십시오.

## 4. 결과 해석

### Slack 알림
매일 아침 전송되는 슬랙 메시지는 다음 정보를 포함합니다.
*   **비중 조절 현황**: 현재 비중, 타겟비중, 예정 비중조절 문구
*   **보유 현황**: 현재 보유 중인 종목 상태와 **연속 보유일** 정보
*   **계좌 요약**: 현재 총 자산, 현금 비중, **평균 수익률** 등


### 로그 파일
실행 중 발생하는 상세 로그는 `logs/` 폴더에 저장됩니다. 오류 발생 시 이 파일을 확인하세요.
