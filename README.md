# 📈 Momentum ETF

> **"추세에 순응하되, 위험은 철저히 관리한다."**
>
> 데이터 기반의 ETF 순위 분석 시스템

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🚀 소개 (Introduction)

**Momentum ETF**는 계좌 설정을 기반으로 동작하는 **ETF 순위 분석 시스템**입니다.
현재 운영 모델은 다음과 같습니다.
* 계좌(DB `account_settings`): 등록된 전체 종목을 직접 관리하고 비중 조절로 운용
* 종목풀(DB `pool_settings`): 순위/전략 계산에 사용하는 종목풀 메타데이터

감정이나 직관에 의존하는 투자를 지양하고, **이동평균(MA)**과 **RSI** 등 기술적 지표를 활용하여 **"현재 계좌 종목의 상대 추세 강도를 비교하는"** 데이터 기반의 의사결정을 지원합니다.

## ✨ 주요 기능 (Key Features)

*   **📊 계좌별 순위 화면**: 이동평균 고정 추세선과 종목풀별 `SHORT_MA_DAYS`/`LONG_MA_DAYS` 설정을 기준으로 추세와 정배열 여부를 즉시 확인합니다.
*   **🟩 실보유 강조 표시**: 실제 보유 종목은 순위 테이블에서 녹색 행으로 즉시 구분합니다.
*   **🛡️ 데이터 안정성 (Robust Caching)**: **Apache Parquet** 포맷을 캐시 엔진으로 도입하여 라이브러리 버전 mismatch로부터 자유롭고 안정적인 데이터 로딩을 보장합니다.
*   **🛠️ 종목 관리**: 계좌별 종목 추가/수정/삭제 및 삭제 종목 복원을 지원합니다.

## 📚 문서 (Documentation)

유지보수용 문서 3개(`docs/`). 사용법·설정값은 적지 않는다 — 코드가 단일 소스다.

*   **[개발자 가이드](docs/developer_guide.md)**: 구조, 화면·모듈, DB 단일 소스, 지켜야 할 규칙
*   **[전략 로직](docs/strategy_logic.md)**: 전략별 규칙과 시도 후 폐기한 것
*   **[서버 인프라](docs/server_infrastructure.md)**: 배포, 배치, 도메인, 환경변수

## ⚡️ 빠른 시작 (Quick Start)

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/momentum-etf.git
cd momentum-etf

# 가상환경 생성 및 활성화 (권장)
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 설정
`.env` 와 `config.py` 를 환경에 맞게 채우고, 계좌·종목풀은 DB(`account_settings`, `pool_settings`)에서 관리합니다. (환경변수는 [서버 인프라](docs/server_infrastructure.md) 참고)

### 3. 실행

```bash
python run_local_dev.py            # FastAPI + Next dev
python infra/server_scheduler.py   # 배치 스케줄러
```

웹에서 다음 기능을 사용합니다.
* `http://localhost:3000/`
* Home
* 자산 관리 / 일별 / 주별 / 월별 / 스냅샷
* 종목 관리 / 종목 비교 / 한국 개별주 / 미국 개별주 / 한국 ETF / 정보

현재 기본 계좌 식별자는 다음과 같습니다.
* 계좌: `kor_account`, `isa_account`, `pension_account`, `core_account`, `aus_account`

## ⚠️ 면책 조항 (Disclaimer)

이 소프트웨어는 투자를 돕기 위한 보조 도구입니다. **최종적인 투자 결정과 그에 따른 책임은 전적으로 사용자에게 있습니다.** 개발자는 이 프로그램을 사용하여 발생한 금전적 손실에 대해 책임지지 않습니다.
