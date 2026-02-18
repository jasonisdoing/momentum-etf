# 📈 Momentum ETF

> **"추세에 순응하되, 위험은 철저히 관리한다."**
>
> 데이터 기반의 반(半)자동 ETF 포트폴리오 운용 시스템

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🚀 소개 (Introduction)

**Momentum ETF**는 **5버킷(Bucket) 분산 투자**를 기반으로 한 **ETF 추세추종(Momentum) 자동화 시스템**입니다.
단순한 모멘텀 전략을 넘어, **[1. 모멘텀, 2. 혁신기술, 3. 시장지수, 4. 배당방어, 5. 대체헷지]**의 5가지 버킷으로 자산을 배분하여 상승장에서는 수익을 극대화하고 하락장에서는 방어력을 높입니다.

감정이나 직관에 의존하는 투자를 지양하고, **이동평균(MA)**과 **RSI** 등 기술적 지표를 활용하여 **"상대적으로 가장 우수한 종목을 선정하고 리밸런싱 때까지 유지하는"** 데이터 기반의 의사결정을 지원합니다.

## ✨ 주요 기능 (Key Features)

*   **📊 통합 추천 시스템 (Unified Recommendation)**: 백테스트 엔진을 기반으로 현재 계좌 상태를 시뮬레이션하여 최적의 매수/매도/교체 신호를 생성하고 Slack으로 알림을 보냅니다.
*   **🛡️ 리스크 관리 (Risk Management)**:
    *   **RSI 과매수 차단**: 과열된 종목의 매수를 막아 고점 추격 매수를 방지합니다.
    *   **쿨다운(Cooldown)**: 매수/매도 후 일정 기간 동일 종목 거래를 제한하여 휩소(whipsaw) 매매를 방지합니다.
*   **🧪 검증된 전략 (Backtesting & Tuning)**: 과거 데이터를 기반으로 전략의 유효성을 검증하고, 시장 상황에 맞는 최적의 파라미터를 자동으로 탐색합니다.

## 📊 성과 (Performance)
*최근 12개월 백테스트 결과 (2024.12.03 기준, 10개 포트폴리오 계좌)*
- **CAGR**: 72.37%
- **MDD**: -11.24%
- **Sharpe Ratio**: 2.82
*(실제 성과는 시장 상황에 따라 달라질 수 있습니다.)*

## 📚 문서 (Documentation)

더 자세한 내용은 `docs/` 폴더의 문서를 참고하세요.

*   **[프로젝트 개요 (Project Overview)](docs/project_overview.md)**: 프로젝트의 철학, 상세 기능, 시스템 구조
*   **[사용자 가이드 (User Guide)](docs/user_guide.md)**: 설치, 설정, 실행 방법, 결과 해석
*   **[전략 로직 (Strategy Logic)](docs/strategy_logic.md)**: MAPS 점수 산정, 매매 조건, 리스크 관리 알고리즘 상세
*   **[개발자 가이드 (Developer Guide)](docs/developer_guide.md)**: 시스템 아키텍처, 데이터 파이프라인, 정합성 원칙

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
`config.py` 및 `zsettings/account/kor_us.json` 파일을 환경에 맞게 수정합니다. (상세 내용은 [사용자 가이드](docs/user_guide.md) 참고)

### 3. 실행

**1. 튜닝 (최적 파라미터 탐색 및 자동 적용)**
```bash
python tune.py kor_us  # 한국 계정 (미국 ETF)
python tune.py us      # 미국 계정
```

**2. 백테스트 (성과 검증)**
```bash
python backtest.py kor_us
python backtest.py us
```

**3. 추천 (매매 신호 생성)**
```bash
python recommend.py kor_us
python recommend.py us
```

## ⚠️ 면책 조항 (Disclaimer)

이 소프트웨어는 투자를 돕기 위한 보조 도구입니다. **최종적인 투자 결정과 그에 따른 책임은 전적으로 사용자에게 있습니다.** 개발자는 이 프로그램을 사용하여 발생한 금전적 손실에 대해 책임지지 않습니다.
