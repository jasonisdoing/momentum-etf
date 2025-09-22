# Momentum ETF

이 프로젝트는 모멘텀 기반의 ETF 투자 전략을 추적하고 백테스트하기 위한 웹 기반 애플리케이션입니다. 한국, 호주, 가상화폐 시장에 대한 포트폴리오 시그널을 모니터링하고, 매매 신호를 확인하며, 슬랙(Slack)을 통해 알림을 받을 수 있는 대시보드를 제공합니다.

## 주요 기능 (Features)

*   **웹 대시보드**: Streamlit을 사용하여 각 국가별 포트폴리오 시그널, 매매 신호, 보유 종목 등을 시각적으로 제공합니다.
*   **자동 시그널 계산**: 스케줄러를 통해 매일 또는 지정된 주기로 포트폴리오 시그널을 자동으로 계산하고 DB에 저장합니다.
*   **슬랙 알림**: 매매 신호 발생 시 슬랙(Slack)으로 실시간 알림을 전송합니다.
*   **백테스팅**: 과거 데이터를 사용하여 투자 전략의 성과를 검증합니다.
*   **유연한 설정**: 웹 UI를 통해 국가별 투자 전략, 초기 자본, 알림 주기 등 대부분의 설정을 관리할 수 있습니다.

## 사전 준비 (Prerequisites)

*   Python 3.10 이상
*   MongoDB Atlas 계정 (무료 클러스터로 충분합니다)
*   (선택) 빗썸(Bithumb) API Key 및 Secret Key (가상화폐 자산 추적 시 필요)
*   (선택) Slack Incoming Webhook URL (알림 기능 사용 시 필요)

## 설치 및 설정 방법 (Setup Instructions)

### 1. 프로젝트 클론

```bash
git clone <repository_url>
cd momentum-etf
```

### 2. Python 가상 환경 설정 및 활성화

프로젝트 의존성을 시스템 라이브러리와 분리하기 위해 가상 환경 사용을 강력히 권장합니다.

```bash
# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화 (macOS/Linux)
source venv/bin/activate

# 가상 환경 활성화 (Windows)
.\venv\Scripts\activate
```

### 3. 의존성 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 4. MongoDB Atlas 설정

이 프로젝트는 MongoDB를 데이터베이스로 사용합니다. MongoDB Atlas에서 무료 클러스터를 생성하여 사용할 수 있습니다.

1.  [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)에 가입하고 로그인합니다.
2.  새로운 프로젝트(Project)를 생성합니다.
3.  **Build a Database**를 클릭하여 무료(M0) 클러스터를 생성합니다. 클라우드 제공자 및 지역은 원하는 대로 선택합니다.
4.  **Database Access** 메뉴로 이동하여 데이터베이스 사용자(Database User)를 생성합니다.
    *   `Username`과 `Password`를 설정하고 꼭 기억해두세요.
5.  **Network Access** 메뉴로 이동하여 IP 주소를 등록합니다.
    *   **Add IP Address**를 클릭합니다.
    *   개발 환경의 IP를 추가하거나, 보안에 유의하여 `0.0.0.0/0` (Allow Access from Anywhere)를 추가할 수 있습니다.
6.  **Database** 메뉴로 돌아와 생성된 클러스터의 **Connect** 버튼을 클릭합니다.
7.  **Drivers**를 선택하고, 나타나는 연결 문자열(Connection String)을 복사합니다. 이 문자열은 다음 단계에서 사용됩니다.

### 5. 환경 변수 설정

프로젝트 루트 디렉터리에 `.env` 파일을 생성하고 아래 내용을 채워넣습니다.

```env
# 4단계에서 복사한 MongoDB 연결 문자열을 붙여넣고, <password> 부분을 실제 비밀번호로 교체하세요.
MONGO_DB_CONNECTION_STRING="mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"

# 사용할 데이터베이스 이름 (기본값: momentum_etf_db)
MONGO_DB_NAME="momentum_etf_db"

# (선택) 국가별 슬랙 알림 웹훅 URL
KOR_SLACK_WEBHOOK="https://hooks.slack.com/services/..."
AUS_SLACK_WEBHOOK="https://hooks.slack.com/services/..."
COIN_SLACK_WEBHOOK="https://hooks.slack.com/services/..."

# (선택) 중요 시스템 로그를 받을 슬랙 웹훅 URL
LOGS_SLACK_WEBHOOK="https://hooks.slack.com/services/..."

# (선택) 빗썸 API 키 (가상화폐 자산 추적용)
BITHUMB_API_KEY="Your_Bithumb_API_Key"
BITHUMB_API_SECRET="Your_Bithumb_Secret_Key"

# (선택) Google Gemini API 키 (ETF 자동 분류용)
GOOGLE_API_KEY="Your_Google_API_Key"
```

## 사용 방법 (Usage)

### 웹 애플리케이션 실행

```bash
streamlit run web_app.py
```

실행 후 터미널에 나타나는 URL(보통 `http://localhost:8501`)을 웹 브라우저에서 열어 대시보드를 확인합니다.

### 스케줄러 실행

시그널 계산 및 알림을 자동으로 실행하려면 스케줄러를 별도의 터미널에서 실행해야 합니다.

```bash
python aps.py
```

스케줄러는 웹 앱의 '알림' 탭에서 설정한 주기에 따라 자동으로 작업을 수행합니다.

### 백테스트 실행

특정 국가의 전략을 백테스트하려면 다음 명령을 사용합니다.

```bash
# 한국(kor) 시장 백테스트
python test.py

# 다른 국가를 지정하려면 인자로 전달 (예: aus, coin)
# python test.py aus
```

## 프로젝트 구조

*   `web_app.py`: Streamlit 웹 UI의 메인 애플리케이션 파일.
*   `signals.py`: 각 국가별 포트폴리오의 현재 상태를 계산하는 핵심 로직.
*   `test.py`: 투자 전략의 성과를 검증하기 위한 백테스팅 스크립트.
*   `aps.py`: `signals.py`를 주기적으로 실행하는 스케줄러.
*   `logic/`: 실제 매매 전략(모멘텀 점수 계산, 매수/매도 결정 등)이 구현된 폴더.
*   `utils/`: 데이터베이스 관리, 데이터 로딩, 슬랙 알림 등 공통 유틸리티 함수.
*   `scripts/`: 빗썸 거래 내역 동기화 등 보조 스크립트.
*   `data/`: 국가별 투자 유니버스(ETF, 코인 목록)를 정의하는 JSON 파일.
*   `requirements.txt`: 프로젝트 실행에 필요한 Python 라이브러리 목록.
*   `.env`: 데이터베이스 연결 정보, API 키 등 민감한 정보를 저장하는 파일.
