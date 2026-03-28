# 04. 현재 진행 상태

## 완료된 것

- 기존 메뉴 구조를 분석했다.
- Node와 Python 사이의 메뉴 소유권 초안을 확정했다.
- URL 구조 v1 초안을 확정했다.
- 기술 검증을 계획보다 먼저 하기로 방향을 확정했다.
- 관련 요약 문서를 `docs/`에 추가했다.
- Node.js + TypeScript 최소 앱 구조를 `web/`에 추가했다.
- Python을 `/py` base path로 띄우는 실행 명령이 실제로 동작함을 확인했다.
- Node 앱이 별도 포트에서 실제 기동됨을 확인했다.
- 하이브리드 compose 문법 검증을 통과했다.
- `http://localhost:8080/` 에서 Node 루트 화면이 실제로 열림을 확인했다.
- `http://localhost:8080/py/` 에서 Python(Streamlit) 앱이 실제로 열림을 확인했다.
- `http://localhost:8080/py/rank` 에서 Python 하위 라우트가 실제로 열림을 확인했다.
- Python 메뉴 재배치를 실제 반영했다.
- Node `자산관리(/cash)`를 실제 MongoDB 저장/조회 화면으로 구현했고 1차 완료 기준을 통과했다.
- Node `스냅샷(/snapshots)` 1차 리스트/상세 조회 화면을 구현했다.
- Node `ETF 마켓(/market)` 1차 조회 화면을 구현했고 실시간 컬럼과 색상 규칙을 복구했다.
- Node `대시보드(/dashboard)` 1차 화면을 구현했다.
- 루트 `/` 를 Odoo 스타일의 앱 런처형 홈으로 교체했다.
- 상단 공통 환율 바를 구현했다.
- 상단 환율 바, 대시보드, 자산관리, ETF 마켓 화면의 ERP 스타일 1차 정리를 마쳤다.
- nginx가 컨테이너 재생성 뒤 바뀐 Docker 내부 IP를 다시 해석하지 못해 `502 Bad Gateway`가 발생하는 문제를 확인하고 수정했다.
- `http://localhost:8080/dashboard` 와 `http://localhost:8080/py/` 가 수정 후 다시 `200 OK`로 복구됨을 확인했다.
- 하이브리드 컨테이너를 재빌드했고 Node/Python/프록시가 모두 정상 기동함을 확인했다.
- 운영 배포를 실제로 적용했고 `https://etf.dojason.com/` 기준으로 Node 메뉴들이 정상 노출됨을 확인했다.
- 운영 `https://etf.dojason.com/py/` 흰 화면 문제를 수정했고 Streamlit 순위 화면이 정상 복구됨을 확인했다.
- Node `벌크 입력(/import)` 1차 화면과 `/api/import/preview`, `/api/import/save`를 구현했다.
- Node `벌크 입력(/import)`에서 실제 TSV 파싱, 미리보기, Mongo 저장이 성공했고 Python 화면과 반영 결과 일치까지 확인했다.
- 루트 앱 런처와 상단 메뉴에서 `Python 순위` 연결을 제거하고 Node 이전 메뉴 중심으로 정리했다.
- Node `종목 관리(/stocks)` 1차 화면과 `/api/stocks`를 구현했다.
- Node `종목 관리(/stocks)`에서 활성 종목 조회, 버킷 변경, 소프트 삭제가 가능하도록 구현했다.
- `삭제된 종목` 기능을 별도 메뉴가 아니라 `종목 관리(/stocks)` 내부 토글로 통합했다.
- `종목 관리(/stocks)`는 Tabler 라이브러리 기반 UI로 전환했고, 좌측 `Edit` 링크 + 모달 편집 패턴을 적용했다.
- Node 인증을 `Google OAuth + 허용 이메일 화이트리스트` 방식으로 전환했다.
- 로컬에서 Google 로그인 흐름과 허용 이메일 인증이 성공적으로 동작함을 확인했다.
- 전역 레이아웃을 Tabler `navbar` 기준으로 재구성하기 시작했다.
- 전역 본문 폭과 주요 화면 헤더/섹션/폼 밀도를 `Tabler 기준`으로 표준화하는 작업을 시작했다.
- 로컬 개발 원칙을 정리했다.
  - 기본 개발은 `npm run dev` + Python 개별 실행
  - Docker 하이브리드 테스트는 프록시/배포 구조 검증이 필요할 때만 수행

## 아직 하지 않은 것

- `스냅샷(/snapshots)` 2차 UX 정리
- `대시보드(/dashboard)` 2차 정리
- 전역 Tabler 표준화 2차 정리
  - 화면별 잔존 커스텀 CSS 축소
  - 페이지 헤더/카드/툴바/테이블 밀도 일관화
- 남아 있는 Python 메뉴의 Node 이관
  - `주별`
  - `시스템정보`
  - `메모`
  - `AI용 요약`

## 현재 상태 한 줄 요약

`하이브리드 1차 배포, Google OAuth 인증 전환, 종목 관리/삭제된 종목 통합, Tabler 표준화 1차가 끝났고 이제 남은 화면 이관과 전역 표준화 정리가 다음 단계다.`

## 지금 시점의 핵심 미해결 기술 이슈

- Node로 아직 옮기지 않은 Python 화면의 저장/조회 규칙을 어떤 순서로 이관할지
- 로컬 개발 시 Docker 없이도 빠르게 검증하되, 배포 구조 회귀는 어떤 시점에 다시 확인할지
- Streamlit 제거 시점을 언제로 잡을지

## 이번 턴에서 실제 확인한 사실

- Streamlit 1.49.1은 `--server.baseUrlPath`를 지원한다.
- `./.venv/bin/python run.py --server.port 8501 --server.baseUrlPath=py --server.headless true` 실행 시
  `/py` 경로로 기동 메시지가 출력되었다.
- Next.js 최소 앱은 의존성 설치와 `npm run build`를 통과했다.
- `npm run start -- --hostname 127.0.0.1 --port 3001` 실행 시 Node 앱이 정상 기동 메시지를 출력했다.
- `docker compose -f docker-compose.hybrid.yml config`는 통과했다.
- `http://localhost:8080/` 경로에서 Node 화면이 실제로 열렸다.
- `http://localhost:8080/py/` 경로에서 Streamlit 화면이 실제로 열렸다.
- `http://localhost:8080/py/rank` 경로에서 Streamlit 하위 라우트가 실제로 열렸다.
- `http://localhost:8080/dashboard` 경로에서 Node 대시보드가 실제로 열렸다.
- Node `자산관리(/cash)` API 경로(`/api/cash/accounts`, `/api/cash/save`)를 추가했고 빌드를 통과했다.
- Node `스냅샷(/snapshots)` API 경로(`/api/snapshots`)를 추가했고 빌드를 통과했다.
- Node `ETF 마켓(/market)` API 경로(`/api/market`)를 추가했고 빌드를 통과했다.
- Node `대시보드(/dashboard)` API 경로(`/api/dashboard`)와 환율 API(`/api/fx`)를 추가했고 빌드를 통과했다.
- Node 자산관리는 Python 기존 화면과 양방향 저장 일치가 실제로 확인되었다.
- 대시보드 금액 가리기 토글, 상단 환율 바, 루트 앱 런처가 로컬에서 실제 동작한다.
- nginx 프록시에 Docker DNS 재해석 설정을 추가하면 컨테이너 재생성 이후 `502` 문제를 복구할 수 있다.
- 하이브리드 컨테이너 재빌드 후 `node_app`, `python_app`, `hybrid_proxy`가 모두 Up 상태로 올라왔다.
- 운영 `https://etf.dojason.com/py/` 흰 화면의 직접 원인은 `/py/_stcore/*` 프록시 경로 손상이었고, `infra/hybrid/nginx.conf` 수정으로 복구됐다.
- 로컬 `web/` 개발 서버는 기본 포트 `3000`을 사용하며, `8080`은 Docker 하이브리드 프록시 전용 포트다.
- 로컬 `npm run dev`에서도 저장소 루트 `.env`와 `zaccounts`를 읽도록 Node 경로를 보정했다.
- Node `벌크 입력(/import)`는 Python과 동일하게 `계좌별 완전 교체`, `최초 매수일 유지` 규칙으로 동작한다.
- Node `종목 관리(/stocks)`는 계좌 선택, 버킷 변경, 소프트 삭제까지 포함한 1차 기능으로 동작한다.
- Node `종목 관리(/stocks)`는 `등록된 종목 / 삭제된 종목` 토글, 복구, 영구 삭제까지 한 화면에서 처리한다.
- Node 로그인은 `.streamlit/secrets.toml` 비밀번호 대신 Google OAuth를 사용한다.
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `AUTH_ALLOWED_EMAILS`, `AUTH_SESSION_SECRET`이 Node 인증 필수 설정값이다.
- 앞으로 Node UI는 커스텀 테마를 계속 덧붙이지 않고 `Tabler 기본 스타일`로 표준화하는 것을 우선한다.

## 이번 턴의 환경 제약

- 이 제약은 있었지만, 사용자가 로컬 브라우저에서 실제 접속 검증을 완료했다.

## 중요 참고

사용자 의도는 명확하다.

- Python과 Node를 서로 연결하는 중간 장치나 브리지 작업은 우선순위가 아니다.
- 그 시간보다 `남은 Python 기능을 Node로 더 빨리 이관`하는 쪽이 이득이다.
- 로컬 개발은 기본적으로 Docker 없이 진행한다.
- UI 차이가 보이면 화면별로 수동 땜질하지 말고 `Tabler 기준 전역 표준화`로 해결한다.
