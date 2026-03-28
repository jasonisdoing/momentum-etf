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
- 초기 하이브리드 공존 검증에서 Python UI 라우팅도 실제로 열림을 확인했다.
- Python 메뉴 재배치를 실제 반영했다.
- Node `자산관리(/cash)`를 실제 MongoDB 저장/조회 화면으로 구현했고 1차 완료 기준을 통과했다.
- Node `스냅샷(/snapshots)` 1차 리스트/상세 조회 화면을 구현했다.
- Node `ETF 마켓(/market)` 1차 조회 화면을 구현했고 실시간 컬럼과 색상 규칙을 복구했다.
- Node `대시보드(/dashboard)` 1차 화면을 구현했다.
- 루트 `/` 를 Odoo 스타일의 앱 런처형 홈으로 교체했다.
- 상단 공통 환율 바를 구현했다.
- 상단 환율 바, 대시보드, 자산관리, ETF 마켓 화면의 ERP 스타일 1차 정리를 마쳤다.
- nginx가 컨테이너 재생성 뒤 바뀐 Docker 내부 IP를 다시 해석하지 못해 `502 Bad Gateway`가 발생하는 문제를 확인하고 수정했다.
- `http://localhost:8080/dashboard` 등 주요 경로가 수정 후 다시 `200 OK`로 복구됨을 확인했다.
- 하이브리드 컨테이너를 재빌드했고 Node/Python/프록시가 모두 정상 기동함을 확인했다.
- 운영 배포를 실제로 적용했고 `https://etf.dojason.com/` 기준으로 Node 메뉴들이 정상 노출됨을 확인했다.
- 운영 흰 화면 문제를 수정했고 서비스가 정상 복구됨을 확인했다.
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
- `PageFrame`, `appPageStack`, `appBannerStack`, `appSection` 기준으로 주요 화면의 타이틀/배너/첫 카드 시작 구조를 공통 패턴으로 정리했다.
- `/stocks`의 티커 컬럼은 고정폭 식별자 폰트(`appCodeText`) 기준으로 표준화했다.
- Node `시스템정보(/system)` 1차 조회 화면과 `/api/system`을 구현했다.
- Node `메모(/note)` 1차 화면과 `/api/note`를 구현했다.
- Node `AI용 요약(/summary)` 1차 화면과 `/api/summary`를 구현했다.
- Node `주별(/weekly)` 화면과 `/api/weekly`를 구현했다.
- `주별`은 조회, 수정, 이번주 데이터 집계를 Node에서 직접 처리하도록 옮겼다.
- `scripts/generate_ai_summary.py`를 추가해 Python의 AI용 요약 생성 규칙을 Node에서 재사용한다.
- Python UI는 제거됐고, 관련 라우팅과 배포 구성도 함께 제거했다.
- `시스템` 실행 버튼(`전체 메타데이터 업데이트`, `전체 가격 캐시 업데이트`, `전체 자산 요약 알림 전송`)을 Node로 옮겼다.
- `주별`, `종목 관리`, `자산 관리`의 편집 진입 방식을 좌측 `Edit` + 공통 `AppModal`로 표준화했다.
- 성공 메시지는 전역 Tabler 토스트로 표준화했고, `[그룹-메뉴] 메시지` 규칙을 적용했다.
- 로그인 화면을 Tabler `sign-in` 구조로 정리했다.
- 저장소에서 Python UI/Streamlit 관련 실행 코드, 배포 라우팅, 문서/주석/requirements 흔적을 제거했다.
- 로컬 개발 원칙을 정리했다.
  - 기본 개발은 `npm run dev` + Python 개별 실행
  - Docker 하이브리드 테스트는 프록시/배포 구조 검증이 필요할 때만 수행

## 아직 하지 않은 것

- 운영 배포 최종 반영과 실제 서비스 회귀 확인
- `스냅샷(/snapshots)` 2차 UX 정리
- `대시보드(/dashboard)` 2차 정리
- 전역 Tabler 표준화 2차 정리
  - 화면별 잔존 커스텀 CSS 축소
  - 페이지 헤더/카드/툴바/테이블 밀도 일관화
  - 모바일 헤더/메뉴/본문 레이어 완성도 보정
- `주별(/weekly)`의 집계 정확도 2차 보정
- `AI용 요약`의 Python 생성 로직을 FastAPI 내부 서비스 호출 이후에도 더 줄일지 판단
- 시스템 배치/요약/캐시 업데이트의 Python 런타임 의존 축소 여부 판단

## 현재 상태 한 줄 요약

`모든 사용자 메뉴 이관, Python UI 제거, 내부 API FastAPI 이관은 끝났고, 남은 작업은 완성도 보정·배포 정리·Python 런타임 축소다.`

## 지금 시점의 핵심 미해결 기술 이슈

- 로컬 개발 시 Docker 없이도 빠르게 검증하되, 배포 구조 회귀는 어떤 시점에 다시 확인할지
- 남은 Python 런타임 의존을 어디까지 줄일지
- 전역 Tabler 표준화를 어디까지 공통 컴포넌트로 강제할지

## 이번 턴에서 실제 확인한 사실

- 초기 공존 검증용 Python 실행 확인을 마쳤다.
- Next.js 최소 앱은 의존성 설치와 `npm run build`를 통과했다.
- `npm run start -- --hostname 127.0.0.1 --port 3001` 실행 시 Node 앱이 정상 기동 메시지를 출력했다.
- `docker compose -f docker-compose.hybrid.yml config`는 통과했다.
- `http://localhost:8080/` 경로에서 Node 화면이 실제로 열렸다.
- 초기 공존 검증에서 Python UI 라우팅이 실제로 동작함을 확인했다.
- `http://localhost:8080/dashboard` 경로에서 Node 대시보드가 실제로 열렸다.
- Node `자산관리(/cash)` API 경로(`/api/cash/accounts`, `/api/cash/save`)를 추가했고 빌드를 통과했다.
- Node `스냅샷(/snapshots)` API 경로(`/api/snapshots`)를 추가했고 빌드를 통과했다.
- Node `ETF 마켓(/market)` API 경로(`/api/market`)를 추가했고 빌드를 통과했다.
- Node `대시보드(/dashboard)` API 경로(`/api/dashboard`)와 환율 API(`/api/fx`)를 추가했고 빌드를 통과했다.
- Node 자산관리는 Python 기존 화면과 양방향 저장 일치가 실제로 확인되었다.
- 대시보드 금액 가리기 토글, 상단 환율 바, 루트 앱 런처가 로컬에서 실제 동작한다.
- nginx 프록시에 Docker DNS 재해석 설정을 추가하면 컨테이너 재생성 이후 `502` 문제를 복구할 수 있다.
- 하이브리드 컨테이너 재빌드와 프록시 복구 검증을 완료했다.
- 운영 흰 화면의 직접 원인을 프록시 경로 손상으로 확인했고 수정으로 복구했다.
- 로컬 `web/` 개발 서버는 기본 포트 `3000`을 사용하며, `8080`은 Docker 하이브리드 프록시 전용 포트다.
- 로컬 `npm run dev`에서도 저장소 루트 `.env`와 `zaccounts`를 읽도록 Node 경로를 보정했다.
- Node `벌크 입력(/import)`는 Python과 동일하게 `계좌별 완전 교체`, `최초 매수일 유지` 규칙으로 동작한다.
- Node `종목 관리(/stocks)`는 계좌 선택, 버킷 변경, 소프트 삭제까지 포함한 1차 기능으로 동작한다.
- Node `종목 관리(/stocks)`는 `등록된 종목 / 삭제된 종목` 토글, 복구, 영구 삭제까지 한 화면에서 처리한다.
- Node 로그인은 기존 비밀번호 방식 대신 Google OAuth를 사용한다.
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `AUTH_ALLOWED_EMAILS`, `AUTH_SESSION_SECRET`이 Node 인증 필수 설정값이다.
- `AI용 요약(/summary)`는 Next가 FastAPI 내부 API를 호출하고, FastAPI가 Python 생성 서비스를 재사용하는 구조다.
- 앞으로 Node UI는 커스텀 테마를 계속 덧붙이지 않고 `Tabler 기본 스타일`로 표준화하는 것을 우선한다.
- 주요 화면은 `PageFrame > appPageStack > appSection > appCard` 구조로 맞추고, 화면별 임시 여백 보정을 늘리지 않는다.
- `Edit + 공통 AppModal`은 앞으로 그리드 수정 진입의 기본 표준이다.
- 컬럼 수가 많은 화면(`weekly`, `stocks`, `market`)은 최대폭, 그 외 화면은 고정폭 기준을 사용한다.
- FastAPI를 도입하면 `Next가 인증과 프론트`, `FastAPI가 내부 도메인 API`를 맡는 구조로 간다.
- FastAPI 내부 API 연동을 위해 `FASTAPI_INTERNAL_URL`, `FASTAPI_INTERNAL_TOKEN` 환경변수가 필요하다.
- `system`, `weekly`, `rank`, `summary`, `dashboard`, `market`, `snapshots`, `stocks`, `cash`, `note`, `import`는 FastAPI 내부 API로 이관을 완료했다.
- 현재 구조는 `Next = 인증/프론트/BFF`, `FastAPI = Python 친화 내부 도메인 API` 기준으로 정리됐다.

## 이번 턴의 환경 제약

- 이 제약은 있었지만, 사용자가 로컬 브라우저에서 실제 접속 검증을 완료했다.

## 중요 참고

사용자 의도는 명확하다.

- Python과 Node를 서로 연결하는 중간 장치나 브리지 작업은 우선순위가 아니다.
- 그 시간보다 `남은 Python 기능을 Node로 더 빨리 이관`하는 쪽이 이득이다.
- 로컬 개발은 기본적으로 Docker 없이 진행한다.
- UI 차이가 보이면 화면별로 수동 땜질하지 말고 `Tabler 기준 전역 표준화`로 해결한다.
