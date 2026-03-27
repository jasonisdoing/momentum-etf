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

## 아직 하지 않은 것

- 운영 서버 또는 스테이징 환경에서의 실제 배포 테스트
- 기존 운영용 `docker-compose.yml` 를 하이브리드 구조로 전환하는 배포 파일 설계
- `스냅샷(/snapshots)` 2차 UX 정리
- Node 1차 메뉴 이후의 다음 이관 대상 결정

## 현재 상태 한 줄 요약

`아키텍처 방향과 Node 1차 메뉴 구현은 끝났고, 이제 가장 중요한 다음 단계는 하이브리드 배포 테스트다.`

## 지금 시점의 핵심 미해결 기술 이슈

- 운영 서버의 기존 단일 컨테이너 배포 구조를 하이브리드 구조로 어떻게 안전하게 바꿀지
- `nginx-proxy` 환경에서 `/` 와 `/py` 경로 분기를 어떻게 적용할지
- 운영 배포 시 컨테이너 재생성 이후에도 프록시가 새 컨테이너를 안정적으로 따라가는지

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

## 이번 턴의 환경 제약

- 이 제약은 있었지만, 사용자가 로컬 브라우저에서 실제 접속 검증을 완료했다.

## 중요 참고

사용자 의도는 명확하다.

- 지금은 계획 고도화보다 `실행 가능한 최소 버전`이 먼저다.
- 문서만 더 늘리는 것은 우선순위가 아니다.
- 이 폴더의 문서는 실행 스파이크를 돕기 위한 기준선으로만 사용한다.
