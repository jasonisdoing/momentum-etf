# 06. 스파이크 구현 메모

## 이번 스파이크에서 추가한 구조

- `web/`
  - Next.js App Router 기반 최소 Node.js + TypeScript 앱
  - 검증용 경로:
    - `/`
    - `/dashboard`
    - `/cash`
    - `/snapshots`
    - `/market`

- `docker-compose.hybrid.yml`
  - 기존 운영용 `docker-compose.yml`와 분리된 하이브리드 스파이크 실행 파일

- `infra/hybrid/nginx.conf`
  - 같은 도메인 경로 구조를 흉내 내기 위한 로컬 프록시 설정
  - `/py/*` -> Python(Streamlit)
  - 나머지 `/` -> Node.js

## 의도

- 운영 배포 파일은 건드리지 않는다.
- 하이브리드 공존 가능성만 별도 세트로 검증한다.

## 실행 의도

하이브리드 compose가 정상 실행되면 다음 구조를 확인할 수 있다.

- `http://localhost:8080/` -> Node.js
- `http://localhost:8080/dashboard` -> Node.js
- `http://localhost:8080/cash` -> Node.js
- `http://localhost:8080/snapshots` -> Node.js
- `http://localhost:8080/market` -> Node.js
- `http://localhost:8080/py/...` -> Python(Streamlit)

## 이번 턴에서 확인된 결과

- `npm install` 완료
- `npm run build` 완료
- Next.js 최소 경로 생성 확인
  - `/`
  - `/dashboard`
  - `/cash`
  - `/snapshots`
  - `/market`
- Python 앱을 `/py` base path로 실행 가능함을 확인
- Node 앱을 별도 포트에서 실행 가능함을 확인
- `http://localhost:8080/` 에서 Node 화면 실제 확인
- `http://localhost:8080/py/` 에서 Streamlit 화면 실제 확인
- `http://localhost:8080/py/rank` 에서 Streamlit 하위 라우트 실제 확인
- Node `자산관리(/cash)` 실기능 구현
- `/api/cash/accounts`, `/api/cash/save` 추가
- Node `스냅샷(/snapshots)` 실기능 구현
- `/api/snapshots` 추가
- Node `ETF 마켓(/market)` 실기능 구현
- `/api/market` 추가
- Node `대시보드(/dashboard)` 실기능 구현
- `/api/dashboard`, `/api/fx` 추가
- 루트 `/` 를 앱 런처형 홈으로 교체
- 상단 공통 환율 바 추가
- Node 1차 메뉴 `/dashboard`, `/cash`, `/snapshots`, `/market`를 모두 실제 데이터 기반으로 확인
- 하이브리드 컨테이너 재빌드 및 기동 확인
- nginx가 컨테이너 재생성 이후 이전 Docker 내부 IP를 계속 바라보며 `502 Bad Gateway`를 내는 문제 확인
- `resolver 127.0.0.11` + 변수 기반 `proxy_pass`로 nginx가 컨테이너 재생성 이후에도 새 IP를 다시 해석하도록 수정
- 수정 후 `/dashboard` 와 `/py/` 모두 `200 OK`로 복구 확인

## 이번 턴에서 막힌 지점

- 초기에는 Docker 데몬과 세션 간 포트 검증 제약이 있었지만, 최종적으로 사용자가 로컬 브라우저에서 실제 공존 검증을 완료했다.

## 다음 검증 우선순위

1. 운영 배포 파일과 `nginx-proxy` 경로 분기 방식 설계
2. 운영 서버 또는 스테이징에서 하이브리드 배포 테스트
3. 배포 후 회귀 확인과 2차 UX 개선 우선순위 정리
