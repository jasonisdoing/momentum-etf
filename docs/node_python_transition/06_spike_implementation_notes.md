# 06. 스파이크 구현 메모

## 이번 스파이크에서 추가한 구조

- `web/`
  - Next.js App Router 기반 최소 Node.js + TypeScript 앱
  - 검증용 경로:
    - `/`
    - `/dashboard`
    - `/import`
    - `/stocks`
    - `/cash`
    - `/snapshots`
    - `/market`

- `docker-compose.hybrid.yml`
  - 기존 운영용 `docker-compose.yml`와 분리된 하이브리드 스파이크 실행 파일

- `infra/hybrid/nginx.conf`
  - 같은 도메인 경로 구조를 흉내 내기 위한 로컬 프록시 설정
  - 나머지 `/` -> Node.js

## 의도

- 초기 스파이크 단계에서는 운영 배포 파일을 건드리지 않는다.
- 이후 실제 운영 배포까지 진행하면서 하이브리드 공존 가능성을 검증했다.

## 실행 의도

하이브리드 compose가 정상 실행되면 다음 구조를 확인할 수 있다.

- `http://localhost:8080/` -> Node.js
- `http://localhost:8080/dashboard` -> Node.js
- `http://localhost:8080/import` -> Node.js
- `http://localhost:8080/stocks` -> Node.js
- `http://localhost:8080/cash` -> Node.js
- `http://localhost:8080/snapshots` -> Node.js
- `http://localhost:8080/market` -> Node.js
- 초기 공존 검증용 라우팅도 함께 확인

## 이번 턴에서 확인된 결과

- `npm install` 완료
- `npm run build` 완료
- Next.js 최소 경로 생성 확인
  - `/`
  - `/dashboard`
  - `/import`
  - `/stocks`
  - `/cash`
  - `/snapshots`
  - `/market`
- Python 앱을 `/py` base path로 실행 가능함을 확인
- Node 앱을 별도 포트에서 실행 가능함을 확인
- `http://localhost:8080/` 에서 Node 화면 실제 확인
- 초기 공존 검증에서 Python UI 라우팅도 실제로 확인
- Node `자산관리(/cash)` 실기능 구현
- `/api/cash/accounts`, `/api/cash/save` 추가
- Node `스냅샷(/snapshots)` 실기능 구현
- `/api/snapshots` 추가
- Node `ETF 마켓(/market)` 실기능 구현
- `/api/market` 추가
- Node `대시보드(/dashboard)` 실기능 구현
- `/api/dashboard`, `/api/fx` 추가
- Node `벌크 입력(/import)` 실기능 구현
- `/api/import/preview`, `/api/import/save` 추가
- Node `종목 관리(/stocks)` 실기능 구현
- `/api/stocks` 추가
- Node `시스템정보(/system)` 1차 조회 화면 구현
- `/api/system` 추가
- Node `메모(/note)` 1차 저장 화면 구현
- `/api/note` 추가
- Node `AI용 요약(/summary)` 1차 화면 구현
- `/api/summary` 추가
- `scripts/generate_ai_summary.py` 추가
- Node `주별(/weekly)` 실기능 구현
- `/api/weekly` 추가
- Node `시스템` 실행 버튼 구현
- `/api/system`에서 메타데이터/가격 캐시/슬랙 요약 실행 추가
- 루트 `/` 를 앱 런처형 홈으로 교체
- 상단 공통 환율 바 추가
- 상단에 CNN 공포탐욕지수 추가
- 토스트를 전역 Tabler 토스트로 표준화
- `stocks`, `weekly`, `cash` 편집 모달을 공통 `AppModal`로 표준화
- Node 1차 메뉴 `/dashboard`, `/import`, `/stocks`, `/cash`, `/snapshots`, `/market`를 모두 실제 데이터 기반으로 확인
- 하이브리드 컨테이너 재빌드 및 기동 확인
- nginx가 컨테이너 재생성 이후 이전 Docker 내부 IP를 계속 바라보며 `502 Bad Gateway`를 내는 문제 확인
- `resolver 127.0.0.11` + 변수 기반 `proxy_pass`로 nginx가 컨테이너 재생성 이후에도 새 IP를 다시 해석하도록 수정
- 수정 후 주요 라우팅이 `200 OK`로 복구 확인
- 운영 `https://etf.dojason.com/` 기준 하이브리드 1차 배포 확인
- 운영 프록시 흰 화면을 `_stcore` 프록시 경로 수정으로 복구 확인
- 로컬 개발 원칙을 `npm run dev` 중심으로 정리
- `web/` 개발 서버에서도 저장소 루트 `.env`와 `zaccounts`를 읽도록 경로 보정
- `벌크 입력`은 Python과 동일하게 계좌별 완전 교체, 최초 매수일 유지 규칙을 따른다.
- Node 인증을 `Google OAuth + 허용 이메일 화이트리스트` 방식으로 전환했다.
- `web/` 개발 서버는 루트 `.env`에서 Google OAuth 인증 설정도 함께 읽는다.
- 주요 화면은 `PageFrame > appPageStack > appSection > appCard` 구조로 표준화하기 시작했다.
- `/stocks` 티커 컬럼은 고정폭 식별자 폰트(`appCodeText`)를 사용한다.
- `주별`의 `이번주 데이터 집계` 버튼은 Python 호출이 아니라 Node에서 직접 처리한다.
- `주별` 집계는 최신 `daily_snapshots` 기준 금액과 환율을 즉시 반영하고, 계산 컬럼은 Node에서 다시 계산한다.
- Python UI, `/py` 라우팅, `python_app`, Streamlit 관련 코드/문서 흔적을 제거했다.
- 내부 API는 FastAPI 기준으로 분리하기 시작했다.
  - 완료: `system`, `weekly`, `rank`, `summary`, `dashboard`, `market`
- 현재 호출 구조는 `브라우저 -> Next -> FastAPI`다.
- 운영과 로컬 모두 `FASTAPI_INTERNAL_URL`, `FASTAPI_INTERNAL_TOKEN`이 필요하다.

## 이번 턴에서 막힌 지점

- 초기에는 Docker 데몬과 세션 간 포트 검증 제약이 있었지만, 최종적으로 사용자가 로컬 브라우저에서 실제 공존 검증을 완료했다.

## 다음 검증 우선순위

1. `주별` 집계 정확도 보정
2. `스냅샷`, `대시보드` UX 완성도 보정
3. 남은 API의 FastAPI 이관 우선순위 결정
4. 전역 Tabler 표준화 지속
5. 운영 배포 회귀 확인
