# 05. 다음 작업 순서

## 바로 다음 액션

`Google OAuth 인증`, `모든 사용자 메뉴 Node 이관`, `시스템 버튼 이관`, `Python UI 제거`, `Streamlit 흔적 정리`까지 끝났다.
이제 다음 작업은 `전역 Tabler 표준화 지속 + 주별/스냅샷/대시보드 완성도 보정 + FastAPI 내부 API 순차 이관 + 배포 정리`다.

## Step 1. 전역 Tabler 표준화 지속

목표:

- 화면별 커스텀 보정 대신 전역 기준을 먼저 정리한다.
- Navbar, page header, section/card, table, form control, button 높이와 간격을 Tabler 기준으로 계속 통일한다.

검증 포인트:

- 화면마다 헤더/툴바/카드 밀도가 다르지 않은가
- 같은 종류의 버튼과 입력 높이가 일관적인가
- 새 화면도 별도 커스텀 없이 Tabler 조합으로 바로 구성 가능한가

## Step 2. 완성도 보정

목표:

- `주별`, `스냅샷`, `대시보드`는 이미 Node로 옮겼다.
- 다음은 계산 정확도와 UX 완성도를 보정한다.

검증 포인트:

- 실제 데이터 조회 성공
- 저장/수정 액션이 Mongo 반영에 성공
- `주별` 집계 결과가 기존 규칙과 얼마나 일치하는가
- `스냅샷`, `대시보드`의 테이블/카드 UX가 충분히 정리됐는가

## Step 3. 로컬 개발 검증

목표:

- 로컬에서는 기본적으로 Docker 없이 검증한다.
- Node는 `npm run dev`, Python은 개별 실행으로 빠르게 확인한다.

검증 포인트:

- Node dev 서버에서 `.env`, `zaccounts`, Mongo 연결이 정상인가
- 화면 수정이 즉시 반영되는가

## Step 4. FastAPI 내부 API 이관 지속

현재 완료:

- `system`
- `weekly`
- `rank`
- `summary`
- `dashboard`
- `market`

다음 목표:

- `Next = 인증/프론트`, `FastAPI = 내부 도메인 API` 구조를 유지한다.
- 남은 API 중 FastAPI로 옮길 가치가 큰 대상(`snapshots`, `stocks`, `cash`, `note`, `import`)의 우선순위를 정한다.

검증 후보:

- `GET/POST /internal/system`
- `GET/POST/PATCH /internal/weekly`
- `GET /internal/rank`
- `GET/POST /internal/summary`
- `GET /internal/dashboard`
- `GET /internal/market`

필수 환경변수:

- `FASTAPI_INTERNAL_URL`
- `FASTAPI_INTERNAL_TOKEN`

## 작업자가 이어서 할 때 첫 문장

다음 작업자는 이 폴더를 읽은 뒤,
`전역 UI는 Tabler 기준으로 표준화하고, 화면별 임시 커스텀을 늘리지 말며, 이제는 Node로 옮긴 화면의 완성도 보정과 배포 정리, 남은 Python 런타임 의존 판단에 집중한다.`
