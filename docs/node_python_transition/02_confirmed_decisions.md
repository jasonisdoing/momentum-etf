# 02. 현재까지 확정된 결정

## 메뉴 소유권

### Node.js

- 대시보드
- 자산관리
- 스냅샷
- ETF 마켓

### Python(Streamlit)

- 데이터
  - 주별
  - 벌크 입력
- 계좌분석
  - 순위
  - 종목 관리
  - 삭제된 종목
- 시스템
  - 시스템정보
  - 메모
  - AI용 요약

## 명확히 확정된 포인트

- `원금/현금` 기능의 소유권은 Node.js 쪽이다.
- Python에서는 `원금/현금` 메뉴를 제거하는 방향이다.
- 스냅샷은 1차에서 Node.js에 `리스트 중심 단순 화면`으로 먼저 둔다.
- Node 화면은 가능하면 Python 계산 로직 또는 Python API를 재사용한다.
- 아직 이전하지 않은 기능은 Python에서 유지한다.

## URL 구조 v1

### Node.js

- `/dashboard`
- `/cash`
- `/snapshots`
- `/market`

### Python(Streamlit)

- `/py/weekly`
- `/py/import`
- `/py/rank`
- `/py/stocks`
- `/py/deleted`
- `/py/system`
- `/py/note`
- `/py/summary-for-ai`

### 향후 API

- `/api/*`

## 제거/이전 대상

Python에서 소유권을 넘길 대상:

- 대시보드
- 원금/현금
- 스냅샷
- ETF 마켓
