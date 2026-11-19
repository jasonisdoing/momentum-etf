# 맥미니 운영 서버 이전 계획 (임시 문서)

> 완료 후 변경사항을 [server_infrastructure.md](file:///Users/jason/DEV/momentum-etf/docs/server_infrastructure.md)에 반영하고 이 문서는 삭제합니다.

## 목표 구성

| 구분 | 역할 | 실행 항목 |
| --- | --- | --- |
| **맥미니** | 24시간 운영 서버 | MongoDB · Node(3000) · FastAPI(8000) · Cron/Worker · `cloudflared` · GitHub Runner |
| **맥북 프로** | 개발 전용 | `run_local_dev.py`(`APP_TYPE=Local`, cron 비활성) · 필요 시 LAN IP(`:27017`)로 맥미니 DB 접속 |
| **오라클 VM** | 이전 완료 후 해제 | 기존 컨테이너 중지 후 다른 프로젝트에 활용 |

---

## 단계별 진행 체크리스트

### 1단계: 사전 준비 (맥북 프로 + 맥미니)
- [x] 맥북 프로에서 현재 작업 중인 변경사항을 `upgrade` 브랜치에 커밋·푸시
- [x] 맥미니 시스템 설정: 전원 어댑터 연결 시 **자동 잠자기 끄기** + **정전 후 자동 시동** 활성화
- [x] 맥미니 기본 도구 확인/설치: `git`, `python 3.12`, `node`, `docker`(Docker 방식 사용 시) 또는 `.venv`

### 2단계: MongoDB 데이터 이전
- [x] 맥미니에 MongoDB 실행 (포트 `27017`)
- [x] 오라클 VM에서 쓰기/배치 컨테이너(`scheduler`, `fastapi_app`, `node_app`) 중지 (`mongodb`만 유지해 정합성 보장)
- [x] 오라클 VM(또는 기존 `27017` 터널)에서 전체 백업: `mongodump`
- [x] 맥미니 MongoDB로 복원: `mongorestore`
- [x] 컬렉션 건수·인덱스·`.env` 접속 정보(`MONGO_DB_*`) 검증
- [x] 맥북 프로 `.env`의 `MONGO_DB_HOST`를 맥미니 LAN IP(`192.168.68.55`)로 변경해 접속 테스트

### 3단계: 앱 및 24시간 배치 실행
- [x] 맥미니에 `momentum-etf` 저장소 클론 (`upgrade` 브랜치) 및 `.env` 배치 (`APP_BASE_URL=https://invest.jason.ai.kr`)
- [x] 실행 방식 확정 및 `launchd` 상시 실행 등록 (FastAPI `:8000` · Next.js `:3000` · 배치 `server_scheduler.py`)
- [x] 맥미니 로컬 서비스·배치 동작 확인 (`http://127.0.0.1:8000/docs` · `http://127.0.0.1:3000/login` `HTTP 200`, 스케줄러 큐 워커/크론 기동) — *Google OAuth 로그인 후 대시보드·사고팔기 화면은 4단계 터널 연결 후 확인*

### 4단계: Cloudflare Tunnel (`cloudflared`) 연결
- [x] 맥미니에 `cloudflared` 설치 및 Cloudflare 로그인·터널 생성
- [x] 터널 라우팅 설정: `https://invest.jason.ai.kr` → 맥미니 `http://127.0.0.1:3000`
- [x] Cloudflare DNS에서 `invest` 레코드를 기존 오라클 VM A 레코드에서 **Cloudflare Tunnel CNAME**으로 전환
- [x] `cloudflared`를 `launchd` 서비스로 등록(재부팅 시 자동 시작)

### 5단계: 운영 안정성 점검 (외부 접속·배치·중복·연동 확인)
- [x] 슬랙 버튼(`/api/slack/interactions`) 동작 확인 *(외부 웹 접속·구글 로그인은 추후 확인)*
- [x] 오라클 VM 잔여 배치/컨테이너(`scheduler` 등) 완전 중지 확인 — 중복 슬랙 발송 차단
- [ ] 맥미니 `launchd` 서비스(FastAPI·Next·`server_scheduler.py`·`cloudflared`) 로그 및 에러 유무 점검
- [ ] 한국장·미국장 주요 배치(`cache_refresh_*`, `strategy_trade_notify`, `strategy_mix_notify`, `broker_balance_sync`, `slack_mock_order_notify` 등) 정상 실행·중복 없음 확인
- [ ] 맥북 프로(`APP_TYPE=Local`)와 맥미니 동시 사용 시 락·큐 충돌 없음 확인

### 6단계: GitHub Self-Hosted Runner 자동 배포
- [ ] 맥미니에 GitHub Actions Self-Hosted Runner 설치 및 `launchd` 서비스 등록
- [ ] `.github/workflows/deploy.yml`을 맥미니 러너(`runs-on: self-hosted`) 기반 `git pull` → 빌드/재시작으로 전환
- [ ] 테스트 커밋 push 후 자동 배포 정상 완료 확인

### 7단계: 마무리 및 정리
- [ ] 맥미니에서 하루(한국장·미국장 배치 포함) 이상 무중단 동작 확인
- [ ] 오라클 VM에서 기존 `momentum-etf` 잔여 리소스 최종 정리
- [ ] `docs/server_infrastructure.md`에 맥미니·Cloudflare Tunnel·Self-Hosted Runner 구성 반영
- [ ] 이 임시 문서(`docs/mac_mini_migration_plan.md`) 삭제
