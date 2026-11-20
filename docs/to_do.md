# 할 일

## [완료] 맥미니 scheduler launchd 종료 유예 시간(ExitTimeOut) 늘리기

- **무엇:** 맥미니의 launchd 설정 `com.momentum-etf.scheduler` (`/Users/jason/Library/LaunchAgents/com.momentum-etf.scheduler.plist`) 에 `ExitTimeOut` 을 90(초)으로 설정.
- **왜:** 배포(`launchctl kickstart -k`)·재시작 때 scheduler 는 종료 신호(SIGTERM)를 받으면 data 레인 배치는 바로
  중단해 대기열로 되돌리고, light 레인 배치(사고팔기 신호·자동 주문·주문 결과 알림 등)는 **끝날 때까지 기다린다**
  (`infra/server_scheduler.py`). `launchd` 기본값(20초)이면 대기 중 강제 종료될 수 있어 유예 시간을 늘림.
- **결과 (2026-11-20):**
  - plist 에 `<key>ExitTimeOut</key><integer>90</integer>` 반영 후 `bootout`/`bootstrap` 재적용 완료.
  - `launchctl print gui/$(id -u)/com.momentum-etf.scheduler` 확인 결과 macOS `launchd` 상한으로 **`exit timeout = 60`** 적용됨 (기본 20초 → 60초, 90·1200 모두 60으로 제한됨).
  - 보통 `light` 레인은 20초 안팎이라 60초로 충분하며, 추후 60초 초과 케이스까지 막으려면 `.github/workflows/deploy.yml` 에서 `launchctl kickstart -k` 대신 `SIGTERM` 전송 후 최대 90초 대기하는 방식으로 우회 가능.
