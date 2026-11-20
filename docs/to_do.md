# 할 일

## 맥미니 scheduler launchd 종료 유예 시간(ExitTimeOut) 90초로 늘리기

- **무엇:** 맥미니의 launchd 설정 `com.momentum-etf.scheduler` 에 `ExitTimeOut` 을 90(초)으로 넣는다.
  이 설정 파일은 저장소 밖(맥미니)에 있고 `infra/cron/install.sh` 에도 이 값이 없어 맥미니에서 직접 바꿔야 한다.
- **왜:** 배포(`launchctl kickstart -k`)·재시작 때 scheduler 는 종료 신호(SIGTERM)를 받으면 data 레인 배치는 바로
  중단해 대기열로 되돌리고, light 레인 배치(사고팔기 신호·자동 주문·주문 결과 알림 등)는 **끝날 때까지 기다린다**
  (`infra/server_scheduler.py`). light 는 중간에 끊으면 슬랙 메시지 중복 등 정합성이 깨질 수 있기 때문이다.
  그런데 launchd 는 종료 신호 뒤 `ExitTimeOut`(기본 20초)이 지나면 강제 종료(SIGKILL)한다. light 배치는 보통
  20초 안팎이지만 `strategy_trade_notify` 는 최대 61초까지 걸린 기록이 있어(2026-11 실측) 기본값이면 기다리는
  도중 강제 종료될 수 있다. 90초면 최장 실측치에 여유를 둔 값이다.
- **어떻게:**
  1. plist 위치 확인: `launchctl print gui/$(id -u)/com.momentum-etf.scheduler | grep path`
  2. 그 plist 의 최상위 `<dict>` 안에 추가:
     ```xml
     <key>ExitTimeOut</key>
     <integer>90</integer>
     ```
  3. 다시 읽기:
     ```bash
     launchctl bootout gui/$(id -u) <plist 경로>
     launchctl bootstrap gui/$(id -u) <plist 경로>
     ```
  4. 확인: `launchctl print gui/$(id -u)/com.momentum-etf.scheduler | grep -i "exit timeout"`
- **완료 후:** 이 항목을 지운다.
