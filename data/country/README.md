# country

국가별 거래일 캘린더 파일을 저장하는 폴더입니다.

현재 런타임에서는 `pandas_market_calendars`를 직접 사용하지 않고, 아래 파일만 읽습니다.

- `data/country/kor/market_calendars.json`
- `data/country/au/market_calendars.json`
- `data/country/us/market_calendars.json`

## 운영 방식

거래일 캘린더 JSON은 Git으로 관리하고 배포합니다.
서버와 로컬이 같은 파일을 읽어야 배치·백테스트·화면의 거래일 기준이 동일해집니다.

`scripts/update_market_calendars.py`는 `pandas_market_calendars`로 국가별 거래일을 생성해 JSON 파일을 덮어쓰는 수동 갱신 보조 스크립트입니다.
자동 배치에는 연결하지 않습니다.

- 시작일: `CACHE_START_DATE`
- 기본 종료일: 실행일 기준 다음 해 `12-31`

```bash
python scripts/update_market_calendars.py
```

생성 후에는 실제 거래소 휴장일과 일치하는지 확인하고, 문제가 없을 때 JSON 파일을 커밋합니다.

## 파일 형식

```json
{
  "country_code": "kor",
  "calendar": "XKRX",
  "source": "pandas_market_calendars",
  "start_date": "2020-01-01",
  "end_date": "2027-12-31",
  "updated_at": "2026-04-08T22:39:30",
  "trading_days": ["2020-01-02", "2020-01-03"]
}
```

## 주의

- 파일이 없으면 `get_trading_days()`는 즉시 에러를 발생시킵니다.
- 요청 구간이 파일의 `start_date ~ end_date` 범위를 벗어나도 즉시 에러를 발생시킵니다.
