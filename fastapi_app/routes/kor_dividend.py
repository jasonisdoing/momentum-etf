"""한국 배당주 화면(`/kor-dividend`) 조회 API.

값은 배치(`scripts/update_kor_dividend_stocks.py`)가 만들어 둔 것을 읽고, 가격이 필요한
파생값(배당률·자사주률·점수)만 조회 시점에 계산한다. 필터·정렬은 화면이 한다.
"""

from fastapi import APIRouter, Depends, HTTPException

from fastapi_app.dependencies import require_internal_token

router = APIRouter(prefix="/internal/kor-dividend", tags=["kor-dividend"])


@router.get("")
def get_kor_dividend(_: None = Depends(require_internal_token)) -> dict:
    from utils.kor_dividend_service import load_kor_dividend_rows

    try:
        return load_kor_dividend_rows()
    except LookupError as exc:
        # 배치가 아직 안 돌았다는 뜻 — 서버 오류가 아니라 준비 안 됨이다.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
