import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from utils.env import load_env_if_present

from .routes.backtest import router as backtest_router
from .routes.cash import router as cash_router
from .routes.dashboard import router as dashboard_router
from .routes.import_data import router as import_router
from .routes.market import router as market_router
from .routes.note import router as note_router
from .routes.rank import router as rank_router
from .routes.snapshots import router as snapshots_router
from .routes.stocks import router as stocks_router
from .routes.summary import router as summary_router
from .routes.system import router as system_router
from .routes.weekly import router as weekly_router

load_env_if_present()

logger = logging.getLogger(__name__)

app = FastAPI(title="Momentum ETF Internal API")


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    """클라이언트 입력 오류 → 400."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_request: Request, exc: RuntimeError) -> JSONResponse:
    """서버 내부 오류 → 500. 스택트레이스는 로그에만 남긴다."""
    logger.exception("RuntimeError in request handler")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.include_router(backtest_router)
app.include_router(cash_router)
app.include_router(dashboard_router)
app.include_router(import_router)
app.include_router(market_router)
app.include_router(note_router)
app.include_router(rank_router)
app.include_router(snapshots_router)
app.include_router(stocks_router)
app.include_router(summary_router)
app.include_router(system_router)
app.include_router(weekly_router)


@app.get("/internal/health")
def health() -> dict[str, str]:
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        return JSONResponse(status_code=503, content={"status": "error", "detail": "DB 연결 실패"})

    db.command("ping")
    return {"status": "ok"}
