from fastapi import FastAPI

from utils.env import load_env_if_present

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

app = FastAPI(title="Momentum ETF Internal API")
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
    return {"status": "ok"}
