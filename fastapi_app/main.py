from fastapi import FastAPI

from utils.env import load_env_if_present

from .routes.rank import router as rank_router
from .routes.summary import router as summary_router
from .routes.system import router as system_router
from .routes.weekly import router as weekly_router

load_env_if_present()

app = FastAPI(title="Momentum ETF Internal API")
app.include_router(rank_router)
app.include_router(summary_router)
app.include_router(system_router)
app.include_router(weekly_router)


@app.get("/internal/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
