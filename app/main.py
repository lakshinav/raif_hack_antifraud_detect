import collections.abc
import contextlib
import os

from fastapi import FastAPI

from app.logging_config import app_logger
from app.models import load_llm
from app.routers import check_router, health_router


@contextlib.asynccontextmanager
async def run_lifespan(fastapi_app: FastAPI) -> collections.abc.AsyncIterator[None]:
    fastapi_app.state.llm_client = load_llm()

    server_port = os.getenv("DEV_PORT", "8787")
    app_logger.info("server_url_configured", port=server_port, url=f"http://localhost:{server_port}")
    app_logger.info("docs_url_configured", port=server_port, url=f"http://localhost:{server_port}/docs")

    yield


application = FastAPI(
    title="Red Flag Detector API",
    version="1.0.0",
    description="API для детекции red flags в диалогах.",
    lifespan=run_lifespan,
)

application.include_router(health_router)
application.include_router(check_router)
