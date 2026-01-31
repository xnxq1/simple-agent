from contextlib import asynccontextmanager
from logging import getLogger

from fastapi import FastAPI

from app.infra.config import settings

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application started")
    yield
    logger.info("Application stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )

    @app.get("/")
    async def root():
        return {
            "status": "ok",
        }

    return app
