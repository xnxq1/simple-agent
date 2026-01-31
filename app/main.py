from contextlib import asynccontextmanager
from logging import getLogger

from fastapi import APIRouter, FastAPI

from app.infra.config import Settings

logger = getLogger(__name__)


class AppBuilder:
    def __init__(self, routers: list[APIRouter], settings: Settings):
        self.settings = settings
        self.routers = routers

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        logger.info("Application started")
        yield
        logger.info("Application stopped")

    def create_app(self) -> FastAPI:
        app = FastAPI(
            title=self.settings.app_name,
            description=self.settings.app_name,
            version=self.settings.app_version,
            lifespan=self.lifespan,
        )
        for router in self.routers:
            app.include_router(router)

        @app.get("/")
        async def root():
            return {
                "status": "ok",
            }

        return app
