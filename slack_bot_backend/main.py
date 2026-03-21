"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import router as api_router
from .config import Settings, get_settings
from .dependencies import build_service_container
from .logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(resolved_settings.log_level)
        if not hasattr(app.state, "services"):
            app.state.services = build_service_container(resolved_settings)
        yield

    app = FastAPI(title=resolved_settings.app_name, lifespan=lifespan)
    app.include_router(api_router, prefix=resolved_settings.api_prefix)
    return app


app = create_app()
