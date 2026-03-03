"""FastAPI app factory."""

from fastapi import FastAPI

from app.api.routes import health, voices, dataset


def create_app() -> FastAPI:
    app = FastAPI(title="VoiceLab", version="0.1.0")
    app.include_router(health.router)
    app.include_router(voices.router)
    app.include_router(dataset.router)
    return app
