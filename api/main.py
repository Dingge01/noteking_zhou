"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.video import router as video_router
from api.routes.config import router as config_router

app = FastAPI(
    title="NoteKing API",
    description=(
        "The ultimate video/blog to learning notes API. "
        "Supports 30+ platforms with 13 output templates."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_router)
app.include_router(config_router)


@app.get("/")
async def root():
    return {
        "name": "NoteKing API",
        "version": "0.1.0",
        "docs": "/docs",
        "templates": "/api/v1/templates",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


def start():
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
