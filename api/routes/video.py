"""Video processing API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, BackgroundTasks

from api.models.schemas import (
    VideoRequest, VideoResponse, BatchRequest, BatchResponse,
    TemplateInfo, ErrorResponse,
)
from core.config import AppConfig
from core import summarize
from core.templates import TEMPLATE_LIST

router = APIRouter(prefix="/api/v1", tags=["video"])

_config = AppConfig.load()


@router.post("/summarize", response_model=VideoResponse)
async def summarize_video(req: VideoRequest):
    """Process a single video and generate notes."""
    try:
        result = summarize(
            url=req.url,
            template=req.template,
            config=_config,
            custom_prompt=req.custom_prompt,
            use_cache=req.use_cache,
        )
        return VideoResponse(
            title=result.get("title", ""),
            content=result.get("content", ""),
            template=result.get("template", req.template),
            source=result.get("source", ""),
            platform=result.get("platform", ""),
            url=result.get("url", req.url),
            duration=result.get("duration", 0),
            output_file=result.get("output_file", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchResponse)
async def batch_summarize(req: BatchRequest):
    """Process a playlist/collection."""
    try:
        result = summarize(
            url=req.url,
            template=req.template,
            config=_config,
        )
        return BatchResponse(
            title=result.get("title", ""),
            content=result.get("content", ""),
            template=result.get("template", req.template),
            total=result.get("total", 1),
            completed=result.get("completed", 1),
            failed=result.get("failed", []),
            output_file=result.get("output_file", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates", response_model=list[TemplateInfo])
async def list_templates():
    """List all available output templates."""
    return [TemplateInfo(**t) for t in TEMPLATE_LIST]


@router.get("/info")
async def get_video_info_endpoint(url: str):
    """Get video metadata without processing."""
    from core.downloader import get_video_info as fetch_info

    try:
        meta = fetch_info(url, _config)
        return {
            "title": meta.title,
            "uploader": meta.uploader,
            "duration": meta.duration,
            "thumbnail": meta.thumbnail,
            "has_subtitles": meta.has_subtitles,
            "is_playlist": meta.is_playlist,
            "entry_count": meta.entry_count,
            "chapters": meta.chapters,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transcript")
async def get_transcript(url: str):
    """Get only the transcript text."""
    from core import get_transcript as fetch_transcript

    try:
        text = fetch_transcript(url, _config)
        return {"url": url, "transcript": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
