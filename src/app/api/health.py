from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
    return {"status": "ok", "llm_mode": request.app.state.draft_service.mode}
