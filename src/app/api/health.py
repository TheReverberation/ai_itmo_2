from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_draft_service
from app.services.llm import DraftService

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(llm: DraftService = Depends(get_draft_service)) -> dict:
    return {
        "status": "ok",
        "llm_mode": llm.mode,
        "llm_breaker": "open" if llm.breaker_open else "closed",
        "llm_calls_today": llm.calls_today,
    }
