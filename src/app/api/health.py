from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
    llm = request.app.state.draft_service
    return {
        "status": "ok",
        "llm_mode": llm.mode,
        "llm_breaker": "open" if llm.breaker_open else "closed",
        "llm_calls_today": llm.calls_today,
    }
