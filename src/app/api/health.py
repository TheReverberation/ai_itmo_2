from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_dedup, get_draft_service, get_llm_budget
from app.services.llm import DraftService
from app.services.loadguard import IncidentDeduplicator, LLMBudget

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(
    llm: DraftService = Depends(get_draft_service),
    budget: LLMBudget = Depends(get_llm_budget),
    dedup: IncidentDeduplicator = Depends(get_dedup),
) -> dict:
    return {
        "status": "ok",
        "llm_mode": llm.mode,
        "llm_breaker": "open" if llm.breaker_open else "closed",
        "llm_calls_today": llm.calls_today,
        "llm_budget_spent_rub": budget.spent_rub,
        "llm_budget_limit_rub": budget.limit_rub,
        "dedup_active_clusters": dedup.active_clusters,
    }
