from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_dedup, get_draft_service, get_kb_index, get_llm_budget
from app.config import Settings, get_settings
from app.db import get_session
from app.models import Ticket
from app.schemas import DecisionRead, TicketCreate, TicketRead
from app.services.llm import DraftService
from app.services.loadguard import IncidentDeduplicator, LLMBudget
from app.services.pipeline import process_ticket
from app.services.retrieval import KnowledgeBaseIndex

router = APIRouter(tags=["tickets"])


@router.post("/tickets", response_model=DecisionRead, status_code=201)
async def create_ticket(
    payload: TicketCreate,
    session: AsyncSession = Depends(get_session),
    kb: KnowledgeBaseIndex = Depends(get_kb_index),
    llm: DraftService = Depends(get_draft_service),
    settings: Settings = Depends(get_settings),
    dedup: IncidentDeduplicator = Depends(get_dedup),
    budget: LLMBudget = Depends(get_llm_budget),
) -> DecisionRead:
    """Принимает тикет, прогоняет через пайплайн и возвращает решение."""
    ticket = Ticket(
        ticket_id=payload.ticket_id or f"t-{uuid.uuid4().hex[:8]}",
        channel=payload.channel,
        user_id=payload.user_id,
        text=payload.text,
        meta=payload.metadata,
    )
    session.add(ticket)
    try:
        await session.flush()
    except IntegrityError as exc:
        # клиентский ticket_id уже занят (например, повторная отправка) — 409,
        # а не 500; идемпотентный replay — задача продовой интеграции (SELF_REVIEW)
        await session.rollback()
        raise HTTPException(
            status_code=409, detail="тикет с таким ticket_id уже существует"
        ) from exc
    decision = await process_ticket(
        ticket, kb, llm, session, settings, dedup=dedup, budget=budget
    )
    return DecisionRead.from_decision(decision)


@router.get("/tickets/{ticket_id}", response_model=TicketRead)
async def get_ticket(
    ticket_id: str,
    session: AsyncSession = Depends(get_session),
) -> Ticket:
    ticket = await session.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="тикет не найден")
    return ticket
