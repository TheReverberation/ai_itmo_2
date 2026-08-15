"""FastAPI-приложение: PoC-пайплайн обработки тикетов как сервис."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import decisions, health, kb, tickets
from app.config import get_settings
from app.db import init_db, session_factory
from app.models import KBArticle
from app.services.llm import DraftService
from app.services.loadguard import IncidentDeduplicator, LLMBudget
from app.services.retrieval import KnowledgeBaseIndex

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


async def seed_kb_if_empty(session: AsyncSession) -> None:
    if (await session.scalars(select(KBArticle))).first() is not None:
        return
    articles = json.loads((DATA_DIR / "kb.json").read_text(encoding="utf-8"))
    session.add_all(KBArticle(**a) for a in articles)
    await session.commit()


async def build_kb_index(session: AsyncSession) -> KnowledgeBaseIndex:
    rows = (await session.scalars(select(KBArticle).order_by(KBArticle.id))).all()
    return KnowledgeBaseIndex(
        [{"id": a.id, "title": a.title, "body": a.body} for a in rows]
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_db()
    async with session_factory() as session:
        await seed_kb_if_empty(session)
        app.state.kb_index = await build_kb_index(session)
    app.state.draft_service = DraftService(
        settings, available=not settings.llm_simulate_down
    )
    # предохранители пиковой нагрузки: ₽-бюджет LLM и дедупликация инцидентов
    app.state.llm_budget = LLMBudget(
        settings.llm_budget_rub, settings.llm_cost_per_call_rub
    )
    app.state.dedup = IncidentDeduplicator(
        settings.dedup_similarity_threshold, settings.dedup_window_seconds
    )
    yield


app = FastAPI(title="Ticket pipeline", version="0.2.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(tickets.router)
app.include_router(decisions.router)
app.include_router(kb.router)
