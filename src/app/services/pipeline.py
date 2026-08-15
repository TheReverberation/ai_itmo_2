"""Полный путь одного тикета: PII → классификация → retrieval → черновик/эскалация.

Решение пишется в append-only таблицу decisions (аналог прежнего decision log).
"""
from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import Decision, Ticket
from app.services.classifier import classify
from app.services.llm import DraftService, LLMUnavailableError
from app.services.pii import mask_pii
from app.services.retrieval import KnowledgeBaseIndex


async def process_ticket(
    ticket: Ticket,
    kb: KnowledgeBaseIndex,
    llm: DraftService,
    session: AsyncSession,
    settings: Settings,
) -> Decision:
    masked, pii = mask_pii(ticket.text)
    cls = classify(masked)

    decision = Decision(
        decision_id=str(uuid.uuid4())[:8],
        ticket_id=ticket.ticket_id,
        pii_masked=pii,
        topic=cls["topic"],
        risk=cls["risk"],
        risk_reasons=cls["risk_reasons"],
        confidence=cls["confidence"],
    )

    if cls["risk"] == "high":
        decision.action = "escalate_to_operator"
        decision.reason = f"рискованная категория: {', '.join(cls['risk_reasons'])}"
    elif cls["confidence"] < settings.confidence_threshold:
        decision.action = "escalate_to_operator"
        decision.reason = f"low confidence ({cls['confidence']} < {settings.confidence_threshold})"
    else:
        article, score = kb.search(masked)
        if article is None or score < settings.retrieval_threshold:
            decision.action = "route_to_queue"
            decision.reason = "нет релевантной статьи KB"
        else:
            decision.kb_article_id = article["id"]
            decision.kb_title = article["title"]
            decision.kb_score = score
            try:
                draft = await llm.draft(masked, article)
                if cls["topic"] in settings.never_auto_topics:
                    decision.action = "suggest_to_operator"
                    decision.draft = draft
                    decision.reason = "категория из списка «только с оператором»"
                else:
                    decision.action = "auto_draft"
                    decision.draft = draft
                    decision.reason = "типовой тикет, высокая уверенность"
            except LLMUnavailableError:
                decision.action = "route_to_queue"
                decision.reason = ("LLM недоступен → graceful degradation: "
                                   "шаблон-подтверждение пользователю, тикет оператору")

    session.add(decision)
    await session.commit()
    return decision
