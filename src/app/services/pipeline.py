"""Полный путь одного тикета: PII → классификация → [дедуп] → retrieval →
черновик (с output gate) / эскалация.

Решение пишется в append-only таблицу decisions (аналог прежнего decision log).

Предохранители пиковой нагрузки (services/loadguard.py) опциональны:
- dedup: последователь кластера инцидента переиспользует ответ лидера без LLM;
- budget: исчерпан ₽-бюджет → ступень «retrieval-only» лестницы деградации
  (шаблон из KB-статьи без LLM-вызова), а не отказ в обслуживании.
"""
from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import Decision, Ticket
from app.services.classifier import classify
from app.services.llm import (
    DraftService,
    LLMUnavailableError,
    passes_output_gate,
    retrieval_only_draft,
)
from app.services.loadguard import IncidentDeduplicator, LLMBudget
from app.services.pii import mask_pii
from app.services.retrieval import KnowledgeBaseIndex


async def process_ticket(
    ticket: Ticket,
    kb: KnowledgeBaseIndex,
    llm: DraftService,
    session: AsyncSession,
    settings: Settings,
    dedup: IncidentDeduplicator | None = None,
    budget: LLMBudget | None = None,
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
        # Дедупликация при инцидентах: рискованные тикеты сюда не доходят,
        # последователь кластера с готовым ответом обслуживается без LLM.
        dd = None
        if dedup is not None:
            dd = dedup.assign(ticket.ticket_id, cls["topic"], masked)
            if not dd.is_leader and dd.cached_answer is not None:
                decision.action = "suggest_to_operator"
                decision.draft = dd.cached_answer
                decision.reason = (
                    f"инцидент-дедупликация: кластер {dd.cluster_id} "
                    f"(размер {dd.cluster_size}), переиспользован ответ лидера "
                    f"{dd.leader_ticket_id} без LLM-вызова"
                )
                session.add(decision)
                await session.commit()
                return decision

        article, score = kb.search(masked)
        if article is None or score < settings.retrieval_threshold:
            decision.action = "route_to_queue"
            decision.reason = "нет релевантной статьи KB"
        else:
            decision.kb_article_id = article["id"]
            decision.kb_title = article["title"]
            decision.kb_score = score
            if budget is not None and not budget.can_spend():
                # Лестница деградации: LLM → retrieval-only шаблон → оператор.
                decision.action = "suggest_to_operator"
                decision.draft = retrieval_only_draft(article)
                decision.reason = (
                    f"₽-бюджет LLM исчерпан ({budget.spent_rub:.0f} из "
                    f"{budget.limit_rub:.0f} ₽) → retrieval-only черновик без LLM"
                )
            else:
                try:
                    draft = await llm.draft(masked, article)
                    if budget is not None:
                        budget.charge()
                    if not passes_output_gate(draft, article):
                        # Output gate: PII-плейсхолдеры или отсутствие grounding —
                        # черновик не уходит никому, тикет — оператору.
                        decision.action = "escalate_to_operator"
                        decision.reason = ("output gate: черновик не прошёл проверку "
                                           "(PII/groundedness) → оператор, не пользователь")
                    else:
                        if dd is not None and dd.is_leader:
                            # Только лидер кэширует ответ — последователь не
                            # перезаписывает кластерный ответ своим.
                            dedup.set_answer(dd.cluster_id, draft)
                        if cls["topic"] in settings.never_auto_topics:
                            decision.action = "suggest_to_operator"
                            decision.draft = draft
                            decision.reason = "категория из списка «только с оператором»"
                        elif "card" in pii:
                            # Платёжные PII — только через оператора, даже в маске
                            decision.action = "suggest_to_operator"
                            decision.draft = draft
                            decision.reason = ("в тикете платёжные данные (карта) → "
                                               "только через оператора")
                        else:
                            decision.action = "auto_draft"
                            decision.draft = draft
                            decision.reason = "типовой тикет, высокая уверенность"
                except LLMUnavailableError as exc:
                    decision.action = "route_to_queue"
                    decision.reason = (f"LLM недоступен ({exc}) → graceful degradation: "
                                       "шаблон-подтверждение пользователю, тикет оператору")

    session.add(decision)
    await session.commit()
    return decision
