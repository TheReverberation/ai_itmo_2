"""End-to-end пайплайн PoC.

Шаги (соответствуют минимальному сценарию из задания):
1. Вход — mock-ticket.
2. PII-маскирование, определение темы и риска.
3. Поиск релевантного фрагмента базы знаний.
4. Черновик ответа или маршрут обработки.
5. Запись лога решения (audit).
6. Рискованный / low-confidence тикет эскалируется оператору,
   а не закрывается автоматически.

Упрощение PoC: шаги вызываются синхронно по порядку; в целевой архитектуре
шаги 3–4 асинхронные, за очередью (см. docs/architecture.md).
"""
from pathlib import Path

from .audit import AuditLog
from .classifier import classify
from .generator import generate_draft
from .loadguard import IncidentDeduplicator, LLMBudget
from .pii import mask_pii
from .retrieval import KnowledgeBase

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"


class TicketPipeline:
    def __init__(self, kb_path: Path | None = None, out_dir: Path | None = None,
                 llm_available: bool = True,
                 budget: LLMBudget | None = None,
                 deduplicator: IncidentDeduplicator | None = None):
        self.kb = KnowledgeBase(kb_path or DATA_DIR / "kb.json")
        out = Path(out_dir or OUT_DIR)
        self.audit = AuditLog(out / "audit_log.jsonl")
        self.operator_inbox = AuditLog(out / "operator_inbox.jsonl")
        self.llm_available = llm_available
        # Опциональные предохранители пиковой нагрузки (см. loadguard.py).
        self.budget = budget
        self.deduplicator = deduplicator

    def process(self, ticket: dict) -> dict:
        """Обрабатывает тикет, возвращает итоговое решение."""
        # 1-2. PII + классификация (быстрый путь).
        masked_text, pii_found = mask_pii(ticket["text"])
        cls = classify(masked_text, pii_found)

        decision = {
            "ticket_id": ticket["id"],
            "channel": ticket.get("channel", "unknown"),
            "topic": cls.topic,
            "confidence": cls.confidence,
            "risky": cls.risky,
            "risk_reasons": cls.risk_reasons,
            "pii_masked": pii_found,
        }

        # 6. Рискованный / low-confidence -> оператор, без автоответа.
        if cls.risky:
            decision.update(action="escalate_to_operator", draft=None, kb_source=None)
            self.operator_inbox.write(
                {"ticket_id": ticket["id"], "masked_text": masked_text,
                 "topic": cls.topic, "risk_reasons": cls.risk_reasons}
            )
            self.audit.write(decision)
            return decision

        # Дедупликация при инцидентах: если тикет — «последователь» уже
        # известного кластера с готовым ответом, переиспользуем его БЕЗ LLM.
        if self.deduplicator is not None:
            dd = self.deduplicator.assign(ticket["id"], cls.topic, masked_text)
            decision.update(cluster_id=dd.cluster_id, cluster_size=dd.cluster_size)
            if not dd.is_leader and dd.cached_answer is not None:
                decision.update(action="draft_for_operator", draft=dd.cached_answer,
                                kb_source=None, generator_status="dedup_cached")
                self.operator_inbox.write(
                    {"ticket_id": ticket["id"], "masked_text": masked_text,
                     "topic": cls.topic, "draft": dd.cached_answer,
                     "cluster_id": dd.cluster_id, "note": "dedup_cached"}
                )
                self.audit.write(decision)
                return decision

        # 3. Retrieval (в целевой архитектуре — асинхронно).
        results = self.kb.search(masked_text, top_k=1)
        score, article = results[0] if results else (0.0, None)

        # Бюджет LLM: если денег на вызов не хватает, деградируем до
        # retrieval-only (флаг llm_available=False для этого тикета).
        llm_ok = self.llm_available
        if self.budget is not None and not self.budget.can_spend():
            llm_ok = False
            decision["budget_exhausted"] = True

        # 4. Черновик ответа (mock-LLM с флагом доступности/бюджета).
        # masked_text передаём для изоляции в промпте (см. generator._build_prompt).
        gen = generate_draft(cls.topic, article, score, llm_ok, masked_text)
        # Списываем бюджет только за реальный LLM-вызов (draft_ready).
        if self.budget is not None and gen["status"] == "draft_ready":
            self.budget.charge()
        # Лидер кластера кэширует свой ответ для последователей.
        if self.deduplicator is not None and gen.get("draft"):
            self.deduplicator.set_answer(decision["cluster_id"], gen["draft"])

        if gen["status"] in ("draft_ready", "fallback_retrieval_only"):
            # Suggest-режим: черновик уходит оператору на подтверждение,
            # автоотправка пользователю — только на этапе 3 раскатки.
            decision.update(action="draft_for_operator", draft=gen["draft"],
                            kb_source=gen["source"], retrieval_score=score,
                            generator_status=gen["status"])
            self.operator_inbox.write(
                {"ticket_id": ticket["id"], "masked_text": masked_text,
                 "topic": cls.topic, "draft": gen["draft"], "kb_source": gen["source"]}
            )
        else:
            decision.update(action="escalate_to_operator", draft=None,
                            kb_source=None, retrieval_score=score,
                            generator_status=gen["status"])
            self.operator_inbox.write(
                {"ticket_id": ticket["id"], "masked_text": masked_text,
                 "topic": cls.topic, "note": gen["status"]}
            )

        # 5. Лог решения.
        self.audit.write(decision)
        return decision
