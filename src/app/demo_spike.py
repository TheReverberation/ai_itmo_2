"""Demo пиковой нагрузки: дедупликация инцидента + ₽-бюджет LLM.

Моделируем всплеск: волна почти одинаковых тикетов об одном сбое доставки
+ несколько прочих. Показываем, что:
  - лидер кластера генерируется через LLM (тратит бюджет),
  - последователи переиспользуют ответ лидера БЕЗ LLM-вызова (дедуп),
  - при исчерпании ₽-бюджета генерация деградирует до retrieval-only шаблона.

Запуск:
    uv run python -m app.demo_spike
"""
from __future__ import annotations

import asyncio

from sqlalchemy import select

from app.config import get_settings
from app.db import init_db, session_factory
from app.main import build_kb_index, seed_kb_if_empty
from app.models import Ticket
from app.services.llm import DraftService
from app.services.loadguard import IncidentDeduplicator, LLMBudget
from app.services.pipeline import process_ticket


def make_incident_burst() -> list[dict]:
    """15 почти одинаковых тикетов об одном сбое доставки + 3 прочих."""
    burst = []
    variants = [
        "Заказ не пришёл, трек доставки не обновляется, где посылка?",
        "Заказ не пришёл! Трек доставки завис, где моя посылка?",
        "трек доставки не обновляется уже день, заказ не пришёл, где посылка",
        "Где посылка? Заказ не пришёл, трек доставки не обновляется совсем.",
    ]
    for i in range(15):
        burst.append({
            "id": f"spike-{i:02d}",
            "channel": ["web", "chat", "email", "mobile"][i % 4],
            "text": variants[i % len(variants)],
        })
    # Прочие тикеты других тем — не должны попасть в кластер доставки.
    burst.append({"id": "spike-oth-1", "channel": "chat",
                  "text": "Хочу отменить подписку и поменять тариф, как это сделать?"})
    burst.append({"id": "spike-oth-2", "channel": "mobile",
                  "text": "Приложение вылетает при запуске после обновления, ошибка."})
    # К этому моменту бюджет уже исчерпан → retrieval-only деградация.
    burst.append({"id": "spike-oth-3", "channel": "web",
                  "text": "Не могу войти, забыл пароль, как восстановить доступ?"})
    return burst


async def run() -> None:
    settings = get_settings()
    await init_db()
    tickets = make_incident_burst()
    # Бюджета хватает только на 3 реальных LLM-вызова (по 3 ₽) — дальше деградация.
    budget = LLMBudget(limit_rub=9.0, cost_per_call_rub=3.0)
    dedup = IncidentDeduplicator(settings.dedup_similarity_threshold,
                                 settings.dedup_window_seconds)

    async with session_factory() as session:
        await seed_kb_if_empty(session)
        kb = await build_kb_index(session)
        llm = DraftService(settings, available=True)

        existing = set((await session.scalars(select(Ticket.ticket_id))).all())
        for t in tickets:
            if t["id"] not in existing:
                session.add(Ticket(ticket_id=t["id"], channel=t["channel"],
                                   text=t["text"]))
        await session.commit()

        print(f"=== Пик: {len(tickets)} тикетов, бюджет {budget.limit_rub:.0f} ₽ "
              f"({budget.cost_per_call_rub:.0f} ₽/LLM-вызов) ===\n")

        dedup_hits = degraded = 0
        for t in tickets:
            ticket = await session.get(Ticket, t["id"])
            d = await process_ticket(ticket, kb, llm, session, settings,
                                     dedup=dedup, budget=budget)
            if "дедупликация" in d.reason:
                dedup_hits += 1
                tag = "♻ переиспользован ответ кластера (без LLM)"
            elif "бюджет LLM исчерпан" in d.reason:
                degraded += 1
                tag = "⚠ бюджет исчерпан → retrieval-only шаблон"
            elif d.draft is not None:
                tag = "🤖 LLM-вызов (списан бюджет)"
            else:
                tag = ""
            print(f"[{d.ticket_id}] тема={d.topic:<14} действие={d.action:<21} {tag}")

        cluster_sizes = {
            c.id: len(c.members)
            for c in dedup._clusters  # демо показывает внутренности дедупа
        }

        n_clusters = len(cluster_sizes)
        n_clustered = sum(cluster_sizes.values())
        dedup_ratio = n_clustered / n_clusters if n_clusters else 0.0
        biggest = max(cluster_sizes.values()) if cluster_sizes else 0

        print(f"\nИтого: LLM-вызовов={budget.calls}, из кэша дедупа={dedup_hits}, "
              f"деградаций по бюджету={degraded}")
        print(f"Кластеров: {n_clusters}; dedup ratio (тикетов/кластер): "
              f"{dedup_ratio:.1f}; крупнейший кластер: {biggest} тикетов")
        print(f"Потрачено бюджета: {budget.spent_rub:.0f} / {budget.limit_rub:.0f} ₽")
        print(f"Без дедупа и бюджета потребовалось бы до {len(tickets)} LLM-вызовов.")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
