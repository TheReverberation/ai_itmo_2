#!/usr/bin/env python3
"""Demo пиковой нагрузки: дедупликация инцидента + бюджет LLM.

Моделируем всплеск: много почти одинаковых тикетов об одном сбое доставки
+ несколько прочих. Показываем, что:
  - лидер кластера генерируется через LLM (тратит бюджет),
  - последователи переиспользуют ответ лидера БЕЗ LLM-вызова (dedup_cached),
  - при исчерпании бюджета генерация деградирует до retrieval-only.

Запуск из корня репозитория:
    python3 poc/demo_loadguard.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.loadguard import IncidentDeduplicator, LLMBudget  # noqa: E402
from app.pipeline import TicketPipeline  # noqa: E402


def make_incident_burst() -> list:
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
            "id": f"inc-{i:02d}",
            "channel": ["web", "chat", "email", "mobile"][i % 4],
            "text": variants[i % len(variants)],
        })
    # Прочие тикеты другой темы — не должны попасть в кластер доставки.
    burst.append({"id": "oth-1", "channel": "chat",
                  "text": "Хочу отменить подписку и поменять тариф, как это сделать?"})
    burst.append({"id": "oth-2", "channel": "mobile",
                  "text": "Приложение вылетает при запуске после обновления, ошибка."})
    return burst


def main() -> None:
    tickets = make_incident_burst()
    # Бюджета хватает только на 3 реальных LLM-вызова (по 3 ₽) — дальше деградация.
    budget = LLMBudget(limit_rub=9.0, cost_per_call_rub=3.0)
    dedup = IncidentDeduplicator(similarity_threshold=0.6, window_seconds=600)

    with tempfile.TemporaryDirectory() as tmp:
        pipeline = TicketPipeline(out_dir=Path(tmp), llm_available=True,
                                  budget=budget, deduplicator=dedup)

        print(f"=== Пик: {len(tickets)} тикетов, бюджет {budget.limit_rub:.0f} ₽ "
              f"({budget.cost_per_call_rub:.0f} ₽/LLM-вызов) ===\n")

        llm_calls = dedup_hits = degraded = 0
        cluster_sizes: dict[str, int] = {}  # cluster_id -> итоговый размер
        for t in tickets:
            d = pipeline.process(t)
            status = d.get("generator_status", "-")
            tag = ""
            if status == "dedup_cached":
                dedup_hits += 1
                tag = "♻ переиспользован ответ кластера (без LLM)"
            elif status == "draft_ready":
                llm_calls += 1
                tag = "🤖 LLM-вызов (списан бюджет)"
            elif status == "fallback_retrieval_only":
                degraded += 1
                tag = "⚠ бюджет исчерпан → retrieval-only"
            cluster = d.get("cluster_id", "-")
            if cluster != "-":
                cluster_sizes[cluster] = d.get("cluster_size", cluster_sizes.get(cluster, 0))
            print(f"[{d['ticket_id']}] тема={d['topic']:<12} "
                  f"кластер={cluster:<6} размер={d.get('cluster_size','-'):<3} {tag}")

        # Метрики дедупа (см. docs/monitoring.md): dedup ratio = тикетов на кластер.
        n_clusters = len(cluster_sizes)
        n_clustered = sum(cluster_sizes.values())
        dedup_ratio = n_clustered / n_clusters if n_clusters else 0.0
        biggest = max(cluster_sizes.values()) if cluster_sizes else 0

        print(f"\nИтого: LLM-вызовов={llm_calls}, из кэша дедупа={dedup_hits}, "
              f"деградаций по бюджету={degraded}")
        print(f"Кластеров: {n_clusters}; dedup ratio (тикетов/кластер): "
              f"{dedup_ratio:.1f}; крупнейший кластер: {biggest} тикетов")
        print(f"Потрачено бюджета: {budget.spent_rub:.0f} / {budget.limit_rub:.0f} ₽ "
              f"({budget.calls} вызовов)")
        print(f"Без дедупа и бюджета потребовалось бы до {len(tickets)} LLM-вызовов — "
              f"экономия налицо.")


if __name__ == "__main__":
    main()
