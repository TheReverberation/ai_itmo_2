#!/usr/bin/env python3
"""Тесты предохранителей пиковой нагрузки: бюджет LLM + дедупликация.

Запуск из корня репозитория:
    python3 -m unittest discover poc -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.loadguard import IncidentDeduplicator, LLMBudget  # noqa: E402
from app.pipeline import TicketPipeline  # noqa: E402


class TestLLMBudget(unittest.TestCase):
    def test_stops_when_exhausted(self):
        b = LLMBudget(limit_rub=6.0, cost_per_call_rub=3.0)
        self.assertTrue(b.can_spend()); b.charge()
        self.assertTrue(b.can_spend()); b.charge()
        self.assertFalse(b.can_spend(), "после 2 вызовов бюджет должен кончиться")
        self.assertEqual(b.calls, 2)
        self.assertEqual(b.remaining(), 0.0)


class TestDeduplicator(unittest.TestCase):
    def test_similar_tickets_share_cluster(self):
        d = IncidentDeduplicator(similarity_threshold=0.5)
        r1 = d.assign("a", "delivery", "заказ не пришёл трек не обновляется где посылка")
        r2 = d.assign("b", "delivery", "заказ не пришёл трек доставки не обновляется посылка")
        self.assertTrue(r1.is_leader)
        self.assertFalse(r2.is_leader, "похожий тикет — последователь того же кластера")
        self.assertEqual(r1.cluster_id, r2.cluster_id)
        self.assertEqual(r2.cluster_size, 2)

    def test_different_topic_new_cluster(self):
        d = IncidentDeduplicator(similarity_threshold=0.5)
        r1 = d.assign("a", "delivery", "заказ не пришёл где посылка")
        r2 = d.assign("b", "payment", "заказ не пришёл где посылка")  # тот же текст, др. тема
        self.assertNotEqual(r1.cluster_id, r2.cluster_id, "blocking по теме разводит кластеры")

    def test_follower_reuses_leader_answer(self):
        d = IncidentDeduplicator(similarity_threshold=0.5)
        r1 = d.assign("a", "delivery", "заказ не пришёл трек не обновляется посылка")
        d.set_answer(r1.cluster_id, "Проверьте раздел Мои заказы")
        r2 = d.assign("b", "delivery", "заказ не пришёл трек доставки не обновляется посылка")
        self.assertEqual(r2.cached_answer, "Проверьте раздел Мои заказы")

    def test_follower_before_leader_answer_has_no_cache(self):
        """Гонка: последователь пришёл РАНЬШЕ, чем лидер сгенерировал ответ.

        cached_answer ещё None — пайплайн должен пойти обычным путём (retrieval),
        а не отдать пустой черновик из кэша.
        """
        d = IncidentDeduplicator(similarity_threshold=0.5)
        d.assign("a", "delivery", "заказ не пришёл трек не обновляется посылка")
        early = d.assign("b", "delivery", "заказ не пришёл трек доставки не обновляется посылка")
        self.assertFalse(early.is_leader, "второй тикет — последователь того же кластера")
        self.assertIsNone(early.cached_answer, "ответа лидера ещё нет — кэш пуст")


class TestPipelineDedup(unittest.TestCase):
    def test_follower_served_from_cache_without_llm(self):
        """Последователь кластера получает ответ без LLM-вызова (dedup_cached)."""
        dedup = IncidentDeduplicator(similarity_threshold=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            p = TicketPipeline(out_dir=Path(tmp), deduplicator=dedup)
            leader = p.process({"id": "L", "channel": "web",
                                "text": "Заказ не пришёл, трек доставки не обновляется, где посылка?"})
            follower = p.process({"id": "F", "channel": "chat",
                                  "text": "Заказ не пришёл! Трек доставки завис, где посылка?"})
            self.assertEqual(leader["generator_status"], "draft_ready")
            self.assertEqual(follower["generator_status"], "dedup_cached")
            self.assertEqual(leader["cluster_id"], follower["cluster_id"])
            self.assertIsNotNone(follower["draft"])


class TestPipelineBudget(unittest.TestCase):
    def test_budget_exhaustion_degrades_to_retrieval_only(self):
        """Разные тикеты (без дедупа): после исчерпания бюджета — retrieval-only."""
        budget = LLMBudget(limit_rub=3.0, cost_per_call_rub=3.0)  # хватает на 1 вызов
        with tempfile.TemporaryDirectory() as tmp:
            p = TicketPipeline(out_dir=Path(tmp), budget=budget)
            first = p.process({"id": "1", "channel": "web",
                               "text": "Заказ не пришёл, трек доставки не обновляется, где посылка?"})
            second = p.process({"id": "2", "channel": "mobile",
                                "text": "Приложение вылетает при запуске после обновления, ошибка."})
            self.assertEqual(first["generator_status"], "draft_ready")
            self.assertEqual(second["generator_status"], "fallback_retrieval_only")
            self.assertTrue(second.get("budget_exhausted"))
            self.assertEqual(budget.calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
