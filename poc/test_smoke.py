#!/usr/bin/env python3
"""Smoke-тесты PoC (stdlib unittest, без внешних зависимостей).

Запуск из корня репозитория:
    python3 -m unittest discover poc -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.classifier import classify  # noqa: E402
from app.pii import mask_pii  # noqa: E402
from app.pipeline import DATA_DIR, TicketPipeline  # noqa: E402


def make_pipeline(tmpdir: str, llm_available: bool = True) -> TicketPipeline:
    return TicketPipeline(out_dir=Path(tmpdir), llm_available=llm_available)


class TestPII(unittest.TestCase):
    def test_masks_email_phone_card(self):
        masked, found = mask_pii(
            "почта a@b.ru, тел +7 916 123-45-67, карта 4276 1600 1234 5678"
        )
        self.assertEqual(sorted(found), ["CARD", "EMAIL", "PHONE"])
        self.assertNotIn("a@b.ru", masked)
        self.assertNotIn("4276", masked)


class TestClassifier(unittest.TestCase):
    def test_payment_is_always_risky(self):
        cls = classify("двойное списание оплаты с карты, верните деньги", [])
        self.assertEqual(cls.topic, "payment")
        self.assertTrue(cls.risky, "платёжная категория не должна автозакрываться")

    def test_unknown_is_risky(self):
        cls = classify("птичка синичка на подоконнике", [])
        self.assertEqual(cls.topic, "unknown")
        self.assertTrue(cls.risky)


class TestHappyPath(unittest.TestCase):
    def test_delivery_ticket_gets_draft(self):
        """Happy path: типовой безопасный тикет -> черновик оператору (suggest)."""
        with tempfile.TemporaryDirectory() as tmp:
            decision = make_pipeline(tmp).process(
                {"id": "h1", "channel": "web",
                 "text": "Заказ не пришёл, трек доставки не обновляется, где посылка?"}
            )
            self.assertEqual(decision["action"], "draft_for_operator")
            self.assertIsNotNone(decision["draft"])
            self.assertEqual(decision["kb_source"], "kb-001")
            # Лог решения записан.
            audit = Path(tmp) / "audit_log.jsonl"
            self.assertTrue(audit.exists())
            record = json.loads(audit.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["ticket_id"], "h1")


class TestRiskyPath(unittest.TestCase):
    def test_risky_ticket_escalated_not_autoclosed(self):
        """Risky path (сценарий №6): рискованный тикет -> оператор, без автоответа."""
        with tempfile.TemporaryDirectory() as tmp:
            decision = make_pipeline(tmp).process(
                {"id": "r1", "channel": "email",
                 "text": "Двойное списание оплаты! Карта 4276 1600 1234 5678, "
                         "буду жаловаться юристу."}
            )
            self.assertEqual(decision["action"], "escalate_to_operator")
            self.assertIsNone(decision["draft"])
            self.assertTrue(decision["risky"])
            self.assertIn("CARD", decision["pii_masked"])
            # PII замаскирован в очереди оператора.
            inbox = (Path(tmp) / "operator_inbox.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("4276 1600 1234 5678", inbox)


class TestLLMFallback(unittest.TestCase):
    def test_llm_down_degrades_gracefully(self):
        """Fallback: LLM недоступен -> retrieval-only ответ или эскалация."""
        with tempfile.TemporaryDirectory() as tmp:
            decision = make_pipeline(tmp, llm_available=False).process(
                {"id": "f1", "channel": "web",
                 "text": "Заказ не пришёл, трек доставки не обновляется, где посылка?"}
            )
            self.assertIn(decision["action"], ("draft_for_operator", "escalate_to_operator"))
            self.assertIn(decision["generator_status"],
                          ("fallback_retrieval_only", "fallback_escalate"))


class TestAllMockTickets(unittest.TestCase):
    def test_full_dataset_runs_end_to_end(self):
        tickets = json.loads((DATA_DIR / "tickets.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = make_pipeline(tmp)
            for ticket in tickets:
                decision = pipeline.process(ticket)
                self.assertIn(decision["action"],
                              ("draft_for_operator", "escalate_to_operator"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
