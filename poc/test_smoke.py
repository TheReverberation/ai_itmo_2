"""Smoke-тесты PoC: happy path, обязательная эскалация рискованного тикета,
low-confidence, PII-маскирование, suggest-режим для категорий «только
с оператором» и деградация при недоступном LLM.

Запуск: python3 -m unittest discover poc
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import KnowledgeBase, MockLLM, load_json, mask_pii, process_ticket

HERE = Path(__file__).parent


class SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase(load_json(HERE / "data" / "kb.json"))
        cls.tickets = {t["id"]: t for t in load_json(HERE / "data" / "tickets.json")}

    def setUp(self):
        self.log = Path(tempfile.mktemp(suffix=".jsonl"))

    def run_ticket(self, ticket_id, llm_available=True):
        llm = MockLLM(available=llm_available)
        return process_ticket(self.tickets[ticket_id], self.kb, llm, self.log)

    def test_happy_path_typical_ticket_gets_draft(self):
        d = self.run_ticket("t-001")  # забыл пароль
        self.assertEqual(d["action"], "auto_draft")
        self.assertEqual(d["topic"], "account_access")
        self.assertIn("пароль", d["draft"].lower())

    def test_risky_payment_ticket_is_escalated_not_answered(self):
        d = self.run_ticket("t-002")  # двойное списание, номер карты
        self.assertEqual(d["action"], "escalate_to_operator")
        self.assertEqual(d["risk"], "high")
        self.assertIsNone(d["draft"])

    def test_low_confidence_goes_to_operator(self):
        d = self.run_ticket("t-004")  # бессодержательный текст
        self.assertEqual(d["action"], "escalate_to_operator")
        self.assertIn("confidence", d["reason"])

    def test_pii_is_masked_before_processing(self):
        masked, found = mask_pii(self.tickets["t-002"]["text"])
        self.assertIn("card", found)
        self.assertNotIn("4276", masked)

    def test_never_auto_topic_goes_to_suggest_not_auto(self):
        d = self.run_ticket("t-007")  # нейтральный платёжный вопрос (где чек)
        self.assertEqual(d["action"], "suggest_to_operator")
        self.assertEqual(d["topic"], "payment")
        self.assertIsNotNone(d["draft"])  # черновик есть, но уходит оператору, не пользователю

    def test_llm_down_degrades_gracefully(self):
        d = self.run_ticket("t-005", llm_available=False)  # типовой тикет, LLM лежит
        self.assertEqual(d["action"], "route_to_queue")
        self.assertIn("LLM недоступен", d["reason"])

    def test_decisions_are_logged(self):
        self.run_ticket("t-001")
        self.run_ticket("t-002")
        records = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 2)
        self.assertTrue(all("action" in r and "ts" in r for r in records))


if __name__ == "__main__":
    unittest.main()
