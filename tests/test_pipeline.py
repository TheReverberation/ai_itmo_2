"""Порт smoke-тестов PoC: happy path, эскалация риска, low-confidence,
PII-маскирование, suggest-режим и деградация при недоступном LLM."""
import pytest
from sqlalchemy import func, select

from app.models import Decision, Ticket
from app.services.llm import DraftService
from app.services.pii import mask_pii
from app.services.pipeline import process_ticket


@pytest.fixture
def run_ticket(seeded_session, kb_index, settings):
    async def _run(ticket_id: str, llm_available: bool = True) -> Decision:
        ticket = await seeded_session.get(Ticket, ticket_id)
        llm = DraftService(settings, available=llm_available)
        return await process_ticket(ticket, kb_index, llm, seeded_session, settings)

    return _run


async def test_happy_path_typical_ticket_gets_draft(run_ticket):
    d = await run_ticket("t-001")  # забыл пароль
    assert d.action == "auto_draft"
    assert d.topic == "account_access"
    assert "пароль" in d.draft.lower()


async def test_risky_payment_ticket_is_escalated_not_answered(run_ticket):
    d = await run_ticket("t-002")  # двойное списание, номер карты
    assert d.action == "escalate_to_operator"
    assert d.risk == "high"
    assert d.draft is None


async def test_low_confidence_goes_to_operator(run_ticket):
    d = await run_ticket("t-004")  # бессодержательный текст
    assert d.action == "escalate_to_operator"
    assert "confidence" in d.reason


async def test_pii_is_masked_before_processing(tickets_data):
    masked, found = mask_pii(tickets_data["t-002"]["text"])
    assert "card" in found
    assert "4276" not in masked


async def test_never_auto_topic_goes_to_suggest_not_auto(run_ticket):
    d = await run_ticket("t-007")  # нейтральный платёжный вопрос (где чек)
    assert d.action == "suggest_to_operator"
    assert d.topic == "payment"
    assert d.draft is not None  # черновик есть, но уходит оператору, не пользователю


async def test_llm_down_degrades_gracefully(run_ticket):
    d = await run_ticket("t-005", llm_available=False)  # типовой тикет, LLM лежит
    assert d.action == "route_to_queue"
    assert "LLM недоступен" in d.reason


async def test_decisions_are_logged(run_ticket, seeded_session):
    await run_ticket("t-001")
    await run_ticket("t-002")
    count = await seeded_session.scalar(select(func.count()).select_from(Decision))
    assert count == 2
    rows = (await seeded_session.scalars(select(Decision))).all()
    assert all(r.action and r.ts for r in rows)
