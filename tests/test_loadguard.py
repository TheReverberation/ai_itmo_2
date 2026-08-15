"""Предохранители пиковой нагрузки: ₽-бюджет LLM и дедупликация инцидентов.

Юнит-контракты loadguard + интеграция с пайплайном: последователь кластера
обслуживается без LLM, исчерпанный бюджет деградирует до retrieval-only.
"""
import pytest

from app.models import Ticket
from app.services.llm import DraftService
from app.services.loadguard import IncidentDeduplicator, LLMBudget
from app.services.pipeline import process_ticket

DELIVERY_A = "Заказ не пришёл, трек доставки не обновляется, где посылка?"
DELIVERY_B = "Заказ не пришёл! Трек доставки завис, где моя посылка?"


def test_budget_stops_when_exhausted():
    b = LLMBudget(limit_rub=6.0, cost_per_call_rub=3.0)
    assert b.can_spend()
    b.charge()
    assert b.can_spend()
    b.charge()
    assert not b.can_spend(), "после 2 вызовов бюджет должен кончиться"
    assert b.calls == 2
    assert b.remaining() == 0.0


def test_similar_tickets_share_cluster():
    d = IncidentDeduplicator(similarity_threshold=0.5)
    r1 = d.assign("a", "delivery", "заказ не пришёл трек не обновляется где посылка")
    r2 = d.assign("b", "delivery", "заказ не пришёл трек доставки не обновляется посылка")
    assert r1.is_leader
    assert not r2.is_leader, "похожий тикет — последователь того же кластера"
    assert r1.cluster_id == r2.cluster_id
    assert r2.cluster_size == 2


def test_different_topic_opens_new_cluster():
    d = IncidentDeduplicator(similarity_threshold=0.5)
    r1 = d.assign("a", "delivery", "заказ не пришёл где посылка")
    r2 = d.assign("b", "payment", "заказ не пришёл где посылка")  # тот же текст, др. тема
    assert r1.cluster_id != r2.cluster_id, "blocking по теме разводит кластеры"


def test_follower_reuses_leader_answer():
    d = IncidentDeduplicator(similarity_threshold=0.5)
    r1 = d.assign("a", "delivery", "заказ не пришёл трек не обновляется посылка")
    d.set_answer(r1.cluster_id, "Проверьте раздел Мои заказы")
    r2 = d.assign("b", "delivery", "заказ не пришёл трек доставки не обновляется посылка")
    assert r2.cached_answer == "Проверьте раздел Мои заказы"


def test_follower_before_leader_answer_has_no_cache():
    # гонка: последователь пришёл РАНЬШЕ, чем лидер сгенерировал ответ;
    # cached_answer ещё None — пайплайн идёт обычным путём, а не отдаёт пустоту
    d = IncidentDeduplicator(similarity_threshold=0.5)
    d.assign("a", "delivery", "заказ не пришёл трек не обновляется посылка")
    early = d.assign("b", "delivery", "заказ не пришёл трек доставки не обновляется посылка")
    assert not early.is_leader
    assert early.cached_answer is None


def test_expired_cluster_is_forgotten():
    d = IncidentDeduplicator(similarity_threshold=0.5, window_seconds=600)
    d.assign("a", "delivery", "заказ не пришёл где посылка", now=1000.0)
    late = d.assign("b", "delivery", "заказ не пришёл где посылка", now=1000.0 + 601)
    assert late.is_leader, "кластер старше окна протух — открывается новый"


@pytest.fixture
def run_spike_ticket(seeded_session, kb_index, settings):
    async def _run(ticket_id: str, text: str, *, dedup=None, budget=None):
        ticket = Ticket(ticket_id=ticket_id, channel="chat", text=text)
        seeded_session.add(ticket)
        await seeded_session.commit()
        llm = DraftService(settings, available=True)
        return await process_ticket(ticket, kb_index, llm, seeded_session,
                                    settings, dedup=dedup, budget=budget)

    return _run


async def test_pipeline_follower_served_from_cache_without_llm(run_spike_ticket):
    dedup = IncidentDeduplicator(similarity_threshold=0.5)
    leader = await run_spike_ticket("sp-l", DELIVERY_A, dedup=dedup)
    follower = await run_spike_ticket("sp-f", DELIVERY_B, dedup=dedup)
    assert leader.action == "auto_draft"
    assert follower.action == "suggest_to_operator"
    assert "дедупликация" in follower.reason
    assert follower.draft == leader.draft, "последователь получает ответ лидера"


async def test_pipeline_budget_exhaustion_degrades_to_retrieval_only(run_spike_ticket):
    budget = LLMBudget(limit_rub=3.0, cost_per_call_rub=3.0)  # хватает на 1 вызов
    first = await run_spike_ticket("sp-1", DELIVERY_A, budget=budget)
    second = await run_spike_ticket(
        "sp-2", "Приложение вылетает при запуске после обновления, ошибка.",
        budget=budget,
    )
    assert first.action == "auto_draft"
    assert budget.calls == 1
    assert second.action == "suggest_to_operator"
    assert "бюджет LLM исчерпан" in second.reason
    assert second.draft is not None, "retrieval-only шаблон, а не отказ в обслуживании"


async def test_pipeline_risky_ticket_is_never_deduplicated(run_spike_ticket):
    dedup = IncidentDeduplicator(similarity_threshold=0.5)
    d = await run_spike_ticket(
        "sp-r", "С карты дважды списали оплату, верните деньги!", dedup=dedup,
    )
    assert d.action == "escalate_to_operator"
    assert dedup.active_clusters == 0, "рискованные тикеты не попадают в кластеры"
