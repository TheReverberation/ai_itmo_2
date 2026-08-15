"""Circuit breaker и дневной бюджет LLM: fail-fast и деградация по стоимости."""
import pytest

from app.config import Settings
from app.services.llm import DraftService, LLMUnavailableError

ARTICLE = {"id": "kb-1", "title": "Тестовая статья", "body": "Тело статьи."}


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, openrouter_api_key=None, **overrides)


async def test_breaker_opens_after_consecutive_failures_and_fails_fast():
    llm = DraftService(make_settings(llm_breaker_failure_threshold=3), available=False)
    for _ in range(3):
        with pytest.raises(LLMUnavailableError, match="LLM API недоступен"):
            await llm.draft("текст", ARTICLE)
    assert llm.breaker_open
    llm.available = True  # API «ожил», но окно ожидания ещё не прошло
    with pytest.raises(LLMUnavailableError, match="circuit breaker"):
        await llm.draft("текст", ARTICLE)


async def test_breaker_half_open_probe_closes_after_window():
    settings = make_settings(llm_breaker_failure_threshold=1, llm_breaker_reset_seconds=0.0)
    llm = DraftService(settings, available=False)
    with pytest.raises(LLMUnavailableError):
        await llm.draft("текст", ARTICLE)
    assert llm.breaker_open
    llm.available = True
    draft = await llm.draft("текст", ARTICLE)  # окно 0 сек → сразу half-open проба
    assert draft
    assert not llm.breaker_open


async def test_budget_exhaustion_degrades_generation():
    llm = DraftService(make_settings(llm_daily_budget_calls=2))
    await llm.draft("текст", ARTICLE)
    await llm.draft("текст", ARTICLE)
    with pytest.raises(LLMUnavailableError, match="бюджет"):
        await llm.draft("текст", ARTICLE)
    assert not llm.breaker_open  # бюджет — политика, а не сбой API
