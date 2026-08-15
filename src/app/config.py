"""Конфигурация приложения через переменные окружения / .env."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "sqlite+aiosqlite:///./app.db"

    # LLM: при наличии ключа — OpenRouter, иначе детерминированный fake-режим
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-4o-mini"
    llm_simulate_down: bool = False

    # деградация: circuit breaker и дневной бюджет генерации (docs/monitoring.md)
    llm_breaker_failure_threshold: int = 3
    llm_breaker_reset_seconds: float = 30.0
    llm_daily_budget_calls: int = 1000

    # политика решений (см. docs/architecture.md)
    confidence_threshold: float = 0.55
    retrieval_threshold: float = 0.15
    never_auto_topics: set[str] = {"payment"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
