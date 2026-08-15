"""Конфигурация приложения через переменные окружения / .env."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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

    # предохранители пиковой нагрузки (services/loadguard.py):
    # ₽-бюджет LLM (исчерпан → retrieval-only черновик) и дедупликация инцидентов
    llm_budget_rub: float = 300.0
    llm_cost_per_call_rub: float = 3.0
    dedup_similarity_threshold: float = 0.6
    dedup_window_seconds: int = 600

    # политика решений (см. docs/architecture.md); порог confidence выбран
    # на demo-данных как компромисс «доля автоматизации vs ошибка» (docs/ml.md)
    confidence_threshold: float = 0.65
    retrieval_threshold: float = 0.15
    # NoDecode: разбираем значение из окружения сами (см. валидатор ниже)
    never_auto_topics: Annotated[set[str], NoDecode] = {"payment"}

    @field_validator("never_auto_topics", mode="before")
    @classmethod
    def _accept_comma_separated(cls, value: object) -> object:
        # в .env привычнее «payment,legal», чем JSON-список: принимаем оба вида,
        # иначе привычная запись роняет старт невнятной SettingsError
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                return json.loads(value)
            return {item.strip() for item in value.split(",") if item.strip()}
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
