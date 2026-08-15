"""Генерация черновика ответа.

Упрощение PoC: шаблонный mock-«LLM», собирающий ответ из фрагмента KB.
В целевой архитектуре: LLM API за circuit breaker, вход — только
маскированный текст + retrieval-контекст (см. docs/architecture.md).

Флаг llm_available моделирует недоступность LLM API для fallback-сценария.
"""

RETRIEVAL_THRESHOLD = 0.1  # ниже — нет grounding, генерацию не запускаем


def generate_draft(topic: str, article: dict | None, score: float, llm_available: bool = True) -> dict:
    """Возвращает {'status': ..., 'draft': ..., 'source': ...}."""
    if not llm_available:
        # Fallback: LLM недоступен -> retrieval-only ответ или эскалация.
        if article and score >= RETRIEVAL_THRESHOLD:
            return {
                "status": "fallback_retrieval_only",
                "draft": (
                    "Здравствуйте! Возможно, вам поможет статья базы знаний: "
                    f"«{article['title']}». {article['body']}"
                ),
                "source": article["id"],
            }
        return {"status": "fallback_escalate", "draft": None, "source": None}

    if not article or score < RETRIEVAL_THRESHOLD:
        # Нет grounding — не генерируем (риск галлюцинаций), отдаём оператору.
        return {"status": "no_grounding_escalate", "draft": None, "source": None}

    draft = (
        f"Здравствуйте! Спасибо за обращение (тема: {topic}). "
        f"{article['body']} "
        "Если это не решит проблему — ответьте на это сообщение, и мы подключим специалиста."
    )
    return {"status": "draft_ready", "draft": draft, "source": article["id"]}
