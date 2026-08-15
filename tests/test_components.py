"""Юнит-тесты компонентов, которые до этого проверялись только через пайплайн:
retrieval, PII-маскирование и классификатор — каждый на своём контракте.
"""
from app.config import Settings
from app.services.classifier import classify
from app.services.pii import mask_pii
from app.services.retrieval import KnowledgeBaseIndex


def test_retrieval_ranks_relevant_article_first(kb_index, settings):
    article, score = kb_index.search("не могу войти, забыл пароль")
    assert article["id"] == "kb-1"  # «Восстановление пароля»
    assert score > settings.retrieval_threshold


def test_retrieval_returns_nothing_for_offtopic_query(kb_index, settings):
    # ниже порога → пайплайн уводит тикет в очередь, а не отвечает наугад
    article, score = kb_index.search("рецепт борща с пампушками")
    assert article is None
    assert score < settings.retrieval_threshold


def test_retrieval_survives_query_without_tokens(kb_index):
    assert kb_index.search("?!") == (None, 0.0)


def test_empty_index_returns_no_article():
    assert KnowledgeBaseIndex([]).search("забыл пароль") == (None, 0.0)


def test_pii_masks_email_and_phone_together():
    masked, found = mask_pii("почта a@b.com, телефон +7 916 123-45-67")
    assert found == ["email", "phone"]
    assert "a@b.com" not in masked and "916" not in masked
    assert "[EMAIL]" in masked and "[PHONE]" in masked


def test_pii_leaves_clean_text_untouched():
    text = "обычный текст без персональных данных"
    assert mask_pii(text) == (text, [])


def test_classifier_reports_zero_confidence_when_nothing_matched():
    cls = classify("абракадабра")
    assert cls["topic"] == "unknown"
    assert cls["confidence"] == 0.0  # пайплайн эскалирует такой тикет


def test_never_auto_topics_accepts_both_env_forms(monkeypatch):
    # в .env пишут «payment,legal», в JSON-виде — ["payment","legal"]; работать
    # должны оба, иначе опечатка в конфиге роняет старт с невнятной ошибкой
    monkeypatch.setenv("NEVER_AUTO_TOPICS", "payment, legal")
    assert Settings(_env_file=None).never_auto_topics == {"payment", "legal"}
    monkeypatch.setenv("NEVER_AUTO_TOPICS", '["payment","legal"]')
    assert Settings(_env_file=None).never_auto_topics == {"payment", "legal"}


def test_risk_is_detected_independently_of_topic():
    # тема не распознана, но риск-слова есть → эскалация всё равно сработает
    cls = classify("взломали аккаунт, не я заходил")
    assert cls["risk"] == "high"
    assert cls["risk_reasons"] == ["security"]
