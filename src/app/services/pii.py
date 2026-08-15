"""PII-маскирование через Microsoft Presidio: выполняется до любых внешних
вызовов (в т.ч. LLM).

Используются pattern-рекогнайзеры (email / телефон RU / банковская карта
с проверкой Luhn), поэтому NLP-движок — пустой русский токенизатор spaCy,
без скачивания моделей. NER для имён/адресов — целевая архитектура
(docs/architecture.md), добавляется загрузкой ru-модели в тот же движок.
"""
from __future__ import annotations

from functools import lru_cache

import spacy
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    EmailRecognizer,
    PhoneRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

_LANGUAGE = "ru"

# entity Presidio → короткое имя в decision log (совместимо с прежним форматом)
_ENTITY_KINDS = {
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone",
    "CREDIT_CARD": "card",
}

_OPERATORS = {
    entity: OperatorConfig("replace", {"new_value": f"[{kind.upper()}]"})
    for entity, kind in _ENTITY_KINDS.items()
}


class _BlankRuNlpEngine(SpacyNlpEngine):
    """Только токенизация: PII ищут pattern-рекогнайзеры, NER не требуется."""

    def __init__(self):
        super().__init__(models=[{"lang_code": _LANGUAGE, "model_name": "blank"}])

    def load(self) -> None:
        self.nlp = {_LANGUAGE: spacy.blank(_LANGUAGE)}


@lru_cache
def _engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    registry = RecognizerRegistry(supported_languages=[_LANGUAGE])
    registry.add_recognizer(EmailRecognizer(supported_language=_LANGUAGE))
    registry.add_recognizer(
        PhoneRecognizer(supported_language=_LANGUAGE, supported_regions=("RU",))
    )
    registry.add_recognizer(CreditCardRecognizer(supported_language=_LANGUAGE))
    analyzer = AnalyzerEngine(
        nlp_engine=_BlankRuNlpEngine(),
        registry=registry,
        supported_languages=[_LANGUAGE],
    )
    return analyzer, AnonymizerEngine()


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Маскирует PII. Возвращает (текст, найденные типы)."""
    analyzer, anonymizer = _engines()
    results = analyzer.analyze(text=text, language=_LANGUAGE)
    if not results:
        return text, []
    masked = anonymizer.anonymize(
        text=text, analyzer_results=results, operators=_OPERATORS
    ).text
    detected = {r.entity_type for r in results}
    found = [kind for entity, kind in _ENTITY_KINDS.items() if entity in detected]
    return masked, found
