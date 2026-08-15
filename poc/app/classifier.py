"""Классификатор темы и риска.

Упрощение PoC: правила по ключевым словам + псевдо-confidence
(доля «выигравшей» категории среди всех совпадений).
В целевой архитектуре: TF-IDF+логрег -> лёгкий transformer (см. docs/ml.md);
риск-список остаётся детерминированным правилом ПОВЕРХ модели.
"""
from dataclasses import dataclass

TOPIC_KEYWORDS = {
    "delivery": ["достав", "курьер", "заказ не приш", "трек", "посылк"],
    "payment": ["оплат", "спис", "карт", "платеж", "платёж", "возврат", "деньги"],
    "account": ["аккаунт", "парол", "взлом", "логин", "не могу войти", "доступ к акк"],
    "app_issue": ["приложени", "ошибк", "не работает", "вылетает", "грузится", "недоступ"],
    "subscription": ["подписк", "тариф", "продлен", "отмен"],
}

# Категории, которые НИКОГДА не закрываются автоматически (правило сильнее модели).
RISKY_TOPICS = {"payment", "account"}
RISK_KEYWORDS = ["суд", "юрист", "мошенн", "украл", "угрож", "жалоб", "компенсац", "персональн"]

CONFIDENCE_THRESHOLD = 0.7  # ниже — эскалация оператору (low confidence)


@dataclass
class Classification:
    topic: str
    confidence: float
    risky: bool
    risk_reasons: list


def classify(masked_text: str, pii_found: list) -> Classification:
    text = masked_text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits:
            scores[topic] = hits
    if not scores:
        return Classification("unknown", 0.0, True, ["unclassified"])

    topic = max(scores, key=lambda t: scores[t])
    confidence = round(scores[topic] / sum(scores.values()), 2)

    risk_reasons = []
    if topic in RISKY_TOPICS:
        risk_reasons.append(f"risky_topic:{topic}")
    for kw in RISK_KEYWORDS:
        if kw in text:
            risk_reasons.append(f"risk_keyword:{kw}")
    if pii_found:
        risk_reasons.append(f"pii:{','.join(pii_found)}")
    if confidence < CONFIDENCE_THRESHOLD:
        risk_reasons.append(f"low_confidence:{confidence}")

    return Classification(topic, confidence, bool(risk_reasons), risk_reasons)
