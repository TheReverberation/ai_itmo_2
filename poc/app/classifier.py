"""Классификатор темы и риска.

Упрощение PoC: правила по ключевым словам + эвристический confidence.
В целевой архитектуре: TF-IDF+логрег -> лёгкий transformer (см. docs/ml.md);
риск-список остаётся детерминированным правилом ПОВЕРХ модели, а confidence —
калиброванной вероятностью модели (Platt/temperature scaling).

Про confidence в PoC: наивная доля «выигравшей» категории (hits_top / все hits)
давала бы 1.0 при единственном совпадении одного ключевого слова — модель была бы
«уверена» на пустом месте, и ветка low-confidence никогда бы не срабатывала.
Поэтому confidence считается из двух факторов (см. _calibrate_confidence):
  - separation — насколько лидирующая тема оторвалась от конкурентов;
  - evidence   — сколько всего сигналов найдено (насыщается на 2 совпадениях).
Так единичное слабое совпадение честно даёт низкую уверенность и уходит оператору.
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
# На скольких совпадениях считаем «сигнала достаточно» (насыщение evidence-фактора).
_EVIDENCE_SATURATION = 2


@dataclass
class Classification:
    topic: str
    confidence: float
    risky: bool
    risk_reasons: list


def _calibrate_confidence(scores: dict, topic: str) -> float:
    """Грубая замена калиброванной вероятности модели.

    Стоит на двух ногах, каждая в [0..1], итог — их произведение:
      - separation: доля голосов лидера среди всех совпадений. Один в поле —
        1.0; если несколько тем конкурируют — падает.
      - evidence: сколько совпадений у лидера, с насыщением на _EVIDENCE_SATURATION.
        Одно совпадение = 0.5, два и больше = 1.0. Это и «сбивает» ложную
        уверенность единичного ключевого слова ниже порога эскалации.
    """
    top_hits = scores[topic]
    total_hits = sum(scores.values())
    separation = top_hits / total_hits
    evidence = min(top_hits, _EVIDENCE_SATURATION) / _EVIDENCE_SATURATION
    return round(separation * evidence, 2)


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
    confidence = _calibrate_confidence(scores, topic)

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
