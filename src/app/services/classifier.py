"""Классификация темы и риска по правилам (в целевой системе — правила + ML-модель)."""
from __future__ import annotations

TOPIC_KEYWORDS = {
    "account_access": ["пароль", "войти", "вход", "логин", "доступ к аккаунту", "восстановить"],
    "payment": ["оплат", "списал", "деньг", "карта", "платеж", "платёж", "возврат", "чек"],
    "delivery": ["доставк", "заказ не пришел", "заказ не пришёл", "курьер", "трек"],
    "app_issue": ["приложение", "вылетает", "ошибка", "не работает", "не грузится", "зависает"],
    "subscription": ["подписк", "отменить", "тариф", "продлен", "продлён"],
}

RISK_KEYWORDS = {
    "payment_dispute": ["списал", "дважды", "верните", "возврат", "не приходил заказ, а деньги"],
    "personal_data": ["персональн", "удалите мои данные", "утечк"],
    "security": ["взлом", "мошенник", "украли", "не я заходил"],
    "legal": ["суд", "жалобу", "роспотребнадзор", "претензи"],
}


def classify(masked_text: str) -> dict:
    """Тема + риск + confidence."""
    low = masked_text.lower()
    scores = {
        topic: sum(1 for kw in kws if kw in low)
        for topic, kws in TOPIC_KEYWORDS.items()
    }
    topic, hits = max(scores.items(), key=lambda kv: kv[1])
    total_hits = sum(scores.values())
    if total_hits == 0:
        topic, confidence = "unknown", 0.0
    else:
        # суррогат уверенности: доля попаданий в лучшую тему, прижатая к [0.3..0.95]
        confidence = min(0.95, 0.3 + 0.65 * hits / max(3, total_hits + 1))

    risk_reasons = [
        reason for reason, kws in RISK_KEYWORDS.items()
        if any(kw in low for kw in kws)
    ]
    risk = "high" if risk_reasons else "low"
    return {"topic": topic, "risk": risk, "risk_reasons": risk_reasons,
            "confidence": round(confidence, 2)}
