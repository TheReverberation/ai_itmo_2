"""Минимальный PoC пайплайна обработки тикетов.

Только stdlib. Каждый этап — упрощённая версия компонента из
docs/architecture.md; чем заменяется в целевой системе, описано в README.
"""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- PII

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"(?<!\d)(?:\+7|8)[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"),
    "card": re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"),
}


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Маскирует PII до любых внешних вызовов. Возвращает (текст, найденные типы)."""
    found = []
    for kind, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            found.append(kind)
            text = pattern.sub(f"[{kind.upper()}]", text)
    return text, found

# ------------------------------------------------- классификация (правила)

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
    """Тема + риск + confidence. В целевой системе — правила + классическая ML-модель."""
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

# ------------------------------------------------- retrieval (TF-IDF, stdlib)

_token_re = re.compile(r"[а-яa-zё0-9]+")


def _tokens(text: str) -> list[str]:
    # грубый стемминг: обрезаем окончания, чтобы «оплата/оплату/оплатил» совпадали
    return [t[:5] for t in _token_re.findall(text.lower())]


class KnowledgeBase:
    """TF-IDF + косинусная близость по статьям базы знаний."""

    def __init__(self, articles: list[dict]):
        self.articles = articles
        docs = [_tokens(a["title"] + " " + a["body"]) for a in articles]
        self.df = Counter(t for doc in docs for t in set(doc))
        self.n = len(docs)
        self.vecs = [self._tfidf(doc) for doc in docs]

    def _tfidf(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        return {t: (c / len(tokens)) * math.log((1 + self.n) / (1 + self.df[t]))
                for t, c in tf.items()}

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, text: str) -> tuple[dict | None, float]:
        qvec = self._tfidf(_tokens(text))
        best_i, best_s = -1, 0.0
        for i, vec in enumerate(self.vecs):
            s = self._cosine(qvec, vec)
            if s > best_s:
                best_i, best_s = i, s
        return (self.articles[best_i] if best_i >= 0 else None), round(best_s, 2)

# ------------------------------------------------- mock LLM

class MockLLM:
    """Шаблонный «LLM». available=False имитирует недоступность внешнего API."""

    def __init__(self, available: bool = True):
        self.available = available

    def draft(self, masked_text: str, article: dict) -> str:
        if not self.available:
            raise ConnectionError("LLM API недоступен")
        return (
            f"Здравствуйте! Похоже, ваш вопрос касается темы «{article['title']}». "
            f"{article['body']} Если это не решит проблему — ответьте на это "
            f"сообщение, и мы передадим обращение специалисту."
        )

# ------------------------------------------------- политика решений

CONFIDENCE_THRESHOLD = 0.55
RETRIEVAL_THRESHOLD = 0.15
# категории, для которых автоответ запрещён политикой (всегда HITL)
NEVER_AUTO_TOPICS = {"payment"}


def process_ticket(ticket: dict, kb: KnowledgeBase, llm: MockLLM,
                   log_path: Path) -> dict:
    """Полный путь одного тикета: PII → классификация → retrieval → черновик/эскалация → лог."""
    masked, pii = mask_pii(ticket["text"])
    cls = classify(masked)

    decision = {
        "decision_id": str(uuid.uuid4())[:8],
        "ticket_id": ticket["id"],
        "pii_masked": pii,
        **cls,
        "kb_article": None,
        "draft": None,
    }

    if cls["risk"] == "high":
        decision.update(action="escalate_to_operator",
                        reason=f"рискованная категория: {', '.join(cls['risk_reasons'])}")
    elif cls["confidence"] < CONFIDENCE_THRESHOLD:
        decision.update(action="escalate_to_operator",
                        reason=f"low confidence ({cls['confidence']} < {CONFIDENCE_THRESHOLD})")
    else:
        article, score = kb.search(masked)
        if article is None or score < RETRIEVAL_THRESHOLD:
            decision.update(action="route_to_queue", reason="нет релевантной статьи KB")
        else:
            decision["kb_article"] = {"id": article["id"], "title": article["title"], "score": score}
            try:
                draft = llm.draft(masked, article)
                if cls["topic"] in NEVER_AUTO_TOPICS:
                    decision.update(action="suggest_to_operator", draft=draft,
                                    reason="категория из списка «только с оператором»")
                else:
                    decision.update(action="auto_draft", draft=draft,
                                    reason="типовой тикет, высокая уверенность")
            except ConnectionError:
                decision.update(action="route_to_queue",
                                reason="LLM недоступен → graceful degradation: "
                                       "шаблон-подтверждение пользователю, тикет оператору")

    _log(decision, log_path)
    return decision


def _log(decision: dict, log_path: Path) -> None:
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **decision}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
