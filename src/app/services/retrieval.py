"""Retrieval по базе знаний: TF-IDF + косинусная близость.

Осознанное упрощение — в целевой системе embedding-модель + vector store
(см. docs/ml.md), интерфейс search() при замене сохраняется.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_token_re = re.compile(r"[а-яa-zё0-9]+")


def _tokens(text: str) -> list[str]:
    # грубый стемминг: обрезаем окончания, чтобы «оплата/оплату/оплатил» совпадали
    return [t[:5] for t in _token_re.findall(text.lower())]


class KnowledgeBaseIndex:
    """In-memory индекс статей KB; строится при старте приложения."""

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
