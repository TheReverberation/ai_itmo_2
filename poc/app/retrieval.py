"""Retrieval по базе знаний.

Упрощение PoC: TF-IDF (чистый stdlib) + cosine similarity по локальной мини-KB.
В целевой архитектуре: embeddings (e5/sbert) + pgvector/Qdrant.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")


def _tokenize(text: str) -> list:
    # Грубый стемминг: усечение до 6 символов сглаживает окончания русских слов.
    return [t[:6] for t in _TOKEN_RE.findall(text.lower())]


class KnowledgeBase:
    def __init__(self, kb_path: str | Path):
        self.articles = json.loads(Path(kb_path).read_text(encoding="utf-8"))
        self._docs = [_tokenize(a["title"] + " " + a["body"]) for a in self.articles]
        self._idf = self._build_idf()

    def _build_idf(self) -> dict:
        n = len(self._docs)
        df: Counter = Counter()
        for doc in self._docs:
            df.update(set(doc))
        return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}

    def _vector(self, tokens: list) -> dict:
        tf = Counter(tokens)
        return {t: c * self._idf.get(t, 1.0) for t, c in tf.items()}

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
        return num / den if den else 0.0

    def search(self, query: str, top_k: int = 1) -> list:
        """Возвращает [(score, article), ...] по убыванию score."""
        qv = self._vector(_tokenize(query))
        scored = [
            (round(self._cosine(qv, self._vector(doc)), 3), article)
            for doc, article in zip(self._docs, self.articles)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]
