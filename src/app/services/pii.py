"""PII-маскирование: выполняется до любых внешних вызовов (в т.ч. LLM)."""
from __future__ import annotations

import re

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"(?<!\d)(?:\+7|8)[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"),
    "card": re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"),
}


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Маскирует PII. Возвращает (текст, найденные типы)."""
    found = []
    for kind, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            found.append(kind)
            text = pattern.sub(f"[{kind.upper()}]", text)
    return text, found
