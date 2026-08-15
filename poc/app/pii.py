"""PII-маскирование (упрощение PoC: regex; в целевой архитектуре — regex + NER).

Маскирование выполняется ДО классификации и до любого (mock-)LLM вызова.
"""
import re

_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("PHONE", re.compile(r"(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
]


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Возвращает (маскированный текст, список найденных типов PII)."""
    found: list[str] = []
    masked = text
    for label, pattern in _PATTERNS:
        if pattern.search(masked):
            found.append(label)
            masked = pattern.sub(f"<{label}>", masked)
    return masked, found
