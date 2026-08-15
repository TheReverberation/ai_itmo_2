"""Pydantic-схемы API."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # только для аннотаций: схемы не зависят от ORM в рантайме
    from app.models import Decision


class TicketCreate(BaseModel):
    # строки обрезаются по краям: текст из одних пробелов — это пустой тикет (422)
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1)
    channel: str = "api"
    ticket_id: str | None = None
    user_id: str | None = None
    metadata: dict | None = None


class TicketRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ticket_id: str
    channel: str
    user_id: str | None
    text: str
    created_at: datetime
    metadata: dict | None = Field(default=None, validation_alias="meta")


class KBArticleRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    body: str


class KBArticleHit(BaseModel):
    id: str
    title: str
    score: float


class DecisionRead(BaseModel):
    """Формат записи прежнего decision log (kb_article — вложенный объект)."""

    model_config = ConfigDict(from_attributes=True)

    decision_id: str
    ticket_id: str
    ts: datetime
    pii_masked: list[str]
    topic: str
    risk: str
    risk_reasons: list[str]
    confidence: float
    kb_article: KBArticleHit | None = None
    draft: str | None
    action: str
    reason: str

    @classmethod
    def from_decision(cls, d: Decision) -> DecisionRead:
        kb = None
        if d.kb_article_id is not None:
            kb = KBArticleHit(id=d.kb_article_id, title=d.kb_title, score=d.kb_score)
        return cls(
            decision_id=d.decision_id, ticket_id=d.ticket_id, ts=d.ts,
            pii_masked=d.pii_masked, topic=d.topic, risk=d.risk,
            risk_reasons=d.risk_reasons, confidence=d.confidence,
            kb_article=kb, draft=d.draft, action=d.action, reason=d.reason,
        )
