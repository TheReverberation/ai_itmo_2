"""Таблицы: tickets, kb_articles, decisions (append-only decision log)."""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_id: Mapped[str] = mapped_column(String, primary_key=True)
    channel: Mapped[str] = mapped_column(String, default="api")
    user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    # атрибут meta: имя metadata зарезервировано в Declarative
    meta: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)


class KBArticle(Base):
    __tablename__ = "kb_articles"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String)
    body: Mapped[str] = mapped_column(Text)


class Decision(Base):
    """Append-only: нет ни одного пути обновления/удаления записей."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    decision_id: Mapped[str] = mapped_column(String)
    ticket_id: Mapped[str] = mapped_column(ForeignKey("tickets.ticket_id"))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    pii_masked: Mapped[list] = mapped_column(JSON, default=list)
    topic: Mapped[str] = mapped_column(String)
    risk: Mapped[str] = mapped_column(String)
    risk_reasons: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float)
    kb_article_id: Mapped[str | None] = mapped_column(ForeignKey("kb_articles.id"), nullable=True)
    kb_title: Mapped[str | None] = mapped_column(String, nullable=True)
    kb_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    draft: Mapped[str | None] = mapped_column(Text, nullable=True)
    action: Mapped[str] = mapped_column(String)
    reason: Mapped[str] = mapped_column(Text)
