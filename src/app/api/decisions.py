from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Decision
from app.schemas import DecisionRead

router = APIRouter(tags=["decisions"])


@router.get("/decisions", response_model=list[DecisionRead])
async def list_decisions(
    ticket_id: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> list[DecisionRead]:
    """Append-only decision log; только чтение."""
    stmt = select(Decision).order_by(Decision.ts, Decision.id)
    if ticket_id is not None:
        stmt = stmt.where(Decision.ticket_id == ticket_id)
    rows = (await session.scalars(stmt)).all()
    return [DecisionRead.from_decision(d) for d in rows]
