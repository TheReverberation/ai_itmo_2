from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import KBArticle
from app.schemas import KBArticleRead

router = APIRouter(tags=["kb"])


@router.get("/kb", response_model=list[KBArticleRead])
async def list_articles(session: AsyncSession = Depends(get_session)) -> list[KBArticle]:
    return (await session.scalars(select(KBArticle).order_by(KBArticle.id))).all()
