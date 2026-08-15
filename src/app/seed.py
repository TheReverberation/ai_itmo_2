"""Наполнение БД демо-данными: uv run python -m app.seed"""
from __future__ import annotations

import asyncio
import json

from sqlalchemy import select

from app.db import init_db, session_factory
from app.main import DATA_DIR, seed_kb_if_empty
from app.models import Ticket


async def seed_tickets_if_missing(session) -> int:
    tickets = json.loads((DATA_DIR / "tickets.json").read_text(encoding="utf-8"))
    existing = set((await session.scalars(select(Ticket.ticket_id))).all())
    added = 0
    for t in tickets:
        if t["id"] in existing:
            continue
        session.add(Ticket(ticket_id=t["id"], channel=t["channel"], text=t["text"]))
        added += 1
    await session.commit()
    return added


async def main() -> None:
    await init_db()
    async with session_factory() as session:
        await seed_kb_if_empty(session)
        added = await seed_tickets_if_missing(session)
    print(f"БД готова: статьи KB загружены, новых тикетов добавлено: {added}")


if __name__ == "__main__":
    asyncio.run(main())
