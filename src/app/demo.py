"""Demo-прогон: 8 mock-тикетов через пайплайн, решения — в таблицу decisions.

Запуск:
    uv run python -m app.demo             # обычный режим
    uv run python -m app.demo --llm-down  # имитация недоступности LLM API
"""
from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select

from app.config import get_settings
from app.db import init_db, session_factory
from app.main import build_kb_index, seed_kb_if_empty
from app.models import Ticket
from app.seed import seed_tickets_if_missing
from app.services.llm import DraftService
from app.services.pipeline import process_ticket


async def run(llm_down: bool) -> None:
    settings = get_settings()
    await init_db()
    async with session_factory() as session:
        await seed_kb_if_empty(session)
        await seed_tickets_if_missing(session)
        kb = await build_kb_index(session)
        llm = DraftService(settings, available=not llm_down)
        tickets = (await session.scalars(select(Ticket).order_by(Ticket.ticket_id))).all()

        mode = "LLM НЕДОСТУПЕН (деградация)" if llm_down else "обычный режим"
        print(f"=== Обработка {len(tickets)} тикетов, {mode} ===\n")

        for ticket in tickets:
            d = await process_ticket(ticket, kb, llm, session, settings)
            print(f"[{d.ticket_id}] тема={d.topic} риск={d.risk} "
                  f"confidence={d.confidence} PII={d.pii_masked or '—'}")
            print(f"  → действие: {d.action}  ({d.reason})")
            if d.kb_article_id:
                print(f"  → статья KB: {d.kb_title} (score={d.kb_score})")
            if d.draft:
                print(f"  → черновик: {d.draft[:100]}…")
            print()

    print("Все решения записаны в append-only таблицу decisions (app.db); "
          "смотреть: GET /decisions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-down", action="store_true",
                        help="имитировать недоступность LLM API (fallback-путь)")
    args = parser.parse_args()
    asyncio.run(run(args.llm_down))


if __name__ == "__main__":
    main()
