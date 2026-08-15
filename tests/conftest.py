import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.db import Base, get_session
from app.models import KBArticle, Ticket
from app.services.llm import DraftService
from app.services.loadguard import IncidentDeduplicator, LLMBudget
from app.services.retrieval import KnowledgeBaseIndex

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def settings() -> Settings:
    # keyless: fake-LLM, без чтения .env
    return Settings(_env_file=None, openrouter_api_key=None)


@pytest.fixture
def kb_articles() -> list[dict]:
    return json.loads((DATA_DIR / "kb.json").read_text(encoding="utf-8"))


@pytest.fixture
def tickets_data() -> dict[str, dict]:
    raw = json.loads((DATA_DIR / "tickets.json").read_text(encoding="utf-8"))
    return {t["id"]: t for t in raw}


@pytest.fixture
def kb_index(kb_articles) -> KnowledgeBaseIndex:
    return KnowledgeBaseIndex(kb_articles)


@pytest.fixture
async def engine():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session


@pytest.fixture
async def seeded_session(session, kb_articles, tickets_data):
    session.add_all(KBArticle(**a) for a in kb_articles)
    session.add_all(
        Ticket(ticket_id=t["id"], channel=t["channel"], text=t["text"])
        for t in tickets_data.values()
    )
    await session.commit()
    return session


@pytest.fixture
def draft_service(settings) -> DraftService:
    return DraftService(settings, available=True)


@pytest.fixture
async def client(engine, settings, kb_index):
    from app.main import app

    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_session():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    app.state.kb_index = kb_index
    app.state.draft_service = DraftService(settings, available=True)
    app.state.llm_budget = LLMBudget(
        settings.llm_budget_rub, settings.llm_cost_per_call_rub
    )
    app.state.dedup = IncidentDeduplicator(
        settings.dedup_similarity_threshold, settings.dedup_window_seconds
    )

    # KB нужна и в БД для GET /kb
    async with factory() as session:
        session.add_all(
            KBArticle(**a)
            for a in json.loads((DATA_DIR / "kb.json").read_text(encoding="utf-8"))
        )
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()
