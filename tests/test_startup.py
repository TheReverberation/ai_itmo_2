"""Старт приложения (lifespan): наполнение KB и сборка индекса.

Остальные API-тесты подставляют app.state вручную, поэтому реальный путь
старта не проверялся ими вовсе — здесь он выполняется целиком, но на
тестовом движке, без записи в рабочую БД.
"""
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app import main
from app.models import KBArticle


async def test_lifespan_seeds_kb_and_builds_index(engine, monkeypatch):
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def noop_init_db() -> None:
        """таблицы уже созданы фикстурой engine"""

    monkeypatch.setattr(main, "init_db", noop_init_db)
    monkeypatch.setattr(main, "session_factory", factory)

    async with main.lifespan(main.app):
        index = main.app.state.kb_index
        assert len(index.articles) == 6  # статьи прочитаны из data/kb.json
        assert index.search("забыл пароль")[0]["id"] == "kb-1"
        assert main.app.state.draft_service is not None

    async with factory() as session:
        count = await session.scalar(select(func.count()).select_from(KBArticle))
    assert count == 6


async def test_lifespan_does_not_duplicate_existing_kb(engine, monkeypatch, kb_articles):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        session.add_all(KBArticle(**a) for a in kb_articles)
        await session.commit()

    async def noop_init_db() -> None:
        """таблицы уже созданы фикстурой engine"""

    monkeypatch.setattr(main, "init_db", noop_init_db)
    monkeypatch.setattr(main, "session_factory", factory)

    async with main.lifespan(main.app):
        pass

    async with factory() as session:
        count = await session.scalar(select(func.count()).select_from(KBArticle))
    assert count == 6  # повторный старт не задваивает базу знаний
