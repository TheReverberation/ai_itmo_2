"""API-тесты: полный путь через HTTP, keyless (fake-LLM)."""


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_mode"] == "fake"


async def test_post_ticket_happy_path(client):
    resp = await client.post("/tickets", json={
        "channel": "chat",
        "text": "Не могу войти, забыл пароль, почта a@b.com",
    })
    assert resp.status_code == 201
    d = resp.json()
    assert d["action"] == "auto_draft"
    assert d["topic"] == "account_access"
    assert "email" in d["pii_masked"]


async def test_post_risky_ticket_escalates_and_pii_never_leaks(client):
    card = "4276 1234 5678 9012"
    resp = await client.post("/tickets", json={
        "channel": "email",
        "text": f"С карты {card} списали дважды, верните деньги, иначе жалобу подам!",
    })
    assert resp.status_code == 201
    d = resp.json()
    assert d["action"] == "escalate_to_operator"
    assert d["risk"] == "high"
    assert "4276" not in resp.text  # номер карты не утёк в ответ


async def test_decisions_listed(client):
    await client.post("/tickets", json={"text": "Не могу войти, забыл пароль"})
    await client.post("/tickets", json={"text": "приложение вылетает с ошибкой"})
    resp = await client.get("/decisions")
    assert resp.status_code == 200
    decisions = resp.json()
    assert len(decisions) == 2
    assert all("action" in d and "ts" in d for d in decisions)


async def test_kb_listed(client):
    resp = await client.get("/kb")
    assert resp.status_code == 200
    assert len(resp.json()) == 6


async def test_get_ticket_and_404(client):
    created = await client.post(
        "/tickets", json={"text": "как отменить подписку", "ticket_id": "t-x1"}
    )
    assert created.status_code == 201
    resp = await client.get("/tickets/t-x1")
    assert resp.status_code == 200
    assert resp.json()["text"] == "как отменить подписку"
    assert (await client.get("/tickets/nope")).status_code == 404
