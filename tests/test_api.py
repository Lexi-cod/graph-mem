"""Phase 3: MemoryService, LangGraph agent and FastAPI endpoints.

Uses a scripted fake LLM (no Ollama). Run against a dev Neo4j, e.g.
    GRAPHMEM_NEO4J_URI=bolt://localhost:7688 pytest tests/test_api.py
"""

from datetime import datetime, timedelta, timezone

import pytest
import httpx

from graphmem import GraphStore, NodeType
from graphmem.agent import Decision, MemoryAgent
from graphmem.api import create_app
from graphmem.config import Settings
from graphmem.extraction import ExtractedEntity as Ent, ExtractedRelation, Extraction
from graphmem.llm import JsonMeta, LLMResult
from graphmem.service import MemoryService

T0 = datetime(2023, 5, 8, 13, 0, tzinfo=timezone.utc)
P, C = NodeType.PERSON, NodeType.CONCEPT
CID = "api-test"


def rel(src, rt, dst, fact, ev):
    return ExtractedRelation(source=src, relation=rt, target=dst, fact=fact, evidence=ev)


class FakeLLM:
    """Extractions keyed by the CURRENT turn text; the agent's decision is 'remember' when the
    message contains 'I '; answers echo the first memory line."""

    identity = {"provider": "fake", "base_url": "fake", "model": "fake"}

    def __init__(self, extractions):
        self.extractions = extractions
        self.calls: list[str] = []
        self.prompts: list[str] = []

    async def generate_json_meta(self, prompt, schema, *, system=None, purpose="misc", max_output_tokens=None):
        self.calls.append(purpose)
        self.prompts.append(prompt)
        if schema is Extraction:
            text = prompt.rsplit("CURRENT turn (extract only from this):\n", 1)[1]
            return self.extractions.get(text, Extraction()).model_copy(deep=True), JsonMeta()
        if schema is Decision:
            msg = prompt.rsplit(":\n", 1)[1]
            return Decision(remember="I " in msg, reason="test"), JsonMeta()
        raise AssertionError(schema)

    async def generate(self, prompt, *, system=None, purpose="misc", max_output_tokens=None, thinking=None, **kw):
        self.calls.append(purpose)
        mem = prompt.split("Memories:\n", 1)[1].split("\n")
        first = next((l for l in mem if l.startswith("- ")), "(none)")
        return LLMResult(text=f"Noted. {first}", input_tokens=1, output_tokens=1, latency_s=0.0, cached=False)


EXTRACTIONS = {
    "Caroline: I moved to Sweden last month and I love it.": Extraction(
        entities=[Ent(name="Caroline", type=P), Ent(name="Sweden", type=C)],
        relations=[rel("Caroline", "LIVES_IN", "Sweden", "Caroline moved to Sweden last month.",
                       "I moved to Sweden last month")]),
    "Caroline: I live in Boston.": Extraction(
        entities=[Ent(name="Caroline", type=P), Ent(name="Boston", type=C)],
        relations=[rel("Caroline", "LIVES_IN", "Boston", "Caroline lives in Boston.", "I live in Boston")]),
    "Caroline: I adopted a guinea pig named Oscar.": Extraction(
        entities=[Ent(name="Caroline", type=P), Ent(name="Oscar", type=C)],
        relations=[rel("Caroline", "OWNS", "Oscar", "Caroline adopted a guinea pig named Oscar.",
                       "I adopted a guinea pig named Oscar")]),
}


@pytest.fixture
async def svc(embedder, tmp_path):
    s = Settings(cache_dir=str(tmp_path))
    try:
        root = await GraphStore.connect(s, embedder)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Neo4j unavailable: {exc}")
    llm = FakeLLM(EXTRACTIONS)
    service = MemoryService(root, llm, s)
    st = await service.store(CID)
    await st.clear()
    yield service, llm
    await st.clear()
    await root.close()


async def test_update_then_query_with_provenance_and_supersession(svc):
    service, llm = svc
    r1 = await service.update(CID, "Caroline", "I live in Boston.", listener="Melanie", time=T0)
    assert r1.facts_created == 1 and r1.facts == ["Caroline lives in Boston."]
    r2 = await service.update(CID, "Caroline", "I moved to Sweden last month and I love it.",
                              listener="Melanie", time=T0 + timedelta(days=30))
    # relative time rewritten; exclusive LIVES_IN retired the old fact (rules mode)
    assert r2.facts == ["Caroline moved to Sweden in May 2023."] and r2.facts_retired == 1
    # the second extraction saw the first turn as context
    assert "Caroline: I live in Boston." in llm.prompts[-1]

    res = await service.query(CID, "Where does Caroline live?", now=T0 + timedelta(days=31), reinforce=False)
    by = {f.text: f for f in res.facts}
    assert by["Caroline moved to Sweden in May 2023."].is_current
    assert not by["Caroline lives in Boston."].is_current
    assert {t.dia_id for t in res.turns} == {r1.turn_id, r2.turn_id}
    assert "Source turns" in res.to_context()


async def test_conversation_id_validated(svc):
    service, _ = svc
    with pytest.raises(ValueError):
        await service.query("bad id!", "x")


async def test_agent_graph_write_answer_and_prune_schedule(svc):
    service, llm = svc
    agent = MemoryAgent(service, llm, prune_every=2, thinking=False)
    out1 = await agent.run(CID, "Caroline", "I adopted a guinea pig named Oscar.", listener="Melanie",
                           time=T0, turn_index=1)
    assert out1["decision"]["remember"] and out1["written"]["facts_created"] == 1
    assert "Oscar" in out1["answer"]  # answered after the mid-run write + re-retrieval
    assert out1["pruned"] is None  # turn 1: prune not scheduled
    assert list(out1["timings_ms"]) == ["retrieve", "reason", "write", "answer"]

    out2 = await agent.run(CID, "Melanie", "What pet does Caroline have?", listener="Caroline",
                           time=T0 + timedelta(minutes=1), turn_index=2)
    assert not out2["decision"]["remember"] and out2["written"] is None
    assert "Oscar" in out2["answer"]
    # turn 2: prune node runs, but pruning is disabled in the frozen config
    assert out2["pruned"] == {"skipped": True, "reason": "prune_enabled=False"}
    assert llm.calls.count("extraction") == 1


async def test_http_endpoints(svc):
    service, llm = svc
    agent = MemoryAgent(service, llm, prune_every=50, thinking=False)
    # Same event loop as the service's Neo4j driver (TestClient would run the app in another loop).
    transport = httpx.ASGITransport(app=create_app(service=service, agent=agent))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/health")).json() == {"status": "ok"}

        r = await client.post("/memory/update", json={"conversation_id": CID, "speaker": "Caroline",
                                                "text": "I live in Boston.", "listener": "Melanie",
                                                "time": T0.isoformat()})
        assert r.status_code == 200 and r.json()["facts_created"] == 1

        q = await client.post("/memory/query", json={"conversation_id": CID, "query": "Where does Caroline live?",
                                               "now": (T0 + timedelta(days=1)).isoformat(), "reinforce": False})
        body = q.json()
        assert q.status_code == 200 and body["facts"][0]["text"] == "Caroline lives in Boston."
        assert body["facts"][0]["subject"] == "Caroline" and body["facts"][0]["object"] == "Boston"
        assert body["turns"][0]["text"] == "Caroline: I live in Boston."

        p = await client.post("/memory/prune", json={"conversation_id": CID, "threshold": 2.0, "dry_run": True,
                                               "now": (T0 + timedelta(days=1)).isoformat()})
        assert p.status_code == 200 and p.json()["dry_run"] and len(p.json()["archived_facts"]) == 1

        a = await client.post("/agent/turn", json={"conversation_id": CID, "speaker": "Melanie",
                                             "message": "Where does Caroline live?",
                                             "time": (T0 + timedelta(days=2)).isoformat()})
        assert a.status_code == 200 and a.json()["turn_index"] == 1 and "Boston" in a.json()["answer"]
        assert a.json()["remembered"] is False

        bad = await client.post("/memory/query", json={"conversation_id": "no spaces", "query": "x"})
        assert bad.status_code == 422
