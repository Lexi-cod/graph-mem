"""Integration tests for the fact store against a live Neo4j (docker compose up -d)."""

from datetime import datetime, timedelta, timezone

import pytest

from graphmem import FactCreate, NodeCreate, NodeType, RelType, SchemaViolation
from graphmem.decay import decay_score

pytestmark = pytest.mark.integration

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
P, O, C, E = NodeType.PERSON, NodeType.ORGANIZATION, NodeType.CONCEPT, NodeType.EVENT


async def _seed(store):
    alice = await store.add_node(NodeCreate(type=P, name="Alice"), T0)
    acme = await store.add_node(NodeCreate(type=O, name="Acme Corp"), T0)
    paris = await store.add_node(NodeCreate(type=C, name="Paris"), T0)
    hike = await store.add_node(NodeCreate(type=E, name="Alps hiking trip"), T0)

    async def add(rt, oid, text):
        return await store.add_fact(
            FactCreate(relation_type=rt, subject_id=alice.id, object_id=oid, text=text), T0)

    works = await add(RelType.WORKS_AT, acme.id, "Alice works at Acme Corp, a robotics company")
    lives = await add(RelType.LIVES_IN, paris.id, "Alice lives in Paris")
    went = await add(RelType.PARTICIPATED_IN, hike.id, "Alice went hiking in the Alps")
    eng = await store.add_fact(FactCreate(relation_type=RelType.HAS_ATTRIBUTE, subject_id=alice.id,
                                          text="Alice is a software engineer", value="software engineer",
                                          slot="job title"), T0)
    return dict(alice=alice, acme=acme, paris=paris, hike=hike, works=works, lives=lives, went=went, eng=eng)


async def test_schema_is_idempotent(store):
    await store.init_schema()
    await store.init_namespace()
    rows = await store._run(
        "SHOW VECTOR INDEXES YIELD name, state WHERE name = 'fact_embedding_NS_default' RETURN state")
    assert rows == [{"state": "ONLINE"}]


async def test_fact_has_temporal_fields_and_links(store):
    g = await _seed(store)
    f = await store.get_fact(g["works"].id)
    assert f.relation_type is RelType.WORKS_AT and f.subject_id == g["alice"].id and f.object_id == g["acme"].id
    assert f.created_at == T0 and f.last_accessed == T0 and f.access_count == 0
    assert f.validity_end is None and f.weight == 1.0 and f.conversation_id == "default"
    attr = await store.get_fact(g["eng"].id)
    assert attr.object_id is None and attr.value == "software engineer" and attr.slot == "job title"
    raw = await store._run("MATCH (f:Fact {id:$id}) RETURN size(f.embedding) AS d", id=f.id)
    assert raw[0]["d"] == 384


async def test_schema_violation_rejected(store):
    g = await _seed(store)
    with pytest.raises(SchemaViolation):  # WORKS_AT must point at an Organization
        await store.add_fact(FactCreate(relation_type=RelType.WORKS_AT, subject_id=g["alice"].id,
                                        object_id=g["paris"].id, text="x"))
    with pytest.raises(SchemaViolation):  # non-attribute facts need an object
        await store.add_fact(FactCreate(relation_type=RelType.LIKES, subject_id=g["alice"].id, text="x"))


async def test_retrieve_returns_relevant_facts_with_names(store):
    g = await _seed(store)
    res = await store.retrieve("Acme Corp robotics company", k=2, now=T0, expand=False)
    assert res.facts[0].id == g["works"].id
    f = res.facts[0]
    assert (f.subject_name, f.object_name) == ("Alice", "Acme Corp") and not f.via_expansion
    assert "Alice works at Acme Corp" in res.to_context() and "[stated 2024-01-01]" in res.to_context()


async def test_expansion_adds_linked_facts_and_is_toggleable(store):
    g = await _seed(store)
    res = await store.retrieve("hiking in the Alps", k=1, now=T0)
    assert res.facts[0].id == g["went"].id
    expanded = [f for f in res.facts if f.via_expansion]
    assert expanded and all(f.subject_id == g["alice"].id for f in expanded)
    assert len({f.id for f in res.facts}) == len(res.facts)
    res = await store.retrieve("hiking in the Alps", k=1, now=T0, expand=False)
    assert [f.id for f in res.facts] == [g["went"].id]


async def test_cypher_decay_matches_python(store, settings):
    g = await _seed(store)
    await store.retrieve("Alice lives in Paris", k=1, now=T0 + timedelta(days=2), expand=False)
    now = T0 + timedelta(days=12.5)
    got = await store.decay_of(g["lives"].id, now=now)
    want = decay_score(1.0, 1, T0 + timedelta(days=2), now, settings.decay_lambda)
    assert got == pytest.approx(want, rel=1e-9)


async def test_retrieval_reinforces_facts_unless_disabled(store):
    g = await _seed(store)
    t = T0 + timedelta(days=5)
    await store.retrieve("Alice lives in Paris", k=1, now=t, reinforce=False, expand=False)
    assert (await store.get_fact(g["lives"].id)).access_count == 0
    await store.retrieve("Alice lives in Paris", k=1, now=t, expand=False)
    f = await store.get_fact(g["lives"].id)
    assert f.access_count == 1 and f.last_accessed == t


async def test_rerank_prefers_fresher_of_equal_facts(store):
    a = await store.add_node(NodeCreate(type=P, name="Bob"), T0)
    blue = await store.add_node(NodeCreate(type=C, name="blue"), T0)
    mk = lambda t: store.add_fact(FactCreate(relation_type=RelType.LIKES, subject_id=a.id, object_id=blue.id,
                                             text="Bob's favorite color is blue"), t)
    old, new = await mk(T0), await mk(T0 + timedelta(days=60))
    now = T0 + timedelta(days=61)
    res = await store.retrieve("Bob's favorite color", k=2, now=now, reinforce=False, expand=False)
    assert [f.id for f in res.facts] == [new.id, old.id]
    assert res.facts[0].similarity == pytest.approx(res.facts[1].similarity, abs=1e-6)
    assert res.facts[0].decay > res.facts[1].decay and res.facts[0].score <= 1.0


async def test_superseded_facts_penalized_labelled_and_filterable(store):
    g = await _seed(store)
    berlin = await store.add_node(NodeCreate(type=C, name="Berlin"), T0)
    move = T0 + timedelta(days=30)
    await store.invalidate_fact(g["lives"].id, at=move)
    new = await store.add_fact(FactCreate(relation_type=RelType.LIVES_IN, subject_id=g["alice"].id,
                                          object_id=berlin.id, text="Alice lives in Berlin",
                                          validity_start=move), move)
    assert [f.id for f in await store.current_facts(g["alice"].id, RelType.LIVES_IN)] == [new.id]

    later = move + timedelta(days=1)
    res = await store.retrieve("Where does Alice live?", k=4, now=later, reinforce=False, expand=False)
    by = {f.text: f for f in res.facts}
    paris, berlin_f = by["Alice lives in Paris"], by["Alice lives in Berlin"]
    raw = lambda f: store.s.retrieval_alpha * f.similarity + (1 - store.s.retrieval_alpha) * f.decay
    assert paris.score == pytest.approx(raw(paris) - store.s.history_penalty, abs=1e-9)
    assert berlin_f.score == pytest.approx(raw(berlin_f), abs=1e-9)
    assert "Alice lives in Paris [stated 2024-01-01; no longer true as of 2024-01-31]" in res.to_context()

    res = await store.retrieve("Where does Alice live?", k=2, now=later, reinforce=False,
                               expand=False, include_history=False)
    assert "Alice lives in Paris" not in {f.text for f in res.facts}
    res = await store.retrieve("Where does Alice live?", k=2, as_of=T0 + timedelta(days=10), now=later,
                               reinforce=False, expand=False)
    texts = {f.text for f in res.facts}
    assert "Alice lives in Paris" in texts and "Alice lives in Berlin" not in texts


async def test_prune_archives_stale_facts(store):
    g = await _seed(store)
    later = T0 + timedelta(days=100)
    await store.retrieve("Alice lives in Paris", k=1, now=later, expand=False)  # keep one fresh
    dry = await store.prune(threshold=0.05, dry_run=True, now=later)
    stale = {f["id"] for f in dry.archived_facts}
    assert g["lives"].id not in stale and {g["works"].id, g["went"].id} <= stale
    assert (await store.stats())["archived_facts"] == 0

    rep = await store.prune(threshold=0.05, now=later)
    assert {f["id"] for f in rep.archived_facts} == stale
    res = await store.retrieve("Where does Alice work?", k=5, now=later, reinforce=False)
    assert g["works"].id not in {f.id for f in res.facts}
    await store.reinforce_fact(g["works"].id, later)  # a restatement revives it
    res = await store.retrieve("Where does Alice work?", k=5, now=later, reinforce=False)
    assert g["works"].id in {f.id for f in res.facts}


async def test_namespace_scoping_of_facts(store):
    other = await store.scoped("other-ns")
    await other.clear()
    try:
        g = await _seed(store)
        o_alice = await other.add_node(NodeCreate(type=P, name="Alice"), T0)
        o_city = await other.add_node(NodeCreate(type=C, name="Rome"), T0)
        await other.add_fact(FactCreate(relation_type=RelType.LIVES_IN, subject_id=o_alice.id,
                                        object_id=o_city.id, text="Alice lives in Rome"), T0)
        res = await store.retrieve("Where does Alice live?", k=10, now=T0, reinforce=False)
        assert "Alice lives in Rome" not in {f.text for f in res.facts}
        with pytest.raises(KeyError):  # cannot link to an entity of another namespace
            await store.add_fact(FactCreate(relation_type=RelType.LIVES_IN, subject_id=g["alice"].id,
                                            object_id=o_city.id, text="x"))
    finally:
        await other.clear()


async def test_provenance_adds_deduplicated_source_turns(store):
    alice = await store.add_node(NodeCreate(type=P, name="Alice"), T0)
    paris = await store.add_node(NodeCreate(type=C, name="Paris"), T0)
    await store.add_turns([
        {"dia_id": "D1:1", "text": "Alice: I moved to Paris and I love the cafes there.", "session_time": T0},
        {"dia_id": "D1:2", "text": "Bob: That's great!", "session_time": T0},
    ])
    for text, rt, oid in [("Alice lives in Paris", RelType.LIVES_IN, paris.id),
                          ("Alice loves the cafes in Paris", RelType.LIKES, paris.id)]:
        await store.add_fact(FactCreate(relation_type=rt, subject_id=alice.id, object_id=oid,
                                        text=text, turn_id="D1:1"), T0)
    res = await store.retrieve("Where does Alice live?", k=2, now=T0, expand=False, provenance=True)
    assert [t.dia_id for t in res.turns] == ["D1:1"]  # two facts, one shared source turn
    ctx = res.to_context()
    assert ctx.startswith("Facts:\n- ") and "Source turns" in ctx
    assert "- [01 January 2024] Alice: I moved to Paris and I love the cafes there." in ctx
    off = await store.retrieve("Where does Alice live?", k=2, now=T0, expand=False, provenance=False)
    assert off.turns == [] and "Source turns" not in off.to_context()
    # another namespace's turns are never attached
    other = await store.scoped("other-prov")
    try:
        await other.add_turns([{"dia_id": "D1:1", "text": "Zed: elsewhere", "session_time": T0}])
        res = await store.retrieve("Where does Alice live?", k=2, now=T0, expand=False, provenance=True)
        assert [t.text for t in res.turns] == ["Alice: I moved to Paris and I love the cafes there."]
    finally:
        await other.clear()
