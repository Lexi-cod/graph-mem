"""Phase 2: ingestion against live Neo4j with a scripted (fake) LLM, post-extraction
checks, namespace isolation, and offline tests of the LLM client."""

import asyncio
from datetime import date, datetime, timedelta, timezone

import pytest

from graphmem import GraphStore, NodeType, RelType
from graphmem.clock import SimClock
from graphmem.config import Settings
from graphmem.extraction import (
    ExtractedEntity as Ent,
    ExtractedRelation,
    Extraction,
    Extractor,
    TurnInput,
    check_evidence,
    sanitize,
)
from graphmem.ingest import ContradictionVerdict, Ingestor, PairJudgement, ingest_many
from graphmem.llm import LLM, Completion, JsonMeta, OpenAICompatBackend, PreflightError, strip_wrappers
from graphmem.locomo import Conversation, Session, TestSetGuard, Turn, require_dev
from graphmem.schema import name_key
from graphmem.timehints import intervals_compatible, name_interval

T0 = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)  # Monday
P, O, C, E = NodeType.PERSON, NodeType.ORGANIZATION, NodeType.CONCEPT, NodeType.EVENT


def Rel(source, relation, target, fact, evidence=None, **kw):
    """Relation whose evidence defaults to the fact text (tests put it in the turn)."""
    rel = relation.value if isinstance(relation, RelType) else relation
    return ExtractedRelation(source=source, relation=rel, target=target, fact=fact,
                             evidence=evidence if evidence is not None else fact, **kw)


class FakeLLM:
    """Scripted extractions keyed by turn text; contradiction verdicts via callback."""

    identity = {"provider": "fake", "base_url": "fake", "model": "fake"}

    def __init__(self, extractions: dict[str, Extraction], verdict=None):
        self.extractions = extractions
        self.verdict = verdict or (lambda prompt: ContradictionVerdict())
        self.calls: list[str] = []

    async def generate_json_meta(self, prompt, schema, *, system=None, purpose="misc", max_output_tokens=None):
        self.calls.append(purpose)
        if schema is Extraction:
            text = prompt.rsplit("CURRENT turn (extract only from this):\n", 1)[1]
            return self.extractions.get(text, Extraction()).model_copy(deep=True), JsonMeta()
        return self.verdict(prompt), JsonMeta()


def turn(text: str, day: float = 0, speaker="Caroline", listener="Melanie", dia="D1:1",
         context=()) -> TurnInput:
    return TurnInput(dia_id=dia, speaker=speaker, listener=listener, text=f"{speaker}: {text}",
                     session_time=T0 + timedelta(days=day), context=list(context))


def say(speaker: str, text: str) -> str:
    return f"{speaker}: {text}"


@pytest.fixture
async def gstore(embedder, tmp_path):
    s = Settings(cache_dir=str(tmp_path), decay_lambda=0.01)
    try:
        st = await GraphStore.connect(s, embedder, clock=SimClock(), namespace="test-a")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Neo4j unavailable: {exc}")
    await st.clear()
    yield st
    await st.clear()
    await st.close()


def make(gstore, extractions, verdict=None, mode="llm"):
    gstore.s.contradiction_mode = mode
    llm = FakeLLM(extractions, verdict)
    return Ingestor(gstore, Extractor(llm, gstore.s.cache_dir), llm), llm


async def run(ing, *turns):
    for t in turns:
        ex = await ing.extractor.extract(t)
        ing.store.clock.set(t.session_time)
        await ing.apply(t, ex, t.session_time)


async def node_id(gstore, name, t=P):
    return (await gstore.find_by_name(name, t)).id


def ex_rel(src, src_t, rel, dst, dst_t, fact, **kw):
    ents = [Ent(name=src, type=src_t)] + ([] if dst_t is None else [Ent(name=dst, type=dst_t)])
    return Extraction(entities=ents, relations=[Rel(src, rel, dst, fact, **kw)])


# ------------------------------------------------------------------ pure


def test_name_key():
    assert name_key("Melanie's  Kids!") == "melanies kids"


def test_event_date_intervals():
    assert name_interval("Pride parade (15-16 July 2023)") == (date(2023, 7, 15), date(2023, 7, 16))
    assert name_interval("talent show (September 2023)") == (date(2023, 9, 1), date(2023, 9, 30))
    assert name_interval("self-portrait (the week before 23 August 2023)") == (date(2023, 8, 16), date(2023, 8, 22))
    assert name_interval("trip in 2022") == (date(2022, 1, 1), date(2022, 12, 31))
    assert name_interval("birthday concert") is None
    assert not intervals_compatible("poetry reading (13 October 2023)", "poetry reading (6 October 2023)")
    assert intervals_compatible("self-portrait (16 August 2023)", "self-portrait (week before 23 August 2023)")
    assert intervals_compatible("birthday concert (13 August 2023)", "birthday concert")


def test_evidence_matching():
    cur = "Caroline: Right! We could do a family outting, or wanna plan something special. I'm so couragous."
    ok = lambda ev, ctx=(): check_evidence(ev, cur, list(ctx))[0]
    assert ok("We could do a family outting")  # exact
    assert ok("“We could do a family outting”")  # quote chars normalized
    assert ok("we could do a family outing, or wanna plan")  # typo fix -> fuzzy >= 0.9
    assert ok("Right! ... I'm so couragous")  # ellipsis fragments checked separately
    assert not ok("Right! ... we went hiking in the mountains")  # one fragment missing
    # a fuzzy match is rejected when a context turn matches the quote better
    ctx = ["Melanie: We could do a family outing, or wanna plan something"]
    assert not ok("We could do a family outing, or wanna plan something", ctx)


def test_sanitize_drops_and_conversions():
    ctx = ["Caroline: I have a guinea pig named Oscar."]
    t = turn("I went to the support group meeting yesterday. My friend Melanie missed the pride parade. "
             "I don't have time for running anymore. Do you like painting? I'm mentoring a teen.",
             context=ctx)
    ex = Extraction(
        entities=[Ent(name="Caroline", type=P), Ent(name="support group meeting (yesterday)", type=C),
                  Ent(name="Melanie", type=P), Ent(name="pride parade", type=E), Ent(name="Oscar", type=C)],
        relations=[
            Rel("Caroline", "PARTICIPATED_IN", "support group meeting (yesterday)", "Caroline went to the meeting",
                evidence="I went to the support group meeting yesterday"),   # kept; retyped; name rewritten
            Rel("Caroline", "OWNS", "Oscar", "Caroline owns a pet named Oscar",
                evidence="I have a guinea pig named Oscar"),                 # quote from context
            Rel("Melanie", "PARTICIPATED_IN", "pride parade", "Melanie went to the parade",
                evidence="Melanie missed the pride parade"),                 # negated participation
            Rel("Caroline", "FAMILY_OF", "Melanie", "Caroline and Melanie are family",
                evidence="My friend Melanie"),                               # no family word
            Rel("Caroline", "KNOWS", "Melanie", "Caroline knows Melanie", evidence="My friend Melanie"),
            Rel("Caroline", "LIKES", "painting", "Caroline likes painting", evidence="Do you like painting?"),
            Rel("Caroline", "MENTOR_OF", "a teen", "Caroline is mentoring a teen",
                evidence="I'm mentoring a teen"),                            # unknown type -> attribute
            Rel("Jon", "LIKES", "running", "Jon likes running", evidence="running"),  # subject not in turn
        ])
    res = sanitize(ex, t)
    assert sorted(d.reason for d in res.dropped) == sorted([
        "evidence_not_in_turn", "negated_participation", "family_not_stated",
        "knows_between_speakers", "question", "subject_not_in_turn"])
    assert res.converted == [{"original_type": "MENTOR_OF", "source": "Caroline", "target": "a teen",
                              "fact": "Caroline is mentoring a teen"}]
    kept = {r.relation: r for r in res.extraction.relations}
    assert set(kept) == {"PARTICIPATED_IN", "HAS_ATTRIBUTE"}
    assert kept["HAS_ATTRIBUTE"].target == "Caroline is mentoring a teen"
    types = {e.name: e.type for e in res.extraction.entities}
    # name rewritten ("yesterday" -> date), Concept retyped to Event, attribute value not an entity
    assert types == {"Caroline": P, "support group meeting (7 May 2023)": E}
    assert res.time_rewrites >= 1


def test_sanitize_type_violation_dropped():
    t = turn("I work at my uncle Bob's place.")
    ex = Extraction(entities=[Ent(name="Caroline", type=P), Ent(name="uncle Bob", type=P)],
                    relations=[Rel("Caroline", "WORKS_AT", "uncle Bob", "Caroline works for Bob",
                                   evidence="I work at my uncle Bob's place")])
    res = sanitize(ex, t)
    assert res.extraction.relations == [] and [d.reason for d in res.dropped] == ["type_violation"]


def test_simclock_refuses_unset_and_naive():
    c = SimClock()
    with pytest.raises(RuntimeError):
        c.now()
    with pytest.raises(ValueError):
        c.set(datetime(2023, 1, 1))


def test_test_set_guard():
    require_dev("conv-26", ["conv-26", "conv-30"])
    with pytest.raises(TestSetGuard):
        require_dev("conv-41", ["conv-26", "conv-30"])


def test_strip_wrappers():
    assert strip_wrappers('```json\n{"a": 1}\n```') == ('{"a": 1}', True, False)
    assert strip_wrappers('<think>hmm</think>\n{"a": 1}') == ('{"a": 1}', False, True)


# ------------------------------------------------------------------ ingestion


async def test_entity_resolution_exact(gstore):
    t1, t2 = "I went to Boston.", "Boston was great again! caroline! Sweden next."
    ex1 = Extraction(entities=[Ent(name="Caroline", type=P), Ent(name="Boston", type=C)],
                     relations=[Rel("Caroline", "LIKES", "Boston", "Caroline went to Boston", evidence=t1)])
    ex2 = Extraction(entities=[Ent(name="caroline!", type=P), Ent(name="Boston", type=O),
                               Ent(name="Sweden", type=C)],
                     relations=[Rel("caroline!", "MEMBER_OF", "Boston", "x", evidence="Boston was great again"),
                                Rel("caroline!", "LIKES", "Sweden", "y", evidence="Sweden next")])
    ing, _ = make(gstore, {say("Caroline", t1): ex1, say("Caroline", t2): ex2})
    await run(ing, turn(t1), turn(t2, day=1))
    st = ing.stats
    assert st.merged_exact == 1 and st.nodes_created == 4  # Caroline, Boston(C), Boston(O), Sweden
    m = st.merge_log[0]
    assert (m.incoming, m.merged_into, m.method, m.score) == ("caroline!", "Caroline", "exact", 1.0)


async def test_event_merge_requires_overlapping_dates(gstore):
    def ev(name, text):
        return Extraction(entities=[Ent(name="Caroline", type=P), Ent(name=name, type=E)],
                          relations=[Rel("Caroline", "PARTICIPATED_IN", name, text)])
    a = "went to the transgender poetry reading (6 October 2023)"
    b = "went to the transgender poetry reading (13 October 2023)"
    c = "went to the transgender poetry reading event"
    ing, _ = make(gstore, {
        say("Caroline", a): ev("transgender poetry reading (6 October 2023)", a),
        say("Caroline", b): ev("transgender poetry reading (13 October 2023)", b),
        say("Caroline", c): ev("transgender poetry reading", c)})
    gstore.s.event_resolution_threshold = 0.85
    await run(ing, turn(a), turn(b, day=7), turn(c, day=8))
    st = ing.stats
    assert st.blocked_event_merges >= 1 and "disjoint" in st.blocked_merges[0].reason
    emb = [m for m in st.merge_log if m.method == "embedding"]
    assert emb and emb[0].incoming == "transgender poetry reading"  # undated -> compatible


async def test_exclusive_slot_rules_mode_retires_directly(gstore):
    a, b = "I live in Boston", "I moved to Sweden"
    ing, llm = make(gstore, {
        say("Caroline", a): ex_rel("Caroline", P, "LIVES_IN", "Boston", C, "Caroline lives in Boston", evidence=a),
        say("Caroline", b): ex_rel("Caroline", P, "LIVES_IN", "Sweden", C, "Caroline moved to Sweden", evidence=b)},
        mode="rules")
    await run(ing, turn(a), turn(b, day=30))
    assert "contradiction" not in llm.calls
    (r,) = [x for x in ing.stats.retire_log if x.action == "retire"]
    assert (r.reason, r.detail, r.old_fact) == ("schema_rule", "exclusive LIVES_IN", "Caroline lives in Boston")
    cur = await gstore.current_facts(await node_id(gstore, "Caroline"), RelType.LIVES_IN)
    assert [f.text for f in cur] == ["Caroline moved to Sweden"]


async def test_exclusive_slot_llm_mode_needs_llm_confirmation(gstore):
    a, b = "I live in Sweden", "I live in Stockholm"
    seen = []

    def verdict(prompt):
        seen.append(prompt)
        return ContradictionVerdict(judgements=[PairJudgement(pair=0, label="compatible")])

    ing, llm = make(gstore, {
        say("Caroline", a): ex_rel("Caroline", P, "LIVES_IN", "Sweden", C, "Caroline lives in Sweden", evidence=a),
        say("Caroline", b): ex_rel("Caroline", P, "LIVES_IN", "Stockholm", C, "Caroline lives in Stockholm", evidence=b)},
        verdict)
    await run(ing, turn(a), turn(b, day=3))
    assert llm.calls.count("contradiction") == 1 and ing.stats.llm_labels == {"compatible": 1}
    assert "[P0] NEW: Caroline lives in Stockholm\n     EXISTING: Caroline lives in Sweden" in seen[0]
    assert len(await gstore.current_facts(await node_id(gstore, "Caroline"), RelType.LIVES_IN)) == 2


async def test_llm_outdated_and_same(gstore):
    a, b, c = "I work at the bank", "I now work at the dance studio", "I'm working at the dance studio"
    labels = iter(["outdated", "same"])
    v = lambda prompt: ContradictionVerdict(judgements=[PairJudgement(pair=0, label=next(labels))])
    ing, _ = make(gstore, {
        say("Jon", a): ex_rel("Jon", P, "WORKS_AT", "bank", O, "Jon works at a bank", evidence=a),
        say("Jon", b): ex_rel("Jon", P, "WORKS_AT", "dance studio", O, "Jon works at his dance studio", evidence=b),
        say("Jon", c): ex_rel("Jon", P, "WORKS_AT", "studio", O, "Jon is employed at a studio", evidence=c)}, v)
    await run(ing, turn(a, speaker="Jon"), turn(b, day=2, speaker="Jon"), turn(c, day=3, speaker="Jon"))
    assert [(x.action, x.reason) for x in ing.stats.retire_log] == [("retire", "llm"), ("reinforce", "restatement")]
    cur = await gstore.current_facts(await node_id(gstore, "Jon"), RelType.WORKS_AT)
    assert [f.text for f in cur] == ["Jon works at his dance studio"]


async def test_preferences_never_go_to_llm(gstore):
    a, b = "I love painting animals", "I love painting landscapes"
    ing, llm = make(gstore, {
        say("Melanie", a): ex_rel("Melanie", P, "LIKES", "painting animals", C, "Melanie likes painting animals", evidence=a),
        say("Melanie", b): ex_rel("Melanie", P, "LIKES", "painting landscapes", C, "Melanie likes painting landscapes", evidence=b)})
    await run(ing, turn(a, speaker="Melanie"), turn(b, day=1, speaker="Melanie"))
    assert "contradiction" not in llm.calls and ing.stats.retire_log == []
    assert (await gstore.stats())["current_facts"] == 2


async def test_preference_negation_and_opposite_same_object_only(gstore):
    a, b = "I love running", "I love hiking"
    c, d = "I stopped running, I no longer enjoy it", "I really dislike hiking"
    ing, llm = make(gstore, {
        say("Gina", a): ex_rel("Gina", P, "LIKES", "running", C, "Gina loves running", evidence=a),
        say("Gina", b): ex_rel("Gina", P, "LIKES", "hiking", C, "Gina loves hiking", evidence=b),
        say("Gina", c): ex_rel("Gina", P, "LIKES", "running", C, "Gina no longer enjoys running", evidence=c),
        say("Gina", d): ex_rel("Gina", P, "DISLIKES", "hiking", C, "Gina dislikes hiking", evidence=d)})
    await run(ing, *(turn(x, day=i, speaker="Gina") for i, x in enumerate([a, b, c, d])))
    ret = [x for x in ing.stats.retire_log if x.action == "retire"]
    assert [(x.reason, x.old_fact) for x in ret] == [("ended", "Gina loves running"),
                                                     ("schema_rule", "Gina loves hiking")]
    assert "object cos=" in ret[0].detail and "contradiction" not in llm.calls
    gid = await node_id(gstore, "Gina")
    assert [f.text for f in await gstore.current_facts(gid)] == ["Gina dislikes hiking"]


async def test_attribute_facts_and_same_slot_llm(gstore):
    a, b, c = "I'm single right now", "I'm dating someone now", "I realized self-care matters"
    v = lambda prompt: ContradictionVerdict(judgements=[PairJudgement(pair=0, label="outdated")])
    attr = lambda val, fact, ev, slot=None: Extraction(
        entities=[Ent(name="Caroline", type=P)],
        relations=[Rel("Caroline", "HAS_ATTRIBUTE", val, fact, evidence=ev, slot=slot)])
    ing, llm = make(gstore, {
        say("Caroline", a): attr("single", "Caroline is single", a, "relationship status"),
        say("Caroline", b): attr("dating", "Caroline is dating someone", b, "Relationship Status"),
        say("Caroline", c): attr("values self-care", "Caroline realized self-care matters", c)}, v)
    await run(ing, turn(a), turn(b, day=10), turn(c, day=11))
    assert llm.calls.count("contradiction") == 1  # only the same-slot pair; slotless attr not compared
    assert (await gstore.stats())["entities"] == 1  # attributes created no Concept entities
    cid = await node_id(gstore, "Caroline")
    cur = await gstore.current_facts(cid, RelType.HAS_ATTRIBUTE)
    assert {f.text for f in cur} == {"Caroline is dating someone", "Caroline realized self-care matters"}
    assert all(f.subject_id == cid and f.object_id is None for f in cur)
    res = await gstore.retrieve("Is Caroline single?", k=3, now=T0 + timedelta(days=12), reinforce=False)
    by = {f.text: f for f in res.facts}
    assert not by["Caroline is single"].is_current and by["Caroline is dating someone"].is_current


async def test_event_endpoints_never_compared(gstore):
    a, b = "I plan to go to the pride parade", "I plan to go to the art fair"
    ing, llm = make(gstore, {
        say("Caroline", a): ex_rel("Caroline", P, "PLANS", "pride parade (July 2023)", E, "Caroline plans the parade", evidence=a),
        say("Caroline", b): ex_rel("Caroline", P, "PLANS", "art fair (July 2023)", E, "Caroline plans the art fair", evidence=b)})
    await run(ing, turn(a), turn(b, day=1))
    assert "contradiction" not in llm.calls


async def test_mode_off_never_retires(gstore):
    a, b = "I live in Boston", "I moved to Sweden"
    ing, llm = make(gstore, {
        say("Caroline", a): ex_rel("Caroline", P, "LIVES_IN", "Boston", C, "Caroline lives in Boston", evidence=a),
        say("Caroline", b): ex_rel("Caroline", P, "LIVES_IN", "Sweden", C, "Caroline moved to Sweden", evidence=b)},
        mode="off")
    await run(ing, turn(a), turn(b, day=30))
    assert ing.stats.retire_log == [] and "contradiction" not in llm.calls
    assert (await gstore.stats())["current_facts"] == 2


async def test_restatement_reinforces(gstore):
    a, b = "I enjoy pottery", "I really enjoy doing pottery"
    ing, _ = make(gstore, {
        say("Melanie", a): ex_rel("Melanie", P, "LIKES", "pottery", C, "Melanie enjoys pottery", evidence=a),
        say("Melanie", b): ex_rel("Melanie", P, "LIKES", "pottery", C, "Melanie really enjoys doing pottery", evidence=b)})
    await run(ing, turn(a, speaker="Melanie"), turn(b, day=5, speaker="Melanie"))
    assert ing.stats.facts_created == 1 and ing.stats.facts_reinforced == 1
    assert [(x.action, x.reason) for x in ing.stats.retire_log] == [("reinforce", "restatement")]


async def test_extraction_cache_keyed_by_turn_and_endpoint(gstore):
    text = "I like tea"
    exs = {say("Caroline", text): ex_rel("Caroline", P, "LIKES", "tea", C, "Caroline likes tea", evidence=text)}
    ing, llm = make(gstore, exs)
    await ing.extractor.extract(turn(text))
    await ing.extractor.extract(turn(text))
    assert llm.calls == ["extraction"] and ing.extractor.hits == 1
    other = FakeLLM(exs)
    other.identity = {"provider": "fake", "base_url": "http://elsewhere/v1", "model": "fake"}
    await Extractor(other, gstore.s.cache_dir).extract(turn(text))
    assert other.calls == ["extraction"]


# ------------------------------------------------------------------ namespaces


def _conv(cid: str, turns: list[tuple[str, str]], start: datetime) -> Conversation:
    return Conversation(
        sample_id=cid, speaker_a="Caroline", speaker_b="Melanie", qa=[],
        sessions=[Session(index=1, time=start, turns=[
            Turn(dia_id=f"D1:{i + 1}", speaker=sp, text=tx) for i, (sp, tx) in enumerate(turns)])])


async def test_two_conversations_with_same_person_stay_isolated(gstore):
    """Same name ('Caroline') in two namespaces, ingested concurrently: no cross-namespace
    merges, retrieval, contradiction checks, pruning or clearing."""
    a_turn, b_turn = "I live in Boston and I love pottery.", "I live in Paris and I love chess."
    exs = {
        say("Caroline", a_turn): Extraction(
            entities=[Ent(name="Caroline", type=P), Ent(name="Boston", type=C), Ent(name="pottery", type=C)],
            relations=[Rel("Caroline", "LIVES_IN", "Boston", "Caroline lives in Boston", evidence="I live in Boston"),
                       Rel("Caroline", "LIKES", "pottery", "Caroline loves pottery", evidence="I love pottery")]),
        say("Caroline", b_turn): Extraction(
            entities=[Ent(name="Caroline", type=P), Ent(name="Paris", type=C), Ent(name="chess", type=C)],
            relations=[Rel("Caroline", "LIVES_IN", "Paris", "Caroline lives in Paris", evidence="I live in Paris"),
                       Rel("Caroline", "LIKES", "chess", "Caroline loves chess", evidence="I love chess")]),
    }
    llm = FakeLLM(exs)
    s = gstore.s.model_copy(update={"contradiction_mode": "rules"})
    ings = await ingest_many(gstore, [_conv("iso-a", [("Caroline", a_turn)], T0),
                                      _conv("iso-b", [("Caroline", b_turn)], T0 + timedelta(days=1))], llm, s)
    a, b = ings["iso-a"].store, ings["iso-b"].store
    try:
        for ing in ings.values():
            assert ing.stats.nodes_created == 3 and ing.stats.merged_exact == 0
            assert ing.stats.retire_log == []  # exclusive LIVES_IN did not see the other Caroline
        ca, cb = await a.find_by_name("Caroline", P), await b.find_by_name("Caroline", P)
        assert ca.id != cb.id and await a.get_node(cb.id) is None
        assert [f.text for f in await a.current_facts(ca.id, RelType.LIVES_IN)] == ["Caroline lives in Boston"]

        ra = await a.retrieve("Where does Caroline live? chess Paris", k=5, now=T0 + timedelta(days=2),
                              reinforce=False)
        texts = {f.text for f in ra.facts}
        assert texts == {"Caroline lives in Boston", "Caroline loves pottery"}

        await a.prune(threshold=2.0, now=T0 + timedelta(days=2))  # archives every fact in A
        assert (await a.stats())["current_facts"] == 0 and (await b.stats())["current_facts"] == 2
        await a.clear()
        assert (await b.stats())["entities"] == 3
    finally:
        await a.clear()
        await b.clear()


# ------------------------------------------------------------------ LLM client (no network)


class ScriptedBackend:
    provider, base_url, model = "openai-compat", "http://x/v1", "m"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.n = self.active = self.max_active = 0

    async def acomplete(self, req, schema):
        self.n += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        out = self.outputs.pop(0) if self.outputs else '{"entities": [], "relations": []}'
        if isinstance(out, Exception):
            raise out
        return Completion(text=out, input_tokens=100, output_tokens=10)


def _err(code):
    e = RuntimeError(f"HTTP {code}")
    e.code = code
    return e


async def test_llm_retry_cache_fences_and_usage(tmp_path, monkeypatch):
    monkeypatch.setattr("graphmem.llm._retry_delay",
                        lambda exc, attempt: 0.0 if getattr(exc, "code", 0) in (429, 503) else None)
    be = ScriptedBackend([_err(503), _err(429), '```json\n{"entities": [], "relations": []}\n```'])
    llm = LLM(be, str(tmp_path), retry_budget_s=60)
    out, meta = await llm.generate_json_meta("p", Extraction, purpose="extraction")
    assert out == Extraction() and be.n == 3 and meta.fence_stripped and not meta.repaired
    assert await llm.generate_json("p", Extraction, purpose="extraction") == Extraction() and be.n == 3
    u = llm.tracker.summary()["extraction"]
    assert (u["calls"], u["cache_hits"], u["retries"]) == (1, 1, 2)
    assert u["json_ok"] == 2 and u["fences_stripped"] == 2


async def test_llm_concurrency_cap(tmp_path):
    be = ScriptedBackend([])
    llm = LLM(be, str(tmp_path), max_concurrency=4)
    await asyncio.gather(*(llm.generate(f"p{i}") for i in range(20)))
    assert be.n == 20 and be.max_active == 4


async def test_llm_repair_then_fail(tmp_path):
    be = ScriptedBackend(['{"entities": "nope"}', '{"entities": [], "relations": []}'])
    llm = LLM(be, str(tmp_path))
    out, meta = await llm.generate_json_meta("p", Extraction, purpose="extraction")
    assert out == Extraction() and meta.repaired and be.n == 2
    llm2 = LLM(ScriptedBackend(["not json", "still not json"]), str(tmp_path / "2"))
    with pytest.raises(Exception):
        await llm2.generate_json("q", Extraction, purpose="extraction")
    assert llm2.tracker.summary()["extraction"]["json_failed"] == 1


def test_openai_compat_request_body():
    be = OpenAICompatBackend("http://localhost:11435/v1/", "qwen3:14b")
    body = be._body({"system": "sys", "prompt": "hi", "temperature": 0.0, "max_output_tokens": 50}, Extraction)
    assert body["reasoning_effort"] == "none"
    assert body["response_format"]["json_schema"]["schema"] == Extraction.model_json_schema()
    assert "evidence" in Extraction.model_json_schema()["$defs"]["ExtractedRelation"]["required"]


async def test_preflight_fails_fast_when_tunnel_down():
    with pytest.raises(PreflightError, match="SSH tunnel"):
        await OpenAICompatBackend("http://127.0.0.1:9/v1", "qwen3:14b").preflight()


# ------------------------------------------------------------------ ended roles (Door Dash)


def test_says_role_ended():
    from graphmem.extraction import says_role_ended
    for yes in ["Unfortunately, I also lost my job at Door Dash this month.",
                "Since I lost my job at Door Dash, things have been tough.",
                "I quit the bank last week", "I was laid off in March", "I was fired from Acme",
                "I'm no longer at the company", "I left my job to start a studio"]:
        assert says_role_ended(yes), yes
    for no in ["I won't quit - your words motivate me!", "Don't quit, remember!",
               "I opened an online clothing store after losing my job",
               "I've been working hard on my online store", "never quit on your dreams"]:
        assert not says_role_ended(no), no


async def test_door_dash_job_loss_is_ended_not_current(gstore):
    """conv-30: Gina's real turns. The model extracted WORKS_AT Door Dash with ended=False;
    the post-check must turn it into an ended fact, and the store must not keep a current
    WORKS_AT Door Dash, while the (current) online store survives."""
    d6_4 = ("Thanks, Jon! Appreciate your offer. Since I lost my job at Door Dash, things have "
            "been tough. But here's some good news - I've got something to share!")
    d5_5 = "I've been working hard on my online store"
    d14_8 = ("Ha, yeah, Jon. I've been one 'cause I lost my job. I opened an online clothing store "
             "and it's been great! Being my own boss and doing something I love is awesome.")
    ents = lambda org: [Ent(name="Gina", type=P), Ent(name=org, type=O)]
    exs = {
        say("Gina", "I worked at Door Dash delivering food."): Extraction(
            entities=ents("Door Dash"),
            relations=[Rel("Gina", "WORKS_AT", "Door Dash", "Gina works at Door Dash",
                           evidence="I worked at Door Dash")]),
        say("Gina", d5_5): Extraction(
            entities=ents("Gina's online store"),
            relations=[Rel("Gina", "WORKS_AT", "Gina's online store", "Gina works at her online store",
                           evidence=d5_5)]),
        say("Gina", d6_4): Extraction(  # exactly what qwen3:14b produced (ended=False)
            entities=ents("Door Dash"),
            relations=[Rel("Gina", "WORKS_AT", "Door Dash", "Gina lost her job at Door Dash.",
                           evidence="lost my job at Door Dash")]),
        say("Gina", d14_8): Extraction(
            entities=ents("Gina's online store"),
            relations=[Rel("Gina", "WORKS_AT", "Gina's online store",
                           "Gina opened an online clothing store after losing her job.",
                           evidence="I opened an online clothing store")]),
    }
    for mode in ("rules", "llm", "off"):
        await gstore.clear()
        ing, llm = make(gstore, exs, mode=mode)
        t_d6 = turn(d6_4, day=50, speaker="Gina", listener="Jon", dia="D6:4")
        await run(ing, turn("I worked at Door Dash delivering food.", speaker="Gina", listener="Jon", dia="D1:3"),
                  turn(d5_5, day=40, speaker="Gina", listener="Jon", dia="D5:5"), t_d6,
                  turn(d14_8, day=140, speaker="Gina", listener="Jon", dia="D14:8"))
        rec = ing.extractor.records[t_d6.dia_id]
        assert rec.extraction.relations[0].ended  # rule fired on the model's output
        gid = await node_id(gstore, "Gina")
        cur = {f.text for f in await gstore.current_facts(gid, RelType.WORKS_AT)}
        assert "Gina lost her job at Door Dash." not in cur, mode
        assert "Gina works at her online store" in cur, mode  # the store is not retired
        if mode != "off":
            assert "Gina works at Door Dash" not in cur, mode  # ended fact retired the old one
            (r,) = [x for x in ing.stats.retire_log if x.action == "retire"]
            assert (r.reason, r.old_fact) == ("ended", "Gina works at Door Dash")
        assert "contradiction" not in llm.calls or mode == "llm"


def test_partner_of_requires_romantic_evidence():
    t = turn("I partnered with a local artist for some cool designs. My husband loves them.", speaker="Gina",
             listener="Jon")
    ex = Extraction(
        entities=[Ent(name="Gina", type=P), Ent(name="local artist", type=P), Ent(name="Gina's husband", type=P)],
        relations=[Rel("Gina", "PARTNER_OF", "local artist", "Gina partnered with a local artist",
                       evidence="I partnered with a local artist"),
                   Rel("Gina", "PARTNER_OF", "Gina's husband", "Gina is married",
                       evidence="My husband loves them")])
    res = sanitize(ex, t)
    assert [r.relation for r in res.extraction.relations] == ["HAS_ATTRIBUTE", "PARTNER_OF"]
    assert res.converted[0]["original_type"] == "PARTNER_OF(non-romantic)"


async def test_llm_retry_budget_then_gives_up(tmp_path, monkeypatch):
    import httpx
    clock = {"t": 0.0}
    monkeypatch.setattr("graphmem.llm.time.monotonic", lambda: clock["t"])

    async def fake_sleep(d):
        clock["t"] += d

    monkeypatch.setattr("graphmem.llm.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("graphmem.llm.random.uniform", lambda a, b: 0.0)
    be = ScriptedBackend([httpx.ConnectError("tunnel down")] * 100)
    llm = LLM(be, str(tmp_path), retry_budget_s=900)
    with pytest.raises(httpx.ConnectError):
        await llm.generate("p")
    assert 800 < clock["t"] <= 900 and be.n > 5  # kept retrying ~15 min, then failed
