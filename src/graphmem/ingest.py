"""Turn-by-turn ingestion: extract -> resolve entities -> write facts -> retire contradictions.

Everything happens inside one namespace (conversation); see GraphStore. Each extracted
relation becomes a Fact node linked to its subject (and object) entity.

Entity resolution (same type only):
  1. normalized exact name/alias match;
  2. else name-embedding cosine >= resolution_threshold (0.94, provisional); for Events
     the bar is event_resolution_threshold (0.95) AND the date intervals in the two names
     must overlap (or at least one is undated). Blocked Event merges are logged.

Fact handling. Restatements are merged in every mode:
  - same (subject, type, object), or a same-type fact with cosine >= duplicate_fact_cosine
    -> reinforce the existing fact (logged as reason=restatement, action=reinforce).
Retirement depends on settings.contradiction_mode ("llm" | "rules" | "off"); rules run
first and the LLM last:
  - ended: an `ended` fact retires the same (subject, type, object) current fact.
  - preferences (LIKES / DISLIKES / INTERESTED_IN) are NEVER sent to the LLM. They are
    retired only by explicit negation ("no longer", "stopped", "quit", ended=True) or by
    the opposite polarity, and only for the SAME object (object-entity name cosine >= 0.9).
  - slot relations (LIVES_IN, WORKS_AT, STUDIES_AT, PARTNER_OF, MEMBER_OF, PLANS, and
    HAS_ATTRIBUTE with the same non-null slot): candidates are the subject's current
    facts of the same type (same slot), excluding anything with an Event endpoint.
      rules: exclusive types (and same-slot attributes) retire the candidates directly.
      llm:   each (new, candidate) pair is labelled same / compatible / outdated by the LLM.
  - PARTICIPATED_IN and anything with an Event endpoint is never compared.
Retired facts get validity_end = now and stay in the graph as history.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from .clock import SimClock
from .config import Settings
from .extraction import Extraction, ExtractedRelation, Extractor, JsonLLM, TurnInput
from .locomo import Conversation
from .schema import (
    LLM_SLOT_TYPES,
    PREFERENCE_TYPES,
    RELATIONS,
    Fact,
    FactCreate,
    NodeCreate,
    NodeType,
    RelType,
)
from .store import GraphStore
from .timehints import intervals_compatible, name_interval

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- LLM check


class PairJudgement(BaseModel):
    pair: int
    label: Literal["same", "compatible", "outdated"]


class ContradictionVerdict(BaseModel):
    judgements: list[PairJudgement] = Field(default_factory=list)


# Pairs only ever come from slot relations (see module docstring). The "compatible"
# examples are the 16 wrongful retirements from the first conv-26 run (dev data).
CONTRADICTION_SYSTEM = """You compare facts in a person's long-term memory.
For each numbered pair, decide how the EXISTING fact relates to the NEW fact:
- "same": the two say essentially the same thing (paraphrase), or the NEW fact is a
  vaguer restatement of the EXISTING one.
- "compatible": both can be true at the same time: related facts, extra detail,
  a different aspect, two different events/objects/photos/artworks, or two
  preferences that can coexist.
- "outdated": the EXISTING fact is no longer true because of the NEW fact
  (the new fact contradicts it, replaces its value, or says it has ended).
Most pairs are "compatible". Answer "outdated" only if both cannot be true now.

These were all WRONGLY marked outdated in the past; the correct labels are shown:
- NEW "Melanie went on a camping trip with her family on 15 July 2023" / EXISTING "Melanie participated in Melanie's wedding ceremony on 15 July 2023" -> compatible
- NEW "Caroline shared a photo of a painting of a heart on a table" / EXISTING "Caroline created a painting of a person holding a purple bowl" -> compatible
- NEW "Melanie painted a piece inspired by sunsets on 6 October 2023" / EXISTING "Melanie created a painting of a sunset with a palm tree on 10 July 2023" -> compatible
- NEW "Caroline painted a piece for an art show" / EXISTING "Caroline created a painting of a tree with a bright sun for her art show" -> same
- NEW "Caroline finds counseling and mental health work tough but rewarding" / EXISTING "Caroline is interested in working in mental health" -> compatible
- NEW "Melanie is a big fan of pottery" / EXISTING "Melanie likes painting as a fun way to express her feelings" -> compatible
- NEW "Melanie shared a photo of a flower drawing on the ground" / EXISTING "Melanie shared a photo of a plate with a bunch of flowers on it" -> compatible
- NEW "Melanie was at the cafe on the weekend before 13 September 2023" / EXISTING "Melanie went camping with the kids on 9 September 2023" -> compatible
- NEW "Caroline participated in the Pride parade on 11 August 2023" / EXISTING "Caroline attended the LGBTQ+ pride parade the week before 3 July 2023" -> compatible
- NEW "Melanie shared a photo of a plate with a bunch of flowers" / EXISTING "Melanie shared a photo of a blue vase with sunflowers and roses" -> compatible
- NEW "Caroline visited an LGBTQ center, which inspired her painting" / EXISTING "Caroline visited an LGBTQ center on 16 July 2023" -> same
- NEW "Melanie created an abstract painting with a blue background" / EXISTING "Melanie created a painting of a sunset with a pink sky" -> compatible
- NEW "Melanie loves Caroline's painting with a red and blue background" / EXISTING "Melanie admires Caroline's sunset painting" -> compatible
- NEW "Melanie likes painting landscapes and still life" / EXISTING "Melanie likes to paint animals because it is peaceful" -> compatible
- NEW "Melanie shared a photo of a plate with a bunch of flowers" / EXISTING "Melanie shared a photo of a person holding a purple bowl" -> compatible
- NEW "Melanie likes classical music, such as Mozart" / EXISTING "Melanie has a daily me-time routine of running, reading or playing the violin" -> compatible
Real "outdated" examples:
- NEW "Caroline moved to Sweden in October 2023" / EXISTING "Caroline lives in Boston" -> outdated
- NEW "Jon now works as a dance studio owner" / EXISTING "Jon's job title is bank teller" -> outdated
- NEW "Melanie's daughter turned 8 in July 2023" / EXISTING "Melanie's daughter is 7 years old" -> outdated

Judge every pair. Return JSON only."""

_NEGATION = re.compile(
    r"\b(no longer|not anymore|anymore|stopped|quit|quitting|gave up|given up|lost interest|"
    r"don'?t (?:like|enjoy|love) .{0,20}any ?more|not into)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------- logs/stats


class MergeLogEntry(BaseModel):
    turn: str
    time: datetime
    incoming: str
    type: NodeType
    merged_into: str  # existing entity name
    node_id: str
    method: Literal["exact", "embedding"]
    score: float  # 1.0 for exact; name cosine for embedding


class BlockedMerge(BaseModel):
    turn: str
    incoming: str
    candidate: str
    cosine: float
    reason: str


class RetireLogEntry(BaseModel):
    turn: str
    time: datetime
    action: Literal["retire", "reinforce"]
    reason: Literal["restatement", "ended", "schema_rule", "llm"]
    detail: str = ""
    old_fact: str
    old_fact_id: str
    new_fact: str
    score: float  # cosine(old fact, new fact)


class IngestStats(BaseModel):
    conversation: str = ""
    turns: int = 0
    extraction_cache_hits: int = 0
    extraction_failed: int = 0
    entities_kept: int = 0
    relations_kept: int = 0
    facts_dropped: dict[str, int] = Field(default_factory=dict)
    converted_unknown_type: int = 0
    fuzzy_evidence_accepted: int = 0
    nodes_created: int = 0
    merged_exact: int = 0
    merged_embedding: int = 0
    blocked_event_merges: int = 0
    facts_created: int = 0
    facts_reinforced: int = 0
    facts_stored_closed: int = 0  # ended facts with nothing to retire (kept as history)
    retired_ended: int = 0
    retired_schema_rule: int = 0
    retired_llm: int = 0
    llm_checks: int = 0  # contradiction LLM calls
    llm_check_failed: int = 0
    llm_pairs: int = 0
    llm_pairs_unlabelled: int = 0
    llm_labels: dict[str, int] = Field(default_factory=dict)
    apply_s: float = 0.0
    merge_log: list[MergeLogEntry] = Field(default_factory=list, repr=False)
    blocked_merges: list[BlockedMerge] = Field(default_factory=list, repr=False)
    retire_log: list[RetireLogEntry] = Field(default_factory=list, repr=False)


def _iv(name: str) -> str:
    iv = name_interval(name)
    return "undated" if iv is None else f"{iv[0]}..{iv[1]}"


def turn_inputs(conv: Conversation, context_turns: int, sessions: int | None = None) -> list[tuple[TurnInput, datetime]]:
    """Build extractor inputs with their simulated write time (session time + i seconds)."""
    out = []
    people = {conv.speaker_a: conv.speaker_b, conv.speaker_b: conv.speaker_a}
    for sess in conv.sessions[:sessions]:
        rendered = [t.render() for t in sess.turns]
        for i, t in enumerate(sess.turns):
            ti = TurnInput(
                dia_id=t.dia_id,
                speaker=t.speaker,
                listener=people.get(t.speaker, "the other speaker"),
                text=rendered[i],
                session_time=sess.time,
                context=rendered[max(0, i - context_turns):i] if context_turns else [],
            )
            out.append((ti, sess.time + timedelta(seconds=i)))
    return out


# ---------------------------------------------------------------------------- ingestor


class Ingestor:
    def __init__(
        self,
        store: GraphStore,
        extractor: Extractor,
        llm: JsonLLM,
        settings: Settings | None = None,
    ):
        self.store = store
        self.extractor = extractor
        self.llm = llm
        self.s = settings or store.s
        self.stats = IngestStats(conversation=store.namespace)

    @property
    def mode(self) -> str:
        return self.s.contradiction_mode

    # ------------------------------------------------------------ driver

    async def ingest_conversation(
        self,
        conv: Conversation,
        sessions: int | None = None,
        progress: Callable[[str, str, int, int], None] | None = None,
    ) -> IngestStats:
        items = turn_inputs(conv, self.s.extraction_context_turns, sessions)
        # Source turns, for provenance at retrieval time.
        await self.store.add_turns([{"dia_id": ti.dia_id, "text": ti.text, "session_time": ti.session_time}
                                    for ti, _ in items])

        # 1. Extract every turn concurrently (independent of graph state; cached).
        done = 0
        hits0 = self.extractor.hits

        async def one(ti: TurnInput) -> Extraction:
            nonlocal done
            ex = await self.extractor.extract(ti)
            done += 1
            if progress and (done % 50 == 0 or done == len(items)):
                progress(conv.sample_id, "extract", done, len(items))
            return ex

        extractions = await asyncio.gather(*(one(ti) for ti, _ in items))
        st = self.stats
        st.extraction_cache_hits += self.extractor.hits - hits0
        st.extraction_failed += sum(self.extractor.records[ti.dia_id].failed for ti, _ in items)
        st.facts_dropped = self.extractor.drop_counts()
        st.converted_unknown_type = len(self.extractor.converted())
        st.fuzzy_evidence_accepted = len(self.extractor.fuzzy_accepted())

        # 2. Apply strictly in conversation order on simulated time.
        clock = self.store.clock
        t0 = asyncio.get_running_loop().time()
        for n, ((ti, t), ex) in enumerate(zip(items, extractions), 1):
            if isinstance(clock, SimClock):
                clock.set(t)
            await self.apply(ti, ex, t)
            if progress and (n % 50 == 0 or n == len(items)):
                progress(conv.sample_id, "apply", n, len(items))
        st.apply_s += asyncio.get_running_loop().time() - t0
        return st

    # ------------------------------------------------------------ entity resolution

    async def _resolve(self, e, name_vec, turn: TurnInput, now: datetime) -> str:
        st = self.stats
        node, method, score = await self.store.find_by_name(e.name, e.type), "exact", 1.0
        if node is None:
            is_event = e.type is NodeType.EVENT
            thr = self.s.event_resolution_threshold if is_event else self.s.resolution_threshold
            for cand, cos in await self.store.similar_entities(name_vec, e.type, thr, limit=3):
                if is_event and not intervals_compatible(e.name, cand.name):
                    st.blocked_event_merges += 1
                    st.blocked_merges.append(BlockedMerge(
                        turn=turn.dia_id, incoming=e.name, candidate=cand.name, cosine=round(cos, 4),
                        reason=f"date intervals disjoint: {_iv(e.name)} vs {_iv(cand.name)}"))
                    continue
                node, method, score = cand, "embedding", cos
                break
        if node is not None:
            await self.store.merge_entity(node.id, alias=e.name, summary=e.summary, now=now)
            if method == "exact":
                st.merged_exact += 1
            else:
                st.merged_embedding += 1
            st.merge_log.append(MergeLogEntry(
                turn=turn.dia_id, time=now, incoming=e.name, type=e.type,
                merged_into=node.name, node_id=node.id, method=method, score=round(score, 4)))
            return node.id
        node = await self.store.add_node(
            NodeCreate(type=e.type, name=e.name, summary=e.summary), now, name_embedding=name_vec)
        st.nodes_created += 1
        return node.id

    # ------------------------------------------------------------ one turn

    async def apply(self, turn: TurnInput, ex: Extraction, now: datetime) -> None:
        st = self.stats
        st.turns += 1
        st.entities_kept += len(ex.entities)
        st.relations_kept += len(ex.relations)
        if not ex.relations:
            return

        emb = self.store.embedder
        vecs = await emb.aembed([e.name for e in ex.entities])
        ids: dict[str, tuple[str, NodeType, list[float]]] = {}
        for e, name_vec in zip(ex.entities, vecs):
            node_id = await self._resolve(e, name_vec, turn, now)
            ids[e.name.lower()] = (node_id, e.type, name_vec)

        fact_vecs = await emb.aembed([r.fact for r in ex.relations])
        pending: list[tuple[Fact, list[float], list[Fact]]] = []  # (new fact, vec, LLM candidates)
        new_ids: set[str] = set()
        for r, vec in zip(ex.relations, fact_vecs):
            out = await self._apply_relation(r, vec, ids, turn, now, new_ids)
            if out is not None:
                pending.append(out)
        if self.mode == "llm":
            pairs = [(fact, vec, c) for fact, vec, cands in pending for c in cands
                     if c.id not in new_ids][: self.s.contradiction_candidates * 4]
            if pairs:
                await self._llm_check(pairs, turn, now)

    async def _apply_relation(self, r: ExtractedRelation, vec, ids, turn: TurnInput, now: datetime,
                              new_ids: set[str]) -> tuple[Fact, list[float], list[Fact]] | None:
        st = self.stats
        rtype = RelType(r.relation)
        spec = RELATIONS[rtype]
        sid, s_t, _ = ids[r.source.lower()]
        if spec.attribute:
            oid, o_t, o_vec = None, None, None
        else:
            oid, o_t, o_vec = ids[r.target.lower()]
        negated = r.ended or bool(_NEGATION.search(r.evidence) or _NEGATION.search(r.fact))
        has_event = NodeType.EVENT in (s_t, o_t)

        # --- restatement (all modes); a negated fact is never a restatement
        if not negated:
            same = [] if spec.attribute else await self.store.current_facts(sid, rtype, oid)
            if not same:
                same = [f for f, _ in await self.store.similar_current_facts(
                    sid, vec, self.s.duplicate_fact_cosine, limit=3) if f.relation_type == rtype]
            if same:
                await self.store.reinforce_fact(same[0].id, now)
                await self._log(same[0], now, "reinforce", "restatement", "", turn, r.fact, vec)
                st.facts_reinforced += 1
                return None

        # --- deterministic retirement rules
        if self.mode != "off":
            if rtype in PREFERENCE_TYPES and o_vec is not None:
                if negated:
                    # "no longer likes X": retire same-polarity prefs for the same object
                    types = ([RelType.LIKES, RelType.INTERESTED_IN] if rtype is not RelType.DISLIKES
                             else [RelType.DISLIKES])
                    for f, ocos in await self.store.facts_with_similar_object(
                            sid, types, o_vec, self.s.preference_target_cosine):
                        await self._retire(f, now, "ended", turn, r.fact, vec,
                                           f"negated preference; object cos={ocos:.3f}")
                        st.retired_ended += 1
                else:
                    opposite = ([RelType.LIKES, RelType.INTERESTED_IN] if rtype is RelType.DISLIKES
                                else [RelType.DISLIKES])
                    for f, ocos in await self.store.facts_with_similar_object(
                            sid, opposite, o_vec, self.s.preference_target_cosine):
                        await self._retire(f, now, "schema_rule", turn, r.fact, vec,
                                           f"opposite polarity {f.relation_type.value}->{rtype.value}; "
                                           f"object cos={ocos:.3f}")
                        st.retired_schema_rule += 1
            elif r.ended and not spec.attribute:
                for f in await self.store.current_facts(sid, rtype, oid):
                    await self._retire(f, now, "ended", turn, r.fact, vec)
                    st.retired_ended += 1

        # --- write the new fact
        fact = await self.store.add_fact(FactCreate(
            relation_type=rtype, subject_id=sid, object_id=oid, text=r.fact, evidence=r.evidence,
            turn_id=turn.dia_id, validity_start=now,
            slot=r.slot if spec.attribute else None, value=r.target if spec.attribute else None,
        ), now, embedding=vec)
        new_ids.add(fact.id)
        if r.ended or (negated and rtype in PREFERENCE_TYPES):
            await self.store.invalidate_fact(fact.id, at=now)
            st.facts_stored_closed += 1
            return None
        st.facts_created += 1

        # --- slot relations: candidates for rule retirement or LLM adjudication
        if self.mode == "off" or rtype not in LLM_SLOT_TYPES or has_event:
            return None
        if spec.attribute and not r.slot:
            return None
        cands = [f for f in await self.store.current_facts(
            sid, rtype, slot=r.slot if spec.attribute else None, exclude_event_endpoints=True)
            if f.id not in new_ids and (spec.attribute or f.object_id != oid)]
        cands = sorted(cands, key=lambda f: f.validity_start, reverse=True)[: self.s.contradiction_candidates]
        if not cands:
            return None
        if self.mode == "rules":
            if spec.exclusive or spec.attribute:
                for f in cands:
                    why = f"same slot '{r.slot}'" if spec.attribute else f"exclusive {rtype.value}"
                    await self._retire(f, now, "schema_rule", turn, r.fact, vec, why)
                    st.retired_schema_rule += 1
            return None
        return fact, vec, cands

    # ------------------------------------------------------------ logging helpers

    async def _log(self, old: Fact, now: datetime, action: str, reason: str, detail: str,
                   turn: TurnInput, new_fact: str, new_vec: list[float]) -> None:
        old_vec = (await self.store.embedder.aembed([old.text]))[0]
        self.stats.retire_log.append(RetireLogEntry(
            turn=turn.dia_id, time=now, action=action, reason=reason, detail=detail,
            old_fact=old.text, old_fact_id=old.id, new_fact=new_fact,
            score=round(float(np.dot(old_vec, new_vec)), 4)))

    async def _retire(self, f: Fact, now: datetime, reason: str, turn: TurnInput,
                      new_fact: str, new_vec: list[float], detail: str = "") -> None:
        if await self.store.invalidate_fact(f.id, at=now) is not None:
            await self._log(f, now, "retire", reason, detail, turn, new_fact, new_vec)

    # ------------------------------------------------------------ LLM adjudication

    async def _llm_check(self, pairs: list[tuple[Fact, list[float], Fact]],
                         turn: TurnInput, now: datetime) -> None:
        """Label each (new fact, candidate) pair same / compatible / outdated.
        outdated -> retire the existing fact; same -> drop the new duplicate fact and
        reinforce the existing one; compatible -> nothing."""
        st = self.stats
        prompt = f"Session date: {turn.session_time:%d %B %Y}\n\n" + "\n".join(
            f"[P{p}] NEW: {new.text}\n     EXISTING: {old.text}"
            for p, (new, _, old) in enumerate(pairs))
        st.llm_checks += 1
        try:
            verdict, _ = await self.llm.generate_json_meta(
                prompt, ContradictionVerdict, system=CONTRADICTION_SYSTEM, purpose="contradiction",
                max_output_tokens=self.s.contradiction_max_tokens)
        except ValidationError:
            st.llm_check_failed += 1
            return
        labels = {j.pair: j.label for j in verdict.judgements if 0 <= j.pair < len(pairs)}
        st.llm_pairs += len(pairs)
        st.llm_pairs_unlabelled += len(pairs) - len(labels)
        for lab in labels.values():
            st.llm_labels[lab] = st.llm_labels.get(lab, 0) + 1
        retired: set[str] = set()
        dropped: set[str] = set()
        for p in sorted(labels, key=lambda p: labels[p] != "outdated"):
            new, new_vec, old = pairs[p]
            if labels[p] == "outdated" and old.id not in retired:
                retired.add(old.id)
                rt = new.relation_type
                detail = f"exclusive {rt.value}" if RELATIONS[rt].exclusive else rt.value
                await self._retire(old, now, "llm", turn, new.text, new_vec, detail)
                st.retired_llm += 1
            elif labels[p] == "same" and new.id not in dropped and old.id not in retired:
                dropped.add(new.id)
                await self.store.delete_fact(new.id)
                await self.store.reinforce_fact(old.id, now)
                await self._log(old, now, "reinforce", "restatement", "llm: same", turn, new.text, new_vec)
                st.facts_created -= 1
                st.facts_reinforced += 1

    def dump_logs(self) -> dict[str, list[dict[str, Any]]]:
        st = self.stats
        return {
            "merge_log": [m.model_dump(mode="json") for m in st.merge_log],
            "blocked_merges": [b.model_dump(mode="json") for b in st.blocked_merges],
            "retire_log": [r.model_dump(mode="json") for r in st.retire_log],
            "converted": self.extractor.converted(),
            "fuzzy_evidence": self.extractor.fuzzy_accepted(),
        }


# ---------------------------------------------------------------------------- parallel driver


async def ingest_many(
    root: GraphStore,
    convs: list[Conversation],
    llm: JsonLLM,
    settings: Settings,
    sessions: int | None = None,
    clear: bool = True,
    progress: Callable[[str, str, int, int], None] | None = None,
) -> dict[str, Ingestor]:
    """Ingest several conversations concurrently, each in its own namespace with its own
    simulated clock. They share `llm`, whose semaphore caps total LLM concurrency."""

    async def one(conv: Conversation) -> Ingestor:
        store = await root.scoped(conv.sample_id, SimClock())
        if clear:
            await store.clear()
        ing = Ingestor(store, Extractor(llm, settings.cache_dir, settings.extraction_max_tokens),
                       llm, settings)
        await ing.ingest_conversation(conv, sessions=sessions, progress=progress)
        return ing

    ings = await asyncio.gather(*(one(c) for c in convs))
    return {c.sample_id: ing for c, ing in zip(convs, ings)}
