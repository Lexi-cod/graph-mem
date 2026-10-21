"""LLM extraction of entities/relations from a single conversation turn.

The model's raw JSON (Pydantic-validated) is cached on disk keyed by a hash of the
turn (text, speaker, session time, context, prompt version, endpoint, model). The
deterministic post-checks in `sanitize` are re-applied on every load, so changing
a check never requires re-running the LLM. Every dropped fact is logged with a reason.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError
from rapidfuzz import fuzz

from .cache import DiskCache, stable_hash
from .schema import RELATIONS, NodeType, RelType, SchemaViolation, check_relation, name_key
from .timehints import rewrite_relative, time_hints

EXTRACTOR_VERSION = "v5"


class ExtractedEntity(BaseModel):
    name: str = Field(description="Canonical name, e.g. 'Priya', 'animal shelter'")
    type: NodeType
    summary: str = Field("", description="Short description grounded in this turn")


class ExtractedRelation(BaseModel):
    source: str = Field(description="Name of an entity in `entities` (the subject)")
    relation: str = Field(description="One of the allowed relation types")
    target: str = Field(description="Entity name; for HAS_ATTRIBUTE the attribute value text")
    evidence: str = Field(description="Verbatim quote from the CURRENT turn supporting the fact")
    fact: str = Field(description="Self-contained statement with absolute dates where known")
    ended: bool = Field(False, description="True only if the turn says this no longer holds")
    slot: str | None = Field(
        None, description="HAS_ATTRIBUTE only: single-valued slot such as 'age', "
                          "'relationship status', 'job title', 'hometown'; else null")


class Extraction(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class DroppedFact(BaseModel):
    reason: str
    relation: dict[str, Any]


class JsonLLM(Protocol):
    identity: dict[str, str]  # provider / base_url / model

    async def generate_json_meta(self, prompt: str, schema: type, *, system: str | None = None,
                                 purpose: str = "misc", max_output_tokens: int | None = None,
                                 ) -> tuple[BaseModel, Any]: ...


def _relation_doc() -> str:
    lines = []
    for rel, spec in RELATIONS.items():
        if spec.attribute:
            lines.append(f"- {rel.value}: any -> attribute value (text, not an entity)")
            continue
        dom = "/".join(sorted(spec.domain)) if spec.domain else "any"
        rng = "/".join(sorted(spec.range)) if spec.range else "any"
        lines.append(f"- {rel.value}: {dom} -> {rng}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You build a long-term memory of a conversation between two people.
From ONE conversation turn, extract the entities and relations worth remembering
about the people: facts, events, plans, preferences, relationships, possessions,
places, activities, traits and their dates.

Entity types:
- Person: named people. Resolve "I"/"me" to the speaker and "you" to the listener;
  unnamed relatives become e.g. "Priya's daughter".
- Organization: companies, schools, groups, teams, clubs, charities, shelters.
- Event: specific happenings: meetings, trips, races, parties, classes, concerts,
  ceremonies, visits, performances, appointments. Name them with who/what and the
  date when known, e.g. "Priya's charity 5K run (12 March 2023)".
- Concept: places, objects, topics, hobbies, books, pets, artworks, foods, careers.
Dates and times are NEVER entities; they go into `fact` text and Event names.
Use full words for names as the speaker means them ("education", not "edu").

Relation types (source -> target):
{_relation_doc()}
Use RELATED_TO only when nothing else fits. Use only these relation types.

Rules:
- Extract ONLY what the CURRENT turn says. Previous turns are context for resolving
  references; never extract facts that only appear in them.
- `evidence` is an exact, verbatim quote copied from the current turn (it may be
  part of a sentence) that supports the fact. Facts without such a quote are invalid.
- The subject (`source`) must be the speaker, the speaker's relative/possession
  (e.g. "Priya's sister"), or someone/something named in the current turn.
- Feelings, realizations, opinions, traits and states of a person are HAS_ATTRIBUTE:
  `target` is a short value phrase (e.g. "values self-care", "single", "32 years old")
  and is NOT listed in `entities`. Set `slot` only for single-valued attributes
  (age, relationship status, job title, hometown, ...), otherwise leave it null.
- FAMILY_OF only when the turn explicitly states a family relationship
  (mother, son, sister, husband, ...). Friends are not family.
- "missed", "didn't go", "couldn't attend" is NOT PARTICIPATED_IN; record it as
  HAS_ATTRIBUTE (e.g. "missed the pride parade on 15 July 2023") or skip it.
- Doing or attending something specific is PARTICIPATED_IN an Event (create the Event).
- Every relation's source, and every non-attribute target, must be listed in `entities`.
- `fact` is a complete, self-contained sentence using names, never pronouns, and
  keeps specific details (titles, counts, names, places, reasons).
- Every relative time in `fact` MUST be replaced by an absolute one computed from
  the session date. With session date 14 March 2023:
    "yesterday" -> "on 13 March 2023"      "last week" -> "the week before 14 March 2023"
    "last year" -> "in 2022"               "next month" -> "in April 2023"
  When "Resolved times" are listed for the turn, use exactly those values.
- Past events that happened are NOT ended. Set `ended` only when a state or
  relationship has stopped (e.g. "I used to live in Boston", "we broke up").
- Photo descriptions in [brackets] are what the speaker shared; extract what they show
  only if it says something about the people (their artwork, pet, trip, ...).
- Do not extract greetings, thanks, compliments, small talk, or that the two
  speakers know each other. A turn with no lasting information -> empty lists.

Example. Session date Tuesday 14 March 2023. Speaker Priya, listener Tom.
Resolved times for this turn: "last Sunday" = Sunday 12 March 2023
Turn: "Priya: I ran a charity 5K for the animal shelter last Sunday! My sister Ana came too. I missed the after-party though. Honestly it made me realize how much I love helping animals."
-> entities: Priya (Person); Priya's sister Ana (Person);
   Priya's charity 5K run (12 March 2023) (Event); animal shelter (Organization)
-> relations:
   Priya PARTICIPATED_IN Priya's charity 5K run (12 March 2023)
     evidence "I ran a charity 5K for the animal shelter last Sunday"
     fact "Priya ran a charity 5K for an animal shelter on Sunday 12 March 2023."
   Priya's sister Ana PARTICIPATED_IN Priya's charity 5K run (12 March 2023)
     evidence "My sister Ana came too" fact "Priya's sister Ana came to Priya's charity 5K run on 12 March 2023."
   Priya FAMILY_OF Priya's sister Ana  evidence "My sister Ana" fact "Ana is Priya's sister."
   Priya HAS_ATTRIBUTE "missed the 5K after-party"  evidence "I missed the after-party though"
     fact "Priya missed the after-party of her charity 5K run on 12 March 2023."
   Priya HAS_ATTRIBUTE "loves helping animals"  evidence "it made me realize how much I love helping animals"
     fact "Running the charity 5K made Priya realize how much she loves helping animals."

Return JSON only."""


class TurnInput(BaseModel):
    dia_id: str
    speaker: str
    listener: str
    text: str  # already rendered: "Speaker: text [shares a photo: ...]"
    session_time: datetime
    context: list[str] = Field(default_factory=list)  # previous rendered turns in the session

    def prompt(self) -> str:
        ctx = "\n".join(self.context) if self.context else "(start of session)"
        hints = time_hints(self.text, self.session_time)
        resolved = ""
        if hints:
            resolved = "Resolved times for this turn: " + "; ".join(
                f'"{e}" = {v}' for e, v in hints) + "\n\n"
        return (
            f"Session date: {self.session_time:%A %d %B %Y, %H:%M}\n"
            f"Speaker: {self.speaker}   Listener: {self.listener}\n\n"
            f"Previous turns (CONTEXT ONLY - do not extract from these):\n{ctx}\n\n"
            f"{resolved}"
            f"CURRENT turn (extract only from this):\n{self.text}"
        )

    def key(self, llm_identity: dict[str, str]) -> str:
        return stable_hash(
            {"v": EXTRACTOR_VERSION, **llm_identity, "system": SYSTEM_PROMPT, "prompt": self.prompt()}
        )


# ---------------------------------------------------------------------------- post-checks

# All quote-like characters compare equal.
_QUOTES = str.maketrans({c: "'" for c in "’‘“”´`\""})
_ELLIPSIS = re.compile(r"…|\.\.\.")
_FAMILY = re.compile(
    r"\b(mom|mother|mum|dad|father|parents?|sister|brother|siblings?|sons?|daughters?|kids?|"
    r"child|children|husband|wife|spouse|grand(?:ma|pa|mother|father|kids?|children|son|daughter)|"
    r"aunt|uncle|cousin|niece|nephew|in-laws?|family|twins?|stepson|stepdaughter|stepmom|stepdad)\b",
    re.IGNORECASE)
# PARTNER_OF is an exclusive slot, so it must be a romantic/marital partner. Business or
# figurative "partners" (conv-30: "Gina partnered with a local artist") retired real facts.
_ROMANTIC = re.compile(
    r"\b(husband|wife|spouse|hubby|boyfriend|girlfriend|fianc[ée]e?|married|marriage|wedding|"
    r"dating|date[ds]?|romantic|engaged|my partner|our relationship|in a relationship|broke up)\b",
    re.IGNORECASE)
_NOT_ATTENDED = re.compile(
    r"\b(missed|miss|didn'?t (?:go|attend|make it)|did not (?:go|attend|make it)|"
    r"couldn'?t (?:go|attend|make it)|could not (?:go|attend|make it)|wasn'?t able to|"
    r"was not able to|skipped|won'?t be able to)\b", re.IGNORECASE)
# Employment/membership that has ended: "lost my job at X", "quit", "left", "was fired",
# "laid off", "no longer at X". Never a current WORKS_AT (conv-30: "Gina lost her job at
# Door Dash" was stored as a current WORKS_AT and later misused by the contradiction check).
_ENDED_ROLE = re.compile(
    r"\b(lost (?:my |her |his |their |the )?(?:job|position|role)|lost (?:it|that job)|"
    r"quit(?:ting)?|left (?:my |her |his |their |the )?(?:job|position|role|company|team|club|school)|"
    r"(?:was|were|got|been|being) (?:fired|laid off|let go)|laid off|let go from|"
    r"resign(?:ed|ing)|no longer (?:work(?:s|ing)?|employed|at|with|a member))\b",
    re.IGNORECASE)
# Negated uses are not statements that the role ended ("I won't quit", "never quit").
_ENDED_ROLE_BLOCKERS = re.compile(
    r"(?:won'?t|will not|don'?t|do not|never|not|didn'?t|did not)\s+(?:\w+\s+){0,1}$",
    re.IGNORECASE)
ENDED_ROLE_TYPES = frozenset({RelType.WORKS_AT, RelType.STUDIES_AT, RelType.MEMBER_OF})


def says_role_ended(text: str) -> bool:
    """True if `text` states that a job/membership ended ("I lost my job at Door Dash"),
    ignoring negated or subordinate uses ("I won't quit", "...after losing her job")."""
    for m in _ENDED_ROLE.finditer(text):
        if not _ENDED_ROLE_BLOCKERS.search(text[:m.start()]):
            return True
    return False


_EDGE_PUNCT = " '.,;:!?-–—"
EVIDENCE_FUZZY_MIN = 0.9


def _ws(s: str) -> str:
    return " ".join(s.translate(_QUOTES).split()).lower()


def _fragments(evidence: str) -> list[str]:
    """Evidence split at ellipses; each piece must be found in the turn on its own."""
    return [f for f in (p.strip(_EDGE_PUNCT) for p in _ELLIPSIS.split(_ws(evidence))) if f]


def evidence_score(evidence: str, text: str) -> float:
    """1.0 if every fragment is a substring of `text` (after quote/whitespace/case
    normalization); otherwise the worst fragment's fuzzy partial-match ratio in [0, 1]."""
    frags = _fragments(evidence)
    if not frags:
        return 0.0
    t = _ws(text)
    return min(1.0 if f in t else fuzz.partial_ratio(f, t) / 100 for f in frags)


def check_evidence(evidence: str, turn_text: str, context: list[str]) -> tuple[bool, float, float]:
    """Accept an exact match in the current turn; or a fuzzy match >= 0.9 to the current
    turn that also beats the best match against every context turn (so a quote really
    taken from a previous turn is still rejected). Returns (ok, current, best_context)."""
    cur = evidence_score(evidence, turn_text)
    if cur == 1.0:
        return True, cur, 0.0
    ctx = max((evidence_score(evidence, c) for c in context), default=0.0)
    return cur >= EVIDENCE_FUZZY_MIN and cur > ctx, cur, ctx


def evidence_in_turn(evidence: str, turn_text: str) -> bool:
    return check_evidence(evidence, turn_text, [])[0]


def _subject_ok(source: str, turn: TurnInput) -> bool:
    s = name_key(source)
    sp = name_key(turn.speaker)
    return s == sp or s.startswith(sp + "s ") or (bool(s) and s in name_key(turn.text))


class SanitizeResult(BaseModel):
    extraction: Extraction
    dropped: list[DroppedFact] = Field(default_factory=list)
    converted: list[dict[str, Any]] = Field(default_factory=list)  # unknown type -> attribute
    fuzzy_evidence: list[dict[str, Any]] = Field(default_factory=list)  # accepted non-exact quotes
    ended_by_rule: list[dict[str, Any]] = Field(default_factory=list)  # job/membership ended
    time_rewrites: int = 0


def sanitize(ex: Extraction, turn: TurnInput | None = None,
             people: tuple[str, ...] = ()) -> SanitizeResult:
    """Normalize, validate against the current turn, and drop (with a reason) any fact
    that fails a check. Relative times are rewritten to absolute ones in fact text and
    entity names. Without `turn`, only structural checks run (used in tests)."""
    if turn is not None:
        people = (turn.speaker, turn.listener)
    speakers = {p.lower() for p in people}
    speaker_keys = {name_key(p) for p in people}
    res = SanitizeResult(extraction=Extraction())

    def norm_name(name: str) -> str:
        name = " ".join(name.split())
        if turn is not None:
            name, n = rewrite_relative(name, turn.session_time)
            res.time_rewrites += n
        return name

    ents: dict[str, ExtractedEntity] = {}
    for e in ex.entities:
        e.name = norm_name(e.name)
        if e.name and e.name.lower() not in ents:
            ents[e.name.lower()] = e

    def drop(r: ExtractedRelation, reason: str) -> None:
        res.dropped.append(DroppedFact(reason=reason, relation=r.model_dump()))

    kept: list[ExtractedRelation] = []
    for r in ex.relations:
        r.source, r.target = norm_name(r.source), " ".join(r.target.split())
        r.fact, r.evidence = r.fact.strip(), r.evidence.strip()
        r.relation = r.relation.strip().upper().replace(" ", "_")
        if r.slot is not None:
            r.slot = name_key(r.slot) or None
        if not (r.source and r.target and r.fact):
            drop(r, "empty_field")
            continue
        if r.relation not in RelType._value2member_map_:
            # Keep the information as an attribute of the subject; log the original type.
            res.converted.append({"original_type": r.relation, "source": r.source,
                                  "target": r.target, "fact": r.fact})
            r.relation, r.target, r.slot = RelType.HAS_ATTRIBUTE.value, r.fact, None
        rel = RelType(r.relation)
        if not RELATIONS[rel].attribute:
            r.target = norm_name(r.target)
        if turn is not None:
            ok, cur, ctx = check_evidence(r.evidence, turn.text, turn.context)
            if not ok:
                drop(r, "evidence_not_in_turn")
                continue
            if cur < 1.0:
                res.fuzzy_evidence.append({"evidence": r.evidence, "current": round(cur, 3),
                                           "best_context": round(ctx, 3)})
            if r.evidence.rstrip(_EDGE_PUNCT.replace("?", "")).endswith("?"):
                drop(r, "question")
                continue
            if not _subject_ok(r.source, turn):
                drop(r, "subject_not_in_turn")
                continue
            if rel is RelType.KNOWS and {name_key(r.source), name_key(r.target)} == speaker_keys:
                drop(r, "knows_between_speakers")
                continue
            if rel is RelType.FAMILY_OF and not _FAMILY.search(turn.text):
                drop(r, "family_not_stated")
                continue
            if rel is RelType.PARTNER_OF and not _ROMANTIC.search(r.evidence):
                # keep the information, but not as an exclusive romantic slot
                res.converted.append({"original_type": "PARTNER_OF(non-romantic)", "source": r.source,
                                      "target": r.target, "fact": r.fact})
                r.relation, r.target, r.slot = RelType.HAS_ATTRIBUTE.value, r.fact, None
                rel = RelType.HAS_ATTRIBUTE
            if rel is RelType.PARTICIPATED_IN and (
                    _NOT_ATTENDED.search(r.evidence) or _NOT_ATTENDED.search(r.fact)):
                drop(r, "negated_participation")
                continue
            if rel in ENDED_ROLE_TYPES and not r.ended and says_role_ended(r.evidence):
                r.ended = True
                res.ended_by_rule.append({"relation": r.relation, "fact": r.fact, "evidence": r.evidence})
            # The model often leaves "last year" etc. in the fact despite the hints.
            r.fact, n = rewrite_relative(r.fact, turn.session_time)
            res.time_rewrites += n
        if not RELATIONS[rel].attribute:
            r.slot = None
        # declare endpoints the model forgot to list
        names = [r.source] if RELATIONS[rel].attribute else [r.source, r.target]
        for name in names:
            if name.lower() not in ents:
                t = NodeType.PERSON if name.lower() in speakers else NodeType.CONCEPT
                ents[name.lower()] = ExtractedEntity(name=name, type=t)
        kept.append(r)

    # Concept is the catch-all type; if a relation admits exactly one type at an
    # endpoint the model typed as Concept (e.g. PARTICIPATED_IN -> Event), retype it.
    for r in kept:
        spec = RELATIONS[RelType(r.relation)]
        if spec.attribute:
            continue
        for name, allowed in ((r.source, spec.domain), (r.target, spec.range)):
            e = ents[name.lower()]
            if allowed and len(allowed) == 1 and e.type is NodeType.CONCEPT and e.type not in allowed:
                e.type = next(iter(allowed))

    rels: list[ExtractedRelation] = []
    for r in kept:
        rel = RelType(r.relation)
        if not RELATIONS[rel].attribute:
            try:
                check_relation(rel, ents[r.source.lower()].type, ents[r.target.lower()].type)
            except SchemaViolation:
                drop(r, "type_violation")
                continue
            if r.source.lower() == r.target.lower():
                drop(r, "self_relation")
                continue
        rels.append(r)

    # Only entities that some surviving fact is about become nodes.
    used = {r.source.lower() for r in rels} | {
        r.target.lower() for r in rels if not RELATIONS[RelType(r.relation)].attribute}
    res.extraction = Extraction(entities=[e for k, e in ents.items() if k in used], relations=rels)
    return res


# ---------------------------------------------------------------------------- extractor


class ExtractionRecord(BaseModel):
    """Per turn: the model's raw output (cached) and the checked result (recomputed)."""

    dia_id: str
    raw: Extraction
    extraction: Extraction = Field(default_factory=Extraction)
    dropped: list[DroppedFact] = Field(default_factory=list)
    converted: list[dict[str, Any]] = Field(default_factory=list)
    fuzzy_evidence: list[dict[str, Any]] = Field(default_factory=list)
    ended_by_rule: list[dict[str, Any]] = Field(default_factory=list)
    time_rewrites: int = 0  # relative times rewritten in kept facts and entity names
    repaired: bool = False
    fence_stripped: bool = False
    failed: bool = False
    error: str | None = None


class Extractor:
    def __init__(self, llm: JsonLLM, cache_dir: str = "cache", max_output_tokens: int = 2048):
        self.llm = llm
        self.cache = DiskCache(f"{cache_dir}/extraction")
        self.max_output_tokens = max_output_tokens
        self.hits = 0
        self.misses = 0
        self.records: dict[str, ExtractionRecord] = {}  # dia_id -> record (this process)

    async def extract(self, turn: TurnInput) -> Extraction:
        key = turn.key(self.llm.identity)
        if (hit := self.cache.get(key)) is not None:
            self.hits += 1
            rec = ExtractionRecord.model_validate(hit)
        else:
            self.misses += 1
            try:
                ex, meta = await self.llm.generate_json_meta(
                    turn.prompt(), Extraction, system=SYSTEM_PROMPT, purpose="extraction",
                    max_output_tokens=self.max_output_tokens,
                )
                rec = ExtractionRecord(dia_id=turn.dia_id, raw=ex, repaired=meta.repaired,
                                       fence_stripped=meta.fence_stripped)
            except ValidationError as err:
                # Invalid even after repair: record it, ingest nothing for this turn.
                rec = ExtractionRecord(dia_id=turn.dia_id, raw=Extraction(), failed=True,
                                       error=str(err)[:500])
            self.cache.put(key, rec.model_dump(mode="json", include={
                "dia_id", "raw", "repaired", "fence_stripped", "failed", "error"}))
        res = sanitize(rec.raw.model_copy(deep=True), turn)
        rec.extraction, rec.dropped, rec.converted = res.extraction, res.dropped, res.converted
        rec.fuzzy_evidence, rec.time_rewrites = res.fuzzy_evidence, res.time_rewrites
        rec.ended_by_rule = res.ended_by_rule
        self.records[turn.dia_id] = rec
        return rec.extraction

    def validity(self) -> dict[str, float | int]:
        recs = list(self.records.values())
        n = len(recs) or 1
        rep = sum(r.repaired and not r.failed for r in recs)
        fail = sum(r.failed for r in recs)
        return {
            "turns": len(recs),
            "valid_first_try": len(recs) - rep - fail,
            "repaired": rep,
            "failed": fail,
            "fence_stripped": sum(r.fence_stripped for r in recs),
            "valid_rate": round((len(recs) - fail) / n, 4),
            "first_try_rate": round((len(recs) - rep - fail) / n, 4),
            "repair_rate": round(rep / n, 4),
        }

    def converted(self) -> list[dict[str, Any]]:
        return [c | {"turn": d} for d, r in self.records.items() for c in r.converted]

    def ended_by_rule(self) -> list[dict[str, Any]]:
        return [c | {"turn": d} for d, r in self.records.items() for c in r.ended_by_rule]

    def fuzzy_accepted(self) -> list[dict[str, Any]]:
        return [c | {"turn": d} for d, r in self.records.items() for c in r.fuzzy_evidence]

    def time_rewrites(self) -> int:
        return sum(r.time_rewrites for r in self.records.values())

    def drop_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self.records.values():
            for d in rec.dropped:
                out[d.reason] = out.get(d.reason, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))
