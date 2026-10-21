"""Graph schema: node labels, typed relations, and the temporal record models.

Every node carries the shared `:Entity` label plus exactly one type label
(`:Person`, `:Organization`, `:Concept`, `:Event`). The shared label backs a
single vector index and a single id uniqueness constraint. Pruned nodes
additionally get `:Archived`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class NodeType(StrEnum):
    PERSON = "Person"
    ORGANIZATION = "Organization"
    CONCEPT = "Concept"  # also places, objects, topics, attributes
    EVENT = "Event"


class RelType(StrEnum):
    KNOWS = "KNOWS"
    FAMILY_OF = "FAMILY_OF"
    PARTNER_OF = "PARTNER_OF"
    WORKS_AT = "WORKS_AT"
    STUDIES_AT = "STUDIES_AT"
    MEMBER_OF = "MEMBER_OF"
    LIVES_IN = "LIVES_IN"
    LIKES = "LIKES"
    DISLIKES = "DISLIKES"
    INTERESTED_IN = "INTERESTED_IN"
    HAS_ATTRIBUTE = "HAS_ATTRIBUTE"
    OWNS = "OWNS"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    PLANS = "PLANS"
    LOCATED_IN = "LOCATED_IN"
    PART_OF = "PART_OF"
    RELATED_TO = "RELATED_TO"


@dataclass(frozen=True)
class RelSpec:
    domain: frozenset[NodeType] | None  # None = any
    range: frozenset[NodeType] | None
    # A subject holds at most one current edge of this type; a new target supersedes the old.
    exclusive: bool = False
    # For a given (source, target) pair, a current edge of this type conflicts with `opposite`.
    opposite: RelType | None = None
    # Attribute relations are self-loops on the entity; the value lives on the edge.
    attribute: bool = False


_P, _O, _C, _E = NodeType.PERSON, NodeType.ORGANIZATION, NodeType.CONCEPT, NodeType.EVENT


def _s(*types: NodeType) -> frozenset[NodeType]:
    return frozenset(types)


RELATIONS: dict[RelType, RelSpec] = {
    RelType.KNOWS: RelSpec(_s(_P), _s(_P)),
    RelType.FAMILY_OF: RelSpec(_s(_P), _s(_P)),
    RelType.PARTNER_OF: RelSpec(_s(_P), _s(_P), exclusive=True),
    RelType.WORKS_AT: RelSpec(_s(_P), _s(_O)),  # people can hold several jobs
    RelType.STUDIES_AT: RelSpec(_s(_P), _s(_O), exclusive=True),
    RelType.MEMBER_OF: RelSpec(_s(_P, _O), _s(_O)),
    RelType.LIVES_IN: RelSpec(_s(_P, _O), _s(_C, _O), exclusive=True),
    RelType.LIKES: RelSpec(None, None, opposite=RelType.DISLIKES),
    RelType.DISLIKES: RelSpec(None, None, opposite=RelType.LIKES),
    RelType.INTERESTED_IN: RelSpec(_s(_P, _O), None),
    RelType.HAS_ATTRIBUTE: RelSpec(None, None, attribute=True),
    RelType.OWNS: RelSpec(_s(_P, _O), None),
    RelType.PARTICIPATED_IN: RelSpec(_s(_P, _O), _s(_E)),
    RelType.PLANS: RelSpec(_s(_P, _O), _s(_E, _C)),
    RelType.LOCATED_IN: RelSpec(_s(_O, _E, _C), _s(_C, _O)),
    RelType.PART_OF: RelSpec(None, None),
    RelType.RELATED_TO: RelSpec(None, None),
}


# Contradiction policy (see ingest.py).
# Slot relations: a newer value can make an older one untrue. Candidates (same source,
# same type, no Event endpoint; HAS_ATTRIBUTE only with the same slot) go to the LLM.
LLM_SLOT_TYPES = frozenset({
    RelType.LIVES_IN, RelType.WORKS_AT, RelType.STUDIES_AT, RelType.PARTNER_OF,
    RelType.MEMBER_OF, RelType.PLANS, RelType.HAS_ATTRIBUTE,
})
# Preferences: never sent to the LLM; retired only by explicit negation or an
# opposite-polarity fact about the same target.
PREFERENCE_TYPES = frozenset({RelType.LIKES, RelType.DISLIKES, RelType.INTERESTED_IN})


class SchemaViolation(ValueError):
    pass


def check_relation(rel: RelType, src: NodeType, dst: NodeType, self_loop: bool = False) -> None:
    spec = RELATIONS[rel]
    if spec.attribute != self_loop:
        raise SchemaViolation(
            f"{rel} must be a self-loop attribute" if spec.attribute
            else f"{rel} cannot be a self-loop")
    if spec.attribute:
        return
    if spec.domain is not None and src not in spec.domain:
        raise SchemaViolation(f"{rel} cannot start at {src} (allowed: {sorted(spec.domain)})")
    if spec.range is not None and dst not in spec.range:
        raise SchemaViolation(f"{rel} cannot end at {dst} (allowed: {sorted(spec.range)})")


_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def name_key(name: str) -> str:
    """Normalized name for exact-match entity resolution:
    lowercase, punctuation removed, whitespace collapsed ("Melanie's  Kids!" -> "melanies kids")."""
    return " ".join(_NON_WORD.sub("", name.lower()).replace("_", " ").split())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid4().hex


def _to_native(v: Any) -> Any:
    # neo4j.time.DateTime -> datetime.datetime
    return v.to_native() if hasattr(v, "to_native") else v


# ---------------------------------------------------------------- inputs


class NodeCreate(BaseModel):
    type: NodeType
    name: str = Field(min_length=1)
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)
    weight: float = Field(1.0, gt=0)
    validity_start: datetime | None = None  # defaults to `now`
    id: str | None = None


class FactCreate(BaseModel):
    relation_type: RelType
    subject_id: str
    object_id: str | None = None  # None for attribute facts (HAS_ATTRIBUTE)
    text: str  # natural-language statement
    evidence: str | None = None  # verbatim quote from the source turn
    turn_id: str | None = None  # provenance (conversation turn id)
    slot: str | None = None  # HAS_ATTRIBUTE: single-valued attribute slot, e.g. "age"
    value: str | None = None  # HAS_ATTRIBUTE: the attribute value
    weight: float = Field(1.0, gt=0)
    validity_start: datetime | None = None
    id: str | None = None


# ---------------------------------------------------------------- stored records


class TemporalRecord(BaseModel):
    id: str
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    validity_start: datetime
    validity_end: datetime | None = None  # None = currently valid
    weight: float = 1.0
    embedding: list[float] | None = Field(default=None, repr=False)

    @field_validator(
        "created_at", "last_accessed", "validity_start", "validity_end", mode="before"
    )
    @classmethod
    def _native_dt(cls, v: Any) -> Any:
        return _to_native(v)

    @property
    def is_current(self) -> bool:
        return self.validity_end is None


class Node(TemporalRecord):
    type: NodeType
    name: str
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)
    archived: bool = False

    @classmethod
    def from_neo4j(cls, props: dict[str, Any], labels: list[str]) -> Node:
        types = [l for l in labels if l in NodeType._value2member_map_]
        return cls(**props, type=types[0], archived="Archived" in labels)


class Fact(TemporalRecord):
    """A fact node: (:Fact)-[:SUBJECT]->(:Entity), and -[:OBJECT]->(:Entity) if any."""

    relation_type: RelType
    text: str
    subject_id: str
    object_id: str | None = None
    conversation_id: str | None = None
    evidence: str | None = None
    turn_id: str | None = None
    slot: str | None = None
    value: str | None = None
    archived: bool = False


# ---------------------------------------------------------------- retrieval output


class RetrievedFact(BaseModel):
    id: str
    relation_type: RelType
    text: str
    subject_id: str
    subject_name: str
    object_id: str | None = None
    object_name: str | None = None
    turn_id: str | None = None
    validity_start: datetime
    validity_end: datetime | None
    similarity: float
    decay: float
    score: float
    via_expansion: bool = False

    _dt = field_validator("validity_start", "validity_end", mode="before")(
        lambda v: _to_native(v)
    )

    @property
    def is_current(self) -> bool:
        return self.validity_end is None

    def validity_label(self) -> str:
        """e.g. "[stated 2023-05-08]" or "[stated 2023-05-08; no longer true as of 2023-10-01]".
        validity_start is when the fact entered memory (the session it was said in)."""
        if self.validity_end is None:
            return f"[stated {self.validity_start:%Y-%m-%d}]"
        return (f"[stated {self.validity_start:%Y-%m-%d}; "
                f"no longer true as of {self.validity_end:%Y-%m-%d}]")


class SourceTurn(BaseModel):
    """A conversation turn that a retrieved fact was extracted from (provenance)."""

    dia_id: str
    text: str  # rendered: "Speaker: text [shares a photo: ...]"
    session_time: datetime

    _dt = field_validator("session_time", mode="before")(lambda v: _to_native(v))


class RetrievalResult(BaseModel):
    query: str
    facts: list[RetrievedFact]  # top-k by score, then expansion facts by score
    turns: list[SourceTurn] = Field(default_factory=list)  # provenance, deduplicated
    db_ms: float  # time spent in the Neo4j transaction only
    embed_ms: float

    def to_context(self) -> str:
        """Compact text rendering for an LLM prompt: one fact per line, then (with
        provenance) the distinct source turns, in order of first use by a fact."""
        lines = [f"- {f.text} {f.validity_label()}" for f in self.facts]
        if not self.turns:
            return "\n".join(lines)
        return "\n".join(["Facts:", *lines, "", "Source turns (where these facts were said):",
                          *(f"- [{t.session_time:%d %B %Y}] {t.text}" for t in self.turns)])


class PruneReport(BaseModel):
    threshold: float
    dry_run: bool
    archived_facts: list[dict[str, Any]]
