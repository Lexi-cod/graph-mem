"""MemoryService: the operations behind the HTTP API and the agent.

One service manages many conversations (namespaces). Writes to a conversation are
serialized by a per-conversation lock, because fact application must happen in turn
order (resolution and contradiction handling depend on graph state). Time defaults to
the wall clock but every operation accepts an explicit timestamp.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .config import Settings
from .extraction import Extractor, JsonLLM, TurnInput
from .ingest import Ingestor
from .schema import PruneReport, RetrievalResult
from .store import GraphStore

CONVERSATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class UpdateResult(BaseModel):
    conversation_id: str
    turn_id: str
    time: datetime
    facts_created: int
    facts_reinforced: int
    facts_retired: int
    entities_created: int
    entities_merged: int
    facts: list[str] = Field(default_factory=list)  # texts of facts written by this turn
    dropped: list[dict[str, Any]] = Field(default_factory=list)  # [{reason, fact}]


def _utc(t: datetime | None) -> datetime:
    if t is None:
        return datetime.now(timezone.utc)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


class MemoryService:
    def __init__(self, root: GraphStore, llm: JsonLLM, settings: Settings | None = None):
        self.root = root
        self.llm = llm
        self.s = settings or root.s
        self._stores: dict[str, GraphStore] = {}
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @staticmethod
    def check_id(conversation_id: str) -> str:
        if not CONVERSATION_ID.match(conversation_id):
            raise ValueError("conversation_id must match ^[A-Za-z0-9_-]{1,64}$")
        return conversation_id

    async def store(self, conversation_id: str) -> GraphStore:
        cid = self.check_id(conversation_id)
        if cid not in self._stores:
            self._stores[cid] = await self.root.scoped(cid)
        return self._stores[cid]

    async def query(self, conversation_id: str, query: str, *, k: int | None = None,
                    as_of: datetime | None = None, now: datetime | None = None,
                    provenance: bool | None = None, include_history: bool | None = None,
                    reinforce: bool = True) -> RetrievalResult:
        st = await self.store(conversation_id)
        return await st.retrieve(query, k=k, as_of=_utc(as_of) if as_of else None, now=_utc(now),
                                 provenance=provenance, include_history=include_history,
                                 reinforce=reinforce)

    async def update(self, conversation_id: str, speaker: str, text: str, *,
                     listener: str | None = None, time: datetime | None = None,
                     turn_id: str | None = None) -> UpdateResult:
        """Store one conversation turn: extract facts (with the previous turns of this
        conversation as context), resolve entities, write facts, retire contradictions."""
        st = await self.store(conversation_id)
        now = _utc(time)
        turn_id = turn_id or f"api-{now:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
        async with self._locks[conversation_id]:
            prev = await st.recent_turns(self.s.extraction_context_turns, before=now)
            rendered = f"{speaker}: {text}"
            ti = TurnInput(dia_id=turn_id, speaker=speaker, listener=listener or "the other speaker",
                           text=rendered, session_time=now, context=[t["text"] for t in prev])
            await st.add_turns([{"dia_id": turn_id, "text": rendered, "session_time": now}])
            extractor = Extractor(self.llm, self.s.cache_dir, self.s.extraction_max_tokens)
            ing = Ingestor(st, extractor, self.llm, self.s)  # fresh stats = this turn's delta
            ex = await extractor.extract(ti)
            await ing.apply(ti, ex, now)
        rec = extractor.records[turn_id]
        stats = ing.stats
        return UpdateResult(
            conversation_id=conversation_id, turn_id=turn_id, time=now,
            facts_created=stats.facts_created, facts_reinforced=stats.facts_reinforced,
            facts_retired=stats.retired_ended + stats.retired_schema_rule + stats.retired_llm,
            entities_created=stats.nodes_created,
            entities_merged=stats.merged_exact + stats.merged_embedding,
            facts=[r.fact for r in ex.relations],
            dropped=[{"reason": d.reason, "fact": d.relation.get("fact")} for d in rec.dropped],
        )

    async def prune(self, conversation_id: str, *, threshold: float | None = None,
                    dry_run: bool = False, now: datetime | None = None) -> PruneReport:
        st = await self.store(conversation_id)
        async with self._locks[conversation_id]:
            return await st.prune(threshold=threshold, dry_run=dry_run, now=_utc(now))
