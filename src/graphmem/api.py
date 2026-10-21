"""HTTP API for GraphMem.

    POST /memory/query   retrieve facts (+ provenance) for a query
    POST /memory/update  store one conversation turn (extract -> resolve -> write -> retire)
    POST /memory/prune   archive decayed facts (dry_run supported)
    POST /agent/turn     run the LangGraph agent on one message
    GET  /health

    uvicorn graphmem.api:app --port 8000
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import MemoryAgent
from .config import Settings
from .schema import PruneReport
from .service import CONVERSATION_ID, MemoryService, UpdateResult

ConvId = Field(pattern=CONVERSATION_ID.pattern)


class QueryRequest(BaseModel):
    conversation_id: str = ConvId
    query: str = Field(min_length=1)
    k: int | None = Field(None, ge=1, le=100)
    as_of: datetime | None = None  # hard validity filter
    now: datetime | None = None  # query time (default: wall clock)
    provenance: bool | None = None  # default: frozen config (on)
    include_history: bool | None = None
    reinforce: bool = True


class FactOut(BaseModel):
    text: str
    relation_type: str
    subject: str
    object: str | None
    stated: datetime
    no_longer_true_since: datetime | None
    score: float
    via_expansion: bool


class TurnOut(BaseModel):
    turn_id: str
    text: str
    time: datetime


class QueryResponse(BaseModel):
    facts: list[FactOut]
    turns: list[TurnOut]
    context: str
    db_ms: float
    embed_ms: float


class UpdateRequest(BaseModel):
    conversation_id: str = ConvId
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)
    listener: str | None = None
    time: datetime | None = None
    turn_id: str | None = None


class PruneRequest(BaseModel):
    conversation_id: str = ConvId
    threshold: float | None = Field(None, ge=0)
    dry_run: bool = False
    now: datetime | None = None


class AgentRequest(BaseModel):
    conversation_id: str = ConvId
    speaker: str = Field(min_length=1)
    message: str = Field(min_length=1)
    listener: str | None = None
    time: datetime | None = None


class AgentResponse(BaseModel):
    answer: str
    remembered: bool
    decision_reason: str
    written: dict[str, Any] | None
    retrieved: int
    pruned: dict[str, Any] | None
    turn_index: int
    timings_ms: dict[str, float]


def create_app(service: MemoryService | None = None, agent: MemoryAgent | None = None,
               prune_every: int = 50) -> FastAPI:
    """With no arguments, connects to Neo4j and the LLM from Settings at startup.
    Tests pass a ready service/agent."""
    state: dict[str, Any] = {"service": service, "agent": agent, "turns": defaultdict(int)}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = None
        if state["service"] is None:
            from .llm import make_llm
            from .store import GraphStore

            s = Settings()
            owned = await GraphStore.connect(s)
            llm = make_llm(s)
            state["service"] = MemoryService(owned, llm, s)
            state["agent"] = MemoryAgent(state["service"], llm, prune_every=prune_every)
        yield
        if owned is not None:
            await owned.close()

    app = FastAPI(title="GraphMem", version="0.1.0", lifespan=lifespan)

    def svc() -> MemoryService:
        return state["service"]

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/memory/query", response_model=QueryResponse)
    async def query(req: QueryRequest) -> QueryResponse:
        r = await svc().query(req.conversation_id, req.query, k=req.k, as_of=req.as_of, now=req.now,
                              provenance=req.provenance, include_history=req.include_history,
                              reinforce=req.reinforce)
        return QueryResponse(
            facts=[FactOut(text=f.text, relation_type=f.relation_type.value, subject=f.subject_name,
                           object=f.object_name, stated=f.validity_start, no_longer_true_since=f.validity_end,
                           score=round(f.score, 4), via_expansion=f.via_expansion) for f in r.facts],
            turns=[TurnOut(turn_id=t.dia_id, text=t.text, time=t.session_time) for t in r.turns],
            context=r.to_context(), db_ms=round(r.db_ms, 2), embed_ms=round(r.embed_ms, 2))

    @app.post("/memory/update", response_model=UpdateResult)
    async def update(req: UpdateRequest) -> UpdateResult:
        try:
            return await svc().update(req.conversation_id, req.speaker, req.text, listener=req.listener,
                                      time=req.time, turn_id=req.turn_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/memory/prune", response_model=PruneReport)
    async def prune(req: PruneRequest) -> PruneReport:
        return await svc().prune(req.conversation_id, threshold=req.threshold, dry_run=req.dry_run, now=req.now)

    @app.post("/agent/turn", response_model=AgentResponse)
    async def agent_turn(req: AgentRequest) -> AgentResponse:
        state["turns"][req.conversation_id] += 1
        n = state["turns"][req.conversation_id]
        out = await state["agent"].run(req.conversation_id, req.speaker, req.message, listener=req.listener,
                                       time=req.time, turn_index=n)
        d = out.get("decision", {})
        return AgentResponse(answer=out.get("answer", ""), remembered=bool(d.get("remember")),
                             decision_reason=d.get("reason", ""), written=out.get("written"),
                             retrieved=out.get("retrieved", 0), pruned=out.get("pruned"), turn_index=n,
                             timings_ms=out.get("timings_ms", {}))

    return app


app = create_app()
