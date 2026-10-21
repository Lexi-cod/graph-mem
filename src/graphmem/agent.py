"""LangGraph agent over GraphMem.

    START -> retrieve -> reason -> [write] -> answer -> [prune] -> END

- retrieve: memory retrieval for the incoming message (frozen retrieval config).
- reason:   one structured LLM call deciding whether the message states lasting
            information about the people that should be remembered.
- write:    (optional, mid-run) ingest the message as a conversation turn, then
            re-retrieve so the answer can use what was just stored.
- answer:   reply grounded in the retrieved memories.
- prune:    every `prune_every` turns, archive decayed facts. Pruning is off by default
            (prune_enabled=False: on LoCoMo it trades accuracy for graph size); the node
            then records that it skipped.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .service import MemoryService

REASON_SYSTEM = """You manage the long-term memory of a conversational assistant.
Given a new message and what memory already contains, decide whether the message states
lasting information about the people in the conversation (facts, events, plans,
preferences, relationships, changes) that should be remembered. Greetings, thanks,
compliments, small talk and pure questions do not need to be remembered.
Return JSON only."""

AGENT_ANSWER_SYSTEM = """You are a friendly conversational assistant with long-term memory.
Reply to the latest message in a few sentences. Use the retrieved memories when they are
relevant; they are marked with the date they were said, and "no longer true" marks
outdated facts. Do not invent facts that are not in the memories or the message."""


class Decision(BaseModel):
    remember: bool = Field(description="The message states lasting information worth storing")
    reason: str = Field("", description="One short sentence")


class AgentState(TypedDict, total=False):
    conversation_id: str
    speaker: str
    listener: str | None
    message: str
    time: datetime | None
    turn_index: int
    context: str
    retrieved: int
    decision: dict[str, Any]
    written: dict[str, Any] | None
    answer: str
    pruned: dict[str, Any] | None
    timings_ms: dict[str, float]


class MemoryAgent:
    def __init__(self, service: MemoryService, llm, prune_every: int = 50,
                 thinking: bool | None = None, answer_max_tokens: int = 512):
        self.svc = service
        self.llm = llm
        self.s = service.s
        self.prune_every = prune_every
        self.thinking = self.s.answer_thinking if thinking is None else thinking
        self.answer_max_tokens = answer_max_tokens
        self.graph = self._build()

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("retrieve", self.retrieve)
        g.add_node("reason", self.reason)
        g.add_node("write", self.write)
        g.add_node("answer", self.answer)
        g.add_node("prune", self.prune)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "reason")
        g.add_conditional_edges("reason", lambda st: "write" if st["decision"]["remember"] else "answer",
                                {"write": "write", "answer": "answer"})
        g.add_edge("write", "answer")
        g.add_conditional_edges("answer", self._should_prune, {"prune": "prune", "end": END})
        g.add_edge("prune", END)
        return g.compile()

    def _should_prune(self, st: AgentState) -> str:
        n = st.get("turn_index", 0)
        return "prune" if self.prune_every > 0 and n > 0 and n % self.prune_every == 0 else "end"

    @staticmethod
    def _timed(st: AgentState, name: str, t0: float) -> dict[str, float]:
        return {**st.get("timings_ms", {}), name: round((time.perf_counter() - t0) * 1000, 1)}

    # ------------------------------------------------------------ nodes

    async def retrieve(self, st: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        r = await self.svc.query(st["conversation_id"], st["message"], now=st.get("time"))
        return {"context": r.to_context(), "retrieved": len(r.facts), "timings_ms": self._timed(st, "retrieve", t0)}

    async def reason(self, st: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        prompt = (f"Memory (retrieved):\n{st.get('context') or '(empty)'}\n\n"
                  f"New message from {st['speaker']}:\n{st['message']}")
        d, _ = await self.llm.generate_json_meta(prompt, Decision, system=REASON_SYSTEM,
                                                 purpose="agent_reason", max_output_tokens=256)
        return {"decision": d.model_dump(), "timings_ms": self._timed(st, "reason", t0)}

    async def write(self, st: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        res = await self.svc.update(st["conversation_id"], st["speaker"], st["message"],
                                    listener=st.get("listener"), time=st.get("time"))
        r = await self.svc.query(st["conversation_id"], st["message"], now=st.get("time"))
        return {"written": res.model_dump(mode="json"), "context": r.to_context(), "retrieved": len(r.facts),
                "timings_ms": self._timed(st, "write", t0)}

    async def answer(self, st: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        prompt = (f"Memories:\n{st.get('context') or '(none)'}\n\n"
                  f"{st['speaker']}: {st['message']}\nReply:")
        r = await self.llm.generate(prompt, system=AGENT_ANSWER_SYSTEM, purpose="agent_answer",
                                    thinking=self.thinking, max_output_tokens=(
                                        self.s.answer_thinking_max_tokens if self.thinking
                                        else self.answer_max_tokens))
        text = r.text.split("</think>")[-1].strip()
        return {"answer": text, "timings_ms": self._timed(st, "answer", t0)}

    async def prune(self, st: AgentState) -> dict[str, Any]:
        if not self.s.prune_enabled:
            return {"pruned": {"skipped": True, "reason": "prune_enabled=False"}}
        t0 = time.perf_counter()
        rep = await self.svc.prune(st["conversation_id"], now=st.get("time"))
        return {"pruned": {"skipped": False, "archived": len(rep.archived_facts), "threshold": rep.threshold},
                "timings_ms": self._timed(st, "prune", t0)}

    # ------------------------------------------------------------ entry point

    async def run(self, conversation_id: str, speaker: str, message: str, *, listener: str | None = None,
                  time: datetime | None = None, turn_index: int = 0) -> AgentState:
        return await self.graph.ainvoke({
            "conversation_id": MemoryService.check_id(conversation_id), "speaker": speaker,
            "listener": listener, "message": message, "time": time, "turn_index": turn_index,
            "timings_ms": {}, "written": None, "pruned": None})
