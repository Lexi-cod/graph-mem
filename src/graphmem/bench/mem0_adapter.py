"""Mem0 (open source) wired to the same local LLM and embedder as GraphMem.

- LLM: a custom Mem0 provider ("graphmem") that routes every Mem0 call through our
  cached LLM client (same qwen3:14b endpoint, thinking off, disk cache, usage stats
  under purpose "mem0").
- No wall-clock time: Mem0 puts today's date into its extraction prompt ("Observation
  Date" / "Current Date"). The provider replaces it with the simulated session date
  (`set_session_date`) before the call, which also keeps the cache deterministic.
  Mem0 still stamps created_at/updated_at metadata with the wall clock; those fields are
  not used for ranking and are not shown to the answer model.
- Embedder: Mem0's HuggingFace provider with sentence-transformers/all-MiniLM-L6-v2.
- Vector store: Mem0's default (Qdrant, embedded on disk), one collection per conversation.
  BM25 keyword search (fastembed) and entity boosts (spaCy) are installed and active.
"""

from __future__ import annotations

import os

# Mem0 sends usage telemetry (PostHog) by default; keep benchmark runs offline.
os.environ.setdefault("MEM0_TELEMETRY", "False")

import re
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase
from mem0.utils.factory import LlmFactory

from ..llm import LLM

_STATE: dict[str, Any] = {"llm": None, "max_prompt_tokens": 0, "calls": 0}
# The simulated session date is per thread, so several conversations can ingest concurrently.
_LOCAL = threading.local()
_LOCK = threading.Lock()


def prompt_stats() -> dict[str, int]:
    """Largest prompt (server-counted tokens) Mem0 has sent; compare with the server's num_ctx."""
    return {"max_prompt_tokens": _STATE["max_prompt_tokens"], "calls": _STATE["calls"]}


def configure(llm: LLM) -> None:
    """Route Mem0's "ollama" provider (the same Ollama endpoint/model we use) through our
    cached client. Mem0's config validator only accepts whitelisted provider names, so we
    override the "ollama" entry rather than registering a new name."""
    _STATE["llm"] = llm
    LlmFactory.provider_to_class["ollama"] = (f"{__name__}.CachedMem0LLM", BaseLlmConfig)


def set_session_date(d: date) -> None:
    _LOCAL.date = d


class CachedMem0LLM(LLMBase):
    def __init__(self, config: BaseLlmConfig | None = None):
        super().__init__(config)

    def generate_response(self, messages: list[dict[str, str]], tools=None, tool_choice="auto",
                          response_format=None, **kwargs) -> str:
        if tools:
            raise NotImplementedError("graph memory / tool calls are not used in this benchmark")
        llm: LLM = _STATE["llm"]
        sim: date | None = getattr(_LOCAL, "date", None)
        if llm is None or sim is None:
            raise RuntimeError("mem0_adapter not configured / session date not set")
        today = datetime.now(timezone.utc).date().isoformat()

        def fix(text: str) -> str:
            return text.replace(today, sim.isoformat())

        system = "\n\n".join(fix(m["content"]) for m in messages if m["role"] == "system") or None
        prompt = "\n\n".join(fix(m["content"]) for m in messages if m["role"] != "system")
        json_object = bool(response_format and response_format.get("type") == "json_object")
        r = llm.generate_sync(prompt, system=system, purpose="mem0", temperature=0.0,
                              max_output_tokens=self.config.max_tokens, json_object=json_object)
        with _LOCK:
            _STATE["calls"] += 1
            _STATE["max_prompt_tokens"] = max(_STATE["max_prompt_tokens"], r.input_tokens)
        return r.text


def build_memory(conv_id: str, root: str | Path = "cache/mem0"):
    """A Mem0 Memory whose vector store and history DB live under `root`/<conv_id>."""
    from mem0 import Memory

    root = Path(root) / conv_id
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "llm": {"provider": "ollama", "config": {"model": "qwen3:14b", "max_tokens": 2000}},
        "embedder": {"provider": "huggingface",
                     "config": {"model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dims": 384}},
        "vector_store": {"provider": "qdrant",
                         "config": {"collection_name": re.sub(r"\W", "_", conv_id), "path": str(root / "qdrant"),
                                    "embedding_model_dims": 384, "on_disk": True}},
        "history_db_path": str(root / "history.db"),
    }
    return Memory.from_config(config)


class TimedStore:
    """Wraps a Mem0 vector store to time the store calls inside search() (the part
    comparable to GraphMem's DB-only latency)."""

    def __init__(self, memory):
        self.memory = memory
        self.ms = 0.0
        vs = memory.vector_store
        for name in ("search", "keyword_search"):
            orig = getattr(vs, name, None)
            if orig is not None:
                setattr(vs, name, self._wrap(orig))

    def _wrap(self, fn):
        def timed(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                self.ms += (time.perf_counter() - t0) * 1000
        return timed
