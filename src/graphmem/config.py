from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All knobs are overridable via GRAPHMEM_* env vars or a .env file."""

    model_config = SettingsConfigDict(
        env_prefix="GRAPHMEM_", env_file=".env", extra="ignore", populate_by_name=True
    )

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphmem-dev"
    neo4j_database: str = "neo4j"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Decay: weight * (1 + 0.1*ln(1+access_count)) * exp(-lambda * dt_days)
    # 0 by default: a dev sweep (conv-26 + conv-30) found no setting beating 0 by > 5 points;
    # decay is retrieval-neutral here and only used for the optional pruning curve.
    decay_lambda: float = Field(0.0, ge=0)

    # Rerank: alpha*similarity + (1-alpha)*min(decay, 1)
    # 1.0 = the exact configuration evaluated on dev (with lambda=0, alpha only rescales
    # similarity relative to history_penalty).
    retrieval_alpha: float = Field(1.0, ge=0, le=1)
    retrieval_k: int = Field(10, ge=1)
    # Fact vector search over-fetches k * this before filtering/reranking.
    fact_overfetch: int = Field(20, ge=1)
    # 1-hop expansion: for entities of the top-k facts, add their next-best facts.
    expand: bool = True
    expand_per_entity: int = Field(2, ge=0)
    expand_max: int | None = None  # total expansion facts; None = k
    # Provenance: append the source turns (speaker, session date, text) of retrieved facts.
    # On by default: +9.0 / +9.4 overall on dev (14B / 32B judge), temporal not hurt.
    provenance: bool = True
    # Superseded/ended facts are returned by default, with this subtracted from their score.
    include_history: bool = True
    history_penalty: float = Field(0.15, ge=0)

    # Pruning is off by default; on LoCoMo it trades accuracy for graph size (dev curve).
    prune_enabled: bool = False
    prune_threshold: float = Field(0.05, ge=0)

    # ---------------------------------------------------------------- LLM (OpenAI-compatible)
    llm_base_url: str = "http://localhost:11435/v1"
    llm_model: str = "qwen3:14b"
    llm_api_key: str | None = None  # Ollama ignores it; vLLM may require one
    llm_max_concurrency: int = Field(4, ge=1)  # match server parallelism (OLLAMA_NUM_PARALLEL)
    # Retry transient LLM errors (incl. connection errors) with backoff for up to this long.
    llm_retry_budget_s: float = Field(900.0, ge=0)
    llm_timeout_s: float = 600.0
    # Sends reasoning_effort="none" (Qwen3 thinking off; verified on Ollama's /v1 endpoint).
    llm_disable_thinking: bool = True
    extraction_max_tokens: int = 2048
    # Answer step only (all systems): Qwen3 thinking on/off, and output budgets.
    answer_thinking: bool = True  # Phase 4: thinking on for every system
    answer_max_tokens: int = 64
    answer_thinking_max_tokens: int = 4096  # thinking tokens count against the budget
    contradiction_max_tokens: int = 1024
    cache_dir: str = "cache"

    # ---------------------------------------------------------------- judge
    # Local judge, run as a separate pass after all answers exist. Planned: a ~32B-class
    # instruct model (4-bit) on the same Ollama server; the answer model is unloaded first so
    # both are never loaded together. Unset = the answer model (llm_model). Thinking off.
    judge_model: str | None = "qwen3-32b-judge"
    judge_base_url: str | None = None  # unset = llm_base_url

    # ---------------------------------------------------------------- ingestion
    # Entity resolution: normalized exact name/alias match first; else same-type
    # name-embedding cosine >= this.
    # Locked at 0.94: 9/10 embedding merges correct on dev (conv-26 5/6, conv-30 4/4).
    resolution_threshold: float = Field(0.94, ge=0, le=1)
    # A new fact with cosine >= this to a current fact of the same type is a restatement.
    duplicate_fact_cosine: float = Field(0.95, ge=0, le=1)
    # Current facts with cosine >= this to a new fact are sent to the LLM contradiction check.
    # Contradiction handling (ablation switch for Phase 4):
    #   llm   = deterministic rules + LLM adjudication of slot relations
    #   rules = deterministic rules only (exclusive slots retire directly)
    #   off   = never retire anything (restatements are still merged)
    # Default "rules": LLM retirement precision on dev was 0/5 (conv-30); "llm" is an ablation.
    contradiction_mode: Literal["llm", "rules", "off"] = "rules"
    contradiction_candidates: int = Field(5, ge=0)
    # Event nodes merge by embedding only at >= this name cosine AND compatible dates.
    event_resolution_threshold: float = Field(0.95, ge=0, le=1)
    # "Same target" for LIKES/DISLIKES/INTERESTED_IN retirement: target-entity name cosine.
    preference_target_cosine: float = Field(0.9, ge=0, le=1)
    # Previous turns in the same session given to the extractor for coreference.
    extraction_context_turns: int = Field(3, ge=0)

    # ---------------------------------------------------------------- evaluation
    # Dev conversations: the only ones used for tuning and prompt iteration.
    # Every other LoCoMo conversation is test-only.
    dev_conversations: list[str] = ["conv-26", "conv-30"]
