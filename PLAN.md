# GraphMem plan

## Done
- Phase 1: temporal store (decay, rerank, validity, prune), now facts-as-nodes:
  `(:Fact)-[:SUBJECT|OBJECT]->(:Entity)`, per-conversation namespaces.
- Phase 2: LLM extraction (local qwen3:14b via Ollama, OpenAI-compatible), evidence-checked
  post-processing, entity resolution, rules-first contradiction handling (`llm|rules|off`),
  parallel multi-conversation ingestion.
- Dev split: conv-26, conv-30 (`Settings.dev_conversations`). The other 8 LoCoMo
  conversations are test-only.

## Config FROZEN (after dev-only tuning on conv-26 + conv-30)
`frozen_config.json` records all settings, prompt hashes and package source hashes;
`python -m graphmem.freeze check` must pass before any test-set run. Key values: qwen3:14b
answers (thinking on), qwen3-32b-judge (thinking off), contradiction `rules`, lambda 0,
alpha 1.0, k 10, 1-hop expansion on, provenance on, pruning off, merge thresholds
0.94 / 0.95 (Events, overlapping dates). No further tuning on any conversation.

## Phase 3 (branch `phase3`, built while Phase 4 ran from tag `phase4-frozen`)
- `graphmem.service.MemoryService`: query / update (one turn: extract -> resolve -> write ->
  retire, previous turns of the conversation as extraction context) / prune; per-conversation
  write lock.
- `graphmem.agent.MemoryAgent` (LangGraph): retrieve -> reason -> [write] -> answer -> [prune
  every N turns]; prune node honors `prune_enabled` (off in the frozen config).
- `graphmem.api` (FastAPI): POST /memory/query, /memory/update, /memory/prune, /agent/turn;
  GET /health. Run: `uvicorn graphmem.api:app --port 8000` (install extra `api`).
- Not merged into `main` until Phase 4 finishes (merging changes source hashes, so the
  freeze check would then report drift by design).

## Next
- Phase 3: LangGraph agent (retrieve -> reason -> optional write -> answer; prune every N
  turns) + FastAPI (`/memory/query`, `/memory/update`, `/memory/prune`).
- Phase 4: LoCoMo benchmark on the 8 test conversations: GraphMem (prune off / on),
  Mem0 (same Ollama model + MiniLM), FAISS RAG over raw turns. Judge + F1 per category,
  retrieval latency (DB-only and end-to-end), context tokens, wall time and tokens/s.
  Ablations: contradiction mode (llm / rules / off), 1-hop expansion on / off.
- Sweep lambda / alpha / prune threshold on dev only (conv-26, conv-30).

## Evaluation protocol (all local, no paid APIs)
- Answer model: qwen3:14b (Ollama, thinking off by default; `GRAPHMEM_ANSWER_THINKING` flag).
- Judge: separate pass after all answers exist (`scripts/dev_eval.py judge`), model set by
  `GRAPHMEM_JUDGE_MODEL`. Plan: a ~32B-class instruct model at 4-bit on the same Ollama
  server, thinking off. The answer model is unloaded first; both are never loaded together.
  Report agreement with the 14B judge (raw %, Cohen's kappa) and a 50-row hand-check sample.
- Contradiction mode default `rules`; `llm` and `off` are ablations.

## Later (noted, not built)
- Second benchmark: LongMemEval subset, question types knowledge-update and
  temporal-reasoning. Before committing, measure GraphMem ingestion time on 5 questions
  (each LongMemEval question carries its own long haystack) and report it, since
  ingestion cost per question may be prohibitive with a local 14B model.
