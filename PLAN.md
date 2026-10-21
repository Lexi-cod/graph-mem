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
- Merged into `main` after Phase 4 finished (the freeze check on `main` now reports drift by
  design; check out `phase4-frozen` to reproduce the test run).

## Status
- Phase 4 done (results/phase4/, README.md); config was frozen at tag `phase4-frozen` (ce3dc69).
- Phase 3 merged into `main`; live smoke test in results/phase3_smoke.md.
- LongMemEval: deferred (not now).

## Later (noted, not built)
- Second benchmark: LongMemEval subset, question types knowledge-update and
  temporal-reasoning. Before committing, measure GraphMem ingestion time on 5 questions
  (each LongMemEval question carries its own long haystack) and report it, since
  ingestion cost per question may be prohibitive with a local 14B model.
