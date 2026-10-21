# GraphMem

GraphMem is long-term memory for LLM agents, stored as a temporal knowledge graph in Neo4j.
Each conversation turn is turned into small, dated **facts**. The facts are linked to the
people and things they mention, and each fact keeps a pointer to the turn it came from
(**provenance**). At question time, GraphMem retrieves facts and shows the answer model
both the facts and the turns they came from.

The repository includes a benchmark harness that compares GraphMem with **Mem0**
(open source) and a **FAISS** baseline over raw turns on **LoCoMo**. Everything runs
locally: `qwen3:14b` on Ollama answers questions, and local judges grade the answers.
There are no paid APIs.

**Main result** (LoCoMo test set: 8 conversations, 1,307 questions; 32B judge):

- GraphMem scores **57.9%**.
- That is a **statistical tie with Mem0 at k=10** (59.0%) and **well above FAISS**
  (44.8% at k=10, 52.7% at matched context size).
- Given the **same context budget, Mem0 is better** (64.1%; −6.2 points for GraphMem,
  p<0.001).
- GraphMem's consistent advantage is **temporal questions**: +10.1 points over Mem0 (k=10)
  and +13.2 over FAISS (matched context).
- GraphMem ingestion uses **2.8× fewer LLM input tokens** than Mem0.

---

## Architecture

```
turn ──► extraction (qwen3:14b, JSON schema) ──► post-checks ──► entity resolution ──► facts
                                                                                        │
question ──► vector search over Fact nodes ──► rerank ──► 1-hop expansion ──► provenance ──► answer
```

**Graph model.** Every conversation lives in its own namespace: a label `:NS_<conversation>`
plus its own vector indexes. Searches never mix conversations.

```
(:Entity:Person|Organization|Concept|Event:NS_x {name, name_key, aliases, name_embedding})
(:Fact:NS_x {text, embedding, relation_type, validity_start, validity_end, evidence,
             turn_id, weight, access_count, last_accessed, [slot, value]})
(:Turn:NS_x {dia_id, text, session_time})                    -- for provenance
(:Fact)-[:SUBJECT]->(:Entity)   (:Fact)-[:OBJECT]->(:Entity)  -- OBJECT is absent for attributes
```

**Ingestion** (`graphmem.extraction`, `graphmem.ingest`)

- **Extraction.** Each turn is extracted with a Pydantic JSON schema. The model sees the
  three previous turns as context only. Relative dates are resolved by code
  ("last Friday" → "the Friday before 15 July 2023"), and those resolutions are passed to
  the model as hints.
- **Post-checks.** Deterministic checks drop and log extracted facts that:
  - have no quote from the *current* turn as evidence (fuzzy ≥ 0.9, and the quote must
    match the current turn better than any context turn);
  - come from questions;
  - claim that the two speakers know each other;
  - claim a family relationship the turn never states;
  - turn "missed it" into "attended".

  Two further checks change a fact instead of dropping it:
  - a non-romantic "partner" (e.g. a business partner) becomes an attribute;
  - "lost / quit / left my job" closes the job fact instead of keeping it current.
- **Entity resolution.** An extracted name first matches an existing entity exactly
  (after normalization). Otherwise it merges if the name embeddings have cosine ≥ 0.94.
  Events need cosine ≥ 0.95 *and* overlapping date intervals.
- **Contradictions** (`rules` mode, the default):
  - A restated fact reinforces the existing one instead of duplicating it.
  - Ended facts close the matching current fact.
  - Single-valued relations (`LIVES_IN`, `STUDIES_AT`, `PARTNER_OF`) replace the old
    value. `LIKES`/`DISLIKES` replace each other only for the same object.

  A replaced fact gets `validity_end` and stays in the graph as history. An `llm` mode
  (the LLM decides each pair) and an `off` mode exist for ablations.

**Retrieval** (`GraphStore.retrieve`)

1. Vector search over the conversation's facts (over-fetch k×20), filtered by
   conversation, validity and archive status.
2. Score each fact: `α·cosine + (1−α)·min(decay, 1)`, minus 0.15 if the fact is no longer
   true.
3. Keep the top k=10 facts. Then **1-hop expansion**: for each entity linked to those
   facts, add its two next-best facts.
4. **Provenance**: add the distinct source turns of all returned facts, with speaker and
   session date.

Old (superseded) facts stay visible, labelled "no longer true as of …". `as_of=` filters
to the facts valid at a given time.

**Decay**: `weight · (1 + 0.1·ln(1+access_count)) · exp(−λ·Δt_days)`. The frozen
configuration uses **λ=0**: decay plays no part in retrieval, and pruning is off (see
Limitations).

**Agent and API** (`graphmem.agent`, `graphmem.api`)

- **Agent** (LangGraph): `retrieve → reason → [write] → answer → [prune every N turns]`.
  The `reason` step decides whether the message should be remembered. If so, `write`
  stores it mid-run, and the agent retrieves again before answering.
- **API** (FastAPI): `POST /memory/query`, `POST /memory/update`, `POST /memory/prune`,
  `POST /agent/turn`, `GET /health`.
- [results/phase3_smoke.md](results/phase3_smoke.md) has a transcript of a run against the
  real stack.

---

## Reproduce

**Prerequisites**

- Python 3.11 and Docker.
- An Ollama server with `qwen3:14b` and a judge model.
  - The judge used here is `qwen3-32b-judge`: `qwen3:32b` Q4_K_M with `num_ctx 1024`.
  - Server settings: `OLLAMA_CONTEXT_LENGTH=16384`, `OLLAMA_NUM_PARALLEL=4`.
  - Mem0's prompts reach about 9.2k tokens, so a context window of 8k is too small for it.
  - All runs here used a single RTX 4090. Both models fit fully on the GPU, but never at
    the same time.

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev,bench,api]"
uv pip install "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
curl -L -o data/locomo/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
docker compose up -d                                  # Neo4j 5.26
export GRAPHMEM_LLM_BASE_URL=http://localhost:11435/v1   # OpenAI-compatible endpoint
pytest -q                                             # 77 tests (needs Neo4j; no LLM)
```

**Dev work** (conv-26 and conv-30 only; all tuning happened here):

```bash
python scripts/phase2_ingest.py --conv conv-26 conv-30
python scripts/dev_eval.py generate --run sweep --conv conv-26 conv-30 --grid   # decay sweep
python scripts/dev_eval.py judge && python scripts/dev_eval.py final
```

**Test-set benchmark (Phase 4).** The script refuses to run unless the frozen config
matches (`python -m graphmem.freeze check`) and the working tree is a clean checkout of
the tag. It can be resumed after a crash: ingestion, answers and judgments are all cached
or checkpointed.

```bash
git checkout phase4-frozen
python scripts/phase4.py      # ingest -> answer (5 systems) -> judge (32B, then 14B) -> report
```

**API**

```bash
uvicorn graphmem.api:app --port 8000
curl -X POST localhost:8000/agent/turn -H 'content-type: application/json' \
     -d '{"conversation_id":"demo","speaker":"Ana","message":"I adopted a cat named Miso yesterday."}'
```

**Commit mapping.**
- `phase4-frozen` points to **`ce3dc69`**.
- The Phase 4 run recorded commit `d754326` in `results/phase4/state.json`. That is the
  pre-rewrite hash of `ce3dc69`: the history was later rewritten to change a commit
  message only, and the file trees are identical.
- Phase 4 results were committed in `1ac9060`.

---

## Results: LoCoMo test set (Phase 4)

**Setup**
- 8 test conversations (conv-41 … conv-50), 1,307 questions. Category 5 (adversarial, no
  gold answer) is excluded.
- Every system uses the same answer model (`qwen3:14b`, thinking on) and the same answer
  prompt.
- Retrieval: k=10 for GraphMem. The baselines run at k=10, and at k=24, which is
  "token-matched": chosen on dev so that the answer prompt has the same mean size as
  GraphMem's (~1,300 tokens).
- Cells show judge accuracy % with a 95% bootstrap CI over questions.

| System | Judge | Single-hop (727) | Multi-hop (239) | Temporal (258) | Open-domain (83) | **Overall** | F1 |
|---|---|---|---|---|---|---|---|
| **GraphMem** | 32B | 63.1 [59.4, 66.6] | 35.1 [29.3, 41.0] | **70.2** [64.3, 75.6] | **39.8** [30.1, 49.4] | **57.9 [55.2, 60.6]** | 46.5 |
| | 14B | 55.8 [52.1, 59.6] | 27.6 [22.2, 33.1] | **59.3** [53.1, 65.1] | **33.7** [24.1, 43.4] | **50.0 [47.4, 52.7]** | |
| Mem0, k=10 | 32B | 69.9 [66.7, 73.0] | 32.6 [26.8, 38.5] | 60.1 [54.3, 65.9] | 36.1 [26.5, 47.0] | 59.0 [56.3, 61.6] | 44.1 |
| | 14B | 59.7 [56.1, 63.1] | 25.9 [20.1, 31.4] | 46.1 [40.3, 51.9] | 25.3 [16.9, 34.9] | 48.7 [46.0, 51.3] | |
| Mem0, token-matched (k=24) | 32B | **73.5** [70.4, 76.5] | **43.1** [36.4, 49.4] | 66.7 [60.9, 72.1] | 34.9 [25.3, 44.6] | **64.1 [61.5, 66.6]** | 47.3 |
| | 14B | **64.5** [61.2, 67.8] | **33.9** [28.0, 39.7] | 50.0 [43.8, 56.2] | 25.3 [16.9, 34.9] | **53.6 [50.9, 56.2]** | |
| FAISS, k=10 | 32B | 53.1 [49.4, 56.5] | 20.5 [15.5, 25.9] | 48.1 [41.9, 53.9] | 32.5 [22.9, 42.2] | 44.8 [42.2, 47.5] | 36.8 |
| | 14B | 45.3 [41.5, 48.7] | 16.3 [11.7, 21.3] | 39.9 [34.1, 45.7] | 28.9 [19.3, 38.6] | 37.9 [35.3, 40.6] | |
| FAISS, token-matched (k=24) | 32B | 60.1 [56.5, 63.4] | 31.4 [25.5, 37.2] | 57.0 [51.2, 62.8] | 36.1 [26.5, 45.8] | 52.7 [50.2, 55.4] | 42.7 |
| | 14B | 51.6 [48.0, 55.2] | 25.9 [20.5, 31.4] | 46.9 [40.7, 52.7] | 31.3 [21.7, 41.0] | 44.7 [42.2, 47.3] | |

The judges agree on 90.0% of answers (Cohen's κ = 0.80 over 4,720 answers). The 14B judge
is the stricter one: there were 464 answers it marked wrong that 32B accepted, and only 8
the other way. Both judges rank the systems the same way.

### Paired bootstrap: GraphMem minus baseline (accuracy points, 10,000 resamples)

| Baseline | Judge | Single-hop | Multi-hop | Temporal | Open-domain | **Overall** |
|---|---|---|---|---|---|---|
| Mem0, k=10 | 32B | **−6.7** [−10.5, −3.0] | +2.5 [−3.8, +8.8] | **+10.1** [+3.9, +16.3] | +3.6 [−6.0, +13.3] | −1.1 [−3.8, +1.8], p=.45 |
| | 14B | −3.9 [−7.7, 0.0] | +1.7 [−3.8, +7.1] | **+13.2** [+7.0, +19.8] | **+8.4** [+2.4, +15.7] | +1.3 [−1.4, +4.1], p=.36 |
| Mem0, token-matched (k=24) | 32B | **−10.3** [−14.2, −6.5] | **−7.9** [−14.6, −1.7] | +3.5 [−2.7, +9.7] | +4.8 [−4.8, +14.5] | **−6.2 [−9.0, −3.4], p<.001** |
| | 14B | **−8.7** [−12.7, −4.7] | **−6.3** [−12.1, −0.4] | **+9.3** [+2.7, +16.3] | **+8.4** [+1.2, +16.9] | **−3.6 [−6.4, −0.8], p=.016** |
| FAISS, k=10 | 32B | **+10.0** [+5.9, +14.3] | **+14.6** [+8.8, +20.5] | **+22.1** [+15.5, +28.7] | +7.2 [0.0, +15.7] | **+13.1 [+10.1, +16.1], p<.001** |
| | 14B | **+10.6** [+6.5, +14.9] | **+11.3** [+5.9, +16.7] | **+19.4** [+12.4, +26.4] | +4.8 [−2.4, +13.3] | **+12.1 [+9.2, +15.1], p<.001** |
| FAISS, token-matched (k=24) | 32B | +3.0 [−1.1, +7.2] | +3.8 [−2.1, +9.6] | **+13.2** [+6.6, +20.2] | +3.6 [−3.6, +12.0] | **+5.2 [+2.3, +8.2], p<.001** |
| | 14B | +4.3 [0.0, +8.5] | +1.7 [−3.8, +7.1] | **+12.4** [+5.8, +19.0] | +2.4 [−6.0, +10.8] | **+5.3 [+2.4, +8.3], p=.001** |

### Cost (test set)

| | GraphMem | Mem0 | FAISS |
|---|---|---|---|
| Ingestion LLM calls | 5,094 extractions (0 contradiction calls in `rules` mode) | 2,609 `add()` calls | none (embedding only) |
| Ingestion tokens in / out | **8.2M / 0.88M** | 23.3M / 0.25M | none |
| Largest prompt | ~1.6k tokens | 9,235 tokens | n/a |
| Ingestion wall time | 3,650 s (8 conversations, 4 LLM slots) | 2,526 s (4 conversations at a time) | seconds |
| Memory size | 2,753 entities, 6,189 facts (29 superseded) | 5,100 memories | 5,094 turns |
| Retrieval-only p50 / p95 | 47 / 72 ms (Neo4j) | 42 / 72 ms (k=10), 41 / 74 ms (k=24), embedded Qdrant | 0.2 / 0.3 ms (in process) |
| End-to-end retrieval p50 / p95 | 64 / 94 ms | 140 / 226 ms, 135 / 224 ms | 16 / 28 ms, 14 / 25 ms |
| Answer prompt tokens per question | 1,314 | 676 (k=10), 1,271 (k=24) | 700 (k=10), 1,329 (k=24) |
| Answer step, seconds per question (thinking on) | 7.1 | 6.4, 7.4 | 6.8, 7.8 |

Full outputs: [results/phase4/](results/phase4/) (`final_table.md`, `paired_phase4.md`,
`judge_sample_50_test.md`, `state.json`, `run.log`).

---

## Limitations

- **Mem0 wins at matched context.**
  - Given the same answer-prompt budget (~1.3k tokens), Mem0 beats GraphMem by 6.2 points
    with the 32B judge and 3.6 with the 14B judge. Both are significant.
  - The gap is on single-hop and multi-hop questions. Mem0 writes long, self-contained
    memories, and more of them help directly.
  - GraphMem only ties Mem0 at k=10 and keeps a lead on temporal questions.
  - On dev, adding provenance (source turns) improved GraphMem by about 9 points. Even so,
    with provenance GraphMem only reaches parity with Mem0 at k=10.

- **Decay hurt on LoCoMo.** Every LoCoMo question is about history spanning months, so
  favouring recent memories only costs recall.
  - An early dev run with λ=0.05/day and α=0.7 scored 13.8%, against 43.4% with decay off
    (conv-26, older pipeline). Facts from early sessions never came back.
  - The dev sweep below (conv-26 / conv-30 / both; 233 questions; thinking off; 14B judge)
    found no setting that beats λ=0 by more than noise. The best setting is +1.3 points,
    about 3 questions. λ=0 was frozen; the λ=0 row and α=1 column are identical by
    construction.

    | λ \ α | 0.85 | 0.90 | 0.95 | 1.0 |
    |---|---|---|---|---|
    | 0 | 46.7 / 42.0 / **45.1** | same | same | same |
    | 0.001 | 45.4 / 39.5 / 43.3 | 46.1 / 39.5 / 43.8 | 48.7 / 42.0 / **46.4** | 45.1 |
    | 0.005 | 43.4 / 42.0 / 42.9 | 42.8 / 43.2 / 42.9 | 46.1 / 40.7 / 44.2 | 45.1 |
    | 0.01 | 38.2 / 39.5 / 38.6 | 41.4 / 40.7 / 41.2 | 44.7 / 39.5 / 42.9 | 45.1 |

  - Pruning by decay (α=1 retrieval; both dev conversations) cost about 1 point of
    accuracy for every 2% of the graph removed:

    | λ | Prune threshold | Accuracy % | Facts kept | Archived |
    |---|---|---|---|---|
    | any | off | 45.1 | 1,152 | 0 |
    | 0.01 | 0.05 / 0.1 | 45.1 | 1,152 | 0 |
    | 0.01 | 0.2 | 38.6 | 1,007 | 145 |
    | 0.05 | 0.05 | 22.3 | 484 | 669 |
    | 0.05 | 0.1 | 16.7 | 328 | 825 |
    | 0.05 | 0.2 | 9.9 | 210 | 943 |

  - Decay and pruning remain in the code, but **LoCoMo cannot show a benefit from them**.
    A knowledge-update benchmark would be needed to evaluate them fairly.

- **Superseded facts are rare in LoCoMo.** Only 29 of 6,218 test-set facts (0.5%) were
  ever superseded. LoCoMo barely tests the temporal-validity machinery (validity ranges,
  `as_of`, contradiction handling).
  - On dev, the LLM contradiction mode had poor precision: 0 of 5 retirements were correct
    on conv-30, and 1 of 20 in an earlier version. That is why `rules` is the default.

- **The judges are local, so the numbers are not comparable to published LoCoMo results.**
  - Published results use a GPT-4o-mini judge (e.g. the Mem0 paper). Here both judges are
    Qwen3 (14B and 32B, Q4_K_M, thinking off).
  - The judges agree well with each other (κ = 0.80), but absolute accuracy depends on
    the judge: the two differ by 7–10 points.
  - Only comparisons *within* this setup are meaningful. Hand-check samples are in
    `results/dev_eval/judge_sample_50*.md` and `results/phase4/judge_sample_50_test.md`.

- **Small development set.** All tuning used 2 conversations (233 questions), where a
  difference of ±6 points overall is within noise. The configuration was then frozen
  (`frozen_config.json`) before the single test-set run.

- **Extraction misses.** From a dev audit of 20 single-hop failures:
  - 6 were facts that lost a key detail during extraction;
  - 5 were facts extracted but not retrieved;
  - 5 were answer errors;
  - 2 were never extracted;
  - 2 were dropped by a post-check. One of these is a known false positive: a statement
    followed by "…right?" is treated as a question.

- **Latency comparisons depend on the architecture.** GraphMem queries Neo4j over the
  network. Mem0 uses embedded Qdrant and FAISS runs in process, all on a single Windows
  machine.

- **Mem0 lost one batch.** One of its 2,609 `add()` calls returned malformed JSON, which
  Mem0 itself could not parse, so that batch stored no memories.

- **Agent prompt (Phase 3).** The smoke test shows the agent sometimes answers in a
  participant's voice instead of as an assistant ("Caroline: Oscar is doing great…").
  It sometimes adds small invented details, and once emitted a stray non-English
  character. The agent's answer prompt was not tuned; it is not part of the benchmark.

---

## Layout

```
src/graphmem/       store, schema, extraction, ingest, timehints, llm, freeze, service, agent, api
src/graphmem/bench/ qa (answer/judge/F1), faiss_rag, mem0_adapter, mem0_ingest, baselines
scripts/            phase2_ingest, dev_eval, token_match, mem0_ingest, phase4, phase3_smoke
results/            dev and test reports (large JSONL answers/judgments are not committed)
frozen_config.json  settings, baselines, prompt and source hashes for the Phase 4 run
PLAN.md             project history and plan
```
