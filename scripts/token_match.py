"""Choose token-matched k for the FAISS and Mem0 baselines (retrieval only; no answers).

Prompt tokens are counted with the Qwen3 tokenizer, calibrated against the server-counted
prompt tokens of existing dev answers (constant chat-template offset). For each candidate
k, the mean answer-prompt size over the 233 dev questions is computed; the k closest to
GraphMem+provenance's mean (from dev answers) within +/-10% is chosen.

    .venv/Scripts/python scripts/token_match.py
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from transformers import AutoTokenizer

from graphmem import Embedder
from graphmem.bench import mem0_adapter as ma
from graphmem.bench.faiss_rag import FaissRAG
from graphmem.bench.qa import ANSWER_SYSTEM, answer_prompt
from graphmem.locomo import load_locomo

DEV = ("conv-26", "conv-30")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")


def count(context: str, question: str) -> int:
    return len(tok(ANSWER_SYSTEM + answer_prompt(question, context))["input_ids"])


rows = [json.loads(l) for f in Path("results/dev_eval/answers").glob("*.jsonl")
        for l in f.open(encoding="utf-8")]
rows = [r for r in rows if r["thinking"] and r["conv"] in DEV]
offsets = [r["prompt_tokens"] - count(r["context"], r["question"]) for r in rows[:400]]
offset = statistics.median(offsets)
spread = (min(offsets), max(offsets))
target = statistics.mean(r["prompt_tokens"] for r in rows if r["system"] == "gm:l=0.0,a=1.0,prov=1")
print(f"calibration: server = tokenizer + {offset} (range {spread}); target (GraphMem+provenance) = {target:.0f}")

convs = {c.sample_id: c for c in load_locomo() if c.sample_id in DEV}
qs = [(cid, q.question) for cid, c in convs.items() for q in c.qa if q.category != 5 and q.answer]
emb = Embedder()


def mean_tokens(contexts):
    return statistics.mean(count(ctx, q) + offset for ctx, (_, q) in zip(contexts, qs))


out = {"target_prompt_tokens": round(target, 1), "tolerance": 0.10, "offset": offset, "faiss": {}, "mem0": {}}
rags = {cid: FaissRAG(c, emb) for cid, c in convs.items()}
for k in range(10, 41, 2):
    out["faiss"][k] = round(mean_tokens([rags[cid].retrieve(q, k)["context"] for cid, q in qs]), 1)
print("faiss:", out["faiss"])

from graphmem import Settings
from graphmem.llm import make_llm

ma.configure(make_llm(Settings()))  # search makes no LLM calls; required to build Mem0
mems = {cid: ma.build_memory(cid) for cid in DEV}
for k in range(10, 41, 2):
    ctxs = []
    for cid, q in qs:
        res = mems[cid].search(q, top_k=k, filters={"user_id": cid})["results"]
        ctxs.append("\n".join(f"- {x['memory']} [stated {(x.get('metadata') or {}).get('session_date', '?')}]"
                              for x in res))
    out["mem0"][k] = round(mean_tokens(ctxs), 1)
print("mem0:", out["mem0"])

for name in ("faiss", "mem0"):
    k, v = min(out[name].items(), key=lambda kv: abs(kv[1] - target))
    ok = abs(v - target) <= 0.10 * target
    out[f"{name}_matched_k"] = k
    print(f"{name}: matched k = {k} -> {v:.0f} tokens ({100 * (v / target - 1):+.1f}% vs target) "
          f"{'OK' if ok else 'OUTSIDE +/-10%'}")
Path("results/dev_eval/token_match.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
