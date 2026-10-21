"""Dev-set evaluation in two passes, so the answer model and the judge are never loaded
together on the Ollama server.

  generate  retrieval (sequential, for clean latency) + answers. No judging.
  judge     separate pass with GRAPHMEM_JUDGE_MODEL (thinking off). Other models are
            unloaded first and residency is checked via /api/ps.
  report    tables from answers + judgments; judge agreement and a 50-row hand-check sample.

System specs:
  gm:l=<lambda>,a=<alpha>[,prune=<threshold>][,expand=0]   GraphMem fact retrieval
  faiss                                                    dense RAG over raw turns
  mem0                                                     Mem0 OSS (scripts/mem0_ingest.py first)

Examples:
  dev_eval.py generate --run sweep --conv conv-26 conv-30 --grid
  dev_eval.py generate --run thinking --conv conv-26 --systems "gm:l=0,a=1" faiss --thinking
  dev_eval.py judge
  dev_eval.py report
"""

import argparse
import asyncio
import json
import logging
import math
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from graphmem import GraphStore, Settings
from graphmem.bench.faiss_rag import FaissRAG
from graphmem.bench.qa import answer, judge, token_f1
from graphmem.llm import PreflightError, UsageTracker, make_judge, make_llm, same_model
from graphmem.locomo import CATEGORIES, TestSetGuard, load_locomo, require_dev

OUT = Path("results/dev_eval")
GRID_L = [0.0, 0.001, 0.005, 0.01]
GRID_A = [0.85, 0.9, 0.95, 1.0]
PRUNE_L = [0.01, 0.05]
PRUNE_T = [0.05, 0.1, 0.2]
CATS = [CATEGORIES[c] for c in (4, 1, 2, 3)]


def parse_spec(spec: str) -> tuple[str, dict]:
    kind, _, rest = spec.partition(":")
    params = {}
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=")
        params[k] = float(v)
    return kind, params


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(p / 100 * len(xs)) - 1))] if xs else float("nan")


def evidence_ids(ev: list[str]) -> set[str]:
    return {x for e in ev for x in re.split(r"[;,\s]+", e) if x}


# ---------------------------------------------------------------------------- residency


async def ensure_only(llm, keep: str) -> None:
    """Unload every model except `keep` from the Ollama server and verify."""
    be = llm.backend
    for m in await be.loaded_models():
        if not same_model(m, keep):
            print(f"  unloading {m}")
            await be.unload(m)
    for _ in range(30):
        loaded = [m for m in await be.loaded_models() if not same_model(m, keep)]
        if not loaded:
            return
        await asyncio.sleep(1)
    raise RuntimeError(f"models still loaded: {loaded}")


# ---------------------------------------------------------------------------- generate


async def cmd_generate(args, s: Settings) -> None:
    specs = list(args.systems or [])
    if args.grid:
        specs += ["faiss"] + [f"gm:l={l},a={a}" for l in GRID_L for a in GRID_A]
    if args.prune_grid:
        specs += [f"gm:l={l},a=1,prune={t}" for l in PRUNE_L for t in PRUNE_T]
    specs = list(dict.fromkeys(specs))
    for cid in args.conv:
        try:
            require_dev(cid, s.dev_conversations, args.allow_test)
        except TestSetGuard as exc:
            sys.exit(str(exc))

    tracker = UsageTracker()
    llm = make_llm(s, tracker)
    try:
        print("answer LLM:", await llm.preflight(), f"thinking={'on' if args.thinking else 'off'}")
    except PreflightError as exc:
        sys.exit(f"Preflight failed: {exc}")
    await ensure_only(llm, s.llm_model)

    convs = {c.sample_id: c for c in load_locomo() if c.sample_id in args.conv}
    root = await GraphStore.connect(s)
    rows, meta = [], {"run": args.run, "thinking": args.thinking, "k": args.k, "answer_llm": llm.identity,
                      "systems": specs, "answer_phase": {}}
    for cid, conv in convs.items():
        store = await root.scoped(cid)
        st = await store.stats()
        if st["current_facts"] == 0:
            sys.exit(f"{cid} is not ingested")
        now = conv.last_time
        qas = [q for q in conv.qa if q.category != 5 and q.answer]
        rag = mem = timed = None
        await store.retrieve("warmup", k=args.k, now=now, reinforce=False)
        for spec in specs:
            kind, p = parse_spec(spec)
            kk = int(p.get("k", args.k))  # per-system k, e.g. "faiss:k=24"
            graph_facts, archived = st["current_facts"], 0
            if kind == "gm" and p.get("prune") is not None:
                await store.unarchive_all()
                rep = await store.prune(threshold=p["prune"], now=now, decay_lambda=p["l"])
                archived = len(rep.archived_facts)
                graph_facts = (await store.stats())["current_facts"]
            if kind == "faiss" and rag is None:
                rag = FaissRAG(conv, store.embedder)
                rag.retrieve("warmup", kk)
            if kind == "mem0" and mem is None:
                from graphmem.bench import mem0_adapter as ma
                ma.configure(llm)
                mem = ma.build_memory(cid)
                timed = ma.TimedStore(mem)
                mem.search("warmup", top_k=kk, filters={"user_id": cid})
                graph_facts = len(mem.get_all(filters={"user_id": cid}, top_k=100000)["results"])

            # ---- retrieval, sequential
            retrieved = []
            for q in qas:
                if kind == "gm":
                    t0 = time.perf_counter()
                    r = await store.retrieve(q.question, k=kk, now=now, reinforce=False,
                                             alpha=p["a"], decay_lambda=p["l"],
                                             expand=bool(p.get("expand", 1)),
                                             provenance=bool(p.get("prov", 0)))
                    e2e = (time.perf_counter() - t0) * 1000
                    retrieved.append(dict(context=r.to_context(), db_ms=r.db_ms, e2e_ms=e2e,
                                          turn_ids=[f.turn_id for f in r.facts], n_items=len(r.facts)))
                elif kind == "faiss":
                    r = rag.retrieve(q.question, kk)
                    retrieved.append(dict(context=r["context"], db_ms=r["search_ms"],
                                          e2e_ms=r["search_ms"] + r["embed_ms"], turn_ids=r["turn_ids"],
                                          n_items=len(r["turn_ids"])))
                elif kind == "mem0":
                    timed.ms = 0.0
                    t0 = time.perf_counter()
                    res = mem.search(q.question, top_k=kk, filters={"user_id": cid})["results"]
                    e2e = (time.perf_counter() - t0) * 1000
                    ctx = "\n".join(
                        f"- {x['memory']} [stated {(x.get('metadata') or {}).get('session_date', '?')}]"
                        for x in res)
                    retrieved.append(dict(context=ctx, db_ms=timed.ms, e2e_ms=e2e, n_items=len(res),
                                          turn_ids=[t for x in res for t in
                                                    ((x.get("metadata") or {}).get("turn_ids") or "").split(",")]))
                else:
                    sys.exit(f"unknown system {spec}")
            if kind == "gm" and p.get("prune") is not None:
                await store.unarchive_all()

            # ---- answers, concurrent (answer-phase wall time recorded per system)
            t0 = time.perf_counter()
            ans = await asyncio.gather(*(
                answer(llm, q.question, ret["context"], thinking=args.thinking,
                       max_tokens=s.answer_max_tokens, thinking_max_tokens=s.answer_thinking_max_tokens)
                for q, ret in zip(qas, retrieved)))
            wall = time.perf_counter() - t0
            uncached = sum(not a.cached for a in ans)
            meta["answer_phase"][f"{cid}|{spec}"] = {
                "wall_s": round(wall, 1), "uncached": uncached, "n": len(ans),
                "sum_latency_s": round(sum(a.latency_s for a in ans), 1),
                "mean_output_tokens": round(statistics.mean(a.output_tokens for a in ans), 1)}
            for i, (q, ret, a) in enumerate(zip(qas, retrieved, ans)):
                ev = evidence_ids(q.evidence)
                rows.append({
                    "run": args.run, "conv": cid, "system": spec, "thinking": args.thinking, "qid": i,
                    "category": q.category_name, "question": q.question, "gold": q.answer, "pred": a.text,
                    "f1": token_f1(a.text, q.answer), "answer_latency_s": a.latency_s,
                    "answer_cached": a.cached, "prompt_tokens": a.prompt_tokens,
                    "output_tokens": a.output_tokens, "db_ms": ret["db_ms"], "e2e_ms": ret["e2e_ms"],
                    "n_items": ret["n_items"], "evidence": sorted(ev),
                    "evidence_hit": bool(ev & {t for t in ret["turn_ids"] if t}),
                    "graph_facts": graph_facts, "archived_facts": archived, "context": ret["context"]})
            print(f"  {cid:<8} {spec:<28} answers {wall:6.1f}s ({uncached} new)  "
                  f"db p50 {pct([r['db_ms'] for r in retrieved], 50):6.1f} ms")
    await root.close()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "answers").mkdir(exist_ok=True)
    with (OUT / "answers" / f"{args.run}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta["usage"] = tracker.summary()
    (OUT / "answers" / f"{args.run}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'answers' / (args.run + '.jsonl')} ({len(rows)} rows)")


# ---------------------------------------------------------------------------- judge


def judge_tag(model: str) -> str:
    return re.sub(r"[^\w.-]", "_", model)


async def cmd_judge(args, s: Settings) -> None:
    if args.judge_model:
        s = s.model_copy(update={"judge_model": args.judge_model})
    tracker = UsageTracker()
    jllm, rec = make_judge(s, tracker)
    try:
        print("judge:", await jllm.preflight(), "(thinking off)")
    except PreflightError as exc:
        sys.exit(f"Judge preflight failed: {exc}")
    await ensure_only(jllm, jllm.model)  # never both loaded
    # Load the judge with one tiny uncached request, then require 100% GPU residency.
    await jllm.backend.acomplete({"system": None, "prompt": "Reply OK.", "temperature": 0.0,
                                  "max_output_tokens": 4}, None)
    res = [m for m in await jllm.backend.residency() if same_model(m["name"], jllm.model)]
    if not res:
        sys.exit(f"judge model {jllm.model} not resident after warmup")
    m = res[0]
    others = [x["name"] for x in await jllm.backend.residency() if not same_model(x["name"], jllm.model)]
    gpu = m.get("size_vram", 0) / max(1, m.get("size", 1))
    print(f"  /api/ps: {m['name']} size={m.get('size', 0) / 2**30:.1f} GiB "
          f"vram={m.get('size_vram', 0) / 2**30:.1f} GiB ({100 * gpu:.1f}% GPU), "
          f"context_length={m.get('context_length')}, other models loaded: {others or 'none'}")
    if gpu < 1.0 or others:
        sys.exit("ABORT: judge is not 100% on GPU (CPU offload) or another model is loaded")
    files = sorted((OUT / "answers").glob("*.jsonl"))
    if args.runs:
        files = [f for f in files if f.stem in args.runs]
    triples = {}
    for f in files:
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            triples[(r["question"], r["gold"], r["pred"])] = None
    keys = list(triples)
    print(f"judging {len(keys)} unique (question, gold, prediction) triples from {len(files)} run(s)")
    t0 = time.perf_counter()
    labels = await asyncio.gather(*(judge(jllm, q, g, p) for q, g, p in keys))
    out = OUT / "judgments" / f"{judge_tag(jllm.model)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out.exists():
        for line in out.open(encoding="utf-8"):
            j = json.loads(line)
            existing[(j["question"], j["gold"], j["pred"])] = j["correct"]
    existing.update(dict(zip(keys, labels)))
    with out.open("w", encoding="utf-8") as fh:
        for (q, g, p), c in existing.items():
            fh.write(json.dumps({"question": q, "gold": g, "pred": p, "correct": c,
                                 "judge": rec}, ensure_ascii=False) + "\n")
    u = tracker.summary().get("judge", {})
    print(f"wrote {out}: {len(existing)} judgments; {time.perf_counter() - t0:.0f}s, "
          f"calls={u.get('calls')} cached={u.get('cache_hits')}")


# ---------------------------------------------------------------------------- report


def load_rows(runs=None) -> list[dict]:
    rows = []
    for f in sorted((OUT / "answers").glob("*.jsonl")):
        if runs and f.stem not in runs:
            continue
        rows += [json.loads(line) for line in f.open(encoding="utf-8")]
    return rows


def load_judgments(tag: str) -> dict:
    path = OUT / "judgments" / f"{tag}.jsonl"
    return {(j["question"], j["gold"], j["pred"]): j["correct"]
            for j in map(json.loads, path.open(encoding="utf-8"))} if path.exists() else {}


def acc(rows, J) -> float:
    vals = [J[(r["question"], r["gold"], r["pred"])] for r in rows]
    return 100 * sum(vals) / len(vals) if vals else float("nan")


def cmd_report(args, s: Settings) -> None:
    tag = judge_tag(args.judge or s.judge_model or s.llm_model)
    J = load_judgments(tag)
    rows = [r for r in load_rows() if (r["question"], r["gold"], r["pred"]) in J]
    print(f"judge: {tag}; {len(rows)} judged rows")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["run"], r["conv"], r["system"], r["thinking"])].append(r)
    out_lines = []

    def emit(line=""):
        print(line)
        out_lines.append(line)

    # sweep table
    sweep = [k for k in groups if k[0] == "sweep"]
    if sweep:
        emit("\n### 4a  decay sweep: overall judge accuracy % (conv-26 / conv-30 / both), rows lambda, cols alpha")
        emit("  " + f"{'lambda':<8}" + "".join(f"{'a=' + str(a):>24}" for a in GRID_A))
        for lam in GRID_L:
            line = f"  {lam:<8}"
            for a in GRID_A:
                spec = f"gm:l={lam},a={a}"
                g26 = groups.get(("sweep", "conv-26", spec, False), [])
                g30 = groups.get(("sweep", "conv-30", spec, False), [])
                line += f"{acc(g26, J):>8.1f} /{acc(g30, J):>5.1f} /{acc(g26 + g30, J):>5.1f}"
            emit(line)
        f26 = groups.get(("sweep", "conv-26", "faiss", False), [])
        f30 = groups.get(("sweep", "conv-30", "faiss", False), [])
        emit(f"  faiss   {acc(f26, J):>8.1f} /{acc(f30, J):>5.1f} /{acc(f26 + f30, J):>5.1f}")

    prune = [k for k in groups if k[0] == "prune"]
    if prune:
        emit("\n### 4b  decay-for-prune-only (alpha=1 retrieval): accuracy vs graph size, both dev convs")
        emit(f"  {'lambda':<8}{'threshold':<11}{'acc %':>8}{'facts kept':>12}{'archived':>10}")
        base = groups.get(("sweep", "conv-26", "gm:l=0.0,a=1.0", False), []) + \
            groups.get(("sweep", "conv-30", "gm:l=0.0,a=1.0", False), [])
        if base:
            kept = sum({(r["conv"]): r["graph_facts"] for r in base}.values())
            emit(f"  {'-':<8}{'off':<11}{acc(base, J):>8.1f}{kept:>12}{0:>10}")
        for lam in PRUNE_L:
            for t in PRUNE_T:
                spec = f"gm:l={lam},a=1,prune={t}"
                g = groups.get(("prune", "conv-26", spec, False), []) + groups.get(("prune", "conv-30", spec, False), [])
                if g:
                    kept = sum({r["conv"]: r["graph_facts"] for r in g}.values())
                    arch = sum({r["conv"]: r["archived_facts"] for r in g}.values())
                    emit(f"  {lam:<8}{t:<11}{acc(g, J):>8.1f}{kept:>12}{arch:>10}")

    # main conv-26 comparison
    emit("\n### conv-26 systems: judge acc % per category, F1, latency, answer step")
    emit(f"  {'system':<24}{'think':<6}" + "".join(f"{c:>12}" for c in CATS) +
         f"{'overall':>9}{'F1':>6}{'db p50/p95 ms':>16}{'e2e p50/p95 ms':>17}{'ans wall s':>11}"
         f"{'ans s/q':>8}{'out tok':>8}{'prompt tok':>11}")
    meta = {}
    for f in (OUT / "answers").glob("*.meta.json"):
        m = json.loads(f.read_text(encoding="utf-8"))
        meta[m["run"]] = m
    shown = set()
    for key in sorted(groups):
        run, cid, spec, th = key
        if cid != "conv-26" or (run == "sweep" and spec not in ("faiss", "gm:l=0.0,a=1.0")) or run == "prune":
            continue
        if (spec, th) in shown:
            continue
        shown.add((spec, th))
        g = groups[key]
        ph = meta.get(run, {}).get("answer_phase", {}).get(f"{cid}|{spec}", {})
        line = f"  {spec:<24}{'on' if th else 'off':<6}"
        line += "".join(f"{acc([r for r in g if r['category'] == c], J):>12.1f}" for c in CATS)
        line += f"{acc(g, J):>9.1f}{100 * statistics.mean(r['f1'] for r in g):>6.1f}"
        line += f"{pct([r['db_ms'] for r in g], 50):>8.1f}/{pct([r['db_ms'] for r in g], 95):<7.1f}"
        line += f"{pct([r['e2e_ms'] for r in g], 50):>9.1f}/{pct([r['e2e_ms'] for r in g], 95):<7.1f}"
        line += f"{ph.get('wall_s', float('nan')):>11.1f}"
        line += f"{statistics.mean(r['answer_latency_s'] for r in g):>8.2f}"
        line += f"{statistics.mean(r['output_tokens'] for r in g):>8.0f}"
        line += f"{statistics.mean(r['prompt_tokens'] for r in g):>11.0f}"
        emit(line)
    emit("  (ans wall s = wall time of the concurrent answer phase when uncached; ans s/q = mean "
         "per-request latency; out tok includes thinking tokens)")

    # judge agreement
    if args.compare:
        a_tag, b_tag = args.compare
        A, B = load_judgments(a_tag), load_judgments(b_tag)
        both = [k for k in A if k in B]
        agree = sum(A[k] == B[k] for k in both)
        pa = sum(A[k] for k in both) / len(both)
        pb = sum(B[k] for k in both) / len(both)
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = (agree / len(both) - pe) / (1 - pe) if pe < 1 else float("nan")
        conf = Counter((A[k], B[k]) for k in both)
        emit(f"\n### judge agreement {a_tag} vs {b_tag}: {len(both)} triples, raw {100 * agree / len(both):.1f}%, "
             f"Cohen's kappa {kappa:.3f}; {a_tag} correct-rate {100 * pa:.1f}%, {b_tag} {100 * pb:.1f}%")
        emit(f"  confusion (A, B): {dict(conf)}")
        rng = random.Random(0)
        sample = rng.sample(both, min(50, len(both)))
        lines = [f"# Judge hand-check sample (50 rows, seed 0)\n\nA = {a_tag}, B = {b_tag}. "
                 "Fill the `you` column with C/W.\n",
                 "| # | question | gold | prediction | A | B | you |", "|---|---|---|---|---|---|---|"]
        for i, (q, g, p) in enumerate(sample, 1):
            cell = lambda x: str(x).replace("|", "/").replace("\n", " ")
            lines.append(f"| {i} | {cell(q)} | {cell(g)} | {cell(p)} | {'C' if A[(q, g, p)] else 'W'} | "
                         f"{'C' if B[(q, g, p)] else 'W'} |  |")
        (OUT / "judge_sample_50.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        emit(f"  wrote {OUT / 'judge_sample_50.md'}")
    (OUT / f"report_{tag}.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")


FINAL_SYSTEMS = [("FAISS RAG", "faiss"), ("GraphMem (lambda=0)", "gm:l=0.0,a=1.0"),
                 ("GraphMem + provenance", "gm:l=0.0,a=1.0,prov=1"), ("Mem0", "mem0")]


def bootstrap_ci(vals: list[bool], b: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile CI, resampling questions with replacement."""
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return 100 * sum(vals) / n, 100 * means[int(0.025 * b)], 100 * means[int(0.975 * b) - 1]


def kappa(a: list[bool], b: list[bool]) -> tuple[float, float]:
    n = len(a)
    agree = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 100 * agree, (agree - pe) / (1 - pe) if pe < 1 else float("nan")


def cmd_final(args, s: Settings) -> None:
    judges = args.judges
    J = {t: load_judgments(t) for t in judges}
    rows = [r for r in load_rows() if r["thinking"] and r["conv"] in args.convs]
    # one row per (system, conv, qid): the latest run wins
    pick: dict = {}
    for r in rows:
        pick[(r["system"], r["conv"], r["qid"])] = r
    lines = []

    def emit(x=""):
        print(x)
        lines.append(x)

    emit(f"# Results: {', '.join(args.convs)}; thinking ON; judge accuracy % [95% bootstrap CI over questions]")
    emit(f"judges: {', '.join(judges)}\n")
    hdr = "| System | Judge | n | " + " | ".join(CATS) + " | **Overall** | F1 |"
    emit(hdr)
    emit("|" + "---|" * (hdr.count("|") - 1))
    table_triples = set()
    for label, spec in args.systems_resolved:
        sys_rows = sorted((r for (sp, _, _), r in pick.items() if sp == spec), key=lambda r: (r["conv"], r["qid"]))
        if not sys_rows:
            emit(f"| {label} | - | 0 | (no thinking-on answers) |")
            continue
        for r in sys_rows:
            table_triples.add((r["question"], r["gold"], r["pred"]))
        for t in judges:
            judged = [r for r in sys_rows if (r["question"], r["gold"], r["pred"]) in J[t]]
            cells = []
            for c in CATS + ["overall"]:
                sub = judged if c == "overall" else [r for r in judged if r["category"] == c]
                m, lo, hi = bootstrap_ci([J[t][(r["question"], r["gold"], r["pred"])] for r in sub])
                cells.append(f"{m:.1f} [{lo:.1f}, {hi:.1f}]" + (f" (n={len(sub)})" if c != "overall" else ""))
            f1 = 100 * statistics.mean(r["f1"] for r in judged) if judged else float("nan")
            emit(f"| {label} | {t} | {len(judged)} | " + " | ".join(cells[:-1]) +
                 f" | **{cells[-1]}** | {f1:.1f} |")

    emit("\n## Retrieval latency and answer step (thinking on)")
    emit("| System | retrieval-only p50 / p95 ms | end-to-end retrieval p50 / p95 ms | answer s/q | answer out tok | prompt tok |")
    emit("|---|---|---|---|---|---|")
    for label, spec in args.systems_resolved:
        g = [r for (sp, _, _), r in pick.items() if sp == spec]
        if g:
            emit(f"| {label} | {pct([r['db_ms'] for r in g], 50):.1f} / {pct([r['db_ms'] for r in g], 95):.1f} | "
                 f"{pct([r['e2e_ms'] for r in g], 50):.1f} / {pct([r['e2e_ms'] for r in g], 95):.1f} | "
                 f"{statistics.mean(r['answer_latency_s'] for r in g):.2f} | "
                 f"{statistics.mean(r['output_tokens'] for r in g):.0f} | {statistics.mean(r['prompt_tokens'] for r in g):.0f} |")

    if len(judges) == 2:
        a, b = judges
        for name, keys in (("final-table answers", sorted(table_triples)),
                           ("all dev answers", sorted(set(J[a]) & set(J[b])))):
            keys = [k for k in keys if k in J[a] and k in J[b]]
            raw, kp = kappa([J[a][k] for k in keys], [J[b][k] for k in keys])
            conf = Counter(("C" if J[a][k] else "W") + ("C" if J[b][k] else "W") for k in keys)
            emit(f"\njudge agreement on {name}: {len(keys)} triples, raw {raw:.1f}%, Cohen's kappa {kp:.3f}; "
                 f"{a} C-rate {100 * statistics.mean(J[a][k] for k in keys):.1f}%, "
                 f"{b} C-rate {100 * statistics.mean(J[b][k] for k in keys):.1f}%; "
                 f"confusion ({a[:8]}/{b[:8]}): {dict(conf)}")
        # The hand-check sample is drawn from the answers of `sample_systems` only, so adding
        # a system to the table does not change an already-delivered sample.
        pool = {(r["question"], r["gold"], r["pred"]) for (sp, _, _), r in pick.items()
                if sp in args.sample_systems}
        keys = sorted(k for k in pool if k in J[a] and k in J[b])
        sample = random.Random(0).sample(keys, min(50, len(keys)))
        cell = lambda x: str(x).replace("|", "/").replace("\n", " ")
        md = [f"# Judge hand-check sample (50 answers from {', '.join(args.sample_systems)}, seed 0)\n",
              f"A = `{a}`, B = `{b}`. Put C (correct) or W (wrong) in the `you` column.\n",
              "| # | question | gold | prediction | A | B | you |", "|---|---|---|---|---|---|---|"]
        for i, (q, g, p) in enumerate(sample, 1):
            md.append(f"| {i} | {cell(q)} | {cell(g)} | {cell(p)} | {'C' if J[a][(q, g, p)] else 'W'} | "
                      f"{'C' if J[b][(q, g, p)] else 'W'} |  |")
        target = OUT / args.sample_out
        if target.exists() and not args.overwrite_sample:
            emit(f"\n{target} exists; not overwritten (it may hold your hand-check marks). "
                 "Use --sample-out NAME or --overwrite-sample.")
        else:
            target.write_text("\n".join(md) + "\n", encoding="utf-8")
            emit(f"\nwrote {target} ({sum(J[a][k] != J[b][k] for k in sample)} of 50 rows "
                 f"where the judges disagree)")
    (OUT / "final_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def paired_bootstrap(diffs: list[int], b: int = 10000, seed: int = 0) -> tuple[float, float, float, float]:
    """Mean of per-question differences (A correct - B correct) in points, 95% percentile CI,
    and a two-sided bootstrap p-value for 'no difference'."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    m = sum(diffs) / n
    p = 2 * min(sum(x <= 0 for x in means), sum(x >= 0 for x in means)) / b
    return 100 * m, 100 * means[int(0.025 * b)], 100 * means[int(0.975 * b) - 1], min(1.0, p)


def thinking_rows(spec: str, convs=None) -> dict:
    out = {}
    for r in load_rows():
        if r["thinking"] and r["system"] == spec and (convs is None or r["conv"] in convs):
            out[(r["conv"], r["qid"])] = r  # latest run wins
    return out


def cmd_paired(args, s: Settings) -> None:
    lines = []

    def emit(x=""):
        print(x)
        lines.append(x)

    emit(f"# Paired bootstrap over questions ({', '.join(args.convs)}; thinking on): A minus B, accuracy points")
    emit("10,000 resamples; [95% CI]; p = two-sided bootstrap p-value\n")
    for a_label, a_spec, b_label, b_spec in args.pairs_resolved:
        A, B = thinking_rows(a_spec, args.convs), thinking_rows(b_spec, args.convs)
        keys = sorted(set(A) & set(B))
        for tag in args.judges:
            J = load_judgments(tag)
            emit(f"## {a_label} vs {b_label}, judge {tag} (n={len(keys)})")
            emit("| category | n | A acc | B acc | A-B [95% CI] | p |")
            emit("|---|---|---|---|---|---|")
            for c in CATS + ["overall"]:
                ks = [k for k in keys if c == "overall" or A[k]["category"] == c]
                ca = [J[(A[k]["question"], A[k]["gold"], A[k]["pred"])] for k in ks]
                cb = [J[(B[k]["question"], B[k]["gold"], B[k]["pred"])] for k in ks]
                m, lo, hi, p = paired_bootstrap([int(x) - int(y) for x, y in zip(ca, cb)])
                emit(f"| {c} | {len(ks)} | {100 * statistics.mean(ca):.1f} | {100 * statistics.mean(cb):.1f} | "
                     f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}] | {p:.3f} |")
            emit()
    (OUT / f"paired_{args.out}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/dev_eval", help="answers/judgments/report directory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--run", required=True)
    g.add_argument("--conv", nargs="+", default=["conv-26"])
    g.add_argument("--systems", nargs="*")
    g.add_argument("--grid", action="store_true")
    g.add_argument("--prune-grid", action="store_true")
    g.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                   help="answer-step thinking (default: GRAPHMEM_ANSWER_THINKING)")
    g.add_argument("--k", type=int, default=10)
    g.add_argument("--allow-test", action="store_true")
    j = sub.add_parser("judge")
    j.add_argument("--judge-model")
    j.add_argument("--runs", nargs="*")
    r = sub.add_parser("report")
    r.add_argument("--judge")
    r.add_argument("--compare", nargs=2, metavar=("JUDGE_A", "JUDGE_B"))
    pr = sub.add_parser("paired")
    pr.add_argument("--pairs", nargs="+", default=["gm:l=0.0,a=1.0|mem0", "gm:l=0.0,a=1.0|faiss"],
                    help="A|B system specs")
    pr.add_argument("--judges", nargs="+", default=["qwen3_14b", "qwen3-32b-judge"])
    pr.add_argument("--out", default="systems")
    pr.add_argument("--convs", nargs="+", default=["conv-26", "conv-30"])
    f = sub.add_parser("final")
    f.add_argument("--judges", nargs="+", default=["qwen3_14b", "qwen3-32b-judge"])
    f.add_argument("--sample-systems", nargs="+", default=["faiss", "gm:l=0.0,a=1.0", "mem0"])
    f.add_argument("--sample-out", default="judge_sample_50.md")
    f.add_argument("--overwrite-sample", action="store_true")
    f.add_argument("--convs", nargs="+", default=["conv-26", "conv-30"])
    f.add_argument("--systems", nargs="+", default=None, help="LABEL=SPEC entries (default: dev systems)")
    args = ap.parse_args()
    global OUT
    OUT = Path(args.out_dir)
    logging.basicConfig(level=logging.WARNING)
    s = Settings()
    if args.cmd == "generate":
        if args.thinking is None:
            args.thinking = s.answer_thinking
        await cmd_generate(args, s)
    elif args.cmd == "judge":
        await cmd_judge(args, s)
    elif args.cmd == "paired":
        names = {spec: label for label, spec in FINAL_SYSTEMS}
        args.pairs_resolved = []
        for pair in args.pairs:
            a, b = pair.split("|")
            args.pairs_resolved.append((names.get(a, a), a, names.get(b, b), b))
        cmd_paired(args, s)
    elif args.cmd == "final":
        args.systems_resolved = ([tuple(x.split("=", 1)) for x in args.systems] if args.systems
                                 else FINAL_SYSTEMS)
        cmd_final(args, s)
    else:
        cmd_report(args, s)


if __name__ == "__main__":
    asyncio.run(main())
