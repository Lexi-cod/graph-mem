"""Phase 4: LoCoMo test-set benchmark (the 8 non-dev conversations), resumable.

Runs from the `phase4-frozen` git tag with the frozen config; refuses otherwise.
Stages (each skips completed work on restart):
  1. GraphMem ingestion (all test conversations in parallel; LLM concurrency 4).
  2. Mem0 ingestion (4 conversations at a time; per-add checkpoints).
  3. Answers, thinking on: GraphMem (frozen), Mem0 k=10, Mem0-matched, FAISS k=10,
     FAISS-matched.
  4. Judge passes: qwen3-32b-judge (14B evicted, 100% GPU verified), then qwen3:14b.
  5. Report: per-category table with 95% CIs (both judges), paired bootstrap of GraphMem
     vs each baseline, latency, ingestion tokens/time, prompt tokens; freeze check.

    .venv/Scripts/python scripts/phase4.py            # run / resume everything
    .venv/Scripts/python scripts/phase4.py --stage report
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from graphmem import GraphStore, Settings
from graphmem.bench import mem0_adapter as ma
from graphmem.bench.baselines import BASELINES
from graphmem.bench.mem0_ingest import ingest_conversation as mem0_ingest
from graphmem.freeze import check as freeze_check
from graphmem.ingest import ingest_many
from graphmem.llm import PreflightError, UsageTracker, make_llm
from graphmem.locomo import load_locomo, split

OUT = Path("results/phase4")
TAG = "phase4-frozen"
GM = "gm:l=0.0,a=1.0,prov=1"
STAGES = ["ingest-graphmem", "ingest-mem0", "answer", "judge", "report"]


def systems() -> list[tuple[str, str]]:
    b = BASELINES
    return [("GraphMem (frozen)", GM), (f"Mem0 (k{b['mem0_k']})", f"mem0:k={b['mem0_k']}"),
            (f"Mem0 token-matched (k{b['mem0_matched_k']})", f"mem0:k={b['mem0_matched_k']}"),
            (f"FAISS (k{b['faiss_k']})", f"faiss:k={b['faiss_k']}"),
            (f"FAISS token-matched (k{b['faiss_matched_k']})", f"faiss:k={b['faiss_matched_k']}")]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def guard(allow_dirty: bool) -> dict:
    problems = freeze_check()
    if problems:
        sys.exit("frozen config drifted:\n  " + "\n  ".join(problems))
    head, tag = git("rev-parse", "HEAD"), git("rev-parse", f"{TAG}^{{commit}}")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if (head != tag or dirty) and not allow_dirty:
        sys.exit(f"not running from a clean checkout of {TAG} (HEAD {head[:10]}, tag {tag[:10]}, "
                 f"dirty={bool(dirty)})")
    return {"commit": head, "tag": TAG}


def load_state() -> dict:
    p = OUT / "state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_state(st: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "state.json.tmp"
    tmp.write_text(json.dumps(st, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT / "state.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def stage_ingest_graphmem(s: Settings, convs, st: dict) -> None:
    done = st.setdefault("graphmem_ingest", {})
    todo = [c for c in convs if c.sample_id not in done]
    if not todo:
        log("GraphMem ingestion: all done")
        return
    tracker = UsageTracker()
    llm = make_llm(s, tracker)
    log(f"GraphMem ingestion: {[c.sample_id for c in todo]}")
    root = await GraphStore.connect(s)
    t0 = time.perf_counter()
    ings = await ingest_many(root, todo, llm, s,
                             progress=lambda cid, stage, i, n: log(f"  {cid} {stage} {i}/{n}") if i == n else None)
    wall = time.perf_counter() - t0
    usage = tracker.summary()
    for cid, ing in ings.items():
        done[cid] = {"stats": ing.stats.model_dump(exclude={"merge_log", "retire_log", "blocked_merges"}),
                     "graph": await ing.store.stats(), "turns": ing.stats.turns}
        logs = ing.dump_logs()
        d = OUT / "graphmem_logs" / cid
        d.mkdir(parents=True, exist_ok=True)
        for name, rows in logs.items():
            (d / f"{name}.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows),
                                             encoding="utf-8")
    st.setdefault("graphmem_ingest_runs", []).append(
        {"convs": [c.sample_id for c in todo], "wall_s": round(wall, 1), "usage": usage})
    save_state(st)
    await root.close()
    log(f"GraphMem ingestion done in {wall:.0f}s")


def stage_ingest_mem0(s: Settings, convs, st: dict) -> None:
    done = st.setdefault("mem0_ingest", {})
    todo = [c for c in convs if c.sample_id not in done]
    if not todo:
        log("Mem0 ingestion: all done")
        return
    tracker = UsageTracker()
    llm = make_llm(s, tracker)
    ma.configure(llm)
    log(f"Mem0 ingestion: {[c.sample_id for c in todo]} (4 at a time)")
    t0 = time.perf_counter()

    def one(conv):
        r = mem0_ingest(conv, OUT, progress=lambda cid, i, n: log(f"  mem0 {cid} {i}/{n} batches"))
        log(f"  mem0 {conv.sample_id} done: {r.get('memories_total')} memories, {r['wall_s']:.0f}s")
        return conv.sample_id, r

    with ThreadPoolExecutor(max_workers=4) as pool:
        for cid, r in pool.map(one, todo):
            done[cid] = {k: v for k, v in r.items() if k != "add_s"}
            save_state(st)
    wall = time.perf_counter() - t0
    st.setdefault("mem0_ingest_runs", []).append(
        {"convs": [c.sample_id for c in todo], "wall_s": round(wall, 1), "usage": tracker.summary(),
         "max_prompt_tokens": ma.prompt_stats()["max_prompt_tokens"]})
    save_state(st)
    log(f"Mem0 ingestion done in {wall:.0f}s; max prompt {ma.prompt_stats()['max_prompt_tokens']} tokens")


def dev_eval(*args: str) -> None:
    cmd = [sys.executable, "scripts/dev_eval.py", "--out-dir", str(OUT), *args]
    log("$ " + " ".join(cmd[1:]))
    subprocess.run(cmd, check=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    ap.add_argument("--allow-dirty", action="store_true", help="development only; never for the real run")
    args = ap.parse_args()
    s = Settings()
    if None in BASELINES.values():
        sys.exit("BASELINES matched k not set")
    st = load_state()
    st["provenance"] = guard(args.allow_dirty)
    save_state(st)
    _, test = split(load_locomo(), s.dev_conversations)
    ids = [c.sample_id for c in test]
    log(f"test conversations: {ids}")
    llm = make_llm(s)
    try:
        log("LLM: " + await llm.preflight())
    except PreflightError as exc:
        sys.exit(f"Preflight failed: {exc}")

    stages = STAGES if args.stage == "all" else [args.stage]
    if "ingest-graphmem" in stages:
        await stage_ingest_graphmem(s, test, st)
    if "ingest-mem0" in stages:
        await asyncio.to_thread(stage_ingest_mem0, s, test, st)
    if "answer" in stages:
        dev_eval("generate", "--run", "phase4", "--conv", *ids, "--allow-test", "--thinking",
                 "--systems", *[spec for _, spec in systems()])
    if "judge" in stages:
        dev_eval("judge", "--judge-model", "qwen3-32b-judge")
        dev_eval("judge", "--judge-model", s.llm_model)
    if "report" in stages:
        labels = [f"{label}={spec}" for label, spec in systems()]
        dev_eval("final", "--convs", *ids, "--systems", *labels, "--sample-systems",
                 *[spec for _, spec in systems()], "--sample-out", "judge_sample_50_test.md")
        dev_eval("paired", "--convs", *ids, "--out", "phase4", "--pairs",
                 *[f"{GM}|{spec}" for _, spec in systems()[1:]])
        problems = freeze_check()
        st["freeze_check_at_end"] = "OK" if not problems else problems
        save_state(st)
        log("freeze check at end: " + ("OK" if not problems else f"{len(problems)} DIFFERENCES"))


if __name__ == "__main__":
    asyncio.run(main())
