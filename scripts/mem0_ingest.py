"""Ingest a LoCoMo conversation into Mem0 (OSS) with the same LLM and embedder as GraphMem.

Turns are added in batches of 2 messages (as in Mem0's own LoCoMo evaluation), each
prefixed with its session date and speaker; the simulated session date replaces Mem0's
wall-clock "current date". Mem0's add() is sequential by design (each call reads the
memories written by the previous one), so ingestion uses one LLM request at a time.

    .venv/Scripts/python scripts/mem0_ingest.py --conv conv-26 --fresh
"""

import argparse
import asyncio
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

from graphmem import Settings
from graphmem.bench import mem0_adapter as ma
from graphmem.llm import PreflightError, UsageTracker, make_llm
from graphmem.locomo import TestSetGuard, load_locomo, require_dev


async def preflight(llm, model: str) -> None:
    try:
        print("LLM:", await llm.preflight())
    except PreflightError as exc:
        sys.exit(f"Preflight failed: {exc}")
    be = llm.backend
    for m in await be.loaded_models():
        if m != model:
            print(f"unloading {m}")
            await be.unload(m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="conv-26")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--fresh", action="store_true", help="delete the existing Mem0 store first")
    ap.add_argument("--num-ctx", type=int, default=8192, help="server context window, for the overflow check")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()

    s = Settings()
    try:
        require_dev(args.conv, s.dev_conversations, args.allow_test)
    except TestSetGuard as exc:
        sys.exit(str(exc))
    tracker = UsageTracker()
    llm = make_llm(s, tracker)
    asyncio.run(preflight(llm, s.llm_model))
    ma.configure(llm)

    conv = next(c for c in load_locomo() if c.sample_id == args.conv)
    root = Path("cache/mem0")
    if args.fresh:
        shutil.rmtree(root / args.conv, ignore_errors=True)
    mem = ma.build_memory(args.conv, root)

    n_turns = len(conv.turns)
    add_s: list[float] = []
    added = 0
    t0 = time.perf_counter()
    done = 0
    for sess in conv.sessions:
        ma.set_session_date(sess.time.date())
        for i in range(0, len(sess.turns), args.batch):
            batch = sess.turns[i:i + args.batch]
            msgs = [{"role": "user", "name": t.speaker,
                     "content": f"[{sess.time:%d %B %Y}] {t.render()}"} for t in batch]
            meta = {"session_date": sess.time.date().isoformat(),
                    "turn_id": batch[0].dia_id, "turn_ids": ",".join(t.dia_id for t in batch)}
            ta = time.perf_counter()
            res = mem.add(msgs, user_id=args.conv, metadata=meta)
            add_s.append(time.perf_counter() - ta)
            added += sum(r.get("event") == "ADD" for r in res.get("results", []))
            done += len(batch)
            if done % 50 < args.batch or done == n_turns:
                print(f"  {done:>4}/{n_turns} turns  {time.perf_counter() - t0:7.1f}s  memories+={added}  "
                      f"max prompt {ma.prompt_stats()['max_prompt_tokens']} tok", flush=True)
    wall = time.perf_counter() - t0
    total = len(mem.get_all(filters={"user_id": args.conv}, top_k=100000)["results"])
    ps = ma.prompt_stats()
    u = tracker.summary().get("mem0", {})
    report = {
        "conv": args.conv, "turns": n_turns, "batch": args.batch, "add_calls": len(add_s),
        "wall_s": round(wall, 1), "add_p50_s": round(statistics.median(add_s), 2),
        "add_p95_s": round(sorted(add_s)[int(0.95 * len(add_s)) - 1], 2),
        "memories_added": added, "memories_total": total, "llm": llm.identity, "usage_mem0": u,
        "max_prompt_tokens": ps["max_prompt_tokens"], "server_num_ctx": args.num_ctx,
        "prompt_overflow_risk": ps["max_prompt_tokens"] >= args.num_ctx - 64,
    }
    print(json.dumps(report, indent=2))
    out = Path("results/dev_eval") / f"mem0_ingest_{args.conv}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["prompt_overflow_risk"]:
        print(f"WARNING: Mem0 prompts reached {ps['max_prompt_tokens']} tokens vs num_ctx {args.num_ctx}; "
              "the server may have truncated them.")


if __name__ == "__main__":
    main()
