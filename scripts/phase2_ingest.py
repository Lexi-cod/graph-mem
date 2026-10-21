"""Phase 2: ingest LoCoMo dev conversations into GraphMem on simulated time.

Each conversation goes into its own namespace; several conversations are ingested
concurrently with a shared LLM concurrency cap (GRAPHMEM_LLM_MAX_CONCURRENCY).
Clears only the namespaces being ingested. Re-running is free: every LLM call and
extraction is cached. Refuses test conversations unless --allow-test.

    .venv/Scripts/python scripts/phase2_ingest.py --conv conv-26
    .venv/Scripts/python scripts/phase2_ingest.py --conv conv-26 conv-30 --mode rules
"""

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

from graphmem import GraphStore, Settings
from graphmem.extraction import SYSTEM_PROMPT
from graphmem.ingest import ingest_many, turn_inputs
from graphmem.llm import PreflightError, make_judge, make_llm
from graphmem.locomo import TestSetGuard, load_locomo, require_dev

FULL_LOCOMO_TURNS = 5882


def usage_table(rows: dict[str, dict]) -> None:
    cols = [("calls", "calls"), ("cache_hits", "cached"), ("retries", "retry"),
            ("input_tokens", "tok_in"), ("output_tokens", "tok_out"),
            ("mean_output_tokens", "mean_out"), ("max_output_tokens", "max_out"),
            ("wall_s", "wall_s"), ("request_s", "req_s"),
            ("out_tok_per_s_stream", "tok/s/strm"), ("out_tok_per_s_aggregate", "tok/s/agg"),
            ("reasoning_responses", "thinking")]
    print(f"  {'purpose':<14}" + "".join(f"{h:>11}" for _, h in cols))
    for name, r in rows.items():
        print(f"  {name:<14}" + "".join(f"{str(r[c]):>11}" for c, _ in cols))


def dump_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows),
                    encoding="utf-8")


def fmt_rel(r) -> str:
    tgt = f'"{r.target}"' if r.relation == "HAS_ATTRIBUTE" else r.target
    slot = f" [slot={r.slot}]" if r.slot else ""
    end = " [ENDED]" if r.ended else ""
    return f"{r.source} -{r.relation}-> {tgt}{slot}{end}\n        fact: {r.fact}\n        evidence: \"{r.evidence}\""


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", nargs="+", default=["conv-26"])
    ap.add_argument("--sessions", type=int, default=None, help="only the first N sessions")
    ap.add_argument("--mode", choices=["llm", "rules", "off"], default=None,
                    help="contradiction mode (default: settings)")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    s = Settings()
    if args.mode:
        s = s.model_copy(update={"contradiction_mode": args.mode})
    try:
        for cid in args.conv:
            require_dev(cid, s.dev_conversations, args.allow_test)
    except TestSetGuard as exc:
        sys.exit(str(exc))

    llm = make_llm(s)
    try:
        print("LLM:", await llm.preflight())
    except PreflightError as exc:
        sys.exit(f"Preflight failed: {exc}")
    _, judge = make_judge(s)
    print(f"judge (for later phases): {judge['model']} @ {judge['base_url']}")
    print(f"contradiction_mode={s.contradiction_mode}  llm_concurrency={s.llm_max_concurrency}")

    convs = [c for c in load_locomo() if c.sample_id in args.conv]
    root = await GraphStore.connect(s)
    n_turns_all = 0
    for c in convs:
        sess = c.sessions[: args.sessions]
        n = sum(len(x.turns) for x in sess)
        n_turns_all += n
        print(f"{c.sample_id}: {c.speaker_a} & {c.speaker_b}, {len(sess)} sessions, {n} turns, "
              f"{sess[0].time:%Y-%m-%d} .. {sess[-1].time:%Y-%m-%d}")

    t0 = time.perf_counter()
    stage_t: dict[tuple[str, str], float] = {}

    def progress(cid, stage, i, n):
        stage_t[(cid, stage)] = time.perf_counter() - t0
        print(f"  {cid:<8}{stage:<8}{i:>5}/{n}  {stage_t[(cid, stage)]:7.1f}s", flush=True)

    ings = await ingest_many(root, convs, llm, s, sessions=args.sessions, progress=progress)
    wall = time.perf_counter() - t0
    usage = llm.tracker.summary()

    print(f"\n### timing\n  total wall {wall:.1f}s for {n_turns_all} turns "
          f"({len(convs)} conversation(s) in parallel)")
    for c in convs:
        ex_t = stage_t.get((c.sample_id, "extract"), 0.0)
        print(f"  {c.sample_id}: extract done at {ex_t:.1f}s, apply done at "
              f"{stage_t.get((c.sample_id, 'apply'), 0.0):.1f}s (apply {ings[c.sample_id].stats.apply_s:.1f}s)")
    rate = wall / n_turns_all
    print(f"  {rate:.2f} s/turn -> extrapolated full LoCoMo ({FULL_LOCOMO_TURNS} turns) at this "
          f"rate: {rate * FULL_LOCOMO_TURNS / 3600:.2f} h  (single-conversation rate; parallel "
          f"runs overlap the sequential apply stage, see note)")

    print("\n### LLM usage by purpose (thinking = responses carrying reasoning content; must be 0)")
    usage_table(usage)

    rng = random.Random(args.seed)
    out_root = Path("results")
    for c in convs:
        ing = ings[c.sample_id]
        st, ex, store = ing.stats, ing.extractor, ing.store
        validity = ex.validity()
        graph = await store.stats()
        counts = await store.counts_by_type()
        print(f"\n================ {c.sample_id} ================")
        print("### extraction JSON validity")
        print("  " + "  ".join(f"{k}={v}" for k, v in validity.items()))
        if validity["repair_rate"] > 0.05:
            print(f"  FLAG: repair rate {validity['repair_rate']:.1%} exceeds 5%")
        con = usage.get("contradiction", {})
        if con:
            print(f"  contradiction JSON: ok={con['json_ok']} repaired={con['json_repaired']} "
                  f"failed={con['json_failed']}")
        print("### facts dropped by post-checks (reason: count)")
        print("  " + (", ".join(f"{k}: {v}" for k, v in st.facts_dropped.items()) or "none"))
        print(f"  relative times rewritten (facts + entity names): {ex.time_rewrites()}")
        conv_ = ex.converted()
        print(f"  unknown relation types converted to HAS_ATTRIBUTE: {len(conv_)} "
              f"{dict(Counter(c['original_type'] for c in conv_))}")
        fz = ex.fuzzy_accepted()
        print(f"  evidence accepted by fuzzy match (not exact): {len(fz)}")
        for z in fz[:8]:
            print(f"      [{z['turn']}] cur={z['current']} ctx={z['best_context']} {z['evidence'][:110]!r}")
        print("### ingestion stats")
        for k, v in st.model_dump(exclude={"merge_log", "retire_log", "blocked_merges",
                                           "facts_dropped"}).items():
            print(f"  {k:<22}{v}")
        print(f"### graph\n  {graph}")
        print(f"  entities by type: {counts['entities']}")
        print(f"  facts by relation type: {counts['facts']}")
        print(f"  RELATED_TO facts: {counts['facts'].get('RELATED_TO', 0)}   "
              f"LLM contradiction calls: {st.llm_checks}  (pairs {st.llm_pairs}, labels {st.llm_labels})")

        # ---------------------------------------------------------- samples
        items = turn_inputs(c, s.extraction_context_turns, args.sessions)
        print("\n### 20 random extractions (kept facts, then dropped facts with reasons)")
        for ti, _ in rng.sample(items, min(20, len(items))):
            rec = ex.records[ti.dia_id]
            flags = (" [REPAIRED]" if rec.repaired else "") + (" [FAILED]" if rec.failed else "")
            print(f"\n[{ti.dia_id}]{flags} {ti.text}")
            if rec.extraction.entities:
                print("  entities: " + "; ".join(f"{e.name} ({e.type.value})" for e in rec.extraction.entities))
            for r in rec.extraction.relations:
                print("  KEEP " + fmt_rel(r))
            for d in rec.dropped:
                r = d.relation
                print(f"  DROP[{d.reason}] {r['source']} -{r['relation']}-> {r['target']}: {r['fact']}"
                      f"\n        evidence: \"{r['evidence']}\"")
            if not rec.extraction.relations and not rec.dropped:
                print("  (nothing extracted)")

        logs = ing.dump_logs()
        retires = logs["retire_log"]
        retired = [r for r in retires if r["action"] == "retire"]
        print(f"\n### retirements: {len(retired)} retired (by reason "
              f"{dict(Counter(r['reason'] for r in retired))}); {len(retires) - len(retired)} "
              f"restatements reinforced. Showing up to 20 random retirements:")
        for r in rng.sample(retired, min(20, len(retired))):
            d = f" ({r['detail']})" if r["detail"] else ""
            print(f"  {r['turn']:<7} {r['reason']}{d} score={r['score']}\n"
                  f"      old: {r['old_fact']}\n      new: {r['new_fact']}")

        merges = logs["merge_log"]
        emb = [m for m in merges if m["method"] == "embedding"]
        renamed = [m for m in merges if m["method"] == "exact" and m["incoming"] != m["merged_into"]]
        rest = [m for m in merges if m["method"] == "exact" and m["incoming"] == m["merged_into"]]
        pick = rng.sample(emb, min(10, len(emb)))
        for pool in (renamed, rest):
            pick += rng.sample(pool, min(10 - len(pick), len(pool)))
        print(f"\n### ALL {len(emb)} embedding merges (for precision labelling)")
        for m in emb:
            print(f"  {m['turn']:<7} {m['incoming']!r} -> {m['merged_into']!r} [{m['type']}] score={m['score']}")
        print(f"\n### 10 merges (of {len(merges)}: {len(emb)} embedding, {len(renamed)} exact with a "
              f"different surface form, {len(rest)} identical); embedding merges first")
        for m in pick:
            print(f"  {m['turn']:<7} {m['incoming']!r} -> {m['merged_into']!r} [{m['type']}] "
                  f"{m['method']} score={m['score']}")
        print(f"\n### blocked Event merges: {len(logs['blocked_merges'])}")
        for b in logs["blocked_merges"][:10]:
            print(f"  {b['turn']:<7} {b['incoming']!r} x {b['candidate']!r} cos={b['cosine']} ({b['reason']})")

        out = out_root / f"phase2_{c.sample_id}"
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in logs.items():
            dump_jsonl(out / f"{name}.jsonl", rows)
        dump_jsonl(out / "extractions.jsonl", [
            {"text": ti.text, **ex.records[ti.dia_id].model_dump(mode="json")} for ti, _ in items])
        (out / "summary.json").write_text(json.dumps({
            "conv": c.sample_id, "turns": len(items), "llm": llm.identity, "judge": judge,
            "contradiction_mode": s.contradiction_mode, "wall_s": wall,
            "validity": validity, "stats": st.model_dump(exclude={"merge_log", "retire_log", "blocked_merges"}),
            "graph": graph, "counts": counts, "usage": usage,
            "extraction_system_prompt_chars": len(SYSTEM_PROMPT),
            "settings": s.model_dump(exclude={"llm_api_key"}),
        }, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {out}/ (summary.json, extractions/merge_log/blocked_merges/retire_log .jsonl)")
    await root.close()


if __name__ == "__main__":
    asyncio.run(main())
