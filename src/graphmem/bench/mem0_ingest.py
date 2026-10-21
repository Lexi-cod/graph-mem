"""Resumable Mem0 ingestion (used by Phase 4).

Turns are added in batches of 2 messages, each prefixed with its session date and speaker;
the thread-local simulated session date replaces Mem0's wall-clock "current date". A
checkpoint file records the next batch after every add(), so a restart continues where it
stopped (the Mem0 store on disk already holds the earlier batches). Without a checkpoint,
an existing store for the conversation is wiped and ingestion starts fresh.
"""

from __future__ import annotations

import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

from ..locomo import Conversation
from . import mem0_adapter as ma


def _batches(conv: Conversation, size: int):
    for sess in conv.sessions:
        for i in range(0, len(sess.turns), size):
            yield sess, sess.turns[i:i + size]


def ingest_conversation(conv: Conversation, state_dir: Path, store_root: Path = Path("cache/mem0"),
                        batch: int = 2, progress=None) -> dict[str, Any]:
    cid = conv.sample_id
    ckpt = state_dir / f"mem0_ckpt_{cid}.json"
    if ckpt.exists():
        st = json.loads(ckpt.read_text(encoding="utf-8"))
    else:
        shutil.rmtree(store_root / cid, ignore_errors=True)
        st = {"next": 0, "add_s": [], "added": 0, "wall_s": 0.0, "done": False}
    if st["done"]:
        return st
    mem = ma.build_memory(cid, store_root)
    batches = list(_batches(conv, batch))
    t_resume = time.perf_counter()
    wall0 = st["wall_s"]
    for idx in range(st["next"], len(batches)):
        sess, turns = batches[idx]
        ma.set_session_date(sess.time.date())
        msgs = [{"role": "user", "name": t.speaker, "content": f"[{sess.time:%d %B %Y}] {t.render()}"}
                for t in turns]
        meta = {"session_date": sess.time.date().isoformat(), "turn_id": turns[0].dia_id,
                "turn_ids": ",".join(t.dia_id for t in turns)}
        ta = time.perf_counter()
        res = mem.add(msgs, user_id=cid, metadata=meta)
        st["add_s"].append(round(time.perf_counter() - ta, 3))
        st["added"] += sum(r.get("event") == "ADD" for r in res.get("results", []))
        st["next"] = idx + 1
        st["wall_s"] = wall0 + time.perf_counter() - t_resume
        tmp = ckpt.with_suffix(".tmp")
        tmp.write_text(json.dumps(st), encoding="utf-8")
        tmp.replace(ckpt)
        if progress and (idx + 1) % 25 == 0:
            progress(cid, idx + 1, len(batches))
    st["memories_total"] = len(mem.get_all(filters={"user_id": cid}, top_k=100000)["results"])
    st["add_p50_s"] = statistics.median(st["add_s"]) if st["add_s"] else None
    st["done"] = True
    ckpt.write_text(json.dumps(st), encoding="utf-8")
    return st
