"""Phase 3 smoke test against the real stack: uvicorn + Neo4j + Ollama (qwen3:14b).

Starts the API server, sends 5 /agent/turn calls (three carry new facts and should trigger
the agent's mid-run write) and 3 /memory/query calls in a fresh `smoke-test` namespace,
and writes the transcript to results/phase3_smoke.md.

    .venv/Scripts/python scripts/phase3_smoke.py
"""

import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from graphmem import GraphStore, Settings

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
CID = "smoke-test"
T0 = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)

TURNS = [
    ("Caroline", "Melanie", "Hi Mel! Big news: I adopted a guinea pig named Oscar last weekend. He loves parsley."),
    ("Melanie", "Caroline", "Aww, congrats! How is Oscar settling in?"),
    ("Caroline", "Melanie", "He's great. I also signed up for a pottery class at the community center that meets every Tuesday."),
    ("Caroline", "Melanie", "One more thing: I moved from Boston to Seattle last month to start a new job as a counselor at a youth center."),
    ("Melanie", "Caroline", "Wow, lots of changes! So what's new with you overall?"),
]
QUERIES = [
    "What pet does Caroline have and what does it like?",
    "Where does Caroline live now?",
    "What class did Caroline sign up for and when does it meet?",
]


def block(obj) -> str:
    return "```json\n" + json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n```"


async def reset_namespace() -> None:
    root = await GraphStore.connect(Settings())
    st = await root.scoped(CID)
    await st.clear()
    await root.close()


async def main() -> None:
    s = Settings()
    await reset_namespace()
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "graphmem.api:app", "--port", str(PORT), "--log-level", "warning"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    md = ["# Phase 3 smoke test (real stack)\n",
          f"- API: `uvicorn graphmem.api:app` on port {PORT}; conversation `{CID}` (cleared first)",
          f"- Neo4j: {s.neo4j_uri}; LLM: `{s.llm_model}` at {s.llm_base_url} "
          f"(answer thinking {'on' if s.answer_thinking else 'off'}, extraction/reason thinking off)",
          f"- Frozen retrieval config: k={s.retrieval_k}, provenance={s.provenance}, "
          f"contradiction_mode={s.contradiction_mode}, prune_enabled={s.prune_enabled}",
          f"- Message timestamps are explicit (start {T0.isoformat()}), one minute apart.\n"]
    t_start = time.perf_counter()
    try:
        async with httpx.AsyncClient(base_url=BASE, timeout=600) as c:
            for _ in range(180):
                try:
                    if (await c.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
            else:
                raise RuntimeError("API did not become healthy")
            md.append(f"`GET /health` -> {(await c.get('/health')).json()}\n")

            md.append("## Agent turns (`POST /agent/turn`)\n")
            writes = 0
            for i, (spk, lst, msg) in enumerate(TURNS):
                req = {"conversation_id": CID, "speaker": spk, "listener": lst, "message": msg,
                       "time": (T0 + timedelta(minutes=i)).isoformat()}
                t0 = time.perf_counter()
                r = await c.post("/agent/turn", json=req)
                r.raise_for_status()
                body = r.json()
                writes += bool(body["written"])
                md.append(f"### Turn {i + 1}: {spk}\n> {msg}\n")
                md.append(f"- **remembered (mid-run write):** {body['remembered']} — {body['decision_reason']}")
                if body["written"]:
                    w = body["written"]
                    md.append(f"- **written facts:** {w['facts']}")
                    md.append(f"- write stats: created {w['facts_created']}, reinforced {w['facts_reinforced']}, "
                              f"retired {w['facts_retired']}, entities created {w['entities_created']}, "
                              f"merged {w['entities_merged']}; dropped {w['dropped']}")
                md.append(f"- retrieved facts used: {body['retrieved']}; prune: {body['pruned']}")
                md.append(f"- timings (ms): {body['timings_ms']}; round trip {1000 * (time.perf_counter() - t0):.0f} ms")
                md.append(f"- **answer:** {body['answer']}\n")
                print(f"turn {i + 1}: remembered={body['remembered']} answer={body['answer'][:80]!r}", flush=True)

            md.append("## Memory queries (`POST /memory/query`)\n")
            q_now = (T0 + timedelta(minutes=len(TURNS) + 1)).isoformat()
            for q in QUERIES:
                req = {"conversation_id": CID, "query": q, "k": 5, "now": q_now, "reinforce": False}
                r = await c.post("/memory/query", json=req)
                r.raise_for_status()
                body = r.json()
                md.append(f"### {q}\n")
                md.append(f"- db {body['db_ms']} ms, embed {body['embed_ms']} ms; {len(body['facts'])} facts, "
                          f"{len(body['turns'])} source turns")
                md.append("```text\n" + body["context"] + "\n```\n")
                print(f"query: {q!r}: {len(body['facts'])} facts", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()

    md.append("## Summary\n")
    md.append(f"- 5 agent turns ({writes} with a mid-run write), 3 queries; total wall time "
              f"{time.perf_counter() - t_start:.0f} s (includes server start-up and model warm-up).")
    out = Path("results/phase3_smoke.md")
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out}; mid-run writes: {writes}")


if __name__ == "__main__":
    asyncio.run(main())
