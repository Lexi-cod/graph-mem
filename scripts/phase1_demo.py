"""Phase 1 demo: schema, decay-aware retrieval, temporal validity, pruning, latency.

Runs on simulated time so decay effects are visible. WIPES the Neo4j database.

    docker compose up -d
    .venv/Scripts/python scripts/phase1_demo.py
"""

import asyncio
import random
import statistics
from datetime import datetime, timedelta, timezone

from graphmem import EdgeCreate, GraphStore, NodeCreate, NodeType, RelType, Settings

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
P, O, C, E = NodeType.PERSON, NodeType.ORGANIZATION, NodeType.CONCEPT, NodeType.EVENT


def day(n: float) -> datetime:
    return T0 + timedelta(days=n)


def show(res, title: str) -> None:
    print(f"\n### {title}\nquery: {res.query!r}   (db {res.db_ms:.1f} ms, embed {res.embed_ms:.1f} ms)")
    print(f"{'node':<28}{'type':<14}{'sim':>7}{'decay':>8}{'score':>8}{'acc':>5}")
    for h in res.hits:
        n = h.node
        print(f"{n.name[:27]:<28}{n.type:<14}{h.similarity:>7.3f}{h.decay:>8.3f}{h.score:>8.3f}{n.access_count:>5}")
        for f in h.facts[:3]:
            end = "now" if f.validity_end is None else f"{f.validity_end:%Y-%m-%d}"
            print(f"    - {f.fact}  [{f.validity_start:%Y-%m-%d} .. {end}]  score={f.score:.3f}")


async def main() -> None:
    s = Settings(decay_lambda=0.05, retrieval_alpha=0.7, retrieval_k=4, prune_threshold=0.1)
    store = await GraphStore.connect(s)
    await store.clear()
    print(f"lambda={s.decay_lambda}/day (half-life {0.693 / s.decay_lambda:.1f} d), alpha={s.retrieval_alpha}")

    # --- day 0: initial facts
    mel = await store.add_node(NodeCreate(type=P, name="Melanie", summary="painter, mother of two"), day(0))
    caro = await store.add_node(NodeCreate(type=P, name="Caroline", summary="counselor, Melanie's friend"), day(0))
    pottery = await store.add_node(NodeCreate(type=C, name="pottery", summary="ceramics hobby"), day(0))
    boston = await store.add_node(NodeCreate(type=C, name="Boston", summary="city in Massachusetts"), day(0))
    lgbtq = await store.add_node(NodeCreate(type=O, name="LGBTQ support group"), day(0))
    await store.add_edge(EdgeCreate(type=RelType.KNOWS, source_id=mel.id, target_id=caro.id,
                                    fact="Melanie and Caroline are close friends"), day(0))
    await store.add_edge(EdgeCreate(type=RelType.LIKES, source_id=mel.id, target_id=pottery.id,
                                    fact="Melanie enjoys pottery"), day(0))
    lives = await store.add_edge(EdgeCreate(type=RelType.LIVES_IN, source_id=caro.id, target_id=boston.id), day(0))
    await store.add_edge(EdgeCreate(type=RelType.MEMBER_OF, source_id=caro.id, target_id=lgbtq.id,
                                    fact="Caroline attends an LGBTQ support group"), day(0))

    # --- day 40: a newer, equally-relevant hobby node and a camping event
    painting = await store.add_node(NodeCreate(type=C, name="painting", summary="art hobby, landscapes"), day(40))
    camp = await store.add_node(NodeCreate(type=E, name="family camping trip", summary="camping in the mountains"), day(40))
    await store.add_edge(EdgeCreate(type=RelType.LIKES, source_id=mel.id, target_id=painting.id,
                                    fact="Melanie loves painting landscapes"), day(40))
    await store.add_edge(EdgeCreate(type=RelType.PARTICIPATED_IN, source_id=mel.id, target_id=camp.id,
                                    fact="Melanie went camping with her family"), day(40))

    print("\nstats:", await store.stats())

    # 1) Decay-aware rerank: same query, pure similarity vs blended
    q = "What hobbies does Melanie have?"
    show(await store.retrieve(q, alpha=1.0, now=day(45), reinforce=False), "alpha=1.0 (pure vector similarity)")
    show(await store.retrieve(q, now=day(45), reinforce=False), "alpha=0.7 (similarity + decay)")

    # 2) Access reinforcement: query pottery repeatedly, decay recovers
    for i in range(5):
        await store.retrieve("pottery ceramics", k=1, now=day(46 + i))
    show(await store.retrieve(q, now=day(52), reinforce=False), "after 5 accesses of 'pottery' (days 46-50)")

    # 3) Temporal validity: Caroline moves (manual invalidation; Phase 2 automates this)
    sweden = await store.add_node(NodeCreate(type=C, name="Sweden", summary="country in Scandinavia"), day(60))
    await store.invalidate_edge(lives.id, RelType.LIVES_IN, at=day(60))
    await store.add_edge(EdgeCreate(type=RelType.LIVES_IN, source_id=caro.id, target_id=sweden.id,
                                    fact="Caroline moved to Sweden", validity_start=day(60)), day(60))
    q2 = "Where does Caroline live?"
    show(await store.retrieve(q2, k=2, now=day(61), reinforce=False), "current facts only (default)")
    show(await store.retrieve(q2, k=2, as_of=day(30), now=day(61), reinforce=False), "as_of=2024-01-31")

    # 4) Prune at day 120
    rep = await store.prune(dry_run=True, now=day(120))
    print(f"\n### prune dry-run @ day 120, threshold={rep.threshold}")
    for n in rep.archived_nodes:
        print(f"  node  {n['name']:<24} decay={n['decay']:.3f}")
    for e in rep.archived_edges:
        print(f"  edge  {e['fact']:<44} decay={e['decay']:.3f}")
    rep = await store.prune(now=day(120))
    print("after prune:", await store.stats())

    # 5) Latency on a larger synthetic graph
    rng = random.Random(0)
    words = ("music cooking travel chess hiking yoga guitar soccer reading photography "
             "gardening swimming baking dancing coding volunteering surfing poetry").split()
    people = []
    for i in range(300):
        people.append(await store.add_node(
            NodeCreate(type=P, name=f"Person {i}", summary=f"likes {rng.choice(words)} and {rng.choice(words)}"),
            day(rng.uniform(0, 120))))
    topics = {w: await store.add_node(NodeCreate(type=C, name=w), day(0)) for w in words}
    for p in people:
        for w in rng.sample(words, 3):
            await store.add_edge(EdgeCreate(type=RelType.INTERESTED_IN, source_id=p.id,
                                            target_id=topics[w].id), day(rng.uniform(0, 120)))
    print("\nlatency graph:", await store.stats())
    db, e2e = [], []
    for i in range(200):
        t = asyncio.get_running_loop().time()
        r = await store.retrieve(f"who is into {rng.choice(words)}?", k=5, now=day(121 + i / 100))
        e2e.append((asyncio.get_running_loop().time() - t) * 1000)
        db.append(r.db_ms)
    for name, xs in (("db call", db), ("end-to-end (embed+db)", e2e)):
        xs = sorted(xs)
        print(f"  {name:<24} p50={statistics.median(xs):6.2f} ms  p95={xs[int(0.95 * len(xs)) - 1]:6.2f} ms")

    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
