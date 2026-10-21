"""Async Neo4j-backed temporal fact store, scoped to one namespace (conversation).

Graph model:
    (:Entity:<Type>:NS_x {name, name_key, aliases, name_embedding, ...})
    (:Fact:NS_x {text, embedding, relation_type, validity_start, validity_end, evidence,
                 turn_id, conversation_id, weight, access_count, last_accessed, ...})
    (:Fact)-[:SUBJECT]->(:Entity), and (:Fact)-[:OBJECT]->(:Entity) when there is an object.

Facts are the unit of memory: decay, reinforcement, supersession (validity_end) and
pruning (:Archived) all operate on Fact nodes. Entities exist for resolution and to
link facts (1-hop expansion at retrieval).

Each namespace has its own vector indexes (fact text, entity name). Neo4j 5.26
vector indexes cannot pre-filter, so scoping the index by namespace label is what
makes the conversation filter exact without losing recall to other conversations.
Retrieval still over-fetches k * fact_overfetch candidates (for reranking) and
checks conversation_id explicitly.

Scores: similarity is raw cosine clamped to [0, 1]; decay is capped to [0, 1];
score = alpha*similarity + (1-alpha)*decay - history_penalty (if superseded).
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncManagedTransaction

from .clock import Clock, WallClock
from .config import Settings
from .decay import cypher_cap, cypher_cosine, cypher_decay
from .embedder import Embedder
from .schema import (
    Fact,
    FactCreate,
    Node,
    NodeCreate,
    NodeType,
    PruneReport,
    RelType,
    RetrievalResult,
    RetrievedFact,
    SourceTurn,
    check_relation,
    name_key,
    new_id,
)

DEFAULT_NAMESPACE = "default"

_NODE_PROJ = "{.*, name_embedding: null}"
_FACT_PROJ = "{.*, embedding: null}"


def ns_label(namespace: str) -> str:
    return "NS_" + re.sub(r"\W", "_", namespace)


def _valid(var: str, as_of: bool) -> str:
    if as_of:
        return (
            f"({var}.validity_start <= $as_of"
            f" AND ({var}.validity_end IS NULL OR {var}.validity_end > $as_of))"
        )
    return f"{var}.validity_end IS NULL"


# Recency refresh for a write: never move last_accessed backwards.
def _refresh(var: str) -> str:
    return (
        f"{var}.last_accessed = CASE WHEN {var}.last_accessed < $now "
        f"THEN $now ELSE {var}.last_accessed END"
    )


def _fact_rows(rows: list[dict[str, Any]]) -> list[Fact]:
    return [Fact(**r["f"], subject_id=r["sid"], object_id=r.get("oid")) for r in rows]


class GraphStore:
    def __init__(
        self,
        driver: AsyncDriver,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        clock: Clock | None = None,
        namespace: str = DEFAULT_NAMESPACE,
        owns_driver: bool = True,
    ):
        self.driver = driver
        self.s = settings or Settings()
        self.embedder = embedder or Embedder(self.s.embedding_model)
        self.clock = clock or WallClock()
        self.namespace = namespace
        self.ns = ns_label(namespace)
        self.E = f"Entity:{self.ns}"  # entity label expression for this namespace
        self.F = f"Fact:{self.ns}"  # fact label expression for this namespace
        self.fact_index = f"fact_embedding_{self.ns}"
        self.name_index = f"entity_name_embedding_{self.ns}"
        self._owns_driver = owns_driver

    # ------------------------------------------------------------ lifecycle

    @classmethod
    async def connect(
        cls,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        clock: Clock | None = None,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> GraphStore:
        s = settings or Settings()
        driver = AsyncGraphDatabase.driver(
            s.neo4j_uri,
            auth=(s.neo4j_user, s.neo4j_password),
            notifications_min_severity="OFF",
        )
        await driver.verify_connectivity()
        store = cls(driver, s, embedder, clock, namespace)
        await store.init_schema()
        await store.init_namespace()
        return store

    async def scoped(self, namespace: str, clock: Clock | None = None) -> GraphStore:
        """A store for another namespace sharing this driver, settings and embedder."""
        st = GraphStore(self.driver, self.s, self.embedder, clock or self.clock,
                        namespace, owns_driver=False)
        await st.init_namespace()
        return st

    async def close(self) -> None:
        if self._owns_driver:
            await self.driver.close()

    async def __aenter__(self) -> GraphStore:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    def _now(self, now: datetime | None) -> datetime:
        return now or self.clock.now()

    async def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        records, _, _ = await self.driver.execute_query(
            query, params, database_=self.s.neo4j_database
        )
        return [r.data() for r in records]

    async def init_schema(self) -> None:
        """Idempotent, global: constraints and property indexes."""
        for q in (
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT turn_id IF NOT EXISTS FOR (t:Turn) REQUIRE t.id IS UNIQUE",
            "CREATE INDEX entity_name_key IF NOT EXISTS FOR (n:Entity) ON (n.name_key)",
            "CREATE INDEX fact_relation_type IF NOT EXISTS FOR (f:Fact) ON (f.relation_type)",
        ):
            await self._run(q)

    async def init_namespace(self) -> None:
        """Idempotent: this namespace's vector indexes (facts by text, entities by name)."""
        # Pre-fact-node versions indexed entity text on the same (label, property); an
        # equivalent index makes CREATE ... IF NOT EXISTS a silent no-op, so drop it.
        await self._run(f"DROP INDEX entity_embedding_{self.ns} IF EXISTS")
        for index, prop in ((self.fact_index, "embedding"), (self.name_index, "name_embedding")):
            await self._run(
                f"""CREATE VECTOR INDEX {index} IF NOT EXISTS
                    FOR (n:{self.ns}) ON n.{prop}
                    OPTIONS {{indexConfig: {{
                      `vector.dimensions`: {self.s.embedding_dim},
                      `vector.similarity_function`: 'cosine'}}}}"""
            )
        await self._run("CALL db.awaitIndexes(60)")

    async def clear(self) -> None:
        """Delete this namespace's data (keeps indexes)."""
        await self._batched_delete(f"MATCH (n:{self.ns})")

    async def clear_all(self) -> None:
        """Delete every namespace's data."""
        await self._batched_delete("MATCH (n)")

    async def _batched_delete(self, match: str) -> None:
        # IN TRANSACTIONS requires an auto-commit (implicit) transaction, i.e. session.run.
        async with self.driver.session(database=self.s.neo4j_database) as session:
            res = await session.run(
                f"{match} CALL (n) {{ DETACH DELETE n }} IN TRANSACTIONS OF 5000 ROWS"
            )
            await res.consume()

    # ------------------------------------------------------------ entities

    async def add_node(
        self,
        node: NodeCreate,
        now: datetime | None = None,
        *,
        name_embedding: list[float] | None = None,
    ) -> Node:
        now = self._now(now)
        if name_embedding is None:
            name_embedding = await self.embedder.aembed_one(node.name)
        props = {
            "id": node.id or new_id(),
            "conversation_id": self.namespace,
            "name": node.name,
            "summary": node.summary,
            "aliases": node.aliases,
            "name_key": name_key(node.name),
            "alias_keys": sorted({name_key(a) for a in node.aliases} - {name_key(node.name)}),
            "weight": node.weight,
            "created_at": now,
            "last_accessed": now,
            "access_count": 0,
            "validity_start": node.validity_start or now,
            "validity_end": None,
            "name_embedding": name_embedding,
        }
        label = NodeType(node.type).value
        rows = await self._run(
            f"CREATE (n:{self.E}:{label}) SET n = $props "
            f"RETURN n {_NODE_PROJ} AS n, labels(n) AS labels",
            props=props,
        )
        return Node.from_neo4j(rows[0]["n"], rows[0]["labels"])

    async def merge_entity(
        self,
        node_id: str,
        alias: str | None = None,
        summary: str | None = None,
        now: datetime | None = None,
    ) -> Node:
        """Fold a new mention into an existing entity: record the alias, fill an empty
        summary, and refresh its recency."""
        rows = await self._run(
            f"""
            MATCH (n:{self.E} {{id: $id}})
            SET {_refresh('n')},
                n.aliases = CASE WHEN $alias IS NULL OR $alias = n.name OR $alias IN n.aliases
                            THEN n.aliases ELSE n.aliases + $alias END,
                n.alias_keys = CASE WHEN $akey IS NULL OR $akey = n.name_key
                                    OR $akey IN coalesce(n.alias_keys, [])
                               THEN coalesce(n.alias_keys, []) ELSE coalesce(n.alias_keys, []) + $akey END,
                n.summary = CASE WHEN coalesce(n.summary, '') = '' THEN coalesce($summary, '')
                            ELSE n.summary END
            RETURN n {_NODE_PROJ} AS n, labels(n) AS labels
            """,
            id=node_id,
            alias=alias,
            akey=name_key(alias) if alias else None,
            summary=summary,
            now=self._now(now),
        )
        if not rows:
            raise KeyError(node_id)
        return Node.from_neo4j(rows[0]["n"], rows[0]["labels"])

    async def get_node(self, node_id: str) -> Node | None:
        rows = await self._run(
            f"MATCH (n:{self.E} {{id: $id}}) RETURN n {_NODE_PROJ} AS n, labels(n) AS labels",
            id=node_id,
        )
        return Node.from_neo4j(rows[0]["n"], rows[0]["labels"]) if rows else None

    async def find_by_name(self, name: str, node_type: NodeType) -> Node | None:
        """Same-type entity whose normalized name or alias equals normalize(name).
        A primary-name match wins over an alias match; then the oldest node."""
        key = name_key(name)
        if not key:
            return None
        label = NodeType(node_type).value
        rows = await self._run(
            f"""
            CALL () {{
              MATCH (n:{self.E}:{label} {{name_key: $k}}) RETURN n, 0 AS pri
              UNION
              MATCH (n:{self.E}:{label}) WHERE $k IN n.alias_keys RETURN n, 1 AS pri
            }}
            WITH n, pri WHERE n.validity_end IS NULL
            RETURN n {_NODE_PROJ} AS n, labels(n) AS labels
            ORDER BY pri, n.created_at LIMIT 1
            """,
            k=key,
        )
        return Node.from_neo4j(rows[0]["n"], rows[0]["labels"]) if rows else None

    async def similar_entities(
        self, name_vec: list[float], node_type: NodeType, min_cosine: float, limit: int = 5
    ) -> list[tuple[Node, float]]:
        """Same-type entities in this namespace whose name embedding has cosine >= min_cosine."""
        rows = await self._run(
            f"""
            CALL db.index.vector.queryNodes($index, $pool, $vec) YIELD node AS n, score
            WITH n, {cypher_cosine('score')} AS cos
            WHERE $label IN labels(n) AND cos >= $min AND n.validity_end IS NULL
            RETURN n {_NODE_PROJ} AS n, labels(n) AS labels, cos
            ORDER BY cos DESC LIMIT $limit
            """,
            index=self.name_index,
            pool=limit * 4,
            vec=name_vec,
            label=NodeType(node_type).value,
            min=min_cosine,
            limit=limit,
        )
        return [(Node.from_neo4j(r["n"], r["labels"]), r["cos"]) for r in rows]

    # ------------------------------------------------------------ source turns (provenance)

    async def add_turns(self, turns: list[dict[str, Any]]) -> int:
        """Upsert conversation turns: dicts with dia_id, text (rendered), session_time."""
        rows = await self._run(
            f"""
            UNWIND $turns AS t
            MERGE (n:Turn:{self.ns} {{id: $ns + ':' + t.dia_id}})
            SET n.dia_id = t.dia_id, n.text = t.text, n.session_time = t.session_time,
                n.conversation_id = $ns
            RETURN count(n) AS n
            """,
            turns=turns,
            ns=self.namespace,
        )
        return rows[0]["n"] if rows else 0

    # ------------------------------------------------------------ facts: writes

    async def add_fact(
        self, fact: FactCreate, now: datetime | None = None, *, embedding: list[float] | None = None
    ) -> Fact:
        """Create a Fact node linked to its subject (and object) entities."""
        now = self._now(now)
        ends = await self._run(
            f"MATCH (s:{self.E} {{id: $sid}}) OPTIONAL MATCH (o:{self.E} {{id: $oid}}) "
            "RETURN labels(s) AS s_labels, labels(o) AS o_labels",
            sid=fact.subject_id,
            oid=fact.object_id,
        )
        if not ends or (fact.object_id and ends[0]["o_labels"] is None):
            raise KeyError(f"missing entity in {self.namespace}: {fact.subject_id} / {fact.object_id}")
        o_t = _type_of(ends[0]["o_labels"]) if fact.object_id else None
        check_relation(fact.relation_type, _type_of(ends[0]["s_labels"]), o_t,
                       self_loop=fact.object_id is None)
        if embedding is None:
            embedding = await self.embedder.aembed_one(fact.text)
        props = {
            "id": fact.id or new_id(),
            "conversation_id": self.namespace,
            "relation_type": RelType(fact.relation_type).value,
            "text": fact.text,
            "evidence": fact.evidence,
            "turn_id": fact.turn_id,
            "slot": fact.slot,
            "value": fact.value,
            "weight": fact.weight,
            "created_at": now,
            "last_accessed": now,
            "access_count": 0,
            "validity_start": fact.validity_start or now,
            "validity_end": None,
            "embedding": embedding,
        }
        rows = await self._run(
            f"""
            MATCH (s:{self.E} {{id: $sid}})
            OPTIONAL MATCH (o:{self.E} {{id: $oid}})
            CREATE (f:{self.F}) SET f = $props
            CREATE (f)-[:SUBJECT]->(s)
            FOREACH (x IN CASE WHEN o IS NULL THEN [] ELSE [o] END | CREATE (f)-[:OBJECT]->(x))
            FOREACH (x IN [s] + CASE WHEN o IS NULL THEN [] ELSE [o] END | SET {_refresh('x')})
            RETURN f {_FACT_PROJ} AS f, s.id AS sid, o.id AS oid
            """,
            sid=fact.subject_id,
            oid=fact.object_id,
            props=props,
            now=now,
        )
        return _fact_rows(rows)[0]

    async def reinforce_fact(self, fact_id: str, now: datetime | None = None) -> Fact | None:
        """A restated fact: refresh its recency (and revive it if archived)."""
        rows = await self._run(
            f"""
            MATCH (f:{self.F} {{id: $id}})-[:SUBJECT]->(s)
            OPTIONAL MATCH (f)-[:OBJECT]->(o)
            REMOVE f:Archived
            SET {_refresh('f')}
            FOREACH (x IN [s] + CASE WHEN o IS NULL THEN [] ELSE [o] END | SET {_refresh('x')})
            RETURN f {_FACT_PROJ} AS f, s.id AS sid, o.id AS oid
            """,
            id=fact_id,
            now=self._now(now),
        )
        return _fact_rows(rows)[0] if rows else None

    async def invalidate_fact(self, fact_id: str, at: datetime | None = None) -> Fact | None:
        """Close a fact's validity interval (it stays in the graph as history)."""
        rows = await self._run(
            f"""
            MATCH (f:{self.F} {{id: $id}})-[:SUBJECT]->(s)
            WHERE f.validity_end IS NULL
            OPTIONAL MATCH (f)-[:OBJECT]->(o)
            SET f.validity_end = $at
            RETURN f {_FACT_PROJ} AS f, s.id AS sid, o.id AS oid
            """,
            id=fact_id,
            at=self._now(at),
        )
        return _fact_rows(rows)[0] if rows else None

    async def delete_fact(self, fact_id: str) -> bool:
        """Hard delete. Only for a fact created moments ago that turned out to be a
        duplicate; real retirements use invalidate_fact so history is kept."""
        rows = await self._run(
            f"MATCH (f:{self.F} {{id: $id}}) DETACH DELETE f RETURN 1 AS ok", id=fact_id)
        return bool(rows)

    # ------------------------------------------------------------ facts: reads

    async def get_fact(self, fact_id: str) -> Fact | None:
        rows = await self._run(
            f"MATCH (f:{self.F} {{id: $id}})-[:SUBJECT]->(s) OPTIONAL MATCH (f)-[:OBJECT]->(o) "
            f"RETURN f {_FACT_PROJ} AS f, s.id AS sid, o.id AS oid",
            id=fact_id,
        )
        return _fact_rows(rows)[0] if rows else None

    async def current_facts(
        self,
        subject_id: str,
        relation_type: RelType | None = None,
        object_id: str | None = None,
        *,
        slot: str | None = None,
        exclude_event_endpoints: bool = False,
    ) -> list[Fact]:
        """Currently-valid, non-archived facts with this subject, optionally filtered."""
        rows = await self._run(
            f"""
            MATCH (f:{self.F})-[:SUBJECT]->(s:{self.E} {{id: $sid}})
            WHERE f.validity_end IS NULL AND NOT f:Archived
              AND ($rt IS NULL OR f.relation_type = $rt)
              AND ($slot IS NULL OR f.slot = $slot)
            OPTIONAL MATCH (f)-[:OBJECT]->(o)
            WITH f, s, o
            WHERE ($oid IS NULL OR o.id = $oid)
              AND (NOT $noev OR (NOT s:Event AND (o IS NULL OR NOT o:Event)))
            RETURN f {_FACT_PROJ} AS f, s.id AS sid, o.id AS oid
            ORDER BY f.validity_start
            """,
            sid=subject_id,
            rt=RelType(relation_type).value if relation_type else None,
            slot=slot,
            oid=object_id,
            noev=exclude_event_endpoints,
        )
        return _fact_rows(rows)

    async def similar_current_facts(
        self, subject_id: str, vec: list[float], min_cosine: float, limit: int = 5
    ) -> list[tuple[Fact, float]]:
        """Current facts about this subject whose text embedding has cosine >= min_cosine
        to `vec` (restatement detection)."""
        rows = await self._run(
            f"""
            MATCH (f:{self.F})-[:SUBJECT]->(s:{self.E} {{id: $sid}})
            WHERE f.validity_end IS NULL
            WITH f, s, vector.similarity.cosine(f.embedding, $vec) AS raw
            WITH f, s, {cypher_cosine('raw')} AS cos
            WHERE cos >= $min
            OPTIONAL MATCH (f)-[:OBJECT]->(o)
            RETURN f {_FACT_PROJ} AS f, s.id AS sid, o.id AS oid, cos
            ORDER BY cos DESC LIMIT $limit
            """,
            sid=subject_id,
            vec=vec,
            min=min_cosine,
            limit=limit,
        )
        return [(f, r["cos"]) for f, r in zip(_fact_rows(rows), rows)]

    async def facts_with_similar_object(
        self, subject_id: str, rel_types: list[RelType], object_name_vec: list[float], min_cosine: float
    ) -> list[tuple[Fact, float]]:
        """Current facts of the given types whose OBJECT entity's name embedding has
        cosine >= min_cosine to `object_name_vec` (the "same target" test)."""
        rows = await self._run(
            f"""
            MATCH (f:{self.F})-[:SUBJECT]->(s:{self.E} {{id: $sid}}), (f)-[:OBJECT]->(o:{self.E})
            WHERE f.validity_end IS NULL AND f.relation_type IN $types
            WITH f, s, o, vector.similarity.cosine(o.name_embedding, $vec) AS raw
            WITH f, s, o, {cypher_cosine('raw')} AS cos
            WHERE cos >= $min
            RETURN f {_FACT_PROJ} AS f, s.id AS sid, o.id AS oid, cos
            """,
            sid=subject_id,
            types=[RelType(t).value for t in rel_types],
            vec=object_name_vec,
            min=min_cosine,
        )
        return [(f, r["cos"]) for f, r in zip(_fact_rows(rows), rows)]

    # ------------------------------------------------------------ retrieval

    async def retrieve(
        self,
        query: str,
        k: int | None = None,
        alpha: float | None = None,
        as_of: datetime | None = None,
        reinforce: bool = True,
        now: datetime | None = None,
        include_history: bool | None = None,
        expand: bool | None = None,
        decay_lambda: float | None = None,
        provenance: bool | None = None,
    ) -> RetrievalResult:
        """Fact retrieval.

        1. Vector search over this namespace's facts for k * fact_overfetch candidates,
           filtered by conversation_id, archive status and validity.
        2. score = alpha*cos + (1-alpha)*min(decay,1) - history_penalty if superseded;
           keep the top k.
        3. If `expand`: for each entity linked to a top fact, add its next-best facts
           (expand_per_entity each, expand_max total), scored the same way.
        Superseded facts are included (penalized) unless include_history=False; `as_of`
        is a hard filter on validity at that instant. With `reinforce`, returned facts
        get last_accessed/access_count bumped; benchmarks pass reinforce=False.
        """
        s = self.s
        k = k or s.retrieval_k
        alpha = s.retrieval_alpha if alpha is None else alpha
        expand = s.expand if expand is None else expand
        provenance = s.provenance if provenance is None else provenance
        history = s.include_history if include_history is None else include_history
        now = self._now(now)

        t0 = time.perf_counter()
        qvec = await self.embedder.aembed_one(query)
        embed_ms = (time.perf_counter() - t0) * 1000

        if as_of is not None:
            vfilter = f"AND {_valid('{v}', True)}"
        elif history:
            vfilter = ""
        else:
            vfilter = f"AND {_valid('{v}', False)}"

        def score_expr(v: str, sim_raw: str) -> str:
            return f"""
            WITH {v}, {cypher_cosine(sim_raw)} AS sim, {cypher_decay(v)} AS d
            WITH {v}, sim, {cypher_cap('d')} AS decay
            WITH {v}, sim, decay, $alpha * sim + (1 - $alpha) * decay
                 - CASE WHEN {v}.validity_end IS NULL THEN 0.0 ELSE $penalty END AS score"""

        returning = """
            MATCH (f)-[:SUBJECT]->(subj)
            OPTIONAL MATCH (f)-[:OBJECT]->(obj)
            RETURN f {.*, embedding: null} AS f, subj.id AS sid, subj.name AS sname,
                   obj.id AS oid, obj.name AS oname, sim, decay, score, elementId(f) AS eid
            ORDER BY score DESC"""

        q_top = f"""
            CALL db.index.vector.queryNodes($index, $pool, $qvec) YIELD node AS f, score AS raw
            WHERE f.conversation_id = $cid AND NOT f:Archived {vfilter.format(v='f')}
            {score_expr('f', 'raw')}
            ORDER BY score DESC LIMIT $k
            {returning}"""

        q_expand = f"""
            MATCH (t:{self.F})-[:SUBJECT|OBJECT]->(e:{self.E})
            WHERE t.id IN $top_ids
            WITH DISTINCT e
            MATCH (g:{self.F})-[:SUBJECT|OBJECT]->(e)
            WHERE NOT g.id IN $top_ids AND NOT g:Archived {vfilter.format(v='g')}
            WITH e, g, vector.similarity.cosine(g.embedding, $qvec) AS raw
            {score_expr('g', 'raw').replace('WITH g,', 'WITH e, g,')}
            ORDER BY score DESC
            WITH e, collect({{g: g, score: score}})[0..$per_entity] AS best
            UNWIND best AS b
            WITH b.g AS f, max(b.score) AS _s
            WITH f, vector.similarity.cosine(f.embedding, $qvec) AS raw
            {score_expr('f', 'raw')}
            ORDER BY score DESC LIMIT $expand_max
            {returning}"""

        params = {
            "index": self.fact_index,
            "pool": k * s.fact_overfetch,
            "qvec": qvec,
            "cid": self.namespace,
            "k": k,
            "alpha": alpha,
            "lambda": s.decay_lambda if decay_lambda is None else decay_lambda,
            "now": now,
            "as_of": as_of,
            "penalty": s.history_penalty,
            "per_entity": s.expand_per_entity,
            "expand_max": s.expand_max if s.expand_max is not None else k,
        }

        turns: list[dict[str, Any]] = []

        async def work(tx: AsyncManagedTransaction) -> list[tuple[dict[str, Any], bool]]:
            top = await (await tx.run(q_top, params)).data()
            rows = [(r, False) for r in top]
            if expand and top and params["per_entity"] > 0 and params["expand_max"] > 0:
                ext = await (await tx.run(q_expand, {**params, "top_ids": [r["f"]["id"] for r in top]})).data()
                rows += [(r, True) for r in ext]
            if provenance and rows:
                ids = list(dict.fromkeys(r["f"].get("turn_id") for r, _ in rows if r["f"].get("turn_id")))
                got = await (await tx.run(
                    f"MATCH (t:Turn:{self.ns}) WHERE t.dia_id IN $ids "
                    "RETURN t.dia_id AS dia_id, t.text AS text, t.session_time AS session_time",
                    ids=ids)).data()
                by_id = {g["dia_id"]: g for g in got}
                turns[:] = [by_id[i] for i in ids if i in by_id]
            if reinforce and rows:
                await tx.run(
                    """
                    UNWIND $eids AS eid
                    MATCH (f) WHERE elementId(f) = eid
                    SET f.last_accessed = $now, f.access_count = f.access_count + 1
                    """,
                    eids=[r["eid"] for r, _ in rows],
                    now=now,
                )
            return rows

        t1 = time.perf_counter()
        async with self.driver.session(database=s.neo4j_database) as session:
            rows = await (session.execute_write(work) if reinforce else session.execute_read(work))
        db_ms = (time.perf_counter() - t1) * 1000

        facts = [
            RetrievedFact(
                id=r["f"]["id"], relation_type=r["f"]["relation_type"], text=r["f"]["text"],
                subject_id=r["sid"], subject_name=r["sname"], object_id=r["oid"],
                object_name=r["oname"], turn_id=r["f"].get("turn_id"),
                validity_start=r["f"]["validity_start"], validity_end=r["f"].get("validity_end"),
                similarity=r["sim"], decay=r["decay"], score=r["score"], via_expansion=exp,
            )
            for r, exp in rows
        ]
        return RetrievalResult(query=query, facts=facts, turns=[SourceTurn(**t) for t in turns],
                               db_ms=db_ms, embed_ms=embed_ms)

    # ------------------------------------------------------------ maintenance

    async def prune(
        self, threshold: float | None = None, dry_run: bool = False, now: datetime | None = None,
        decay_lambda: float | None = None,
    ) -> PruneReport:
        """Archive (never delete) facts whose decay < threshold: they get the :Archived
        label and are excluded from retrieval. Reinforcing a fact revives it."""
        threshold = self.s.prune_threshold if threshold is None else threshold
        set_f = "" if dry_run else "SET f:Archived, f.archived_at = $now"
        rows = await self._run(
            f"""
            MATCH (f:{self.F}) WHERE NOT f:Archived
            WITH f, {cypher_decay('f')} AS decay
            WHERE decay < $thr
            {set_f}
            RETURN f.id AS id, f.text AS text, decay
            ORDER BY decay
            """,
            thr=threshold,
            now=self._now(now),
            **{"lambda": self.s.decay_lambda if decay_lambda is None else decay_lambda},
        )
        return PruneReport(threshold=threshold, dry_run=dry_run, archived_facts=rows)

    async def unarchive_all(self) -> int:
        """Restore every archived fact in this namespace (used between prune experiments)."""
        rows = await self._run(
            f"MATCH (f:{self.F}:Archived) REMOVE f:Archived RETURN count(f) AS n")
        return rows[0]["n"]

    async def decay_of(self, fact_id: str, now: datetime | None = None) -> float | None:
        """Raw (uncapped) decay score of a fact."""
        rows = await self._run(
            f"MATCH (f:{self.F} {{id: $id}}) RETURN {cypher_decay('f')} AS d",
            id=fact_id,
            now=self._now(now),
            **{"lambda": self.s.decay_lambda},
        )
        return rows[0]["d"] if rows else None

    async def stats(self) -> dict[str, int]:
        rows = await self._run(
            f"""
            CALL () {{ MATCH (n:{self.E}) RETURN count(n) AS entities }}
            CALL () {{ MATCH (f:{self.F}) WHERE f.validity_end IS NULL AND NOT f:Archived
                        RETURN count(f) AS current_facts }}
            CALL () {{ MATCH (f:{self.F}) WHERE f.validity_end IS NOT NULL
                        RETURN count(f) AS superseded_facts }}
            CALL () {{ MATCH (f:{self.F}:Archived) RETURN count(f) AS archived_facts }}
            RETURN entities, current_facts, superseded_facts, archived_facts
            """
        )
        return rows[0]

    async def counts_by_type(self) -> dict[str, dict[str, int]]:
        ents = await self._run(
            f"MATCH (n:{self.E}) UNWIND labels(n) AS l WITH l WHERE l IN $types "
            "RETURN l, count(*) AS n ORDER BY n DESC",
            types=[t.value for t in NodeType],
        )
        facts = await self._run(
            f"MATCH (f:{self.F}) RETURN f.relation_type AS t, count(*) AS n ORDER BY n DESC")
        return {"entities": {r["l"]: r["n"] for r in ents}, "facts": {r["t"]: r["n"] for r in facts}}

    async def all_facts(self) -> list[Fact]:
        rows = await self._run(
            f"MATCH (f:{self.F})-[:SUBJECT]->(s) OPTIONAL MATCH (f)-[:OBJECT]->(o) "
            f"RETURN f {_FACT_PROJ} AS f, s.id AS sid, o.id AS oid ORDER BY f.validity_start"
        )
        return _fact_rows(rows)


def _type_of(labels: list[str]) -> NodeType:
    return next(NodeType(l) for l in labels if l in NodeType._value2member_map_)
