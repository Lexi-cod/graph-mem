import pytest

from graphmem import Embedder, GraphStore, Settings


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    e = Embedder()
    e.embed(["warmup"])
    return e


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(decay_lambda=0.05, retrieval_alpha=0.7, retrieval_k=5)


@pytest.fixture
async def store(settings, embedder):
    """Live store against docker-compose Neo4j. WARNING: wipes the database."""
    try:
        s = await GraphStore.connect(settings, embedder)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Neo4j unavailable: {exc}")
    await s.clear()
    yield s
    await s.close()
