import asyncio
import threading
from functools import cached_property


class Embedder:
    """sentence-transformers wrapper. Vectors are L2-normalized (cosine == dot)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        # One model instance is shared by concurrent ingestions (asyncio.to_thread).
        self._lock = threading.Lock()

    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name, device="cpu")

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            vecs = self.model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )
        return vecs.tolist()

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed, texts)

    async def aembed_one(self, text: str) -> list[float]:
        return (await self.aembed([text]))[0]
