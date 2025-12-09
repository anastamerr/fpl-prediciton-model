"""
Query embedding helper supporting BGE-small and MPNet models.

Embeddings are lazily loaded to avoid heavy startup costs and can be normalized
for cosine similarity queries in Neo4j vector indexes.
"""

from typing import Dict, Optional
import numpy as np


MODEL_REGISTRY = {
    "bge-small": {"model_id": "BAAI/bge-small-en-v1.5", "dimension": 384},
    "mpnet": {"model_id": "sentence-transformers/all-mpnet-base-v2", "dimension": 768},
}


class QueryEmbedder:
    def __init__(
        self,
        model_alias: str = "bge-small",
        normalize: bool = True,
        cache_enabled: bool = True,
    ) -> None:
        if model_alias not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model alias: {model_alias}")

        self.model_alias = model_alias
        self.normalize = normalize
        self.cache_enabled = cache_enabled
        self._model = None
        self._cache: Dict[str, np.ndarray] = {}

    def switch_model(self, model_alias: str) -> None:
        if model_alias not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model alias: {model_alias}")
        if model_alias == self.model_alias:
            return

        self.model_alias = model_alias
        self._model = None
        self._cache.clear()

    def embed(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("Query text is required to generate embeddings.")

        normalized_query = query.strip()
        if self.cache_enabled and normalized_query in self._cache:
            return self._cache[normalized_query]

        model = self._load_model()
        vector = model.encode(normalized_query)
        embedding = np.asarray(vector, dtype=float)
        if self.normalize:
            embedding = self._normalize(embedding)

        if self.cache_enabled:
            self._cache[normalized_query] = embedding
        return embedding

    def embedding_dimension(self) -> int:
        return MODEL_REGISTRY[self.model_alias]["dimension"]

    def model_id(self) -> str:
        return MODEL_REGISTRY[self.model_alias]["model_id"]

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for QueryEmbedder but is not installed."
            ) from exc

        model_id = self.model_id()
        self._model = SentenceTransformer(model_id)
        return self._model

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
