"""Embedding generation using Cloudflare Workers AI."""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)
DEFAULT_MODEL = "@cf/baai/bge-small-en-v1.5"


class EmbeddingGenerator:
    """Generate normalized 384-dimensional embeddings through Cloudflare."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.provider = os.getenv("EMBEDDING_PROVIDER", "cloudflare").lower()
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        self.embedding_dim = 384
        self.device = device
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        self.timeout = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60"))

        if self.provider != "cloudflare":
            raise ValueError("Set EMBEDDING_PROVIDER=cloudflare.")
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.api_token = os.getenv("CLOUDFLARE_API_TOKEN")
        if not self.account_id or not self.api_token:
            raise ValueError(
                "Cloudflare embeddings require CLOUDFLARE_ACCOUNT_ID and "
                "CLOUDFLARE_API_TOKEN."
            )
        self.url = (
            "https://api.cloudflare.com/client/v4/"
            f"accounts/{self.account_id}/ai/run/{self.model_name}"
        )
        logger.info("Using hosted embedding model: %s", self.model_name)

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)

        size = batch_size or self.batch_size
        batches: List[np.ndarray] = []
        for start in range(0, len(texts), size):
            batch = [text.strip() for text in texts[start:start + size]]
            response = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                },
                json={"text": batch},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("success") is False:
                raise RuntimeError(f"Cloudflare embedding failed: {payload.get('errors')}")
            result = payload.get("result", {})
            vectors = result.get("data") if isinstance(result, dict) else result
            embeddings = np.asarray(vectors, dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Expected {self.embedding_dim}-dimensional embeddings, "
                    f"received shape {embeddings.shape}."
                )
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-12)
            batches.append(embeddings)
        return np.vstack(batches)

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        texts: List[str] = []
        valid_indices: List[int] = []
        for index, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if text:
                texts.append(text)
                valid_indices.append(index)
        if not texts:
            return chunks

        embeddings = self.generate_embeddings(texts)
        valid_index_set = set(valid_indices)
        embedded_chunks: List[Dict[str, Any]] = []
        embedding_index = 0
        for index, chunk in enumerate(chunks):
            copy = chunk.copy()
            copy["metadata"] = chunk.get("metadata", {}).copy()
            if index in valid_index_set:
                copy["embedding"] = embeddings[embedding_index].tolist()
                copy["metadata"].update({
                    "embedding_model": self.model_name,
                    "embedding_dim": self.embedding_dim,
                    "embedded_at": datetime.now().isoformat(),
                    "embedding_status": "success",
                })
                embedding_index += 1
            else:
                copy["embedding"] = None
                copy["metadata"].update({
                    "embedding_status": "failed",
                    "embedding_error": "Empty or invalid text",
                })
            embedded_chunks.append(copy)
        return embedded_chunks

    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        return self.generate_embeddings([query])[0]

    def compute_similarity(self, query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        return np.dot(chunk_embeddings, query_embedding.T).flatten()

    def get_embeddings_from_chunks(self, embedded_chunks: List[Dict[str, Any]]) -> np.ndarray:
        embeddings = [chunk["embedding"] for chunk in embedded_chunks if chunk.get("embedding") is not None]
        return np.asarray(embeddings, dtype=np.float32) if embeddings else np.array([], dtype=np.float32)

    def search_similar_chunks(self, query: str, embedded_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        chunk_embeddings = self.get_embeddings_from_chunks(embedded_chunks)
        if len(chunk_embeddings) == 0:
            return []
        similarities = self.compute_similarity(self.embed_query(query), chunk_embeddings)
        top_indices = set(np.argsort(similarities)[::-1][:top_k].tolist())
        results = []
        valid_index = 0
        for chunk in embedded_chunks:
            if chunk.get("embedding") is not None:
                if valid_index in top_indices:
                    copy = chunk.copy()
                    copy["similarity_score"] = float(similarities[valid_index])
                    results.append(copy)
                valid_index += 1
        return sorted(results, key=lambda item: item["similarity_score"], reverse=True)
