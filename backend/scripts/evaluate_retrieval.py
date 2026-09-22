"""Evaluate tenant-scoped document retrieval against labelled questions.

Run from the backend directory. This deliberately evaluates retrieval only;
answer/citation evaluation needs human-reviewed reference answers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from rag_pipeline.embeddings import EmbeddingGenerator
from rag_pipeline.storage import VectorStore


def reciprocal_rank(results: list[dict], expected_document_ids: set[str]) -> float:
    for rank, result in enumerate(results, start=1):
        if result.get("metadata", {}).get("document_id") in expected_document_ids:
            return 1.0 / rank
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Recall@K and MRR for RAG retrieval.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--user-id", required=True, help="Firebase UID that owns the evaluated documents")
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 11))
    args = parser.parse_args()

    examples = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(examples, list) or not examples:
        raise ValueError("Dataset must be a non-empty JSON array")

    embedder = EmbeddingGenerator()
    store = VectorStore()
    rows = []
    for example in examples:
        query = str(example["query"]).strip()
        expected = set(example["expected_document_ids"])
        if not query or not expected:
            raise ValueError("Every example needs query and expected_document_ids")
        results = store.search_by_query(
            query=query,
            embedder=embedder,
            top_k=args.top_k,
            filter_criteria={"user_id": args.user_id},
        )
        rr = reciprocal_rank(results, expected)
        rows.append({"query": query, "reciprocal_rank": rr, "retrieved": len(results)})

    recall = mean(1.0 if row["reciprocal_rank"] else 0.0 for row in rows)
    mrr = mean(row["reciprocal_rank"] for row in rows)
    print(json.dumps({"examples": len(rows), f"recall_at_{args.top_k}": recall, "mrr": mrr, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
