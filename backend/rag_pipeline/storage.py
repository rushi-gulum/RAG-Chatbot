import os
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

import numpy as np
from qdrant_client import QdrantClient, models
from sqlalchemy.orm import Session

from database import DocumentService, get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Qdrant Cloud vector store.

    Embeddings are generated externally by Cloudflare Workers AI.
    Qdrant is responsible only for vector storage and similarity search.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_documents",
        embedding_function: Optional[Any] = None,
    ):
        self.collection_name = os.getenv(
            "QDRANT_COLLECTION",
            collection_name,
        )

        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not self.qdrant_url:
            raise ValueError("QDRANT_URL is required")

        if not self.qdrant_api_key:
            raise ValueError("QDRANT_API_KEY is required")

        logger.info("Initializing Qdrant Cloud")
        logger.info(f"Qdrant collection: {self.collection_name}")

        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            prefer_grpc=False,  # use HTTP only — avoids gRPC DLL issues on Windows
        )

        self._ensure_collection()

        logger.info("Qdrant Cloud initialized successfully")

        try:
            count = self._count()
            logger.info(
                f"Collection '{self.collection_name}' ready "
                f"with {count} existing vectors"
            )
        except Exception as e:
            logger.warning(f"Could not get collection count: {e}")

    def _ensure_collection(self):
        """Create the collection if it doesn't already exist."""

        if self.client.collection_exists(self.collection_name):
            return

        logger.info(
            f"Creating Qdrant collection '{self.collection_name}'"
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        )

        logger.info(
            f"Created collection '{self.collection_name}' "
            f"with 384-dimensional cosine vectors"
        )

        # Create payload indexes required for filtering.
        # Qdrant Cloud requires an index before any filtered query can run.
        for field in ("user_id", "document_id"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.info(f"Created payload index: {field}")
            except Exception as e:
                logger.warning(f"Could not create index for '{field}': {e}")

    def _count(self) -> int:
        """Return the number of vectors in the collection."""

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count

    def is_document_already_processed(
        self,
        filename: str,
        file_content: bytes,
        db: Session = None,
    ) -> bool:

        if db is None:
            db_gen = get_db()
            db = next(db_gen)

            try:
                return self._check_document_processed(
                    filename,
                    file_content,
                    db,
                )
            finally:
                db.close()

        return self._check_document_processed(
            filename,
            file_content,
            db,
        )

    def _check_document_processed(
        self,
        filename: str,
        file_content: bytes,
        db: Session,
    ) -> bool:

        file_hash = DocumentService.calculate_file_hash(
            file_content
        )

        return DocumentService.is_document_processed(
            db,
            filename,
            file_hash,
        )

    def _clean_metadata(
        self,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Qdrant payload values should be JSON-compatible.
        """

        cleaned = {}

        for key, value in metadata.items():

            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value

            elif isinstance(value, (list, tuple)):
                cleaned[key] = [
                    item
                    for item in value
                    if isinstance(
                        item,
                        (str, int, float, bool),
                    )
                ]

            else:
                cleaned[key] = str(value)

        return cleaned

    def _build_filter(
        self,
        filter_criteria: Optional[Dict[str, Any]] = None,
        document_ids: Optional[Union[str, List[str]]] = None,
    ):
        """
        Convert the existing application's filter format
        into a Qdrant filter.
        """

        conditions = []

        if filter_criteria:

            for field, value in filter_criteria.items():

                if isinstance(value, dict) and "$in" in value:

                    conditions.append(
                        models.FieldCondition(
                            key=field,
                            match=models.MatchAny(
                                any=value["$in"]
                            ),
                        )
                    )

                else:

                    conditions.append(
                        models.FieldCondition(
                            key=field,
                            match=models.MatchValue(
                                value=value
                            ),
                        )
                    )

        if document_ids:

            if isinstance(document_ids, str):
                document_ids = [document_ids]

            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(
                        any=document_ids
                    ),
                )
            )

        if not conditions:
            return None

        return models.Filter(
            must=conditions
        )

    def store_embedded_chunks(
        self,
        embedded_chunks: List[Dict[str, Any]],
        document_id: Optional[str] = None,
        batch_size: int = 100,
        filename: Optional[str] = None,
        file_content: Optional[bytes] = None,
        file_size: Optional[int] = None,
        db: Optional[Session] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        if not embedded_chunks:
            return {
                "status": "error",
                "message": "No chunks provided",
            }

        logger.info(
            f"Storing {len(embedded_chunks)} chunks in Qdrant"
        )

        should_track_in_db = all(
            [
                filename,
                file_content is not None,
                file_size is not None,
            ]
        )

        db_session = None

        if should_track_in_db and db is None:
            db_gen = get_db()
            db_session = next(db_gen)
        else:
            db_session = db

        points = []

        valid_chunks = 0
        failed_chunks = 0

        for chunk in embedded_chunks:

            try:

                embedding = chunk.get("embedding")

                if embedding is None:
                    failed_chunks += 1
                    continue

                chunk_id = (
                    chunk.get("metadata", {})
                    .get("chunk_id")
                )

                if not chunk_id:
                    chunk_id = str(uuid.uuid4())

                text = chunk.get("text", "")

                metadata = chunk.get(
                    "metadata",
                    {}
                ).copy()

                metadata["document_id"] = (
                    document_id
                    or metadata.get(
                        "document_id",
                        "unknown",
                    )
                )

                metadata["stored_at"] = (
                    datetime.now().isoformat()
                )

                metadata["storage_status"] = "stored"

                if user_id:
                    metadata["user_id"] = user_id

                metadata.pop("embedding", None)

                metadata = self._clean_metadata(
                    metadata
                )

                if isinstance(
                    embedding,
                    np.ndarray,
                ):
                    embedding = embedding.tolist()

                if not isinstance(
                    embedding,
                    list,
                ) or not embedding:

                    failed_chunks += 1
                    continue

                if len(embedding) != 384:
                    logger.error(
                        f"Invalid embedding dimension "
                        f"for {chunk_id}: "
                        f"{len(embedding)} instead of 384"
                    )
                    failed_chunks += 1
                    continue

                points.append(
                    models.PointStruct(
                        id=chunk_id,
                        vector=embedding,
                        payload={
                            "text": text,
                            **metadata,
                        },
                    )
                )

                valid_chunks += 1

            except Exception as e:

                logger.error(
                    f"Error preparing chunk: {e}"
                )

                failed_chunks += 1

        if not points:

            return {
                "status": "error",
                "message": "No valid chunks to store",
                "valid_chunks": 0,
                "failed_chunks": failed_chunks,
            }

        stored_count = 0

        try:

            for i in range(
                0,
                len(points),
                batch_size,
            ):

                batch = points[
                    i:i + batch_size
                ]

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True,
                )

                stored_count += len(batch)

                logger.info(
                    f"Stored Qdrant batch "
                    f"{i // batch_size + 1}: "
                    f"{len(batch)} chunks"
                )

            result = {
                "status": "success",
                "message": (
                    f"Successfully stored "
                    f"{stored_count} chunks"
                ),
                "valid_chunks": valid_chunks,
                "failed_chunks": failed_chunks,
                "stored_chunks": stored_count,
                "collection_size": self._count(),
            }

            if should_track_in_db and db_session:

                try:

                    file_hash = (
                        DocumentService
                        .calculate_file_hash(
                            file_content
                        )
                    )

                    DocumentService.mark_document_processed(
                        db_session,
                        filename,
                        file_hash,
                        file_size,
                        valid_chunks,
                        document_id,
                        user_id,
                    )

                except Exception as db_error:

                    logger.warning(
                        "Failed to mark document "
                        f"as processed: {db_error}"
                    )

            logger.info(
                f"Qdrant storage completed: {result}"
            )

            return result

        except Exception as e:

            logger.error(
                f"Failed to store chunks in Qdrant: {e}"
            )

            if should_track_in_db and db_session:

                try:

                    file_hash = (
                        DocumentService
                        .calculate_file_hash(
                            file_content
                        )
                    )

                    DocumentService.mark_document_failed(
                        db_session,
                        filename,
                        file_hash,
                        file_size,
                        document_id,
                    )

                except Exception as db_error:

                    logger.warning(
                        "Failed to mark document "
                        f"as failed: {db_error}"
                    )

            return {
                "status": "error",
                "message": f"Storage failed: {e}",
                "valid_chunks": valid_chunks,
                "failed_chunks": failed_chunks,
            }

        finally:

            if (
                should_track_in_db
                and db is None
                and db_session
            ):

                try:
                    db_session.close()
                except Exception:
                    pass

    def search_similar_chunks(
        self,
        query_embedding: Union[
            np.ndarray,
            List[float],
        ],
        top_k: int = 5,
        filter_criteria: Optional[
            Dict[str, Any]
        ] = None,
        document_ids: Optional[
            Union[str, List[str]]
        ] = None,
    ) -> List[Dict[str, Any]]:

        try:

            if isinstance(
                query_embedding,
                np.ndarray,
            ):
                query_embedding = (
                    query_embedding.tolist()
                )

            query_filter = self._build_filter(
                filter_criteria,
                document_ids,
            )

            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

            similar_chunks = []

            for point in response.points:

                payload = point.payload or {}

                metadata = {
                    key: value
                    for key, value in payload.items()
                    if key != "text"
                }

                similar_chunks.append(
                    {
                        "chunk_id": str(point.id),
                        "text": payload.get(
                            "text",
                            "",
                        ),
                        "metadata": metadata,
                        "similarity_score": float(
                            point.score
                        ),
                        "distance": float(
                            1 - point.score
                        ),
                    }
                )

            logger.info(
                f"Found {len(similar_chunks)} "
                "similar chunks in Qdrant"
            )

            return similar_chunks

        except Exception as e:

            logger.error(
                f"Qdrant search failed: {e}"
            )

            return []

    def search_by_query(
        self,
        query: str,
        embedder,
        top_k: int = 5,
        filter_criteria: Optional[
            Dict[str, Any]
        ] = None,
        document_ids: Optional[
            Union[str, List[str]]
        ] = None,
    ) -> List[Dict[str, Any]]:

        try:

            query_embedding = (
                embedder.embed_query(query)
            )

            return self.search_similar_chunks(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_criteria=filter_criteria,
                document_ids=document_ids,
            )

        except Exception as e:

            logger.error(
                f"Query search failed: {e}"
            )

            return []

    def get_document_chunks(
        self,
        document_id: str,
    ) -> List[Dict[str, Any]]:

        try:

            query_filter = self._build_filter(
                document_ids=document_id
            )

            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )

            chunks = []

            for point in records:

                payload = point.payload or {}

                metadata = {
                    key: value
                    for key, value in payload.items()
                    if key != "text"
                }

                chunks.append(
                    {
                        "chunk_id": str(point.id),
                        "text": payload.get(
                            "text",
                            "",
                        ),
                        "metadata": metadata,
                    }
                )

            logger.info(
                f"Retrieved {len(chunks)} chunks "
                f"for document {document_id}"
            )

            return chunks

        except Exception as e:

            logger.error(
                f"Failed to retrieve document chunks: {e}"
            )

            return []

    def delete_document(
        self,
        document_id: str,
        user_id: str = None,
    ) -> Dict[str, Any]:

        try:

            query_filter = self._build_filter(
                filter_criteria={
                    "document_id": document_id
                },
            )

            if user_id:

                query_filter = self._build_filter(
                    filter_criteria={
                        "document_id": document_id,
                        "user_id": user_id,
                    }
                )

            existing, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=10000,
                with_payload=False,
                with_vectors=False,
            )

            deleted_count = len(existing)

            if deleted_count == 0:

                return {
                    "status": "warning",
                    "message": (
                        f"No chunks found for "
                        f"document {document_id}"
                    ),
                    "deleted_count": 0,
                }

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=query_filter
                ),
                wait=True,
            )

            result = {
                "status": "success",
                "message": (
                    f"Deleted {deleted_count} "
                    f"chunks for document "
                    f"{document_id}"
                ),
                "deleted_count": deleted_count,
                "collection_size": self._count(),
            }

            logger.info(
                f"Document deletion completed: {result}"
            )

            return result

        except Exception as e:

            logger.error(
                f"Failed to delete document: {e}"
            )

            return {
                "status": "error",
                "message": f"Deletion failed: {e}",
                "deleted_count": 0,
            }

    def clear_user_collection(
        self,
        user_id: str,
    ) -> Dict[str, Any]:

        try:

            query_filter = self._build_filter(
                filter_criteria={
                    "user_id": user_id
                }
            )

            existing, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=10000,
                with_payload=False,
                with_vectors=False,
            )

            deleted_count = len(existing)

            if deleted_count == 0:

                return {
                    "status": "success",
                    "message": (
                        "No documents found for user"
                    ),
                    "deleted_count": 0,
                }

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=query_filter
                ),
                wait=True,
            )

            return {
                "status": "success",
                "message": (
                    f"Cleared {deleted_count} "
                    "chunks for user"
                ),
                "deleted_count": deleted_count,
            }

        except Exception as e:

            logger.error(
                f"Failed to clear user collection: {e}"
            )

            return {
                "status": "error",
                "message": f"Clear failed: {e}",
                "deleted_count": 0,
            }

    def get_collection_stats(
        self,
    ) -> Dict[str, Any]:

        try:

            total_chunks = self._count()

            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=min(
                    100,
                    total_chunks,
                ) if total_chunks > 0 else 1,
                with_payload=True,
                with_vectors=False,
            )

            documents = set()
            file_types = set()

            for point in records:

                payload = point.payload or {}

                if "document_id" in payload:
                    documents.add(
                        payload["document_id"]
                    )

                if "file_type" in payload:
                    file_types.add(
                        payload["file_type"]
                    )

            return {
                "total_chunks": total_chunks,
                "unique_documents": len(documents),
                "file_types": list(file_types),
                "collection_name": self.collection_name,
                "persist_directory": "Qdrant Cloud",
            }

        except Exception as e:

            logger.error(
                f"Failed to get collection stats: {e}"
            )

            return {
                "error": str(e)
            }

    def clear_collection(self) -> Dict[str, Any]:

        try:

            self.client.delete_collection(
                self.collection_name
            )

            self._ensure_collection()

            return {
                "status": "success",
                "message": (
                    f"Collection "
                    f"'{self.collection_name}' "
                    "cleared successfully"
                ),
                "collection_size": 0,
            }

        except Exception as e:

            logger.error(
                f"Failed to clear collection: {e}"
            )

            return {
                "status": "error",
                "message": f"Clear failed: {e}",
            }
