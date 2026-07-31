import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
from pathlib import Path
import uuid
from sqlalchemy.orm import Session
from database import DocumentService, get_db

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(
        self, 
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_documents",
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize ChromaDB vector store with FAISS indexing
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_function: Custom embedding function (optional)
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at: {self.persist_directory}")
        logger.info(f"Collection name: {collection_name}")
        
        try:
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity for FAISS
            )
            
            logger.info(f"ChromaDB initialized successfully")
            logger.info(f"Collection '{collection_name}' ready with {self.collection.count()} existing documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    def is_document_already_processed(
        self, 
        filename: str, 
        file_content: bytes,
        db: Session = None
    ) -> bool:
        """
        Check if document has already been processed using database tracking
        
        Args:
            filename: Name of the document file
            file_content: Raw file content for hash calculation
            db: Database session (optional)
            
        Returns:
            True if document is already processed, False otherwise
        """
        if db is None:
            # Get a database session if not provided
            db_gen = get_db()
            db = next(db_gen)
            try:
                return self._check_document_processed(filename, file_content, db)
            finally:
                db.close()
        else:
            return self._check_document_processed(filename, file_content, db)
    
    def _check_document_processed(self, filename: str, file_content: bytes, db: Session) -> bool:
        """Internal method to check if document is processed"""
        file_hash = DocumentService.calculate_file_hash(file_content)
        return DocumentService.is_document_processed(db, filename, file_hash)

    def store_embedded_chunks(
        self, 
        embedded_chunks: List[Dict[str, Any]], 
        document_id: Optional[str] = None,
        batch_size: int = 100,
        filename: Optional[str] = None,
        file_content: Optional[bytes] = None,
        file_size: Optional[int] = None,
        db: Optional[Session] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store embedded chunks in ChromaDB with FAISS indexing and database tracking
        
        Args:
            embedded_chunks: List of chunks with embeddings from embeddings.py
            document_id: Optional document identifier for grouping chunks
            batch_size: Batch size for storage operations
            filename: Document filename for database tracking
            file_content: File content for hash calculation
            file_size: File size for database tracking
            db: Database session for tracking processed documents
            
        Returns:
            Storage result summary
        """
        if not embedded_chunks:
            logger.warning("No chunks provided for storage")
            return {"status": "error", "message": "No chunks provided"}
        
        logger.info(f"Storing {len(embedded_chunks)} embedded chunks...")
        
        # Database tracking setup
        should_track_in_db = all([filename, file_content is not None, file_size is not None])
        db_session = None
        if should_track_in_db and db is None:
            db_gen = get_db()
            db_session = next(db_gen)
        else:
            db_session = db
        
        # Prepare data for ChromaDB
        chunk_ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        valid_chunks = 0
        failed_chunks = 0
        
        for chunk in embedded_chunks:
            try:
                # Check if chunk has valid embedding
                if chunk.get("embedding") is None:
                    failed_chunks += 1
                    continue
                
                # Extract chunk data
                chunk_id = chunk["metadata"].get("chunk_id")
                if not chunk_id:
                    chunk_id = str(uuid.uuid4())
                    logger.warning(f"Generated new chunk_id: {chunk_id}")
                
                embedding = chunk["embedding"]
                text = chunk.get("text", "")
                metadata = chunk["metadata"].copy()
                
                # Add additional metadata
                metadata["document_id"] = document_id or metadata.get("document_id", "unknown")
                metadata["stored_at"] = datetime.now().isoformat()
                metadata["storage_status"] = "stored"
                
                # Add user_id for data isolation
                if user_id:
                    metadata["user_id"] = user_id
                
                # Remove embedding from metadata to avoid duplication
                if "embedding" in metadata:
                    del metadata["embedding"]
                
                # Validate embedding format
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                
                if not isinstance(embedding, np.ndarray) or embedding.size == 0:
                    logger.warning(f"Invalid embedding for chunk {chunk_id}")
                    failed_chunks += 1
                    continue
                
                # Add to batch
                chunk_ids.append(chunk_id)
                embeddings.append(embedding.tolist())
                metadatas.append(metadata)
                documents.append(text)
                
                valid_chunks += 1
                
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                failed_chunks += 1
                continue
        
        if not chunk_ids:
            return {
                "status": "error", 
                "message": "No valid chunks to store",
                "valid_chunks": 0,
                "failed_chunks": failed_chunks
            }
        
        # Store in batches
        stored_count = 0
        try:
            for i in range(0, len(chunk_ids), batch_size):
                batch_end = min(i + batch_size, len(chunk_ids))
                
                batch_ids = chunk_ids[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                batch_metadatas = metadatas[i:batch_end]
                batch_documents = documents[i:batch_end]
                
                # Add to ChromaDB collection
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
                
                stored_count += len(batch_ids)
                logger.info(f"Stored batch {i//batch_size + 1}: {len(batch_ids)} chunks")
            
            result = {
                "status": "success",
                "message": f"Successfully stored {stored_count} chunks",
                "valid_chunks": valid_chunks,
                "failed_chunks": failed_chunks,
                "stored_chunks": stored_count,
                "collection_size": self.collection.count()
            }
            
            # Mark document as processed in database
            if should_track_in_db and db_session:
                try:
                    file_hash = DocumentService.calculate_file_hash(file_content)
                    DocumentService.mark_document_processed(
                        db_session, filename, file_hash, file_size, valid_chunks, document_id, user_id
                    )
                    logger.info(f"Document '{filename}' marked as processed in database")
                except Exception as db_error:
                    logger.warning(f"Failed to mark document as processed in database: {str(db_error)}")
            
            logger.info(f"Storage completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to store chunks in ChromaDB: {str(e)}")
            
            # Mark document as failed in database
            if should_track_in_db and db_session:
                try:
                    file_hash = DocumentService.calculate_file_hash(file_content)
                    DocumentService.mark_document_failed(db_session, filename, file_hash, file_size, document_id)
                    logger.info(f"Document '{filename}' marked as failed in database")
                except Exception as db_error:
                    logger.warning(f"Failed to mark document as failed in database: {str(db_error)}")
            
            return {
                "status": "error",
                "message": f"Storage failed: {str(e)}",
                "valid_chunks": valid_chunks,
                "failed_chunks": failed_chunks
            }
        finally:
            # Close database session if we created it
            if should_track_in_db and db is None and db_session:
                try:
                    db_session.close()
                except:
                    pass

    def search_similar_chunks(
        self, 
        query_embedding: Union[np.ndarray, List[float]], 
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
        document_ids: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using ChromaDB's FAISS indexing
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            filter_criteria: Optional metadata filters
            document_ids: Single document_id (str) or list of document_ids to filter by
            
        Returns:
            List of similar chunks with similarity scores
        """
        if self.collection.count() == 0:
            logger.warning("No documents in collection for search")
            return []
        
        try:
            # Convert embedding to list if it's numpy array
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Prepare final filter criteria
            final_filter = None
            
            # Handle document ID filtering
            if document_ids:
                if isinstance(document_ids, str):
                    # Single document ID
                    doc_filter = {"document_id": document_ids}
                elif isinstance(document_ids, list) and len(document_ids) > 0:
                    # Multiple document IDs - use ChromaDB's $in operator
                    doc_filter = {"document_id": {"$in": document_ids}}
                else:
                    logger.warning("Invalid document_ids provided, ignoring filter")
                    doc_filter = None
                
                # Combine with other filter criteria using $and
                if filter_criteria and doc_filter:
                    final_filter = {
                        "$and": [
                            filter_criteria,
                            doc_filter
                        ]
                    }
                elif filter_criteria:
                    final_filter = filter_criteria
                elif doc_filter:
                    final_filter = doc_filter
            else:
                final_filter = filter_criteria
            
            # Perform similarity search
            # Only pass 'where' parameter if we have actual filters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": min(top_k, self.collection.count()),
                "include": ["metadatas", "documents", "distances"]
            }
            
            if final_filter is not None:
                query_params["where"] = final_filter
            
            results = self.collection.query(**query_params)
            
            # Format results
            similar_chunks = []
            
            if results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    chunk_id = results["ids"][0][i]
                    distance = results["distances"][0][i]
                    similarity_score = 1 - distance  # Convert distance to similarity
                    metadata = results["metadatas"][0][i]
                    text = results["documents"][0][i]
                    
                    similar_chunks.append({
                        "chunk_id": chunk_id,
                        "text": text,
                        "metadata": metadata,
                        "similarity_score": float(similarity_score),
                        "distance": float(distance)
                    })
            
            logger.info(f"Found {len(similar_chunks)} similar chunks")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def search_by_query(
        self, 
        query: str, 
        embedder,  # EmbeddingGenerator instance
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
        document_ids: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using text query (convenience method)
        
        Args:
            query: Text query
            embedder: EmbeddingGenerator instance for encoding query
            top_k: Number of results to return
            filter_criteria: Optional metadata filters
            document_ids: Single document_id (str) or list of document_ids to filter by
            
        Returns:
            List of similar chunks
        """
        try:
            # Generate query embedding
            query_embedding = embedder.embed_query(query)
            
            # Search with embedding
            return self.search_similar_chunks(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_criteria=filter_criteria,
                document_ids=document_ids
            )
            
        except Exception as e:
            logger.error(f"Query search failed: {str(e)}")
            return []

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific document
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of chunks for the document
        """
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas", "documents"]
            )
            
            chunks = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    chunks.append({
                        "chunk_id": results["ids"][i],
                        "text": results["documents"][i],
                        "metadata": results["metadatas"][i]
                    })
            
            logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve document chunks: {str(e)}")
            return []

    def delete_document(self, document_id: str, user_id: str = None) -> Dict[str, Any]:
        """
        Delete all chunks for a specific document
        
        Args:
            document_id: Document identifier
            user_id: User identifier for access control (optional)
            
        Returns:
            Deletion result
        """
        try:
            # Get chunks to delete with user filtering
            where_filter = {"document_id": document_id}
            if user_id:
                where_filter["user_id"] = user_id
                
            chunks_to_delete = self.collection.get(
                where=where_filter,
                include=["metadatas"]
            )
            
            if not chunks_to_delete["ids"]:
                return {
                    "status": "warning",
                    "message": f"No chunks found for document {document_id}",
                    "deleted_count": 0
                }
            
            # Delete chunks
            self.collection.delete(
                where={"document_id": document_id}
            )
            
            deleted_count = len(chunks_to_delete["ids"])
            
            result = {
                "status": "success",
                "message": f"Deleted {deleted_count} chunks for document {document_id}",
                "deleted_count": deleted_count,
                "collection_size": self.collection.count()
            }
            
            logger.info(f"Document deletion completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return {
                "status": "error",
                "message": f"Deletion failed: {str(e)}",
                "deleted_count": 0
            }

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Collection statistics
        """
        try:
            total_chunks = self.collection.count()
            
            # Get sample of metadata to analyze
            sample_results = self.collection.get(
                limit=min(100, total_chunks),
                include=["metadatas"]
            )
            
            # Analyze documents
            documents = set()
            file_types = set()
            
            for metadata in sample_results["metadatas"]:
                if "document_id" in metadata:
                    documents.add(metadata["document_id"])
                if "file_type" in metadata:
                    file_types.add(metadata["file_type"])
            
            stats = {
                "total_chunks": total_chunks,
                "unique_documents": len(documents),
                "file_types": list(file_types),
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
            
            logger.info(f"Collection stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}

    def clear_collection(self) -> Dict[str, Any]:
        """
        Clear all data from the collection
        
        Returns:
            Clear operation result
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            result = {
                "status": "success",
                "message": f"Collection '{self.collection_name}' cleared successfully",
                "collection_size": 0
            }
            
            logger.info(f"Collection cleared: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return {
                "status": "error",
                "message": f"Clear failed: {str(e)}"
            }

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if hasattr(self, 'client') and self.client:
                # ChromaDB handles persistence automatically
                logger.info("VectorStore cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")