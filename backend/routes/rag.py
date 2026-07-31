from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import uuid
from pathlib import Path
import json
from datetime import datetime
from sqlalchemy.orm import Session

# Import database dependencies
from database import get_db, DocumentService
from auth.firebase_auth import require_auth

# Import RAG pipeline components
try:
    from rag_pipeline.preprocessing import DocumentProcessor
    from rag_pipeline.embeddings import EmbeddingGenerator
    from rag_pipeline.storage import VectorStore
    from rag_pipeline.llm_generator import LLMGenerator
except ImportError as e:
    logging.warning(f"RAG pipeline imports failed: {e}")
    DocumentProcessor = None
    EmbeddingGenerator = None
    VectorStore = None
    LLMGenerator = None

logger = logging.getLogger(__name__)

# Pydantic models for API
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5
    document_id: Optional[str] = None  # Keep for backward compatibility
    document_ids: Optional[List[str]] = None  # New field for multiple document search

class SearchResult(BaseModel):
    chunk_id: str
    text: str
    similarity_score: float
    metadata: Dict[str, Any]

class LLMResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    model_used: str
    timestamp: str
    token_usage: Dict[str, int]
    query: str

class ProcessingStatus(BaseModel):
    status: str
    message: str
    document_id: Optional[str] = None
    chunks_processed: int = 0
    chunks_embedded: int = 0
    chunks_stored: int = 0

class CollectionStats(BaseModel):
    total_chunks: int
    unique_documents: int
    collection_name: str
    file_types: List[str]

# Create router
router = APIRouter(prefix="/rag", tags=["RAG Pipeline"])

# Initialize RAG components (will be None if dependencies not installed)
doc_processor = DocumentProcessor() if DocumentProcessor else None
embedder = EmbeddingGenerator() if EmbeddingGenerator else None
vector_store = VectorStore(persist_directory="./chroma_db") if VectorStore else None
llm_generator = LLMGenerator() if LLMGenerator else None

@router.get("/status")
async def get_rag_status():
    """Get RAG pipeline status and availability"""
    return {
        "preprocessing": doc_processor is not None,
        "embeddings": embedder is not None,
        "storage": vector_store is not None,
        "llm": llm_generator is not None,
        "dependencies_installed": all([doc_processor, embedder, vector_store]),
        "llm_available": llm_generator is not None,
        "embedding_model": embedder.model_name if embedder else None,
        "collection_stats": vector_store.get_collection_stats() if vector_store else None
    }

@router.post("/process-document", response_model=ProcessingStatus)
async def process_document_complete(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_auth)
):
    """
    Complete RAG processing: Upload → Process → Embed → Store
    Checks if document is already processed to avoid reprocessing.
    """
    if not all([doc_processor, embedder, vector_store]):
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not available. Please install dependencies: pip install chromadb torch transformers"
        )
    
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read file content for hash checking
        content = await file.read()
        file_size = len(content)
        
        # Check if document is already processed
        if vector_store.is_document_already_processed(file.filename, content, db):
            logger.info(f"Document '{file.filename}' already processed, skipping RAG pipeline")
            return ProcessingStatus(
                status="already_processed",
                message=f"Document '{file.filename}' was already processed and stored",
                chunks_processed=0,
                chunks_embedded=0
            )
        
        # Generate document ID if not provided
        doc_id = document_id or str(uuid.uuid4())
        
        # Save uploaded file temporarily
        upload_path = Path("uploads") / file.filename
        upload_path.parent.mkdir(exist_ok=True)
        
        with open(upload_path, "wb") as buffer:
            buffer.write(content)
        
        # Step 1: Process document into chunks
        logger.info(f"Processing document: {file.filename}")
        chunks = doc_processor.process_document(
            str(upload_path), 
            doc_id, 
            file.filename
        )
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content could be extracted from the document")
        
        # Step 2: Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embedded_chunks = embedder.embed_chunks(chunks)
        embedded_count = len([c for c in embedded_chunks if c.get("embedding") is not None])
        
        # Step 3: Store in vector database with database tracking
        logger.info(f"Storing {embedded_count} embedded chunks")
        user_id = current_user["uid"]
        storage_result = vector_store.store_embedded_chunks(
            embedded_chunks, 
            document_id=doc_id,
            filename=file.filename,
            file_content=content,
            file_size=file_size,
            db=db,
            user_id=user_id
        )
        
        # Clean up uploaded file
        upload_path.unlink(missing_ok=True)
        
        return ProcessingStatus(
            status="success",
            message=f"Document processed successfully",
            document_id=doc_id,
            chunks_processed=len(chunks),
            chunks_embedded=embedded_count,
            chunks_stored=storage_result.get("stored_chunks", 0)
        )
        
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        # Clean up on error
        if 'upload_path' in locals():
            upload_path.unlink(missing_ok=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )

@router.post("/search", response_model=List[SearchResult])
async def search_documents(
    query: SearchQuery,
    current_user: dict = Depends(require_auth)
):
    """
    Search for similar chunks using semantic similarity
    Supports filtering by single document_id or multiple document_ids
    """
    if not all([embedder, vector_store]):
        raise HTTPException(
            status_code=503,
            detail="RAG search not available. Please install dependencies."
        )
    
    try:
        # Handle document ID filtering (support both single and multiple)
        document_filter = None
        if query.document_ids:
            # Use the new multiple document IDs
            document_filter = query.document_ids
        elif query.document_id:
            # Use the single document ID for backward compatibility
            document_filter = query.document_id
        
        # Add user_id filtering for data isolation
        user_id = current_user["uid"]
        
        # For backward compatibility, also search legacy documents without user_id
        # First try user-specific documents
        user_filter = {"user_id": user_id}
        
        # Perform search with user filtering
        results = vector_store.search_by_query(
            query=query.query,
            embedder=embedder,
            top_k=query.top_k,
            filter_criteria=user_filter,
            document_ids=document_filter
        )
        
        # STRICT MODE: If specific documents were selected, don't fall back to other documents
        # Only fall back to legacy documents if NO document filter was specified
        if not results and not document_filter:
            try:
                logger.info("No user documents found, searching legacy documents")
                # Search documents that don't have user_id (legacy documents)
                legacy_results = vector_store.search_by_query(
                    query=query.query,
                    embedder=embedder,
                    top_k=query.top_k,
                    filter_criteria=None,  # No user filter for legacy
                    document_ids=None
                )
                # Filter out any results that do have user_id (keep only true legacy)
                results = [r for r in legacy_results if not r.get('metadata', {}).get('user_id')]
                logger.info(f"Found {len(results)} legacy document results")
            except Exception as e:
                logger.warning(f"Legacy document search failed: {e}")
                results = []
        elif document_filter and not results:
            logger.info(f"No results found in selected documents: {document_filter}")
        
        # Format results
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                chunk_id=result["chunk_id"],
                text=result["text"],
                similarity_score=result["similarity_score"],
                metadata=result["metadata"]
            ))
        
        return search_results
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@router.post("/search-llm", response_model=LLMResponse)
async def search_documents_with_llm(
    query: SearchQuery,
    current_user: dict = Depends(require_auth)
):
    """
    Search for similar chunks and generate AI response using LLM
    Supports filtering by single document_id or multiple document_ids
    """
    if not all([embedder, vector_store, llm_generator]):
        raise HTTPException(
            status_code=503,
            detail="RAG+LLM search not available. Please install dependencies and set GROQ_API_KEY."
        )
    
    try:
        # Handle document ID filtering (support both single and multiple)
        document_filter = None
        if query.document_ids:
            # Use the new multiple document IDs
            document_filter = query.document_ids
        elif query.document_id:
            # Use the single document ID for backward compatibility
            document_filter = query.document_id
        
        # Add user_id filtering for data isolation
        user_id = current_user["uid"]
        user_filter = {"user_id": user_id}
        
        # Perform vector search to get relevant chunks
        results = vector_store.search_by_query(
            query=query.query,
            embedder=embedder,
            top_k=query.top_k,
            filter_criteria=user_filter,
            document_ids=document_filter
        )
        
        # STRICT MODE: If specific documents were selected, don't fall back to other documents
        # Only fall back to legacy documents if NO document filter was specified
        if not results and not document_filter:
            try:
                logger.info("No user documents found for LLM search, trying legacy documents")
                legacy_results = vector_store.search_by_query(
                    query=query.query,
                    embedder=embedder,
                    top_k=query.top_k,
                    filter_criteria=None,
                    document_ids=None
                )
                results = [r for r in legacy_results if not r.get('metadata', {}).get('user_id')]
                logger.info(f"Found {len(results)} legacy document results for LLM")
            except Exception as e:
                logger.warning(f"Legacy document search failed: {e}")
        elif document_filter and not results:
            logger.info(f"No results found in selected documents for LLM: {document_filter}")
        
        # If still no results and specific documents were selected, provide helpful message
        if not results and document_filter:
            logger.warning(f"No relevant content found in selected documents: {document_filter}")
            # Generate a response indicating that the selected documents don't contain relevant information
            return LLMResponse(
                response="I couldn't find any relevant information in the selected documents to answer your question. The documents you've chosen may not contain content related to your query. You could try selecting different documents or asking a different question.",
                sources=[],
                model_used="system_fallback",
                timestamp=datetime.now().isoformat(),
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                query=query.query
            )
        
        # Generate AI response using LLM
        llm_response = llm_generator.generate_response(
            query=query.query,
            retrieved_chunks=results
        )
        
        return LLMResponse(
            response=llm_response["response"],
            sources=llm_response["sources"],
            model_used=llm_response["model_used"],
            timestamp=llm_response["timestamp"],
            token_usage=llm_response["token_usage"],
            query=llm_response.get("query", query.query)  # Fallback to original query
        )
        
    except Exception as e:
        logger.error(f"LLM search failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"LLM search failed: {str(e)}"
        )

@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: str):
    """
    Get all chunks for a specific document
    """
    if not vector_store:
        raise HTTPException(
            status_code=503,
            detail="Vector storage not available"
        )
    
    try:
        chunks = vector_store.get_document_chunks(document_id)
        return {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "chunks": chunks
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve document chunks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve chunks: {str(e)}"
        )

@router.delete("/documents/{document_id}")
async def delete_document_from_storage(
    document_id: str,
    current_user: dict = Depends(require_auth)
):
    """
    Delete document and all its chunks from vector storage
    """
    if not vector_store:
        raise HTTPException(
            status_code=503,
            detail="Vector storage not available"
        )
    
    try:
        user_id = current_user["uid"]
        result = vector_store.delete_document(document_id, user_id)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

@router.get("/collection/stats", response_model=CollectionStats)
async def get_collection_statistics():
    """
    Get vector collection statistics
    """
    if not vector_store:
        raise HTTPException(
            status_code=503,
            detail="Vector storage not available"
        )
    
    try:
        stats = vector_store.get_collection_stats()
        
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        
        return CollectionStats(
            total_chunks=stats["total_chunks"],
            unique_documents=stats["unique_documents"],
            collection_name=stats["collection_name"],
            file_types=stats.get("file_types", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get collection stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )

@router.delete("/collection/clear")
async def clear_collection(current_user: dict = Depends(require_auth)):
    """
    Clear all documents from the vector collection for the current user
    """
    if not vector_store:
        raise HTTPException(
            status_code=503,
            detail="Vector storage not available"
        )
    
    try:
        user_id = current_user["uid"]
        
        # Get all user documents first
        collection = vector_store.collection
        user_docs = collection.get(
            where={"user_id": user_id},
            include=["metadatas"]
        )
        
        if not user_docs["ids"]:
            return {"message": "No documents found for user", "deleted_count": 0}
        
        # Delete user documents
        collection.delete(ids=user_docs["ids"])
        
        result = {
            "status": "success",
            "message": f"Cleared {len(user_docs['ids'])} chunks for user",
            "deleted_count": len(user_docs["ids"])
        }
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear collection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear collection: {str(e)}"
        )

@router.get("/health")
async def rag_health_check():
    """
    Health check for RAG pipeline components
    """
    health_status = {
        "status": "healthy",
        "components": {
            "document_processor": doc_processor is not None,
            "embedding_generator": embedder is not None,
            "vector_store": vector_store is not None
        },
        "ready": all([doc_processor, embedder, vector_store])
    }
    
    if embedder:
        health_status["embedding_info"] = embedder.get_model_info()
    
    if vector_store:
        try:
            stats = vector_store.get_collection_stats()
            health_status["collection_info"] = {
                "total_chunks": stats.get("total_chunks", 0),
                "unique_documents": stats.get("unique_documents", 0)
            }
        except Exception:
            health_status["collection_info"] = {"error": "Could not retrieve stats"}
    
    return health_status