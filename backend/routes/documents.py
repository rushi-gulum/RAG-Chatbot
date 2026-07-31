from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pathlib import Path
from sqlalchemy.orm import Session

# Import database dependencies
from database import get_db, DocumentService
from auth.firebase_auth import require_auth

router = APIRouter(prefix="/documents", tags=["documents"])

# Configure upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_auth)
):
    """
    Upload a PDF document for RAG processing - redirects to RAG pipeline
    """
    try:
        # Import RAG processing function
        from routes.rag import process_document_complete
        
        # Use the existing RAG pipeline for processing
        result = await process_document_complete(file=file, document_id=None, db=db)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "PDF uploaded and processed successfully",
                "file_id": result.document_id,
                "filename": file.filename,
                "chunks_processed": result.chunks_processed,
                "chunks_embedded": result.chunks_embedded,
                "chunks_stored": result.chunks_stored,
                "status": result.status
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the file: {str(e)}"
        )

@router.get("/list")
async def list_uploaded_documents(
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_auth)
):
    """
    List all processed documents from both database and vector store
    """
    try:
        documents_list = []
        
        # First, try to get documents from the database
        try:
            user_id = current_user["uid"]
            db_documents = DocumentService.get_processed_documents(db, user_id)
            for doc in db_documents:
                documents_list.append({
                    "file_id": doc.document_id,
                    "filename": doc.filename,
                    "uploaded_at": doc.processed_date.timestamp() if doc.processed_date else 0,
                    "size_bytes": doc.file_size,
                    "source": "database"
                })
        except Exception as db_error:
            print(f"Database query failed: {db_error}")
        
        # If no documents in database, try to get from vector store
        if not documents_list:
            try:
                from rag_pipeline.storage import VectorStore
                vector_store = VectorStore(persist_directory="./chroma_db")
                collection = vector_store.collection
                
                # Get documents filtered by user_id
                user_id = current_user["uid"]
                results = collection.get(
                    where={"user_id": user_id},
                    include=['metadatas']
                )
                seen_docs = set()
                
                if results and results.get('metadatas'):
                    for metadata in results['metadatas']:
                        if metadata and 'document_id' in metadata:
                            doc_id = metadata['document_id']
                            if doc_id not in seen_docs:
                                seen_docs.add(doc_id)
                                documents_list.append({
                                    "file_id": doc_id,
                                    "filename": metadata.get('filename', 'Unknown'),
                                    "uploaded_at": 0,  # ChromaDB doesn't store upload timestamp
                                    "size_bytes": 0,
                                    "source": "vector_store"
                                })
                
                # If no user-specific documents found, check for legacy documents without user_id
                # (This is for backwards compatibility with existing documents)
                if not documents_list:
                    legacy_results = collection.get(include=['metadatas'])
                    legacy_docs = set()
                    
                    if legacy_results and legacy_results.get('metadatas'):
                        for metadata in legacy_results['metadatas']:
                            # Only show documents that don't have user_id (legacy documents)
                            if metadata and 'document_id' in metadata and 'user_id' not in metadata:
                                doc_id = metadata['document_id']
                                if doc_id not in legacy_docs:
                                    legacy_docs.add(doc_id)
                                    documents_list.append({
                                        "file_id": doc_id,
                                        "filename": metadata.get('filename', 'Unknown'),
                                        "uploaded_at": 0,
                                        "size_bytes": 0,
                                        "source": "legacy_vector_store"
                                    })
            except Exception as vector_error:
                print(f"Vector store query failed: {vector_error}")
        
        # If still no documents, check file system as last resort
        if not documents_list:
            for file_path in UPLOAD_DIR.glob("*.pdf"):
                file_stats = file_path.stat()
                parts = file_path.name.split("_", 1)
                file_id = parts[0] if len(parts) > 1 else "unknown"
                original_name = parts[1] if len(parts) > 1 else file_path.name
                
                documents_list.append({
                    "file_id": file_id,
                    "filename": original_name,
                    "uploaded_at": file_stats.st_ctime,
                    "size_bytes": file_stats.st_size,
                    "source": "filesystem"
                })
        
        return JSONResponse(
            status_code=200,
            content={
                "documents": documents_list,
                "total": len(documents_list)
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while listing documents: {str(e)}"
        )

@router.delete("/delete/{file_id}")
async def delete_document(
    file_id: str, 
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_auth)
):
    """
    Comprehensively delete a document from all storage locations:
    - Vector database (ChromaDB)
    - SQLite database records
    - Physical file system
    """
    try:
        deletion_results = {
            "vector_store": False,
            "database": False,
            "filesystem": False,
            "document_found": False
        }
        
        # 1. Check and delete from database first
        try:
            user_id = current_user["uid"]
            document = DocumentService.get_document_by_id(db, file_id, user_id)
            if document:
                deletion_results["document_found"] = True
                deletion_results["database"] = DocumentService.delete_document(db, file_id)
                print(f"Database deletion for {file_id}: {deletion_results['database']}")
        except Exception as db_error:
            print(f"Database deletion failed: {db_error}")
        
        # 2. Delete from vector store (ChromaDB)
        try:
            from rag_pipeline.storage import VectorStore
            vector_store = VectorStore(persist_directory="./chroma_db")
            
            # Check if document exists and belongs to current user
            collection = vector_store.collection
            results = collection.get(
                where={
                    "document_id": file_id,
                    "user_id": user_id
                },
                include=['metadatas']
            )
            
            if results and results.get('metadatas') and len(results['metadatas']) > 0:
                deletion_results["document_found"] = True
                # Use the existing vector store delete method
                delete_result = vector_store.delete_document(file_id, user_id)
                deletion_results["vector_store"] = delete_result.get("status") == "success"
                print(f"Vector store deletion for {file_id}: {deletion_results['vector_store']}")
                
                if not deletion_results["vector_store"]:
                    print(f"Vector store deletion error: {delete_result.get('message', 'Unknown error')}")
            elif results:
                # Document exists but doesn't belong to user
                raise HTTPException(status_code=403, detail="Access denied: Document not found or you don't have permission")
                    
        except Exception as vector_error:
            print(f"Vector store deletion failed: {vector_error}")
        
        # 3. Delete physical files from filesystem
        try:
            # Look for files with the document ID in the uploads directory
            matching_files = list(UPLOAD_DIR.glob(f"{file_id}_*.pdf"))
            
            for file_path in matching_files:
                deletion_results["document_found"] = True
                file_path.unlink()
                deletion_results["filesystem"] = True
                print(f"Deleted physical file: {file_path}")
                
        except Exception as file_error:
            print(f"Filesystem deletion failed: {file_error}")
        
        # 4. Also check for files in any temporary processing directories
        try:
            # Check if there are any temporary files or processed files
            import glob
            temp_patterns = [
                f"./temp/{file_id}*",
                f"./processed/{file_id}*",
                f"./{file_id}*"
            ]
            
            for pattern in temp_patterns:
                temp_files = glob.glob(pattern)
                for temp_file in temp_files:
                    try:
                        Path(temp_file).unlink()
                        print(f"Deleted temporary file: {temp_file}")
                    except Exception as e:
                        print(f"Could not delete {temp_file}: {e}")
                        
        except Exception as temp_error:
            print(f"Temporary file cleanup failed: {temp_error}")
        
        # Check if document was found in any location
        if not deletion_results["document_found"]:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID '{file_id}' not found in any storage location"
            )
        
        # Determine overall success
        successful_deletions = sum([
            deletion_results["vector_store"],
            deletion_results["database"],
            deletion_results["filesystem"]
        ])
        
        response_data = {
            "message": f"Document deletion completed. Successful deletions: {successful_deletions}/3 locations",
            "file_id": file_id,
            "deletion_details": deletion_results
        }
        
        # If at least one deletion was successful, consider it a success
        if successful_deletions > 0:
            return JSONResponse(
                status_code=200,
                content=response_data
            )
        else:
            return JSONResponse(
                status_code=207,  # Multi-status: partial success
                content={
                    **response_data,
                    "warning": "Document was found but could not be deleted from any location"
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during document deletion: {str(e)}"
        )