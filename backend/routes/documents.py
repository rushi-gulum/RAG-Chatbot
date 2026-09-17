from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pathlib import Path
from sqlalchemy.orm import Session

from database import get_db, DocumentService
from auth.firebase_auth import require_auth

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_auth),
):
    """Upload a PDF and run it through the full RAG pipeline."""
    try:
        from routes.rag import process_document_complete

        result = await process_document_complete(
            file=file,
            document_id=None,
            db=db,
            current_user=current_user,
        )

        return JSONResponse(
            status_code=200,
            content={
                "message": "PDF uploaded and processed successfully",
                "file_id": result.document_id,
                "filename": file.filename,
                "chunks_processed": result.chunks_processed,
                "chunks_embedded": result.chunks_embedded,
                "chunks_stored": result.chunks_stored,
                "status": result.status,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the file: {str(e)}",
        )


@router.get("/list")
async def list_uploaded_documents(
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_auth),
):
    """List all processed documents for the current user."""
    try:
        user_id = current_user["uid"]
        documents_list = []

        # Primary source: relational database (PostgreSQL / SQLite)
        try:
            db_documents = DocumentService.get_processed_documents(db, user_id)
            for doc in db_documents:
                documents_list.append(
                    {
                        "file_id": doc.document_id,
                        "filename": doc.filename,
                        "uploaded_at": (
                            doc.processed_date.timestamp()
                            if doc.processed_date
                            else 0
                        ),
                        "size_bytes": doc.file_size or 0,
                        "source": "database",
                    }
                )
        except Exception as db_error:
            print(f"Database query failed: {db_error}")

        # Fallback: filesystem scan (development / first-run only)
        _SUPPORTED_EXTS = ("*.pdf", "*.docx", "*.txt")
        if not documents_list:
            for file_path in (p for ext in _SUPPORTED_EXTS for p in UPLOAD_DIR.glob(ext)):
                stats = file_path.stat()
                parts = file_path.name.split("_", 1)
                file_id = parts[0] if len(parts) > 1 else "unknown"
                original_name = parts[1] if len(parts) > 1 else file_path.name
                documents_list.append(
                    {
                        "file_id": file_id,
                        "filename": original_name,
                        "uploaded_at": stats.st_ctime,
                        "size_bytes": stats.st_size,
                        "source": "filesystem",
                    }
                )

        return JSONResponse(
            status_code=200,
            content={"documents": documents_list, "total": len(documents_list)},
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while listing documents: {str(e)}",
        )


@router.delete("/delete/{file_id}")
async def delete_document(
    file_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_auth),
):
    """
    Delete a document from all storage locations:
      1. PostgreSQL / SQLite — document metadata record
      2. Qdrant — all vector chunks for that document
      3. Filesystem — any temporary upload files
    """
    try:
        user_id = current_user["uid"]
        deletion_results = {
            "vector_store": False,
            "database": False,
            "filesystem": False,
            "document_found": False,
        }

        # 1. Database
        try:
            document = DocumentService.get_document_by_id(db, file_id, user_id)
            if document:
                if document.user_id and document.user_id != user_id:
                    raise HTTPException(
                        status_code=403,
                        detail="Access denied: document belongs to another user",
                    )
                deletion_results["document_found"] = True
                deletion_results["database"] = DocumentService.delete_document(
                    db, file_id
                )
        except HTTPException:
            raise
        except Exception as db_error:
            print(f"Database deletion failed: {db_error}")

        # 2. Qdrant vector store
        try:
            from rag_pipeline.storage import VectorStore

            vector_store = VectorStore()
            delete_result = vector_store.delete_document(file_id, user_id)

            if delete_result.get("status") == "success":
                deletion_results["document_found"] = True
                deletion_results["vector_store"] = True
            elif delete_result.get("deleted_count", 0) == 0:
                # Not found in Qdrant — not an error if DB already handled it
                pass
            else:
                print(
                    f"Vector store deletion issue: {delete_result.get('message')}"
                )
        except Exception as vector_error:
            print(f"Vector store deletion failed: {vector_error}")

        # 3. Filesystem — any temp upload files matching the document ID
        try:
            for ext in (".pdf", ".docx", ".txt"):
                for file_path in UPLOAD_DIR.glob(f"{file_id}*{ext}"):
                    deletion_results["document_found"] = True
                    file_path.unlink(missing_ok=True)
                    deletion_results["filesystem"] = True
        except Exception as file_error:
            print(f"Filesystem deletion failed: {file_error}")

        if not deletion_results["document_found"]:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{file_id}' not found in any storage location",
            )

        successful = sum(
            [
                deletion_results["vector_store"],
                deletion_results["database"],
                deletion_results["filesystem"],
            ]
        )

        response_data = {
            "message": f"Document deletion completed ({successful}/3 locations cleaned)",
            "file_id": file_id,
            "deletion_details": deletion_results,
        }

        status_code = 200 if successful > 0 else 207
        return JSONResponse(status_code=status_code, content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during document deletion: {str(e)}",
        )
