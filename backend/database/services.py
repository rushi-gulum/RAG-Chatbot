"""
Database services for document and chat management
"""
from sqlalchemy.orm import Session
from .models import ProcessedDocument, ChatSession, ChatMessage, DocumentChunk
from typing import List, Optional
import hashlib
import json
from datetime import datetime

class DocumentService:
    """Service for managing processed documents"""
    
    @staticmethod
    def calculate_file_hash(file_content: bytes) -> str:
        """Calculate SHA256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()
    
    @staticmethod
    def is_document_processed(db: Session, filename: str, file_hash: str) -> bool:
        """Check if document has already been processed"""
        document = db.query(ProcessedDocument).filter(
            ProcessedDocument.filename == filename,
            ProcessedDocument.file_hash == file_hash,
            ProcessedDocument.status == "processed"
        ).first()
        return document is not None
    
    @staticmethod
    def mark_document_processed(
        db: Session, 
        filename: str, 
        file_hash: str, 
        file_size: int, 
        chunk_count: int = 0,
        document_id: str = None,
        user_id: str = None
    ) -> ProcessedDocument:
        """Mark document as processed"""
        document = ProcessedDocument(
            user_id=user_id,
            document_id=document_id,
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            chunk_count=chunk_count,
            status="processed"
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        return document
    
    @staticmethod
    def get_processed_documents(db: Session, user_id: str = None) -> List[ProcessedDocument]:
        """Get all processed documents, optionally filtered by user"""
        query = db.query(ProcessedDocument).filter(
            ProcessedDocument.status == "processed"
        )
        
        if user_id:
            query = query.filter(ProcessedDocument.user_id == user_id)
        
        return query.all()
    
    @staticmethod
    def get_document_by_id(db: Session, document_id: str, user_id: str = None) -> Optional[ProcessedDocument]:
        """Get a specific document by document_id, optionally filtered by user"""
        query = db.query(ProcessedDocument).filter(
            ProcessedDocument.document_id == document_id
        )
        
        if user_id:
            query = query.filter(ProcessedDocument.user_id == user_id)
        
        return query.first()
    
    @staticmethod
    def delete_document(db: Session, document_id: str, user_id: str = None) -> bool:
        """Delete a document from the database"""
        try:
            query = db.query(ProcessedDocument).filter(
                ProcessedDocument.document_id == document_id
            )
            
            if user_id:
                query = query.filter(ProcessedDocument.user_id == user_id)
            
            document = query.first()
            
            if document:
                db.delete(document)
                db.commit()
                return True
            return False
        except Exception as e:
            db.rollback()
            print(f"Database deletion error: {e}")
            return False
    
    @staticmethod
    def mark_document_failed(db: Session, filename: str, file_hash: str, file_size: int, document_id: str = None, user_id: str = None):
        """Mark document processing as failed"""
        document = ProcessedDocument(
            user_id=user_id,
            document_id=document_id,
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            status="failed"
        )
        db.add(document)
        db.commit()

class ChatService:
    """Service for managing chat sessions and messages"""
    
    @staticmethod
    def create_chat_session(db: Session, title: str = "New Chat", user_id: str = None) -> ChatSession:
        """Create a new chat session"""
        session = ChatSession(title=title, user_id=user_id)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    
    @staticmethod
    def get_chat_session(db: Session, session_id: str, user_id: str = None) -> Optional[ChatSession]:
        """Get chat session by ID"""
        query = db.query(ChatSession).filter(
            ChatSession.session_id == session_id,
            ChatSession.is_active == True
        )
        
        if user_id:
            query = query.filter(ChatSession.user_id == user_id)
            
        return query.first()
    
    @staticmethod
    def get_all_chat_sessions(db: Session, user_id: str = None) -> List[ChatSession]:
        """Get all active chat sessions"""
        query = db.query(ChatSession).filter(
            ChatSession.is_active == True
        )
        
        if user_id:
            query = query.filter(ChatSession.user_id == user_id)
            
        return query.order_by(ChatSession.updated_at.desc()).all()
    
    @staticmethod
    def update_session_title(db: Session, session_id: str, title: str, user_id: str = None) -> Optional[ChatSession]:
        """Update chat session title with user ownership validation"""
        query = db.query(ChatSession).filter(
            ChatSession.session_id == session_id
        )
        
        # Add user ownership validation
        if user_id:
            query = query.filter(ChatSession.user_id == user_id)
            
        session = query.first()
        if session:
            session.title = title
            session.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(session)
        return session
    
    @staticmethod
    def delete_chat_session(db: Session, session_id: str, user_id: str = None) -> bool:
        """Delete chat session (soft delete) with user ownership validation"""
        query = db.query(ChatSession).filter(
            ChatSession.session_id == session_id
        )
        
        # Add user ownership validation
        if user_id:
            query = query.filter(ChatSession.user_id == user_id)
            
        session = query.first()
        if session:
            session.is_active = False
            db.commit()
            return True
        return False
    
    @staticmethod
    def add_message(
        db: Session, 
        session_id: str, 
        message_type: str, 
        content: str, 
        sources: List[dict] = None
    ) -> ChatMessage:
        """Add message to chat session"""
        sources_json = json.dumps(sources) if sources else None
        
        message = ChatMessage(
            session_id=session_id,
            message_type=message_type,
            content=content,
            sources_json=sources_json
        )
        db.add(message)
        
        # Update session updated_at
        session = db.query(ChatSession).filter(
            ChatSession.session_id == session_id
        ).first()
        if session:
            session.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(message)
        return message
    
    @staticmethod
    def get_chat_history(db: Session, session_id: str) -> List[ChatMessage]:
        """Get chat history for a session"""
        return db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.timestamp.asc()).all()
    
    @staticmethod
    def format_message_for_api(message: ChatMessage) -> dict:
        """Format database message for API response"""
        sources = None
        if message.sources_json:
            try:
                sources = json.loads(message.sources_json)
            except json.JSONDecodeError:
                sources = None
        
        return {
            "id": message.id,
            "type": message.message_type,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "sources": sources
        }