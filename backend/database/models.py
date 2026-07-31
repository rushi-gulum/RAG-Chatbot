"""
Database models for RAG Chatbot persistent storage
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class ProcessedDocument(Base):
    """Track documents that have been processed through RAG pipeline"""
    __tablename__ = "processed_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=True)  # Firebase user ID
    document_id = Column(String, unique=True, index=True, nullable=True)  # UUID from RAG pipeline
    filename = Column(String, index=True, nullable=False)  # Remove unique constraint as same file can be processed multiple times
    file_hash = Column(String, index=True, nullable=False)  # Hash of file content
    file_size = Column(Integer, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed_date = Column(DateTime, default=datetime.utcnow)
    chunk_count = Column(Integer, default=0)
    status = Column(String, default="processed")  # processed, failed, processing
    
    def __repr__(self):
        return f"<ProcessedDocument(filename='{self.filename}', status='{self.status}')>"

class ChatSession(Base):
    """Store multiple chat sessions"""
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=True)  # Firebase user ID
    session_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationship to messages
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ChatSession(session_id='{self.session_id}', title='{self.title}')>"

class ChatMessage(Base):
    """Store individual messages in chat sessions"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"), nullable=False)
    message_type = Column(String, nullable=False)  # user, assistant
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Store sources as JSON string for assistant messages
    sources_json = Column(Text, nullable=True)
    
    # Relationship to session
    session = relationship("ChatSession", back_populates="messages")
    
    def __repr__(self):
        return f"<ChatMessage(type='{self.message_type}', session='{self.session_id}')>"

class DocumentChunk(Base):
    """Track individual chunks created from documents"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_filename = Column(String, ForeignKey("processed_documents.filename"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    embedding_id = Column(String, nullable=True)  # ChromaDB ID
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<DocumentChunk(document='{self.document_filename}', chunk={self.chunk_index})>"