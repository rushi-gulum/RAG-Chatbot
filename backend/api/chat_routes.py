"""
Chat session management endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from database import get_db, ChatService
from database.models import ChatSession, ChatMessage
from auth.firebase_auth import require_auth

router = APIRouter(prefix="/chat", tags=["chat"])

# Pydantic models for request/response
class ChatSessionCreate(BaseModel):
    title: Optional[str] = "New Chat"

class ChatSessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    is_active: bool
    message_count: int = 0

class MessageCreate(BaseModel):
    content: str
    message_type: str  # "user" or "assistant"
    sources: Optional[List[dict]] = None

class MessageResponse(BaseModel):
    id: int
    type: str
    content: str
    timestamp: datetime
    sources: Optional[List[dict]] = None

class ChatHistoryResponse(BaseModel):
    session_id: str
    title: str
    messages: List[MessageResponse]

@router.post("/sessions", response_model=ChatSessionResponse)
async def create_chat_session(
    session_data: ChatSessionCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_auth)
):
    """Create a new chat session"""
    try:
        user_id = current_user["uid"]
        session = ChatService.create_chat_session(db, session_data.title, user_id)
        return ChatSessionResponse(
            session_id=session.session_id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            is_active=session.is_active,
            message_count=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create chat session: {str(e)}")

@router.get("/sessions", response_model=List[ChatSessionResponse])
async def get_chat_sessions(
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_auth)
):
    """Get all active chat sessions"""
    try:
        user_id = current_user["uid"]
        sessions = ChatService.get_all_chat_sessions(db, user_id)
        return [
            ChatSessionResponse(
                session_id=session.session_id,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
                is_active=session.is_active,
                message_count=len(session.messages) if session.messages else 0
            )
            for session in sessions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat sessions: {str(e)}")

@router.get("/sessions/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_auth)
):
    """Get chat history for a specific session"""
    try:
        user_id = current_user["uid"]
        session = ChatService.get_chat_session(db, session_id, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        messages = ChatService.get_chat_history(db, session_id)
        formatted_messages = [
            MessageResponse(
                id=msg.id,
                type=msg.message_type,
                content=msg.content,
                timestamp=msg.timestamp,
                sources=ChatService.format_message_for_api(msg).get("sources")
            )
            for msg in messages
        ]
        
        return ChatHistoryResponse(
            session_id=session.session_id,
            title=session.title,
            messages=formatted_messages
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")

@router.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def add_message(
    session_id: str,
    message_data: MessageCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_auth)
):
    """Add a message to a chat session"""
    try:
        user_id = current_user["uid"]
        # Check if session exists
        session = ChatService.get_chat_session(db, session_id, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Add message
        message = ChatService.add_message(
            db, 
            session_id, 
            message_data.message_type, 
            message_data.content, 
            message_data.sources
        )
        
        return MessageResponse(
            id=message.id,
            type=message.message_type,
            content=message.content,
            timestamp=message.timestamp,
            sources=ChatService.format_message_for_api(message).get("sources")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")

@router.put("/sessions/{session_id}/title")
async def update_session_title(
    session_id: str,
    title_data: dict,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_auth)
):
    """Update chat session title"""
    try:
        title = title_data.get("title", "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        
        user_id = current_user["uid"]
        session = ChatService.update_session_title(db, session_id, title, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found or you don't have permission")
        
        return {"message": "Session title updated successfully", "title": session.title}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update session title: {str(e)}")

@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_auth)
):
    """Delete a chat session (soft delete)"""
    try:
        user_id = current_user["uid"]
        success = ChatService.delete_chat_session(db, session_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Chat session not found or you don't have permission")
        
        return {"message": "Chat session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chat session: {str(e)}")

@router.get("/sessions/{session_id}/export")
async def export_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_auth)
):
    """Export chat session as JSON"""
    try:
        user_id = current_user["uid"]
        session = ChatService.get_chat_session(db, session_id, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        messages = ChatService.get_chat_history(db, session_id)
        export_data = {
            "session_id": session.session_id,
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "messages": [
                ChatService.format_message_for_api(msg)
                for msg in messages
            ]
        }
        
        return export_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export chat session: {str(e)}")