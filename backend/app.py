from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path
from datetime import datetime

# Configure paths for different environments
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_USERS_DIR = BASE_DIR / "chroma_db_users"

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
CHROMA_USERS_DIR.mkdir(exist_ok=True)

# Import authentication and security
try:
    from auth.firebase_auth import firebase_auth
    from auth.middleware import SecurityMiddleware, setup_rate_limiting
    security_available = True
except ImportError as e:
    print(f"Security modules not available: {e}")
    security_available = False

# Import database initialization
from database import init_database

# Import routes
from routes.documents import router as documents_router
from api.chat_routes import router as chat_router

# Import RAG routes (gracefully handle if dependencies not installed)
try:
    from routes.rag import router as rag_router
    rag_available = True
except ImportError:
    rag_router = None
    rag_available = False

app = FastAPI(title="RAG Chatbot Backend", version="1.0.0") # pyright: ignore[reportUnknownVariableType]

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    try:
        init_database()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        # Don't stop the app, just log the error

# Enable CORS for frontend communication
allowed_origins = [
    "http://localhost:3000",  # React dev server
    "http://127.0.0.1:3000", # Alternative localhost
    "https://*.devtunnels.ms", # VS Code Dev Tunnels
    "https://*.gitpod.io",    # Gitpod
    "https://*.codespaces.githubusercontent.com", # GitHub Codespaces
]

# Add production origin if configured
if os.getenv("FRONTEND_URL"):
    allowed_origins.append(os.getenv("FRONTEND_URL"))

# For development, allow all origins with devtunnels pattern
if os.getenv("NODE_ENV") != "production":
    allowed_origins.append("https://*.inc1.devtunnels.ms")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if os.getenv("NODE_ENV") != "production" else allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add middleware to log CORS requests for debugging
@app.middleware("http")
async def cors_debug_middleware(request, call_next):
    origin = request.headers.get("origin")
    if origin:
        print(f"CORS Request from origin: {origin}")
    response = await call_next(request)
    return response

# Add security middleware
if security_available:
    app.add_middleware(SecurityMiddleware, allowed_origins=allowed_origins)
    limiter = setup_rate_limiting(app)
    print("Security middleware enabled")
else:
    print("Security middleware disabled - install dependencies to enable")

# Include routers
app.include_router(documents_router)
app.include_router(chat_router)

# Include RAG router if available
if rag_available and rag_router:
    app.include_router(rag_router)

@app.get("/")
async def home():
    return {"message": "RAG Chatbot Backend API is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint with CORS info"""
    return {
        "status": "healthy",
        "cors_enabled": True,
        "allowed_origins": allowed_origins if os.getenv("NODE_ENV") == "production" else ["*"],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Configuration for development
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )