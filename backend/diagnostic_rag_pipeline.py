#!/usr/bin/env python3
"""
RAG Pipeline Diagnostic Script
=============================

This script checks each component of the RAG pipeline to identify issues.
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

print("🔍 RAG Pipeline Diagnostic")
print("=" * 50)

# 1. Check Environment Variables
print("\n1️⃣ Environment Variables:")
required_vars = [
    "GROQ_API_KEY",
    "CLOUDFLARE_ACCOUNT_ID", 
    "CLOUDFLARE_API_TOKEN",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_PROVIDER"
]

for var in required_vars:
    value = os.getenv(var)
    if value:
        print(f"   ✅ {var}: {'*' * 10}...{value[-4:] if len(value) > 4 else '***'}")
    else:
        print(f"   ❌ {var}: NOT SET")

# 2. Test Embedding Generator
print("\n2️⃣ Embedding Generator:")
try:
    from rag_pipeline.embeddings import EmbeddingGenerator
    embedder = EmbeddingGenerator()
    print("   ✅ EmbeddingGenerator initialized successfully")
    
    # Test embedding generation
    test_text = "This is a test sentence."
    embedding = embedder.generate_embedding(test_text)
    print(f"   ✅ Embedding generated: {len(embedding)} dimensions")
    
except Exception as e:
    print(f"   ❌ EmbeddingGenerator failed: {e}")

# 3. Test Vector Store
print("\n3️⃣ Vector Store:")
try:
    from rag_pipeline.storage import VectorStore
    vector_store = VectorStore()
    print("   ✅ VectorStore initialized successfully")
    
    # Test connection
    try:
        # This might fail but tells us about connection
        info = vector_store.client.get_collection("rag_documents")
        print(f"   ✅ Qdrant connection successful")
        print(f"   📊 Collection info: {info.vectors_count} vectors")
    except Exception as e:
        print(f"   ⚠️  Qdrant connection issue: {e}")
    
except Exception as e:
    print(f"   ❌ VectorStore failed: {e}")

# 4. Test LLM Generator  
print("\n4️⃣ LLM Generator:")
try:
    from rag_pipeline.llm_generator import LLMGenerator
    llm_generator = LLMGenerator()
    print("   ✅ LLMGenerator initialized successfully")
    
    # Test LLM call
    test_response = llm_generator.generate_response("What is AI?", [])
    print("   ✅ LLM generation test successful")
    
except Exception as e:
    print(f"   ❌ LLMGenerator failed: {e}")

# 5. Test Document Processor
print("\n5️⃣ Document Processor:")
try:
    from rag_pipeline.preprocessing import DocumentProcessor
    processor = DocumentProcessor()
    print("   ✅ DocumentProcessor initialized successfully")
    
except Exception as e:
    print(f"   ❌ DocumentProcessor failed: {e}")

# 6. Test Database Connection
print("\n6️⃣ Database Connection:")
try:
    from database import get_db, DocumentService
    # Test database connection
    print("   ✅ Database modules imported successfully")
    
    # Test document service
    # Note: This requires a DB session, so we'll just check import
    print("   ✅ DocumentService available")
    
except Exception as e:
    print(f"   ❌ Database connection failed: {e}")

# 7. Check for common issues
print("\n7️⃣ Common Issues Check:")

# Check if uploads directory exists
uploads_dir = Path("uploads")
if uploads_dir.exists():
    files = list(uploads_dir.glob("*"))
    print(f"   📁 Uploads directory: {len(files)} files")
else:
    print("   ⚠️  Uploads directory missing")

# Check Python version
import sys
print(f"   🐍 Python version: {sys.version}")

# Check for key dependencies
dependencies = [
    "groq", "qdrant_client", "pypdf", "PyPDF2", 
    "sqlalchemy", "firebase_admin", "httpx"
]

print("\n8️⃣ Dependencies:")
for dep in dependencies:
    try:
        __import__(dep)
        print(f"   ✅ {dep}")
    except ImportError:
        print(f"   ❌ {dep} - NOT INSTALLED")

print("\n" + "=" * 50)
print("🎯 DIAGNOSTIC COMPLETE")
print("=" * 50)

# Summary and recommendations
print("\n💡 RECOMMENDATIONS:")
print("If you see failures above, here's what to check:")
print("1. Environment variables - ensure all required vars are set")
print("2. API connectivity - check internet connection and API keys")
print("3. Dependencies - run: pip install -r requirements.txt")
print("4. Qdrant connection - verify QDRANT_URL and QDRANT_API_KEY")
print("5. Document processing - check if files are being uploaded properly")