#!/usr/bin/env python3
"""
Deployment validation script for RAG Chatbot Backend
Run this to verify all dependencies and configurations are ready for production.
"""
import sys
import os
from pathlib import Path
import importlib
from typing import Dict, List, Tuple

def check_environment_variables() -> List[Tuple[str, bool]]:
    """Check if required environment variables are set"""
    required_vars = [
        "FIREBASE_PROJECT_ID",
        "GROQ_API_KEY", 
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "DATABASE_URL",
        "SECRET_KEY"
    ]
    
    results = []
    for var in required_vars:
        is_set = bool(os.getenv(var))
        results.append((var, is_set))
    
    return results

def check_dependencies() -> List[Tuple[str, bool, str]]:
    """Check if required Python packages are available"""
    required_packages = [
        ("fastapi", "fastapi", "Web framework"),
        ("uvicorn", "uvicorn", "ASGI server"),
        ("firebase_admin", "firebase_admin", "Firebase authentication"),
        ("qdrant_client", "qdrant_client", "Vector database client"),
        ("groq", "groq", "Primary LLM API"),
        ("openai", "openai", "Fallback LLM API"),
        ("tenacity", "tenacity", "Retry mechanism"),
        ("sqlalchemy", "sqlalchemy", "Database ORM"),
        ("psycopg2", "psycopg2", "PostgreSQL driver"),
        ("scikit-learn", "sklearn", "ML utilities for evaluation"),
        ("numpy", "numpy", "Numerical computing"),
    ]
    
    results = []
    for display_name, import_name, description in required_packages:
        try:
            importlib.import_module(import_name.replace("-", "_"))
            results.append((display_name, True, description))
        except ImportError:
            results.append((display_name, False, description))
    
    return results

def check_files() -> List[Tuple[str, bool]]:
    """Check if required files exist"""
    required_files = [
        "requirements.txt",
        "app.py",
        "Procfile",
        "runtime.txt",
        ".env",
        "database/__init__.py",
        "rag_pipeline/__init__.py",
        "routes/__init__.py",
    ]
    
    results = []
    base_dir = Path(__file__).parent
    
    for file_path in required_files:
        file_exists = (base_dir / file_path).exists()
        results.append((file_path, file_exists))
    
    return results

def main():
    """Run deployment validation"""
    print("🚀 RAG Chatbot Deployment Validation")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check environment variables
    print("\n📋 Environment Variables:")
    env_results = check_environment_variables()
    for var, is_set in env_results:
        status = "✅" if is_set else "❌"
        print(f"  {status} {var}")
        if not is_set:
            all_checks_passed = False
    
    # Check dependencies  
    print("\n📦 Dependencies:")
    dep_results = check_dependencies()
    for package, available, description in dep_results:
        status = "✅" if available else "❌"
        print(f"  {status} {package:<20} - {description}")
        if not available:
            all_checks_passed = False
    
    # Check files
    print("\n📁 Required Files:")
    file_results = check_files()
    for file_path, exists in file_results:
        status = "✅" if exists else "❌"
        print(f"  {status} {file_path}")
        if not exists:
            all_checks_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 All checks passed! Ready for deployment.")
        sys.exit(0)
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\n💡 Quick fixes:")
        print("  - Set missing environment variables in .env file")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Create missing files as needed")
        sys.exit(1)

if __name__ == "__main__":
    main()