#!/usr/bin/env python3
"""Fix Qdrant collection indexes"""

import os
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType

def fix_qdrant_indexes():
    print("🔧 Fixing Qdrant Collection Indexes")
    print("=" * 50)
    
    # Connect to Qdrant
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    collection_name = "rag_documents"
    
    try:
        # Check current collection info
        collection_info = client.get_collection(collection_name)
        print(f"✅ Collection '{collection_name}' exists")
        print(f"📊 Vectors count: {collection_info.points_count}")
        
        # Create indexes for filtering fields
        indexes_to_create = [
            ("user_id", PayloadSchemaType.KEYWORD),
            ("document_id", PayloadSchemaType.KEYWORD),
            ("filename", PayloadSchemaType.KEYWORD),
            ("embedding_model", PayloadSchemaType.KEYWORD),
            ("chunk_index", PayloadSchemaType.INTEGER),
        ]
        
        print("\n🏗️ Creating payload indexes...")
        
        for field_name, field_type in indexes_to_create:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                print(f"   ✅ Created index for '{field_name}' ({field_type})")
                
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"   ℹ️  Index for '{field_name}' already exists")
                else:
                    print(f"   ⚠️  Failed to create index for '{field_name}': {e}")
        
        print("\n🧪 Testing search with filters...")
        
        # Test search with user_id filter
        test_results = client.search(
            collection_name=collection_name,
            query_vector=[0.1] * 384,  # Dummy vector for testing
            query_filter={
                "must": [
                    {"key": "user_id", "match": {"value": "BPjsXjxGMIQByQDBatN6kJ9b3sv2"}}
                ]
            },
            limit=1
        )
        
        print(f"   ✅ Filter test successful - found {len(test_results)} results")
        
        print("\n🎉 Qdrant indexes fixed successfully!")
        print("The RAG search should now work properly.")
        
    except Exception as e:
        print(f"❌ Failed to fix indexes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_qdrant_indexes()