#!/usr/bin/env python3
"""Check what's in the vector database"""

import os
from dotenv import load_dotenv
load_dotenv()

from rag_pipeline.storage import VectorStore

def debug_vector_db():
    store = VectorStore()
    
    print("🔍 Vector Database Debug")
    print("=" * 40)
    
    try:
        # Get some vectors to see what user_ids exist
        result = store.client.scroll(
            collection_name='rag_documents',
            limit=10,
            with_payload=True
        )
        
        print(f"Found {len(result[0])} vectors (showing first 10)")
        print()
        
        user_ids = set()
        filenames = set()
        
        for i, point in enumerate(result[0]):
            payload = point.payload
            user_id = payload.get('user_id', 'unknown')
            filename = payload.get('filename', 'unknown')
            chunk_index = payload.get('chunk_index', 'unknown')
            
            user_ids.add(user_id)
            filenames.add(filename)
            
            print(f"Vector {i+1}:")
            print(f"   user_id: {user_id}")
            print(f"   filename: {filename}")
            print(f"   chunk_index: {chunk_index}")
            print(f"   text preview: {payload.get('text', '')[:100]}...")
            print()
        
        print("=" * 40)
        print("SUMMARY:")
        print(f"Unique user_ids: {len(user_ids)}")
        for uid in user_ids:
            print(f"   - {uid}")
        
        print(f"Unique filenames: {len(filenames)}")
        for fname in filenames:
            print(f"   - {fname}")
            
    except Exception as e:
        print(f"Error querying vector database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_vector_db()