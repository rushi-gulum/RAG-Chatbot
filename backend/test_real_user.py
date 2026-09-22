#!/usr/bin/env python3
"""Test RAG search with real user data"""

import os
from dotenv import load_dotenv
load_dotenv()

from rag_pipeline.storage import VectorStore
from rag_pipeline.embeddings import EmbeddingGenerator  
from rag_pipeline.llm_generator import LLMGenerator

def test_rag_search():
    print("🧪 Testing RAG Search with Real User Data")
    print("=" * 50)
    
    # Real user IDs from the database
    real_user_ids = [
        "BPjsXjxGMIQByQDBatN6kJ9b3sv2",
        "jysHfC4OQQRB7cSm0sTCx4JWWZs2"
    ]
    
    # Test query
    test_query = "What is LinkedIn growth strategy?"
    
    try:
        # Initialize components
        embedder = EmbeddingGenerator()
        vector_store = VectorStore()
        llm_generator = LLMGenerator()
        
        print(f"Query: {test_query}")
        print()
        
        for user_id in real_user_ids:
            print(f"👤 Testing user: {user_id}")
            
            # Search documents
            results = vector_store.search_by_query(
                query=test_query,
                embedder=embedder,
                top_k=5,
                filter_criteria={"user_id": user_id}
            )
            
            print(f"   📄 Found {len(results)} results")
            
            if results:
                # Show first result
                first_result = results[0]
                print(f"   📖 Top result: {first_result.get('metadata', {}).get('filename', 'unknown')}")
                print(f"   🎯 Score: {first_result.get('similarity_score', 0):.3f}")
                print(f"   📝 Text preview: {first_result.get('text', '')[:100]}...")
                
                # Generate answer
                try:
                    response = llm_generator.generate_response(test_query, results)
                    answer = response.get('response', 'No response')
                    sources = response.get('sources', [])
                    
                    print(f"   🤖 Generated answer: {answer[:200]}...")
                    print(f"   📚 Sources: {len(sources)}")
                    
                except Exception as e:
                    print(f"   ❌ LLM generation failed: {e}")
            else:
                print(f"   ⚠️  No results found for this user")
            
            print()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_search()