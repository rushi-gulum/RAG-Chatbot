"""
Simple test script for RAGAS evaluation
=======================================

Run this to test RAGAS evaluation with your uploaded documents.
"""

import asyncio
import json
import os
from pathlib import Path

# Set up environment
import sys
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.ragas_evaluation import RAGPipelineEvaluator, print_ragas_report


async def test_ragas_evaluation():
    """Test RAGAS evaluation with sample queries"""
    
    # Sample test queries - replace with your own
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?", 
        "What are neural networks?",
        "Explain deep learning concepts",
        "What is natural language processing?"
    ]
    
    # Use your Firebase UID - get this from your authentication
    # You can find it in the browser console when logged in: user.uid
    user_id = "test-user-123"  # Replace with your actual Firebase UID
    
    print("🧪 Testing RAGAS Evaluation System")
    print("=" * 50)
    print(f"User ID: {user_id}")
    print(f"Test queries: {len(test_queries)}")
    print(f"Groq API available: {'✅' if os.getenv('GROQ_API_KEY') else '❌'}")
    print()
    
    if not os.getenv('GROQ_API_KEY'):
        print("❌ Error: GROQ_API_KEY not found in environment variables")
        print("   Please set your Groq API key to run RAGAS evaluation")
        return
    
    # Initialize evaluator
    try:
        pipeline_evaluator = RAGPipelineEvaluator()
        print("✅ RAG pipeline evaluator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize evaluator: {e}")
        return
    
    # Run evaluation on sample queries
    print("\n🔄 Running RAGAS evaluation...")
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"   {i}/{len(test_queries)}: {query[:50]}...")
        
        try:
            result = await pipeline_evaluator.evaluate_pipeline_query(
                question=query,
                user_id=user_id,
                top_k=5
            )
            
            if "error" in result:
                print(f"      ⚠️  {result['error']}")
            else:
                metrics = result["ragas_metrics"]
                print(f"      ✅ Overall score: {metrics.overall_score:.3f}")
            
            results.append(result)
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            results.append({"error": str(e), "question": query})
    
    # Calculate aggregated results
    valid_results = [r for r in results if "error" not in r]
    
    print(f"\n📊 RAGAS Evaluation Results")
    print("=" * 50)
    
    if not valid_results:
        print("❌ No valid results obtained")
        print("\n💡 Possible issues:")
        print("   - No documents uploaded for this user")
        print("   - User ID doesn't match document ownership")
        print("   - Vector database connectivity issues")
        print("   - LLM API connectivity issues")
        return
    
    # Print individual results
    print(f"✅ Evaluated {len(valid_results)}/{len(test_queries)} queries successfully")
    print()
    
    for i, result in enumerate(valid_results, 1):
        metrics = result["ragas_metrics"]
        print(f"Query {i}: {result['question'][:60]}...")
        print(f"   Faithfulness:    {metrics.faithfulness:.3f}")
        print(f"   Answer Relevancy: {metrics.answer_relevancy:.3f}")
        print(f"   Context Precision: {metrics.context_precision:.3f}")
        print(f"   Context Recall:   {metrics.context_recall:.3f}")
        print(f"   Overall Score:    {metrics.overall_score:.3f}")
        print(f"   Context Count:    {result.get('context_count', 0)}")
        print()
    
    # Calculate averages
    ragas_scores = [r["ragas_metrics"] for r in valid_results]
    
    avg_faithfulness = sum(m.faithfulness for m in ragas_scores) / len(ragas_scores)
    avg_answer_relevancy = sum(m.answer_relevancy for m in ragas_scores) / len(ragas_scores)
    avg_context_precision = sum(m.context_precision for m in ragas_scores) / len(ragas_scores)
    avg_context_recall = sum(m.context_recall for m in ragas_scores) / len(ragas_scores)
    overall_score = sum(m.overall_score for m in ragas_scores) / len(ragas_scores)
    
    print("🎯 OVERALL RAGAS METRICS")
    print("-" * 30)
    print(f"Faithfulness:     {avg_faithfulness:.3f}")
    print(f"Answer Relevancy: {avg_answer_relevancy:.3f}")  
    print(f"Context Precision: {avg_context_precision:.3f}")
    print(f"Context Recall:   {avg_context_recall:.3f}")
    print(f"Overall Score:    {overall_score:.3f}")
    
    # Interpretation
    print("\n📋 INTERPRETATION")
    print("-" * 30)
    
    if overall_score >= 0.8:
        print("🥇 EXCELLENT: Production-ready RAG system!")
    elif overall_score >= 0.7:
        print("🥈 VERY GOOD: High-quality RAG with minor improvements possible")
    elif overall_score >= 0.6:
        print("🥉 GOOD: Solid RAG system with room for improvement")
    elif overall_score >= 0.5:
        print("⚠️  FAIR: Functional but needs significant improvements")
    else:
        print("❌ POOR: Major improvements required before production")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 30)
    
    if avg_faithfulness < 0.6:
        print("📝 Improve Faithfulness:")
        print("   - Better prompting to reduce hallucinations")
        print("   - Higher quality source documents")
        print("   - More conservative generation parameters")
    
    if avg_answer_relevancy < 0.6:
        print("🎯 Improve Answer Relevancy:")
        print("   - Better query understanding")
        print("   - Improved answer generation prompts")
        print("   - Query expansion techniques")
    
    if avg_context_precision < 0.6:
        print("🔍 Improve Context Precision:")
        print("   - Better embedding models")
        print("   - Tune retrieval parameters")
        print("   - Consider hybrid search")
    
    if avg_context_recall < 0.6:
        print("📚 Improve Context Recall:")
        print("   - Increase retrieval top-k")
        print("   - Better chunking strategy")
        print("   - More comprehensive document coverage")
    
    # Save results
    output_file = Path(__file__).parent / "ragas_test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": str(asyncio.get_event_loop().time()),
            "user_id": user_id,
            "query_count": len(valid_results),
            "avg_faithfulness": avg_faithfulness,
            "avg_answer_relevancy": avg_answer_relevancy,
            "avg_context_precision": avg_context_precision,
            "avg_context_recall": avg_context_recall,
            "overall_ragas_score": overall_score,
            "individual_results": results
        }, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_ragas_evaluation())