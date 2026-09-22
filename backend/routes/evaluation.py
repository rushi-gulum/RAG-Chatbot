"""
Evaluation API Routes for RAG System
====================================

Provides REST API endpoints for running evaluations including RAGAS.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import asyncio
import json
from pathlib import Path
import uuid
from datetime import datetime

from auth.firebase_auth import require_auth
from auth.middleware import rate_limit

# Import evaluation modules
try:
    from evaluation.ragas_evaluation import RAGPipelineEvaluator, RAGASResult
    RAGAS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RAGAS evaluation not available: {e}")
    RAGAS_AVAILABLE = False

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# Pydantic models
class EvaluationQuery(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    expected_document_ids: Optional[List[str]] = None
    reference_answer: Optional[str] = None

class QuickEvaluationRequest(BaseModel):
    queries: List[EvaluationQuery] = Field(min_items=1, max_items=20)
    top_k: int = Field(default=5, ge=1, le=10)

class RAGASQuickResponse(BaseModel):
    evaluation_id: str
    timestamp: str
    query_count: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    overall_ragas_score: float
    individual_results: List[Dict[str, Any]]

# In-memory storage for evaluation results (in production, use Redis or database)
evaluation_results = {}


@router.post("/ragas/quick", response_model=RAGASQuickResponse)
@rate_limit("5/minute")  # Limit due to computational cost
async def run_quick_ragas_evaluation(
    request: QuickEvaluationRequest,
    current_user: dict = Depends(require_auth)
):
    """
    Run RAGAS evaluation on a small set of queries for quick feedback
    """
    if not RAGAS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="RAGAS evaluation not available. Install required dependencies."
        )
    
    user_id = current_user["uid"]
    evaluation_id = str(uuid.uuid4())
    
    try:
        # Initialize pipeline evaluator
        pipeline_evaluator = RAGPipelineEvaluator()
        
        # Evaluate each query
        individual_results = []
        for query_data in request.queries:
            result = await pipeline_evaluator.evaluate_pipeline_query(
                question=query_data.query,
                user_id=user_id,
                top_k=request.top_k
            )
            individual_results.append(result)
        
        # Calculate aggregated scores
        valid_results = [r for r in individual_results if "error" not in r]
        if not valid_results:
            raise HTTPException(
                status_code=400,
                detail="No valid evaluation results. Check if documents are uploaded."
            )
        
        ragas_scores = [r["ragas_metrics"] for r in valid_results]
        
        response = RAGASQuickResponse(
            evaluation_id=evaluation_id,
            timestamp=datetime.now().isoformat(),
            query_count=len(valid_results),
            avg_faithfulness=sum(m.faithfulness for m in ragas_scores) / len(ragas_scores),
            avg_answer_relevancy=sum(m.answer_relevancy for m in ragas_scores) / len(ragas_scores),
            avg_context_precision=sum(m.context_precision for m in ragas_scores) / len(ragas_scores),
            avg_context_recall=sum(m.context_recall for m in ragas_scores) / len(ragas_scores),
            overall_ragas_score=sum(m.overall_score for m in ragas_scores) / len(ragas_scores),
            individual_results=[
                {
                    "query": r["question"],
                    "answer": r["answer"],
                    "faithfulness": r["ragas_metrics"].faithfulness,
                    "answer_relevancy": r["ragas_metrics"].answer_relevancy,
                    "context_precision": r["ragas_metrics"].context_precision,
                    "context_recall": r["ragas_metrics"].context_recall,
                    "overall_score": r["ragas_metrics"].overall_score,
                    "context_count": r.get("context_count", 0)
                }
                for r in valid_results
            ]
        )
        
        # Store results for later retrieval
        evaluation_results[evaluation_id] = response
        
        return response
        
    except Exception as e:
        logging.error(f"RAGAS evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.get("/ragas/{evaluation_id}")
async def get_evaluation_result(
    evaluation_id: str,
    current_user: dict = Depends(require_auth)
):
    """Get stored evaluation results by ID"""
    
    if evaluation_id not in evaluation_results:
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    
    return evaluation_results[evaluation_id]


@router.get("/ragas/health")
async def ragas_health_check():
    """Check if RAGAS evaluation is available"""
    
    return {
        "ragas_available": RAGAS_AVAILABLE,
        "status": "ready" if RAGAS_AVAILABLE else "dependencies_missing",
        "message": "RAGAS evaluation ready" if RAGAS_AVAILABLE else "Install ragas dependencies"
    }


class BenchmarkRequest(BaseModel):
    queries: List[str] = Field(min_items=3, max_items=50)
    top_k: int = Field(default=5, ge=1, le=10)

@router.post("/benchmark")
@rate_limit("2/minute")  # Very limited due to computational cost
async def run_benchmark_evaluation(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(require_auth)
):
    """
    Run a comprehensive benchmark evaluation (async background task)
    """
    if not RAGAS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="RAGAS evaluation not available"
        )
    
    evaluation_id = str(uuid.uuid4())
    user_id = current_user["uid"]
    
    # Convert queries to evaluation format
    evaluation_queries = [
        EvaluationQuery(query=q) for q in request.queries
    ]
    
    # Store initial status
    evaluation_results[evaluation_id] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "query_count": len(request.queries)
    }
    
    # Run evaluation in background
    background_tasks.add_task(
        run_background_evaluation,
        evaluation_id,
        evaluation_queries,
        user_id,
        request.top_k
    )
    
    return {
        "evaluation_id": evaluation_id,
        "status": "started",
        "message": f"Benchmark evaluation started for {len(request.queries)} queries",
        "check_url": f"/evaluation/ragas/{evaluation_id}"
    }


async def run_background_evaluation(
    evaluation_id: str,
    queries: List[EvaluationQuery],
    user_id: str,
    top_k: int
):
    """Run evaluation in background and store results"""
    
    try:
        pipeline_evaluator = RAGPipelineEvaluator()
        individual_results = []
        
        for i, query_data in enumerate(queries):
            result = await pipeline_evaluator.evaluate_pipeline_query(
                question=query_data.query,
                user_id=user_id,
                top_k=top_k
            )
            individual_results.append(result)
            
            # Update progress
            evaluation_results[evaluation_id]["progress"] = (i + 1) / len(queries)
        
        # Calculate final results
        valid_results = [r for r in individual_results if "error" not in r]
        ragas_scores = [r["ragas_metrics"] for r in valid_results]
        
        if ragas_scores:
            final_result = {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "query_count": len(valid_results),
                "avg_faithfulness": sum(m.faithfulness for m in ragas_scores) / len(ragas_scores),
                "avg_answer_relevancy": sum(m.answer_relevancy for m in ragas_scores) / len(ragas_scores),
                "avg_context_precision": sum(m.context_precision for m in ragas_scores) / len(ragas_scores),
                "avg_context_recall": sum(m.context_recall for m in ragas_scores) / len(ragas_scores),
                "overall_ragas_score": sum(m.overall_score for m in ragas_scores) / len(ragas_scores),
                "individual_results": individual_results
            }
        else:
            final_result = {
                "status": "failed",
                "error": "No valid evaluation results"
            }
        
        evaluation_results[evaluation_id] = final_result
        
    except Exception as e:
        evaluation_results[evaluation_id] = {
            "status": "failed",
            "error": str(e)
        }


# Performance monitoring endpoint
@router.get("/metrics/performance")
async def get_performance_metrics(
    current_user: dict = Depends(require_auth)
):
    """Get performance metrics for the user's RAG system"""
    
    # This could be expanded to track historical performance
    return {
        "message": "Performance monitoring endpoint",
        "user_id": current_user["uid"],
        "available_metrics": [
            "response_time",
            "retrieval_accuracy", 
            "generation_quality",
            "user_satisfaction"
        ]
    }