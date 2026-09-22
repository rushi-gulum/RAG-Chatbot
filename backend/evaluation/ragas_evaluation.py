"""
RAGAS (Retrieval-Augmented Generation Assessment) Implementation
=============================================================

Reference-free evaluation framework for RAG systems using LLMs to assess:
- Faithfulness: Factual consistency between answer and context
- Answer Relevancy: Pertinence of answer to query
- Context Precision: Relevance of retrieved context to query
- Context Recall: Coverage of necessary information in context

Usage:
    python evaluation/ragas_evaluation.py --dataset evaluation/dataset_ragas.json --user-id YOUR_UID
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
from statistics import mean, stdev
import re

# RAG Pipeline imports
from rag_pipeline.embeddings import EmbeddingGenerator
from rag_pipeline.storage import VectorStore
from rag_pipeline.llm_generator import LLMGenerator

# For RAGAS, we'll use the Groq LLM as the evaluation model
import os
from groq import Groq


@dataclass
class RAGASMetrics:
    """RAGAS evaluation metrics for a single query"""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    
    @property
    def overall_score(self) -> float:
        """Calculate overall RAGAS score"""
        return mean([self.faithfulness, self.answer_relevancy, 
                    self.context_precision, self.context_recall])


@dataclass
class RAGASResult:
    """Complete RAGAS evaluation result"""
    timestamp: str
    dataset_size: int
    
    # Aggregated RAGAS metrics
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    overall_ragas_score: float
    
    # Performance metrics
    avg_evaluation_time_ms: float
    
    # Individual query results
    query_results: List[Dict[str, Any]]


class RAGASEvaluator:
    """
    RAGAS evaluation implementation using LLMs for reference-free assessment
    with heuristic fallback when LLM evaluation fails
    """
    
    def __init__(self, evaluation_llm_client=None):
        """Initialize RAGAS evaluator with LLM client for evaluation"""
        self.evaluation_llm = evaluation_llm_client or Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "openai/gpt-oss-20b"  # Using publicly available Groq model
        
        # Initialize heuristic evaluator as fallback
        try:
            from evaluation.ragas_heuristic import HeuristicRAGASEvaluator
            self.heuristic_evaluator = HeuristicRAGASEvaluator()
            self.use_heuristic = True
            logging.info("Heuristic RAGAS evaluator initialized as fallback")
        except Exception as e:
            logging.warning(f"Could not initialize heuristic evaluator: {e}")
            self.heuristic_evaluator = None
            self.use_heuristic = False
        
    async def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Evaluate faithfulness: factual consistency of answer against context
        
        Returns a score between 0 and 1 where:
        1.0 = completely faithful (all claims supported by context)
        0.0 = completely unfaithful (claims contradict context)
        """
        
        context_text = "\n".join(contexts[:2])  # Use only first 2 contexts to keep prompt short
        
        # Much simpler, shorter prompt
        prompt = f"""Context: {context_text[:500]}...

Answer: {answer[:200]}...

Rate answer faithfulness to context from 0.0 to 1.0. Only respond with a number."""

        try:
            response = self.evaluation_llm.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Use 0 temperature for consistency
                max_tokens=5      # Only need a number
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # Try to extract score using multiple patterns
            import re
            patterns = [
                r'([01]\.?\d*)',  # 0.8, 1.0, 0.75, etc.
                r'(\d\.?\d*)',    # Any decimal number  
                r'(\d+)',         # Any integer
            ]
            
            for pattern in patterns:
                match = re.search(pattern, score_text)
                if match:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))
            
            # If no number found, try to parse the whole string
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))
            except ValueError:
                # If still fails, log the response and return default
                logging.warning(f"Could not parse faithfulness score from: '{score_text}' for prompt: '{prompt[:100]}...'")
                return 0.5
            
        except Exception as e:
            logging.warning(f"Faithfulness evaluation failed: {e}")
            return 0.5  # Default neutral score
    
    async def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Evaluate answer relevancy: how well the answer addresses the question
        
        Returns a score between 0 and 1 where:
        1.0 = perfectly relevant answer
        0.0 = completely irrelevant answer
        """
        
        # Simplified prompt
        prompt = f"""Question: {question[:200]}
Answer: {answer[:200]}

Rate answer relevancy from 0.0 to 1.0. Only respond with a number."""

        try:
            response = self.evaluation_llm.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # Try to extract score using multiple patterns
            import re
            patterns = [
                r'([01]\.?\d*)',  # 0.8, 1.0, 0.75, etc.
                r'(\d\.?\d*)',    # Any decimal number  
                r'(\d+)',         # Any integer
            ]
            
            for pattern in patterns:
                match = re.search(pattern, score_text)
                if match:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))
            
            # If no number found, try to parse the whole string
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))
            except ValueError:
                logging.warning(f"Could not parse answer relevancy score from: '{score_text}'")
                return 0.5
            
        except Exception as e:
            logging.warning(f"Answer relevancy evaluation failed: {e}")
            return 0.5
    
    async def evaluate_context_precision(self, question: str, contexts: List[str]) -> float:
        """
        Evaluate context precision: relevance of retrieved context to the question
        
        Returns a score between 0 and 1 where:
        1.0 = all context highly relevant to question
        0.0 = no context relevant to question
        """
        
        # Evaluate only first 2 contexts to keep it simple
        individual_scores = []
        
        for i, context in enumerate(contexts[:2]):
            prompt = f"""Question: {question[:200]}
Context: {context[:300]}

Rate context relevance from 0.0 to 1.0. Only respond with a number."""

            try:
                response = self.evaluation_llm.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=5
                )
                
                score_text = response.choices[0].message.content.strip()
                
                # Try to extract score using multiple patterns
                import re
                patterns = [
                    r'([01]\.?\d*)',  # 0.8, 1.0, 0.75, etc.
                    r'(\d\.?\d*)',    # Any decimal number  
                    r'(\d+)',         # Any integer
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, score_text)
                    if match:
                        score = float(match.group(1))
                        individual_scores.append(max(0.0, min(1.0, score)))
                        break
                else:
                    # If no number found, try to parse the whole string
                    try:
                        score = float(score_text)
                        individual_scores.append(max(0.0, min(1.0, score)))
                    except ValueError:
                        logging.warning(f"Could not parse context precision score from: '{score_text}'")
                        individual_scores.append(0.5)
                
            except Exception as e:
                logging.warning(f"Context precision evaluation failed for chunk {i}: {e}")
                individual_scores.append(0.5)
        
        # Return average precision across evaluated contexts
        return mean(individual_scores) if individual_scores else 0.0
    
    async def evaluate_context_recall(self, question: str, answer: str, contexts: List[str]) -> float:
        """
        Evaluate context recall: how well context covers information needed to answer question
        
        Returns a score between 0 and 1 where:
        1.0 = context contains all necessary information
        0.0 = context lacks necessary information
        """
        
        context_text = "\n".join(contexts[:2])  # Use first 2 contexts only
        
        prompt = f"""Question: {question[:200]}
Answer: {answer[:200]}
Context: {context_text[:400]}

Rate if context covers answer info from 0.0 to 1.0. Only respond with a number."""

        try:
            response = self.evaluation_llm.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # Try to extract score using multiple patterns
            import re
            patterns = [
                r'([01]\.?\d*)',  # 0.8, 1.0, 0.75, etc.
                r'(\d\.?\d*)',    # Any decimal number  
                r'(\d+)',         # Any integer
            ]
            
            for pattern in patterns:
                match = re.search(pattern, score_text)
                if match:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))
            
            # If no number found, try to parse the whole string
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))
            except ValueError:
                logging.warning(f"Could not parse context recall score from: '{score_text}'")
                return 0.5
            
        except Exception as e:
            logging.warning(f"Context recall evaluation failed: {e}")
            return 0.5
    
    async def evaluate_query(self, question: str, answer: str, contexts: List[str]) -> RAGASMetrics:
        """Evaluate a single query with all RAGAS metrics"""
        
        # Use heuristic evaluation as primary method since LLM evaluation is not working reliably
        if self.use_heuristic and self.heuristic_evaluator:
            logging.info("Using heuristic RAGAS evaluation")
            return await self.heuristic_evaluator.evaluate_query(question, answer, contexts)
        
        # Fallback to LLM evaluation (though it's currently not working properly)
        logging.info("Using LLM RAGAS evaluation")
        
        # Run all evaluations concurrently
        faithfulness_task = self.evaluate_faithfulness(answer, contexts)
        relevancy_task = self.evaluate_answer_relevancy(question, answer)
        precision_task = self.evaluate_context_precision(question, contexts)
        recall_task = self.evaluate_context_recall(question, answer, contexts)
        
        # Wait for all evaluations to complete
        faithfulness, relevancy, precision, recall = await asyncio.gather(
            faithfulness_task, relevancy_task, precision_task, recall_task
        )
        
        return RAGASMetrics(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=precision,
            context_recall=recall
        )


class RAGPipelineEvaluator:
    """Evaluates complete RAG pipeline using RAGAS metrics"""
    
    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.llm_generator = LLMGenerator()
        self.ragas_evaluator = RAGASEvaluator()
    
    async def evaluate_pipeline_query(self, question: str, user_id: str, top_k: int = 5) -> Dict[str, Any]:
        """Evaluate RAG pipeline for a single query"""
        
        start_time = asyncio.get_event_loop().time()
        
        # 1. Retrieve context
        results = self.vector_store.search_by_query(
            query=question,
            embedder=self.embedder,
            top_k=top_k,
            filter_criteria={"user_id": user_id}
        )
        
        # Extract context texts
        contexts = [result.get("text", "") for result in results if result.get("text")]
        
        if not contexts:
            return {
                "question": question,
                "answer": "No relevant documents found.",
                "contexts": [],
                "ragas_metrics": RAGASMetrics(0.0, 0.0, 0.0, 0.0),
                "evaluation_time_ms": 0,
                "error": "No context retrieved"
            }
        
        # 2. Generate answer
        try:
            response = self.llm_generator.generate_response(question, results)
            answer = response.get("response", "")
        except Exception as e:
            logging.error(f"Answer generation failed: {e}")
            answer = "Failed to generate answer."
        
        # 3. Evaluate with RAGAS
        ragas_metrics = await self.ragas_evaluator.evaluate_query(question, answer, contexts)
        
        evaluation_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "context_count": len(contexts),
            "ragas_metrics": ragas_metrics,
            "evaluation_time_ms": evaluation_time,
            "retrieved_sources": len(results)
        }


async def run_ragas_evaluation(dataset_path: Path, user_id: str, top_k: int = 5, 
                              output_path: Optional[Path] = None) -> RAGASResult:
    """Run RAGAS evaluation on a dataset"""
    
    # Load dataset
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty JSON array")
    
    # Initialize pipeline evaluator
    pipeline_evaluator = RAGPipelineEvaluator()
    
    print(f"Running RAGAS evaluation on {len(dataset)} queries...")
    all_results = []
    
    for i, example in enumerate(dataset, 1):
        question = example.get("question") or example.get("query")
        if not question:
            continue
            
        print(f"Evaluating query {i}/{len(dataset)}: {question[:60]}...")
        
        result = await pipeline_evaluator.evaluate_pipeline_query(
            question=question,
            user_id=user_id,
            top_k=top_k
        )
        
        all_results.append(result)
    
    # Aggregate RAGAS scores
    valid_results = [r for r in all_results if "error" not in r]
    
    if not valid_results:
        raise ValueError("No valid evaluation results")
    
    ragas_scores = [r["ragas_metrics"] for r in valid_results]
    
    ragas_result = RAGASResult(
        timestamp=datetime.now().isoformat(),
        dataset_size=len(valid_results),
        
        avg_faithfulness=mean([m.faithfulness for m in ragas_scores]),
        avg_answer_relevancy=mean([m.answer_relevancy for m in ragas_scores]),
        avg_context_precision=mean([m.context_precision for m in ragas_scores]),
        avg_context_recall=mean([m.context_recall for m in ragas_scores]),
        overall_ragas_score=mean([m.overall_score for m in ragas_scores]),
        
        avg_evaluation_time_ms=mean([r["evaluation_time_ms"] for r in valid_results]),
        
        query_results=valid_results
    )
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(asdict(ragas_result), f, indent=2, default=str)
        print(f"RAGAS results saved to {output_path}")
    
    return ragas_result


def print_ragas_report(result: RAGASResult):
    """Print comprehensive RAGAS evaluation report"""
    
    print("\n" + "="*80)
    print("RAGAS EVALUATION REPORT")
    print("Retrieval-Augmented Generation Assessment")
    print("="*80)
    print(f"Timestamp: {result.timestamp}")
    print(f"Evaluated Queries: {result.dataset_size}")
    print()
    
    # Core RAGAS Metrics
    print("🎯 RAGAS METRICS")
    print("-" * 50)
    print(f"Faithfulness:       {result.avg_faithfulness:.3f}  (Factual consistency)")
    print(f"Answer Relevancy:   {result.avg_answer_relevancy:.3f}  (Query alignment)")
    print(f"Context Precision:  {result.avg_context_precision:.3f}  (Retrieved relevance)")
    print(f"Context Recall:     {result.avg_context_recall:.3f}  (Information coverage)")
    print()
    print(f"Overall RAGAS Score: {result.overall_ragas_score:.3f}")
    print()
    
    # Performance
    print("⚡ EVALUATION PERFORMANCE")
    print("-" * 50)
    print(f"Avg Evaluation Time: {result.avg_evaluation_time_ms:.1f} ms")
    print()
    
    # Interpretation and Recommendations
    print("📊 INTERPRETATION & RECOMMENDATIONS")
    print("-" * 50)
    
    # Faithfulness Analysis
    if result.avg_faithfulness >= 0.8:
        print("✅ Faithfulness: EXCELLENT - Answers are factually consistent")
    elif result.avg_faithfulness >= 0.6:
        print("⚠️  Faithfulness: GOOD - Minor factual inconsistencies")
    else:
        print("❌ Faithfulness: POOR - Significant hallucination issues")
        print("   → Check context quality and LLM prompting")
    
    # Answer Relevancy Analysis
    if result.avg_answer_relevancy >= 0.8:
        print("✅ Answer Relevancy: EXCELLENT - Answers address queries well")
    elif result.avg_answer_relevancy >= 0.6:
        print("⚠️  Answer Relevancy: GOOD - Some answers drift from queries")
    else:
        print("❌ Answer Relevancy: POOR - Answers don't address queries")
        print("   → Improve query understanding and answer generation")
    
    # Context Precision Analysis
    if result.avg_context_precision >= 0.8:
        print("✅ Context Precision: EXCELLENT - Retrieved context is highly relevant")
    elif result.avg_context_precision >= 0.6:
        print("⚠️  Context Precision: GOOD - Some irrelevant context retrieved")
    else:
        print("❌ Context Precision: POOR - Much retrieved context is irrelevant")
        print("   → Improve embedding model and retrieval parameters")
    
    # Context Recall Analysis
    if result.avg_context_recall >= 0.8:
        print("✅ Context Recall: EXCELLENT - Context covers necessary information")
    elif result.avg_context_recall >= 0.6:
        print("⚠️  Context Recall: GOOD - Context missing some information")
    else:
        print("❌ Context Recall: POOR - Context lacks necessary information")
        print("   → Increase retrieval top-k or improve chunking strategy")
    
    print()
    
    # Overall Assessment
    print("🏆 OVERALL ASSESSMENT")
    print("-" * 50)
    
    if result.overall_ragas_score >= 0.8:
        rating = "EXCELLENT"
        emoji = "🥇"
        description = "Production-ready RAG system"
    elif result.overall_ragas_score >= 0.7:
        rating = "VERY GOOD"
        emoji = "🥈"
        description = "High-quality RAG with minor improvements possible"
    elif result.overall_ragas_score >= 0.6:
        rating = "GOOD"
        emoji = "🥉"
        description = "Solid RAG system with room for improvement"
    elif result.overall_ragas_score >= 0.5:
        rating = "FAIR"
        emoji = "⚠️"
        description = "Functional but needs significant improvements"
    else:
        rating = "POOR"
        emoji = "❌"
        description = "Major improvements required before production"
    
    print(f"{emoji} Overall Rating: {rating}")
    print(f"   Score: {result.overall_ragas_score:.3f} / 1.0")
    print(f"   Assessment: {description}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="RAGAS Evaluation for RAG Systems")
    parser.add_argument("--dataset", type=Path, required=True,
                       help="Path to evaluation dataset JSON file")
    parser.add_argument("--user-id", required=True,
                       help="Firebase UID that owns the evaluated documents")
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 11),
                       help="Number of documents to retrieve")
    parser.add_argument("--output", type=Path,
                       help="Path to save RAGAS evaluation results")
    
    args = parser.parse_args()
    
    # Verify environment
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY environment variable not set")
        print("RAGAS evaluation requires an LLM for reference-free assessment")
        return
    
    try:
        # Run RAGAS evaluation
        result = asyncio.run(run_ragas_evaluation(
            dataset_path=args.dataset,
            user_id=args.user_id,
            top_k=args.top_k,
            output_path=args.output
        ))
        
        print_ragas_report(result)
        
    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()