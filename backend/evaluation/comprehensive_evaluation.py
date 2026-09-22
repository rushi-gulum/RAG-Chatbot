"""
Comprehensive RAG System Evaluation Suite
=========================================

This module provides multiple evaluation methods for RAG systems:
1. Retrieval Evaluation (Precision, Recall, NDCG, MRR)
2. Generation Quality (BLEU, ROUGE, BERTScore, Faithfulness)
3. End-to-End Pipeline Evaluation
4. Performance Metrics (Latency, Throughput)
5. User Experience Metrics

Usage:
    python evaluation/comprehensive_evaluation.py --dataset evaluation/dataset.json --user-id YOUR_UID
"""

import argparse
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from statistics import mean, stdev
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# RAG Pipeline imports
from rag_pipeline.embeddings import EmbeddingGenerator
from rag_pipeline.storage import VectorStore
from rag_pipeline.llm_generator import LLMGenerator

# Optional evaluation libraries (install with: pip install rouge-score sacrebleu bert-score)
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    import sacrebleu
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False


@dataclass
class EvaluationResult:
    """Complete evaluation results for a RAG system"""
    timestamp: str
    dataset_size: int
    
    # Retrieval Metrics
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    mrr: float
    
    # Generation Quality Metrics
    bleu_score: Optional[float] = None
    rouge_scores: Optional[Dict[str, float]] = None
    bert_score_f1: Optional[float] = None
    faithfulness_score: Optional[float] = None
    
    # Performance Metrics
    avg_retrieval_latency_ms: float = 0.0
    avg_generation_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0
    throughput_qps: float = 0.0
    
    # Quality Metrics
    avg_answer_length: float = 0.0
    citation_coverage: float = 0.0
    source_diversity: float = 0.0


class RetrievalEvaluator:
    """Evaluates retrieval quality using standard IR metrics"""
    
    def __init__(self, embedder: EmbeddingGenerator, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
    
    def evaluate_query(self, query: str, expected_doc_ids: List[str], 
                      user_id: str, top_k: int = 10) -> Dict[str, Any]:
        """Evaluate a single query"""
        start_time = time.time()
        results = self.vector_store.search_by_query(
            query=query,
            embedder=self.embedder,
            top_k=top_k,
            filter_criteria={"user_id": user_id}
        )
        retrieval_time = (time.time() - start_time) * 1000
        
        retrieved_doc_ids = [r.get("metadata", {}).get("document_id") for r in results]
        expected_set = set(expected_doc_ids)
        
        # Calculate metrics
        metrics = {}
        for k in [1, 3, 5, 10]:
            if k <= len(results):
                retrieved_k = set(retrieved_doc_ids[:k])
                precision_k = len(retrieved_k & expected_set) / k if k > 0 else 0
                recall_k = len(retrieved_k & expected_set) / len(expected_set) if expected_set else 0
                metrics[f"precision_at_{k}"] = precision_k
                metrics[f"recall_at_{k}"] = recall_k
        
        # MRR calculation
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in expected_set:
                mrr = 1.0 / (i + 1)
                break
        
        # NDCG calculation (simplified)
        dcg = sum(1.0 / (i + 2) for i, doc_id in enumerate(retrieved_doc_ids) 
                 if doc_id in expected_set)
        idcg = sum(1.0 / (i + 2) for i in range(min(len(expected_set), top_k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return {
            **metrics,
            "mrr": mrr,
            "ndcg": ndcg,
            "retrieval_latency_ms": retrieval_time,
            "retrieved_count": len(results),
            "similarity_scores": [r.get("similarity_score", 0) for r in results]
        }


class GenerationEvaluator:
    """Evaluates generation quality using multiple metrics"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) if ROUGE_AVAILABLE else None
    
    def evaluate_generation(self, generated: str, reference: str, sources: List[Dict]) -> Dict[str, Any]:
        """Evaluate generated text against reference"""
        metrics = {}
        
        # BLEU Score
        if BLEU_AVAILABLE:
            bleu = sacrebleu.sentence_bleu(generated, [reference])
            metrics["bleu"] = bleu.score
        
        # ROUGE Scores  
        if ROUGE_AVAILABLE and self.rouge_scorer:
            rouge_scores = self.rouge_scorer.score(reference, generated)
            metrics["rouge1_f1"] = rouge_scores['rouge1'].fmeasure
            metrics["rouge2_f1"] = rouge_scores['rouge2'].fmeasure
            metrics["rougeL_f1"] = rouge_scores['rougeL'].fmeasure
        
        # BERTScore
        if BERT_SCORE_AVAILABLE:
            P, R, F1 = bert_score([generated], [reference], lang="en")
            metrics["bert_score_f1"] = F1.item()
        
        # Custom metrics
        metrics["answer_length"] = len(generated.split())
        metrics["citation_count"] = len(sources)
        metrics["source_diversity"] = len(set(s.get("filename", "") for s in sources))
        
        # Faithfulness (simple keyword overlap with sources)
        source_text = " ".join(s.get("text", "") for s in sources)
        generated_words = set(generated.lower().split())
        source_words = set(source_text.lower().split())
        overlap = len(generated_words & source_words)
        metrics["faithfulness_score"] = overlap / len(generated_words) if generated_words else 0
        
        return metrics


class EndToEndEvaluator:
    """Evaluates the complete RAG pipeline"""
    
    def __init__(self, embedder: EmbeddingGenerator, vector_store: VectorStore, llm_generator: LLMGenerator):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_generator = llm_generator
        self.retrieval_eval = RetrievalEvaluator(embedder, vector_store)
        self.generation_eval = GenerationEvaluator()
    
    def evaluate_pipeline(self, query: str, expected_doc_ids: List[str], 
                         reference_answer: Optional[str], user_id: str, 
                         top_k: int = 5) -> Dict[str, Any]:
        """Evaluate complete pipeline for one query"""
        
        # 1. Evaluate Retrieval
        start_time = time.time()
        retrieval_metrics = self.retrieval_eval.evaluate_query(query, expected_doc_ids, user_id, top_k)
        
        # 2. Generate Answer
        generation_start = time.time()
        results = self.vector_store.search_by_query(
            query=query,
            embedder=self.embedder,
            top_k=top_k,
            filter_criteria={"user_id": user_id}
        )
        
        try:
            response = self.llm_generator.generate_response(query, results)
            generated_answer = response.get("response", "")
            sources = response.get("sources", [])
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            generated_answer = ""
            sources = []
        
        generation_time = (time.time() - generation_start) * 1000
        total_time = (time.time() - start_time) * 1000
        
        # 3. Evaluate Generation (if reference available)
        generation_metrics = {}
        if reference_answer and generated_answer:
            generation_metrics = self.generation_eval.evaluate_generation(
                generated_answer, reference_answer, sources
            )
        
        return {
            **retrieval_metrics,
            **generation_metrics,
            "generation_latency_ms": generation_time,
            "total_latency_ms": total_time,
            "generated_answer": generated_answer,
            "source_count": len(sources),
            "has_answer": bool(generated_answer.strip())
        }


async def run_comprehensive_evaluation(dataset_path: Path, user_id: str, 
                                     top_k: int = 5, output_path: Optional[Path] = None) -> EvaluationResult:
    """Run comprehensive evaluation on a dataset"""
    
    # Load dataset
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty JSON array")
    
    # Initialize components
    embedder = EmbeddingGenerator()
    vector_store = VectorStore()
    llm_generator = LLMGenerator()
    evaluator = EndToEndEvaluator(embedder, vector_store, llm_generator)
    
    # Run evaluation
    print(f"Running evaluation on {len(dataset)} queries...")
    all_results = []
    
    for i, example in enumerate(dataset, 1):
        print(f"Evaluating query {i}/{len(dataset)}: {example['query'][:50]}...")
        
        query = example["query"]
        expected_doc_ids = example["expected_document_ids"]
        reference_answer = example.get("reference_answer")
        
        result = evaluator.evaluate_pipeline(
            query=query,
            expected_doc_ids=expected_doc_ids,
            reference_answer=reference_answer,
            user_id=user_id,
            top_k=top_k
        )
        
        all_results.append(result)
    
    # Aggregate results
    def safe_mean(values):
        filtered = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]  # Filter None and NaN
        return mean(filtered) if filtered else 0.0
    
    # Calculate aggregated metrics
    evaluation_result = EvaluationResult(
        timestamp=datetime.now().isoformat(),
        dataset_size=len(dataset),
        
        # Retrieval metrics
        precision_at_k={k: safe_mean([r.get(f"precision_at_{k}", 0) for r in all_results]) 
                       for k in [1, 3, 5, 10]},
        recall_at_k={k: safe_mean([r.get(f"recall_at_{k}", 0) for r in all_results]) 
                    for k in [1, 3, 5, 10]},
        ndcg_at_k={k: safe_mean([r.get("ndcg", 0) for r in all_results]) 
                  for k in [1, 3, 5, 10]},  # Simplified for now
        mrr=safe_mean([r.get("mrr", 0) for r in all_results]),
        
        # Generation metrics (if available)
        bleu_score=safe_mean([r.get("bleu") for r in all_results if r.get("bleu")]),
        rouge_scores={
            "rouge1": safe_mean([r.get("rouge1_f1") for r in all_results if r.get("rouge1_f1")]),
            "rouge2": safe_mean([r.get("rouge2_f1") for r in all_results if r.get("rouge2_f1")]),
            "rougeL": safe_mean([r.get("rougeL_f1") for r in all_results if r.get("rougeL_f1")])
        },
        bert_score_f1=safe_mean([r.get("bert_score_f1") for r in all_results if r.get("bert_score_f1")]),
        faithfulness_score=safe_mean([r.get("faithfulness_score", 0) for r in all_results]),
        
        # Performance metrics
        avg_retrieval_latency_ms=safe_mean([r.get("retrieval_latency_ms", 0) for r in all_results]),
        avg_generation_latency_ms=safe_mean([r.get("generation_latency_ms", 0) for r in all_results]),
        avg_total_latency_ms=safe_mean([r.get("total_latency_ms", 0) for r in all_results]),
        throughput_qps=1000 / safe_mean([r.get("total_latency_ms", 1000) for r in all_results]),
        
        # Quality metrics
        avg_answer_length=safe_mean([r.get("answer_length", 0) for r in all_results]),
        citation_coverage=safe_mean([1.0 if r.get("source_count", 0) > 0 else 0.0 for r in all_results]),
        source_diversity=safe_mean([r.get("source_diversity", 0) for r in all_results])
    )
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(asdict(evaluation_result), f, indent=2)
        print(f"Results saved to {output_path}")
    
    return evaluation_result


def print_evaluation_report(result: EvaluationResult):
    """Print a comprehensive evaluation report"""
    
    print("\n" + "="*80)
    print("RAG SYSTEM EVALUATION REPORT")
    print("="*80)
    print(f"Timestamp: {result.timestamp}")
    print(f"Dataset Size: {result.dataset_size} queries")
    print()
    
    # Retrieval Metrics
    print("📊 RETRIEVAL PERFORMANCE")
    print("-" * 40)
    for k in [1, 3, 5, 10]:
        if k in result.precision_at_k:
            print(f"Precision@{k}:  {result.precision_at_k[k]:.3f}")
    print()
    for k in [1, 3, 5, 10]:
        if k in result.recall_at_k:
            print(f"Recall@{k}:     {result.recall_at_k[k]:.3f}")
    print(f"MRR:           {result.mrr:.3f}")
    print()
    
    # Generation Quality
    if result.bleu_score or result.rouge_scores or result.bert_score_f1:
        print("🎯 GENERATION QUALITY")
        print("-" * 40)
        if result.bleu_score:
            print(f"BLEU Score:    {result.bleu_score:.3f}")
        if result.rouge_scores:
            print(f"ROUGE-1:       {result.rouge_scores.get('rouge1', 0):.3f}")
            print(f"ROUGE-2:       {result.rouge_scores.get('rouge2', 0):.3f}")
            print(f"ROUGE-L:       {result.rouge_scores.get('rougeL', 0):.3f}")
        if result.bert_score_f1:
            print(f"BERTScore F1:  {result.bert_score_f1:.3f}")
        if result.faithfulness_score:
            print(f"Faithfulness:  {result.faithfulness_score:.3f}")
        print()
    
    # Performance Metrics
    print("⚡ PERFORMANCE METRICS")
    print("-" * 40)
    print(f"Avg Retrieval Latency: {result.avg_retrieval_latency_ms:.1f} ms")
    print(f"Avg Generation Latency: {result.avg_generation_latency_ms:.1f} ms")
    print(f"Avg Total Latency:     {result.avg_total_latency_ms:.1f} ms")
    print(f"Throughput:            {result.throughput_qps:.2f} queries/sec")
    print()
    
    # Quality Metrics
    print("📝 QUALITY METRICS")
    print("-" * 40)
    print(f"Avg Answer Length:     {result.avg_answer_length:.1f} words")
    print(f"Citation Coverage:     {result.citation_coverage:.1%}")
    print(f"Source Diversity:      {result.source_diversity:.2f}")
    print()
    
    # Overall Assessment
    print("🎖️  OVERALL ASSESSMENT")
    print("-" * 40)
    
    # Simple scoring system
    retrieval_score = (result.precision_at_k.get(5, 0) + result.recall_at_k.get(5, 0) + result.mrr) / 3
    performance_score = min(1.0, 1000 / max(result.avg_total_latency_ms, 100))  # Good if < 1s
    quality_score = (result.citation_coverage + min(1.0, result.avg_answer_length / 100)) / 2
    
    overall_score = (retrieval_score + performance_score + quality_score) / 3
    
    print(f"Retrieval Score:   {retrieval_score:.1%}")
    print(f"Performance Score: {performance_score:.1%}")
    print(f"Quality Score:     {quality_score:.1%}")
    print(f"Overall Score:     {overall_score:.1%}")
    
    if overall_score >= 0.8:
        print("✅ EXCELLENT - Production ready")
    elif overall_score >= 0.6:
        print("⚠️  GOOD - Minor improvements needed")
    elif overall_score >= 0.4:
        print("🔧 FAIR - Significant improvements needed")
    else:
        print("❌ POOR - Major improvements required")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive RAG System Evaluation")
    parser.add_argument("--dataset", type=Path, required=True, 
                       help="Path to evaluation dataset JSON file")
    parser.add_argument("--user-id", required=True, 
                       help="Firebase UID that owns the evaluated documents")
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 11),
                       help="Number of documents to retrieve")
    parser.add_argument("--output", type=Path, 
                       help="Path to save evaluation results JSON")
    args = parser.parse_args()
    
    # Check optional dependencies
    missing_deps = []
    if not ROUGE_AVAILABLE:
        missing_deps.append("rouge-score")
    if not BLEU_AVAILABLE:
        missing_deps.append("sacrebleu")
    if not BERT_SCORE_AVAILABLE:
        missing_deps.append("bert-score")
    
    if missing_deps:
        print(f"⚠️  Optional dependencies missing: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        print("Proceeding with basic evaluation...\n")
    
    # Run evaluation
    try:
        result = asyncio.run(run_comprehensive_evaluation(
            dataset_path=args.dataset,
            user_id=args.user_id,
            top_k=args.top_k,
            output_path=args.output
        ))
        
        print_evaluation_report(result)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()