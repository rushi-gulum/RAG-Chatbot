"""
Heuristic-based RAGAS evaluation 
==================================

This provides RAGAS-like evaluation metrics without requiring LLM calls,
useful when evaluation LLMs are not accessible or as a fallback.

Uses text similarity, keyword matching, and length-based heuristics
to approximate RAGAS metrics.
"""

import re
from dataclasses import dataclass
from typing import List
from statistics import mean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging


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

class HeuristicRAGASEvaluator:
    """Heuristic-based RAGAS evaluation using text similarity metrics"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
    
    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Heuristic faithfulness: How much of answer content appears in contexts
        """
        if not contexts or not answer:
            return 0.0
        
        try:
            # Combine contexts
            context_text = " ".join(contexts)
            
            # Extract key phrases and entities from answer
            answer_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
            context_words = set(re.findall(r'\b\w{4,}\b', context_text.lower()))
            
            if not answer_words:
                return 0.5
            
            # Calculate word overlap
            overlap = len(answer_words.intersection(context_words))
            total_answer_words = len(answer_words)
            
            word_overlap_score = overlap / total_answer_words
            
            # TF-IDF similarity
            try:
                vectors = self.vectorizer.fit_transform([answer, context_text])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            except:
                similarity = 0.0
            
            # Combine scores with weight on word overlap (more reliable)
            faithfulness = (0.7 * word_overlap_score + 0.3 * similarity)
            
            return max(0.0, min(1.0, faithfulness))
            
        except Exception as e:
            logging.warning(f"Heuristic faithfulness calculation failed: {e}")
            return 0.5
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Heuristic answer relevancy: Similarity between question and answer
        """
        if not question or not answer:
            return 0.0
        
        try:
            # Extract key terms
            question_words = set(re.findall(r'\b\w{3,}\b', question.lower()))
            answer_words = set(re.findall(r'\b\w{3,}\b', answer.lower()))
            
            # Basic keyword overlap
            if question_words:
                overlap_score = len(question_words.intersection(answer_words)) / len(question_words)
            else:
                overlap_score = 0.0
            
            # TF-IDF similarity
            try:
                vectors = self.vectorizer.fit_transform([question, answer])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            except:
                similarity = 0.0
            
            # Length penalty - very short answers are likely not relevant
            length_penalty = 1.0
            if len(answer.split()) < 10:
                length_penalty = 0.7
            elif len(answer.split()) < 5:
                length_penalty = 0.5
            
            relevancy = (0.4 * overlap_score + 0.6 * similarity) * length_penalty
            
            return max(0.0, min(1.0, relevancy))
            
        except Exception as e:
            logging.warning(f"Heuristic answer relevancy calculation failed: {e}")
            return 0.5
    
    def evaluate_context_precision(self, question: str, contexts: List[str]) -> float:
        """
        Heuristic context precision: How relevant each context is to question
        """
        if not question or not contexts:
            return 0.0
        
        try:
            question_words = set(re.findall(r'\b\w{3,}\b', question.lower()))
            precision_scores = []
            
            for context in contexts[:5]:  # Limit to top 5
                if not context.strip():
                    precision_scores.append(0.0)
                    continue
                
                context_words = set(re.findall(r'\b\w{3,}\b', context.lower()))
                
                # Keyword overlap
                if question_words:
                    overlap_score = len(question_words.intersection(context_words)) / len(question_words)
                else:
                    overlap_score = 0.0
                
                # TF-IDF similarity
                try:
                    vectors = self.vectorizer.fit_transform([question, context])
                    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                except:
                    similarity = 0.0
                
                # Context length bonus - longer contexts tend to be more informative
                length_bonus = min(1.0, len(context.split()) / 100)  # Bonus for contexts up to 100 words
                
                precision = (0.5 * overlap_score + 0.5 * similarity) * (0.8 + 0.2 * length_bonus)
                precision_scores.append(max(0.0, min(1.0, precision)))
            
            return mean(precision_scores) if precision_scores else 0.0
            
        except Exception as e:
            logging.warning(f"Heuristic context precision calculation failed: {e}")
            return 0.5
    
    def evaluate_context_recall(self, question: str, answer: str, contexts: List[str]) -> float:
        """
        Heuristic context recall: How well contexts cover the information in answer
        """
        if not question or not answer or not contexts:
            return 0.0
        
        try:
            # Extract important terms from answer
            answer_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
            
            # Combine all contexts
            all_context_text = " ".join(contexts)
            context_words = set(re.findall(r'\b\w{4,}\b', all_context_text.lower()))
            
            if not answer_words:
                return 0.5
            
            # Calculate coverage
            covered_words = answer_words.intersection(context_words)
            coverage_ratio = len(covered_words) / len(answer_words)
            
            # TF-IDF based coverage
            try:
                vectors = self.vectorizer.fit_transform([answer, all_context_text])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            except:
                similarity = 0.0
            
            # Context diversity bonus - more contexts might cover more information
            diversity_bonus = min(1.0, len(contexts) / 10)  # Up to 10 contexts
            
            recall = (0.6 * coverage_ratio + 0.4 * similarity) * (0.9 + 0.1 * diversity_bonus)
            
            return max(0.0, min(1.0, recall))
            
        except Exception as e:
            logging.warning(f"Heuristic context recall calculation failed: {e}")
            return 0.5
    
    async def evaluate_query(self, question: str, answer: str, contexts: List[str]) -> dict:
        """
        Evaluate a single query with heuristic RAGAS metrics
        """
        
        faithfulness = self.evaluate_faithfulness(answer, contexts)
        relevancy = self.evaluate_answer_relevancy(question, answer)
        precision = self.evaluate_context_precision(question, contexts)
        recall = self.evaluate_context_recall(question, answer, contexts)
        
        return RAGASMetrics(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=precision,
            context_recall=recall
        )


# Test the heuristic evaluator
if __name__ == "__main__":
    evaluator = HeuristicRAGASEvaluator()
    
    # Test data
    question = "What is LinkedIn growth strategy?"
    answer = "LinkedIn growth strategy involves building technical authority, sharing valuable content, and networking with professionals in your industry."
    contexts = [
        "LinkedIn is a professional networking platform where professionals can connect and build their brand.",
        "For tech professionals, LinkedIn growth requires sharing technical insights and engaging with the community.",
        "Building authority on LinkedIn involves consistent posting and thought leadership."
    ]
    
    print("🧪 Testing Heuristic RAGAS Evaluator")
    print("=" * 40)
    
    import asyncio
    
    async def test():
        metrics = await evaluator.evaluate_query(question, answer, contexts)
        
        print(f"Question: {question}")
        print(f"Answer: {answer[:100]}...")
        print(f"Contexts: {len(contexts)}")
        print()
        print("📊 HEURISTIC RAGAS METRICS")
        print("-" * 30)
        print(f"Faithfulness:     {metrics.faithfulness:.3f}")
        print(f"Answer Relevancy: {metrics.answer_relevancy:.3f}")
        print(f"Context Precision: {metrics.context_precision:.3f}")
        print(f"Context Recall:   {metrics.context_recall:.3f}")
        print(f"Overall Score:    {metrics.overall_score:.3f}")
    
    asyncio.run(test())