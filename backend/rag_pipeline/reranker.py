"""
Cross-Encoder Reranker for RAG Pipeline
Improves context precision by reordering retrieved chunks by true semantic relevance
"""

import logging
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)


class HeuristicReranker:
    """
    Heuristic-based reranker using TF-IDF and semantic similarity
    
    This is a fast, lightweight alternative to neural rerankers like Cohere Rerank
    or BGE-Reranker, suitable for production use without additional API costs.
    """
    
    def __init__(self):
        """Initialize the reranker"""
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=2000,
            ngram_range=(1, 2),  # Include bigrams for better semantic capture
            min_df=1,  # Keep all terms for small context sets
            lowercase=True
        )
    
    def rerank(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks by semantic relevance to query
        
        Args:
            query: The user's search query
            chunks: List of retrieved chunks from vector search
            top_k: Number of top chunks to return
            
        Returns:
            Reranked list of top_k chunks with relevance scores
        """
        if not chunks or not query:
            return chunks[:top_k]
        
        if len(chunks) <= top_k:
            # If we have fewer chunks than requested, return all with scores
            return self._add_relevance_scores(query, chunks)
        
        try:
            # Extract text content from chunks
            chunk_texts = []
            for chunk in chunks:
                # Combine title and content for better matching
                text = chunk.get('text', '')
                if chunk.get('title'):
                    text = f"{chunk['title']} {text}"
                chunk_texts.append(text)
            
            # Create corpus with query + all chunks
            corpus = [query] + chunk_texts
            
            # Compute TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity between query (index 0) and all chunks
            query_vector = tfidf_matrix[0:1]
            chunk_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
            
            # Create tuples of (similarity_score, original_index, chunk)
            scored_chunks = []
            for i, (similarity, chunk) in enumerate(zip(similarities, chunks)):
                # Add additional scoring factors
                relevance_score = self._calculate_enhanced_score(
                    query, chunk, similarity, i
                )
                
                scored_chunks.append((relevance_score, i, chunk))
            
            # Sort by relevance score (descending) and take top_k
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            
            # Extract reranked chunks and add scores
            reranked = []
            for score, original_idx, chunk in scored_chunks[:top_k]:
                chunk_with_score = chunk.copy()
                chunk_with_score['rerank_score'] = float(score)
                chunk_with_score['original_rank'] = original_idx
                reranked.append(chunk_with_score)
            
            logger.info(f"Reranked {len(chunks)} chunks, returning top {len(reranked)}")
            return reranked
            
        except Exception as e:
            logger.warning(f"Reranking failed, returning original order: {e}")
            return chunks[:top_k]
    
    def _calculate_enhanced_score(
        self, 
        query: str, 
        chunk: Dict[str, Any], 
        base_similarity: float, 
        original_position: int
    ) -> float:
        """
        Calculate enhanced relevance score with additional factors
        
        Args:
            query: Search query
            chunk: Document chunk
            base_similarity: TF-IDF cosine similarity
            original_position: Original position from vector search
            
        Returns:
            Enhanced relevance score
        """
        score = base_similarity
        
        # Factor 1: Query term frequency in chunk
        query_terms = set(query.lower().split())
        chunk_text = chunk.get('text', '').lower()
        
        if query_terms:
            term_matches = sum(1 for term in query_terms if term in chunk_text)
            term_coverage = term_matches / len(query_terms)
            score += 0.1 * term_coverage
        
        # Factor 2: Length penalty/bonus (prefer chunks with moderate length)
        text_length = len(chunk.get('text', ''))
        if 200 <= text_length <= 800:  # Sweet spot for context
            score += 0.05
        elif text_length < 100:  # Too short, might lack context
            score -= 0.1
        elif text_length > 1500:  # Too long, might be unfocused
            score -= 0.05
        
        # Factor 3: Slight preference for higher-ranked vector results
        # (But allow reranking to override if semantic match is much better)
        position_bonus = max(0, (10 - original_position) * 0.005)
        score += position_bonus
        
        # Factor 4: Title relevance bonus
        title = chunk.get('title', '')
        if title and any(term in title.lower() for term in query.lower().split()):
            score += 0.15
        
        return score
    
    def _add_relevance_scores(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add relevance scores to chunks without reordering"""
        if not chunks:
            return chunks
        
        try:
            chunk_texts = [chunk.get('text', '') for chunk in chunks]
            corpus = [query] + chunk_texts
            
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            query_vector = tfidf_matrix[0:1]
            chunk_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
            
            scored_chunks = []
            for i, (chunk, similarity) in enumerate(zip(chunks, similarities)):
                chunk_with_score = chunk.copy()
                chunk_with_score['rerank_score'] = float(similarity)
                chunk_with_score['original_rank'] = i
                scored_chunks.append(chunk_with_score)
            
            return scored_chunks
            
        except Exception as e:
            logger.warning(f"Score calculation failed: {e}")
            return chunks


def test_reranker():
    """Test the reranker with sample data"""
    reranker = HeuristicReranker()
    
    query = "LinkedIn growth strategy for engineers"
    chunks = [
        {
            "text": "Social media marketing tips for general audiences and broad engagement tactics",
            "title": "Social Media Basics"
        },
        {
            "text": "LinkedIn algorithm changes in 2026 specifically affect technical professionals and engineers seeking to build authority",
            "title": "LinkedIn Algorithm for Tech Professionals"
        },
        {
            "text": "Facebook advertising strategies for e-commerce businesses and product promotion",
            "title": "Facebook Ads Guide"
        },
        {
            "text": "Engineering career development on LinkedIn requires strategic content creation and networking",
            "title": "LinkedIn for Engineers"
        }
    ]
    
    reranked = reranker.rerank(query, chunks, top_k=3)
    
    print("🔄 Reranking Test Results")
    print("=" * 40)
    for i, chunk in enumerate(reranked, 1):
        print(f"{i}. {chunk['title']} (Score: {chunk['rerank_score']:.3f})")
        print(f"   Original Rank: {chunk['original_rank']}")
        print()


if __name__ == "__main__":
    test_reranker()