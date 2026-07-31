"""
LLM Integration for RAG Pipeline using Groq API
Supports Llama-3-8b-instant for generating responses from retrieved context
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional
from groq import Groq
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMGenerator:
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "llama-3.1-8b-instant",
        max_tokens: int = 1024,
        temperature: float = 0.7
    ):
        """
        Initialize Groq LLM for RAG response generation
        
        Args:
            api_key: Groq API key (if None, will try to get from environment)
            model_name: Groq model to use
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0 = deterministic, 1.0 = creative)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Get API key
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not provided. Set GROQ_API_KEY environment variable or pass api_key parameter."
            )
        
        try:
            # Initialize Groq client
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq LLM initialized successfully with model: {model_name}")
            
            # Test the connection
            self._test_connection()
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {str(e)}")
            raise
    
    def _test_connection(self):
        """Test Groq API connection"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0.1
            )
            logger.info("Groq API connection test successful")
        except Exception as e:
            logger.warning(f"Groq API connection test failed: {str(e)}")
    
    def generate_response(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response using retrieved context chunks
        
        Args:
            query: User's question
            retrieved_chunks: List of relevant chunks from vector search
            system_prompt: Optional custom system prompt
            
        Returns:
            Dictionary with generated response and metadata
        """
        if not retrieved_chunks:
            return {
                "response": "I couldn't find any relevant information to answer your question. Please try rephrasing your query or check if documents are properly uploaded.",
                "sources": [],
                "model_used": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        
        try:
            # Prepare context from retrieved chunks
            context = self._prepare_context(retrieved_chunks)
            
            # Create system prompt
            if not system_prompt:
                system_prompt = self._create_default_system_prompt()
            
            # Create user prompt with query and context
            user_prompt = self._create_user_prompt(query, context)
            
            # Generate response
            logger.info(f"Generating response for query: {query[:50]}...")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False
            )
            
            # Extract response
            generated_text = response.choices[0].message.content
            
            # Extract which sources were actually cited
            cited_sources = self._extract_cited_sources(generated_text, retrieved_chunks)
            
            # Prepare sources information (only cited ones)
            sources = cited_sources
            
            # Get token usage
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            result = {
                "response": generated_text,
                "sources": sources,
                "model_used": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "token_usage": token_usage,
                "query": query
            }
            
            logger.info(f"Response generated successfully. Tokens used: {token_usage['total_tokens']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return {
                "response": f"I encountered an error while generating a response: {str(e)}",
                "sources": self._prepare_sources(retrieved_chunks),
                "model_used": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "error": str(e)
            }
    
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            similarity_score = chunk.get("similarity_score", 0.0)
            
            # Get document information
            filename = metadata.get("filename", "Unknown Document")
            chunk_index = metadata.get("chunk_index", "?")
            
            # Add chunk with source information
            context_parts.append(
                f"**Source {i}** (From: {filename}, Chunk: {chunk_index}, Relevance: {similarity_score:.1%}):\n{text}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _prepare_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare sources information for response"""
        sources = []
        
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            sources.append({
                "filename": metadata.get("filename", "Unknown Document"),
                "chunk_index": metadata.get("chunk_index", 0),
                "similarity_score": chunk.get("similarity_score", 0.0),
                "chunk_id": chunk.get("chunk_id", "")
            })
        
        return sources
    
    def _extract_cited_sources(self, generated_text: str, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract only the sources that were actually cited in the response"""
        # Find all citation numbers in the generated text
        citation_pattern = r'\[(\d+)\]'
        cited_numbers = set()
        
        for match in re.finditer(citation_pattern, generated_text):
            citation_num = int(match.group(1))
            cited_numbers.add(citation_num)
        
        # Return only the sources that were cited (convert to 0-based indexing)
        cited_sources = []
        for citation_num in sorted(cited_numbers):
            # Citation numbers are 1-based, but our list is 0-based
            index = citation_num - 1
            if 0 <= index < len(retrieved_chunks):
                chunk = retrieved_chunks[index]
                metadata = chunk.get("metadata", {})
                cited_sources.append({
                    "filename": metadata.get("filename", "Unknown Document"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "similarity_score": chunk.get("similarity_score", 0.0),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "citation_number": citation_num
                })
        
        return cited_sources
    
    def _create_default_system_prompt(self) -> str:
        """Create default system prompt for RAG responses"""
        return """You are a helpful AI assistant that answers questions based on the provided context from documents.

INSTRUCTIONS:
1. Use ONLY the provided context to answer. Do not invent information.
2. If the context is insufficient, say so clearly and suggest what info is missing.
3. When referencing information from sources, use numbered citations [1], [2], [3], etc.
4. If there's conflicting info, acknowledge it instead of ignoring.
5. Always format the response in a clean, structured way.

CITATION RULES:
- Each source in the context is numbered (Source 1, Source 2, etc.)
- When you reference information from Source 1, add [1] at the end of that sentence
- When you reference information from Source 2, add [2] at the end of that sentence
- You can use multiple citations in one sentence like [1][2] if referencing multiple sources
- ONLY cite sources that you actually reference in your answer

FORMAT YOUR RESPONSE:
- **Direct Answer**: Short, to-the-point definition/answer with citations [1], [2], etc.
- **Explanation**: 
   • Use bullet points or numbered lists for clarity with appropriate citations
   • Highlight important terms in **bold**
   • Keep it concise but informative
   • Add citations [1], [2], etc. after statements that reference specific sources
- If no relevant info found, state "No relevant info found in uploaded documents. Answer provided from general knowledge."

STYLE:
- Be professional but simple
- Avoid large unstructured paragraphs
- Ensure readability for both technical and non-technical users
- Use citations [1], [2], etc. consistently throughout your response
"""
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        """Create user prompt with query and context"""
        return f"""QUESTION: {query}

CONTEXT FROM DOCUMENTS:
{context}

Please answer the question based on the provided context."""
    
    def generate_streaming_response(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        system_prompt: Optional[str] = None
    ):
        """
        Generate streaming response (generator function)
        
        Args:
            query: User's question
            retrieved_chunks: List of relevant chunks from vector search
            system_prompt: Optional custom system prompt
            
        Yields:
            Streaming response chunks
        """
        if not retrieved_chunks:
            yield {
                "content": "I couldn't find any relevant information to answer your question.",
                "done": True
            }
            return
        
        try:
            # Prepare context and prompts
            context = self._prepare_context(retrieved_chunks)
            if not system_prompt:
                system_prompt = self._create_default_system_prompt()
            user_prompt = self._create_user_prompt(query, context)
            
            # Generate streaming response
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield {
                        "content": chunk.choices[0].delta.content,
                        "done": False
                    }
            
            # Send completion signal with sources
            yield {
                "content": "",
                "done": True,
                "sources": self._prepare_sources(retrieved_chunks),
                "model_used": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Streaming response failed: {str(e)}")
            yield {
                "content": f"Error generating response: {str(e)}",
                "done": True,
                "error": str(e)
            }


# Test function
def test_llm_integration():
    """Test LLM integration with sample data"""
    
    # Sample retrieved chunks for testing
    sample_chunks = [
        {
            "chunk_id": "test_1",
            "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans. AI research focuses on developing algorithms that can perform tasks that typically require human intelligence.",
            "similarity_score": 0.95,
            "metadata": {
                "filename": "ai_basics.pdf",
                "chunk_index": 1,
                "document_id": "test_doc_1"
            }
        },
        {
            "chunk_id": "test_2", 
            "text": "Machine learning is a subset of AI that enables computers to learn and make decisions from data without being explicitly programmed for every task.",
            "similarity_score": 0.87,
            "metadata": {
                "filename": "ml_overview.pdf",
                "chunk_index": 3,
                "document_id": "test_doc_2"
            }
        }
    ]
    
    try:
        # Initialize LLM
        llm = LLMGenerator()
        
        # Test query
        query = "What is artificial intelligence and how does it relate to machine learning?"
        
        # Generate response
        result = llm.generate_response(query, sample_chunks)
        
        print("=== LLM Integration Test ===")
        print(f"Query: {query}")
        print(f"Response: {result['response']}")
        print(f"Sources: {len(result['sources'])} documents")
        print(f"Tokens used: {result['token_usage']['total_tokens']}")
        print("=== Test Complete ===")
        
        return True
        
    except Exception as e:
        print(f"LLM test failed: {e}")
        return False


if __name__ == "__main__":
    test_llm_integration()