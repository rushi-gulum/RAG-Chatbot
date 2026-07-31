import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = "intfloat/e5-small", device: Optional[str] = None):
        """
        Initialize embedding generator with E5-small model
        
        Args:
            model_name: Hugging Face model name (default: intfloat/e5-small)
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing embedding model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Get model dimensions
            self.embedding_dim = self.model.config.hidden_size
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise

    def _prepare_text_for_e5(self, text: str, prefix: str = "passage: ") -> str:
        """
        Prepare text for E5 model with appropriate prefix
        
        Args:
            text: Input text
            prefix: Prefix for E5 model ("passage: " for documents, "query: " for queries)
            
        Returns:
            Formatted text with prefix
        """
        # Clean text to remove excessive whitespace
        cleaned_text = " ".join(text.strip().split())
        return prefix + cleaned_text

    def _mean_pooling(self, model_output, attention_mask):
        """
        Apply mean pooling to get sentence embeddings
        
        Args:
            model_output: Model output from transformer
            attention_mask: Attention mask from tokenizer
            
        Returns:
            Mean pooled embeddings
        """
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate_embeddings(self, texts: List[str], batch_size: int = 16, normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        # Prepare texts with E5 prefix
        prepared_texts = [self._prepare_text_for_e5(text) for text in texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(prepared_texts), batch_size):
            batch_texts = prepared_texts[i:i + batch_size]
            
            try:
                # Tokenize batch
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,  # E5-small max length
                    return_tensors='pt'
                ).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                    
                    # Normalize embeddings if requested
                    if normalize:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    # Move to CPU and convert to numpy
                    batch_embeddings = embeddings.cpu().numpy()
                    all_embeddings.append(batch_embeddings)
                    
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(prepared_texts) + batch_size - 1)//batch_size}")
                    
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                raise
        
        # Concatenate all batches
        final_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        
        logger.info(f"Generated embeddings shape: {final_embeddings.shape}")
        return final_embeddings

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for processed document chunks from preprocessing module
        
        Args:
            chunks: List of chunk dictionaries from DocumentProcessor.process_document()
                   Format: [{"text": "...", "metadata": {...}}, ...]
            
        Returns:
            List of chunks with embeddings added to each chunk
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []
        
        logger.info(f"Embedding {len(chunks)} chunks...")
        
        # Extract texts from chunks
        texts = []
        valid_indices = []
        
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if text:
                texts.append(text)
                valid_indices.append(i)
            else:
                logger.warning(f"Chunk {i} has empty or missing text")
        
        if not texts:
            logger.warning("No valid texts found in chunks")
            return chunks
        
        # Generate embeddings
        try:
            embeddings = self.generate_embeddings(texts)
            
            # Create new list with embedded chunks
            embedded_chunks = []
            embedding_idx = 0
            
            for i, chunk in enumerate(chunks):
                # Create a copy of the chunk
                chunk_copy = chunk.copy()
                chunk_copy["metadata"] = chunk["metadata"].copy()
                
                if i in valid_indices:
                    # Add embedding to chunk
                    chunk_copy["embedding"] = embeddings[embedding_idx].tolist()
                    
                    # Add embedding metadata
                    chunk_copy["metadata"]["embedding_model"] = self.model_name
                    chunk_copy["metadata"]["embedding_dim"] = self.embedding_dim
                    chunk_copy["metadata"]["embedded_at"] = datetime.now().isoformat()
                    chunk_copy["metadata"]["embedding_status"] = "success"
                    
                    embedding_idx += 1
                    logger.debug(f"Successfully embedded chunk {i} (chunk_id: {chunk_copy['metadata'].get('chunk_id', 'unknown')})")
                else:
                    # Mark as failed embedding
                    chunk_copy["embedding"] = None
                    chunk_copy["metadata"]["embedding_status"] = "failed"
                    chunk_copy["metadata"]["embedding_error"] = "Empty or invalid text"
                
                embedded_chunks.append(chunk_copy)
            
            logger.info(f"Successfully embedded {len(valid_indices)} out of {len(chunks)} chunks")
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding as numpy array
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Prepare query with E5 prefix for queries
        prepared_query = self._prepare_text_for_e5(query, prefix="query: ")
        
        try:
            # Tokenize
            encoded_input = self.tokenizer(
                prepared_query,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Normalize
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                
                # Convert to numpy
                query_embedding = embedding.cpu().numpy().flatten()
            
            logger.info(f"Generated query embedding with shape: {query_embedding.shape}")
            return query_embedding
            
        except Exception as e:
            logger.error(f"Failed to embed query: {str(e)}")
            raise

    def compute_similarity(self, query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and chunk embeddings
        
        Args:
            query_embedding: Query embedding vector
            chunk_embeddings: Array of chunk embeddings
            
        Returns:
            Array of similarity scores
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute cosine similarity
        similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
        return similarities

    def get_embeddings_from_chunks(self, embedded_chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract embedding vectors from embedded chunks
        
        Args:
            embedded_chunks: List of chunks with embeddings
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = []
        for chunk in embedded_chunks:
            if chunk.get("embedding") is not None:
                embeddings.append(chunk["embedding"])
        
        return np.array(embeddings) if embeddings else np.array([])

    def search_similar_chunks(self, query: str, embedded_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for most similar chunks to a query
        
        Args:
            query: Search query
            embedded_chunks: List of chunks with embeddings
            top_k: Number of top results to return
            
        Returns:
            List of top-k most similar chunks with similarity scores
        """
        # Get query embedding
        query_embedding = self.embed_query(query)
        
        # Get chunk embeddings
        chunk_embeddings = self.get_embeddings_from_chunks(embedded_chunks)
        
        if len(chunk_embeddings) == 0:
            logger.warning("No valid embeddings found in chunks")
            return []
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, chunk_embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Create result list with similarity scores
        results = []
        valid_chunk_idx = 0
        
        for chunk_idx, chunk in enumerate(embedded_chunks):
            if chunk.get("embedding") is not None:
                if valid_chunk_idx in top_indices:
                    chunk_copy = chunk.copy()
                    chunk_copy["similarity_score"] = float(similarities[valid_chunk_idx])
                    results.append(chunk_copy)
                valid_chunk_idx += 1
        
        # Sort by similarity score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        logger.info(f"Found {len(results)} similar chunks for query: '{query[:50]}...'")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "max_sequence_length": 512,
            "model_type": "e5-small",
            "normalization": True,
            "prefix_passage": "passage: ",
            "prefix_query": "query: "
        }

# Test function for integration
def test_embedding_with_preprocessing():
    """
    Test function to demonstrate integration between preprocessing and embedding modules
    """
    from preprocessing import DocumentProcessor
    import uuid
    
    # Initialize both processors
    doc_processor = DocumentProcessor()
    embedder = EmbeddingGenerator()
    
    # Test with your exact file structure
    file_path = "uploads/mem.pdf"
    file_id = str(uuid.uuid4())
    filename = "mem.pdf"
    
    try:
        # Step 1: Process document into chunks
        print("Step 1: Processing document into chunks...")
        chunks = doc_processor.process_document(file_path, file_id, filename)
        print(f"Generated {len(chunks)} chunks")
        
        # Step 2: Embed chunks
        print("Step 2: Generating embeddings for chunks...")
        embedded_chunks = embedder.embed_chunks(chunks)
        print(f"Embedded {len(embedded_chunks)} chunks")
        print(embedded_chunks[0])  # Print first embedded chunk for verification
        
        # Step 3: Test search
        print("Step 3: Testing search functionality...")
        query = "artificial intelligence"
        similar_chunks = embedder.search_similar_chunks(query, embedded_chunks, top_k=3)
        
        print(f"\nQuery: '{query}'")
        print(f"Found {len(similar_chunks)} similar chunks:")
        
        for i, chunk in enumerate(similar_chunks):
            print(f"\nRank {i+1} (Similarity: {chunk['similarity_score']:.4f}):")
            print(f"Chunk ID: {chunk['metadata']['chunk_id']}")
            print(f"Preview: {chunk['text'][:100]}...")
        
        return embedded_chunks
        
    except Exception as e:
        print(f"Error in test: {e}")
        return None

if __name__ == "__main__":
    # Run the integration test
    embedded_chunks = test_embedding_with_preprocessing()
