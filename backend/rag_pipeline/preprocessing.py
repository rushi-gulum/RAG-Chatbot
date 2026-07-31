import PyPDF2
import numpy as np
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
from datetime import datetime

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50, max_tokens_per_chunk=512):
        """
        Initialize document processor for RAG pipeline
        
        Args:
            chunk_size: Number of words per chunk
            chunk_overlap: Number of words to overlap between chunks
            max_tokens_per_chunk: Maximum tokens per chunk for embedding models
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens_per_chunk = max_tokens_per_chunk

    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract raw text from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text while preserving semantic meaning
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
        text = re.sub(r'\n+', '\n', text)  # Normalize multiple newlines
        
        # Remove unwanted special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\\@\#\$\%\^\&\*\+\=\~\`]', '', text)
        
        # Remove page markers that were added during extraction
        text = re.sub(r'\n--- Page \d+ ---\n', '\n', text)
        
        # Clean up extra spaces around punctuation
        text = re.sub(r'\s+([\.!?;:])', r'\1', text)
        text = re.sub(r'([\.!?;:])\s+', r'\1 ', text)
        
        # Normalize quotes
        text = re.sub(r"[‘’]", "'", text)
        text = re.sub(r'[“”]', '"', text)

        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text

    def segment_sentences(self, text: str) -> List[str]:
        """
        Simple sentence segmentation to maintain coherent chunks
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting on common punctuation
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Break text into overlapping chunks with metadata
        
        Args:
            text: Clean text to chunk
            metadata: Base metadata to attach to each chunk
            
        Returns:
            List of chunks with metadata
        """
        if not text:
            return []
        
        # Get sentences for better chunking boundaries
        sentences = self.segment_sentences(text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            # If adding this sentence exceeds chunk size, finalize current chunk
            if current_word_count + sentence_word_count > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk_metadata = self._create_chunk_metadata(chunk_text, metadata, len(chunks))
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
                
                # Start new chunk with overlap
                overlap_words = ' '.join(current_chunk).split()[-self.chunk_overlap:]
                current_chunk = overlap_words + sentence_words
                current_word_count = len(current_chunk)
            else:
                # Add sentence to current chunk
                current_chunk.extend(sentence_words)
                current_word_count += sentence_word_count
        
        # Add final chunk if it has content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = self._create_chunk_metadata(chunk_text, metadata, len(chunks))
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunks

    def _create_chunk_metadata(self, chunk_text: str, base_metadata: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
        """
        Create metadata for a text chunk
        
        Args:
            chunk_text: The chunk text
            base_metadata: Base metadata from document
            chunk_index: Index of this chunk in the document
            
        Returns:
            Metadata dictionary
        """
        metadata = base_metadata.copy() if base_metadata else {}
        
        # Add chunk-specific metadata
        metadata.update({
            "chunk_id": str(uuid.uuid4()),
            "chunk_index": chunk_index,
            "chunk_length": len(chunk_text),
            "word_count": len(chunk_text.split()),
            "processed_at": datetime.now().isoformat()
        })
        
        # Extract simple features
        metadata["has_numbers"] = bool(re.search(r'\d', chunk_text))
        metadata["has_emails"] = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', chunk_text))
        metadata["has_urls"] = bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', chunk_text))
        
        return metadata

    def process_document(self, file_path: str, file_id: str, filename: str) -> List[Dict[str, Any]]:
        """
        Full document processing pipeline
        
        Args:
            file_path: Path to document file
            file_id: Unique identifier for the document
            filename: Original filename
            
        Returns:
            List of processed chunks ready for embedding
        """
        try:
            # Step 1: Extract text
            if file_path.lower().endswith('.pdf'):
                raw_text = self.extract_text_from_pdf(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            if not raw_text.strip():
                raise ValueError("No text could be extracted from the document")
            
            # Step 2: Clean text
            clean_text = self.clean_text(raw_text)
            
            # Step 3: Create base metadata
            base_metadata = {
                "file_id": file_id,
                "filename": filename,
                "file_path": file_path,
                "file_type": "pdf",
                "total_characters": len(clean_text),
                "total_words": len(clean_text.split())
            }
            
            # Step 4: Chunk text with metadata
            chunks = self.chunk_text(clean_text, base_metadata)
            
            # Step 5: Validate chunks
            valid_chunks = []
            for chunk in chunks:
                if self._validate_chunk(chunk):
                    valid_chunks.append(chunk)
            
            return valid_chunks
            
        except Exception as e:
            raise Exception(f"Error processing document {filename}: {str(e)}")

    def _validate_chunk(self, chunk: Dict[str, Any]) -> bool:
        """
        Validate that a chunk meets quality criteria
        
        Args:
            chunk: Chunk dictionary
            
        Returns:
            True if chunk is valid
        """
        text = chunk.get("text", "")
        
        # Check minimum length
        if len(text.strip()) < 10:
            return False
        
        # Check minimum word count
        if len(text.split()) < 5:
            return False
        
        # Check that it's not just numbers or special characters
        if not re.search(r'[a-zA-Z]', text):
            return False
        
        return True

    def get_chunk_preview(self, chunk: Dict[str, Any], max_length: int = 100) -> str:
        """
        Get a preview of chunk text for debugging/display
        
        Args:
            chunk: Chunk dictionary
            max_length: Maximum length of preview
            
        Returns:
            Preview string
        """
        text = chunk.get("text", "")
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def get_processing_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about processed chunks
        
        Args:
            chunks: List of processed chunks
            
        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {"total_chunks": 0}
        
        word_counts = [chunk["metadata"]["word_count"] for chunk in chunks]
        char_counts = [chunk["metadata"]["chunk_length"] for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_words_per_chunk": np.mean(word_counts),
            "min_words_per_chunk": min(word_counts),
            "max_words_per_chunk": max(word_counts),
            "avg_chars_per_chunk": np.mean(char_counts),
            "total_words": sum(word_counts),
            "total_characters": sum(char_counts)
        }




if __name__ == "__main__":
    processor = DocumentProcessor()

    file_path = f"uploads/mem.pdf"  # your test PDF
    file_id = str(uuid.uuid4())       # temporary ID for testing
    filename = "mem.pdf"

    chunks = processor.process_document(file_path, file_id, filename)

    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]): 
        print("\n")# preview first 3 chunks
        print(f"Chunk {i} preview:", processor.get_chunk_preview(chunk))
        print("Metadata:", chunk["metadata"])